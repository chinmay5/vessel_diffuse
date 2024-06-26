import time
import os
import math
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb
from torch import nn

from midi.analysis import non_molecular_visualization
from midi.analysis.non_molecular_visualization import visualize_chains, visualize
from midi.diffusion.two_stage_noise_model import MarginalTwoStageNoiseModel, UniformTwoStageNoiseModel
from midi.metrics.non_molecular_metrics import NonMolecularSamplingMetrics
from midi.models.midi_transformer_model import EquivariantGraphTransformer
from midi.models.transformer_model import GraphTransformer
# from midi.models.egnn_ablation import GraphTransformer
# print("RUNNING ABLATION")
from midi.diffusion.noise_model import DiscreteUniformTransition, MarginalUniformTransition
from midi.diffusion import diffusion_utils
from midi.diffusion.diffusion_utils import mask_distributions, sum_except_batch
from midi.metrics.train_metrics import TrainLoss, ValLoss, TrainMolecularMetrics
from midi.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from midi import utils
import midi.metrics.abstract_metrics as custom_metrics
from midi.diffusion.extra_features import ExtraFeatures
from midi.datasets.adaptive_loader import effective_batch_size
from midi.utils import save_graphs


class FullDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos):
        super().__init__()
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        self.node_dist = nodes_dist
        self.dataset_infos = dataset_infos
        self.extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)
        self.output_dims = dataset_infos.output_dims

        # Train metrics
        self.train_loss = TrainLoss(lambda_train=self.cfg.model.lambda_train
        if hasattr(self.cfg.model, "lambda_train") else self.cfg.train.lambda0,
                                    cfg=cfg, dataset_infos=dataset_infos
                                    )
        self.train_metrics = TrainMolecularMetrics(dataset_infos)

        # Val Metrics
        self.val_metrics = torchmetrics.MetricCollection([custom_metrics.XKl(),
                                                          custom_metrics.EKl(), custom_metrics.ChargesKl()])
        self.val_loss = ValLoss(lambda_train=self.cfg.model.lambda_train
        if hasattr(self.cfg.model, "lambda_train") else self.cfg.train.lambda0,
                                cfg=cfg, dataset_infos=dataset_infos)
        self.val_nll = NLL()

        # Test metrics
        self.test_metrics = torchmetrics.MetricCollection([custom_metrics.XKl(),
                                                           custom_metrics.EKl(), custom_metrics.ChargesKl()])
        self.test_nll = NLL()
        # Let us also put test loss for completeness
        self.test_loss = ValLoss(lambda_train=self.cfg.model.lambda_train
        if hasattr(self.cfg.model, "lambda_train") else self.cfg.train.lambda0,
                                 cfg=cfg, dataset_infos=dataset_infos)
        self.save_hyperparameters(ignore=['train_metrics', 'dataset_infos'])
        self.directed = cfg.dataset.get("is_directed", False)
        print(f"MODEL DIRECTED = {self.directed}")

        # Select either the equivariant or the updated transformer model
        if cfg.model.get("is_equivariant", False):
            print("USING MIDI EQUIVARIANT MODEL")
            self.model = EquivariantGraphTransformer(input_dims=self.input_dims,
                                                     n_layers=cfg.model.n_layers,
                                                     hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                                     hidden_dims=cfg.model.hidden_dims,
                                                     output_dims=self.output_dims,
                                                     # Whether we generate directed graphs or not
                                                     is_directed=self.directed,
                                                     cfg=cfg)
            if cfg.model.transition == 'uniform':
                self.noise_model = DiscreteUniformTransition(output_dims=self.output_dims,
                                                             cfg=cfg)
            else:
                self.noise_model = MarginalUniformTransition(x_marginals=self.dataset_infos.atom_types,
                                                             e_marginals=self.dataset_infos.edge_types,
                                                             charges_marginals=self.dataset_infos.charges_marginals,
                                                             y_classes=self.output_dims.y,
                                                             cfg=cfg)

        else:
            print("USING OUR CUSTOM MODEL")
            self.model = GraphTransformer(input_dims=self.input_dims,
                                          n_layers=cfg.model.n_layers,
                                          hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                          hidden_dims=cfg.model.hidden_dims,
                                          output_dims=self.output_dims,
                                          # Whether we generate directed graphs or not
                                          is_directed=self.directed,
                                          cfg=cfg)
            if cfg.model.transition == 'uniform':
                print("Using Uniform LaPlacian transitions")
                self.noise_model = UniformTwoStageNoiseModel(output_dims=self.output_dims, cfg=cfg,
                                                             dataset_infos=dataset_infos,
                                                             x_marginals=self.dataset_infos.atom_types,
                                                             e_marginals=self.dataset_infos.edge_types,
                                                             charges_marginals=self.dataset_infos.charges_marginals,
                                                             y_classes=self.output_dims.y, )
            else:
                self.noise_model = MarginalTwoStageNoiseModel(x_marginals=self.dataset_infos.atom_types,
                                                              e_marginals=self.dataset_infos.edge_types,
                                                              charges_marginals=self.dataset_infos.charges_marginals,
                                                              y_classes=self.output_dims.y,
                                                              cfg=cfg, dataset_infos=dataset_infos)

        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        z_t = self.noise_model.apply_noise(dense_data)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)

        loss, tl_log_dict = self.train_loss(masked_pred=pred, masked_true=dense_data,
                                            log=i % self.log_every_steps == 0, epoch=self.trainer.current_epoch)

        tm_log_dict = self.train_metrics(masked_pred=pred, masked_true=dense_data,
                                         log=i % self.log_every_steps == 0)
        if tl_log_dict is not None:
            self.log_dict(tl_log_dict, batch_size=self.BS)
        if tm_log_dict is not None:
            self.log_dict(tm_log_dict, batch_size=self.BS)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_metrics.reset()
        self.val_loss.reset()

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        z_t = self.noise_model.apply_noise(dense_data)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        # Computing extra validation loss terms
        self.val_loss(masked_pred=pred, masked_true=dense_data,
                      log=i % self.log_every_steps == 0)
        nll, log_dict = self.compute_val_loss(pred, z_t, clean_data=dense_data, test=False)
        return {'loss': nll}, log_dict

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_metrics.compute()]
        log_dict = {"val/epoch_NLL": metrics[0],
                    
                    "val/X_kl": metrics[1]['XKl'] * self.T,
                    "val/E_kl": metrics[1]['EKl'] * self.T,
                    "val/charges_kl": metrics[1]['ChargesKl'] * self.T
                    }
        vle_log = self.val_loss.log_epoch_metrics()
        self.print(f"VAL Epoch {self.current_epoch} finished: -- "
                   f"X: {vle_log['val_epoch/x_CE'] :.2f} --"
                   f" E: {vle_log['val_epoch/E_CE'] :.2f} --"
                   f" charges: {vle_log['val_epoch/charges_CE'] :.2f} --"
                   f" y: {vle_log['val_epoch/y_CE'] :.2f}")
        # Now we would go ahead and merge the two dictionaries
        # Such a convenient method for merging two dictionaries
        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:,.2f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = ''.join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])
        # We would add the two metrics and log them together in tensorboard
        log_dict.update(vle_log)
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)
        if wandb.run:
            wandb.log(log_dict)

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print(f'Val loss: {val_nll:,.4f} \t Best val loss:  {self.best_val_nll:,.4f}\n')

        self.val_counter += 1
        if self.name == "debug" or (self.val_counter % self.cfg.general.sample_every_val == 0 and
                                    self.current_epoch > 0):
            self.print(f"Sampling start")
            start = time.time()
            gen = self.cfg.general
            samples = self.sample_n_graphs(samples_to_generate=math.ceil(gen.samples_to_generate / max(gen.gpus, 1)),
                                           chains_to_save=gen.chains_to_save if self.local_rank == 0 else 0,
                                           samples_to_save=gen.samples_to_save if self.local_rank == 0 else 0,
                                           test=False)
            del samples
            torch.cuda.empty_cache()
        self.print(f"Val epoch {self.current_epoch} ends")

    def on_test_epoch_start(self):
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        self.test_nll.reset()
        self.test_metrics.reset()
        self.test_loss.reset()

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        z_t = self.noise_model.apply_noise(dense_data)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        nll, log_dict = self.compute_val_loss(pred, z_t, clean_data=dense_data, test=True)
        # We also compute the test loss for completeness
        self.test_loss(masked_pred=pred, masked_true=dense_data,
                       log=i % self.log_every_steps == 0)
        return {'loss': nll}, log_dict

    @torch.enable_grad()
    @torch.inference_mode(False)
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_metrics.compute()]
        test_nll = metrics[0]
        print(f'Test loss: {test_nll :.4f}')
        log_dict = {"test/epoch_NLL": metrics[0],
                    
                    "test/X_kl": metrics[1]['XKl'] * self.T,
                    "test/E_kl": metrics[1]['EKl'] * self.T,
                    "test/charges_kl": metrics[1]['ChargesKl'] * self.T
                    }
        # We include the extra metric
        tle_log = self.test_loss.log_epoch_metrics()
        # Let us update the keys.
        # This is a bit hacky, but perhaps, it is fine for the time being
        tle_log_new = {}
        for key, value in tle_log.items():
            new_key = key.replace("val", "test")
            tle_log_new[new_key] = value
        self.print(f"Epoch {self.current_epoch} finished -- "
                   f"X: {tle_log_new['test_epoch/x_CE'] :.2f} --"
                   f" E: {tle_log_new['test_epoch/E_CE'] :.2f} --"
                   f" y: {tle_log_new['test_epoch/y_CE'] :.2f}")
        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.4f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = ''.join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])
        # Now we would go ahead and merge the two dictionaries
        # Such a convenient method for merging two dictionaries
        log_dict.update(tle_log_new)
        self.log_dict(log_dict, sync_dist=True)
        if wandb.run:
            wandb.log(log_dict)

        print(f"Sampling start on GR{self.global_rank}")
        start = time.time()
        print(f"Samples to generate: {self.cfg.general.final_model_samples_to_generate}")
        print(f"Samples to save: {self.cfg.general.final_model_samples_to_save}")
        samples = self.sample_n_graphs(samples_to_generate=self.cfg.general.final_model_samples_to_generate,
                                       chains_to_save=self.cfg.general.final_model_chains_to_save,
                                       samples_to_save=self.cfg.general.final_model_samples_to_save,
                                       test=True)
        print("Saving the generated graphs")
        save_graphs(samples)
        print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
        print(f"Test ends.")
        del samples
        torch.cuda.empty_cache()

    def kl_prior(self, clean_data, node_mask):
        """Computes the KL between q(z^T | x) and the prior p(z^T) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((clean_data.X.size(0), 1), dtype=torch.long, device=clean_data.X.device)
        Ts = self.T * ones
        Qtb = self.noise_model.get_Qt_bar(t_int=Ts)

        # Compute transition probabilities
        probX = clean_data.X @ Qtb.X + 1e-7  # (bs, n, dx_out)
        probE = clean_data.E @ Qtb.E.unsqueeze(1) + 1e-7  # (bs, n, n, de_out)
        probc = clean_data.charges @ Qtb.charges + 1e-7
        probX = probX / probX.sum(dim=-1, keepdims=True)
        probE = probE / probE.sum(dim=-1, keepdims=True)
        assert probX.shape == clean_data.X.shape

        bs, n, _ = probX.shape
        limit_dist = self.noise_model.get_limit_dist().device_as(probX)

        # Set masked rows , so it doesn't contribute to loss
        probX[~node_mask] = limit_dist.X.float()
        probc[~node_mask] = limit_dist.charges.float()
        diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
        probE[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = limit_dist.E.float()

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist.X[None, None, :], reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist.E[None, None, None, :], reduction='none')
        print(f"This value should be low:-{kl_distance_E.mean()=}")
        kl_distance_c = F.kl_div(input=probc.log(), target=limit_dist.charges[None, None, :], reduction='none')
        return (sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E) +
                + sum_except_batch(kl_distance_c))

    def compute_Lt(self, clean_data, pred, z_t, s_int, node_mask, logger_metric):
        # TODO: ideally all probabilities should be computed in log space
        t_int = z_t.t_int
        pred = utils.PlaceHolder(X=F.softmax(pred.X, dim=-1), charges=F.softmax(pred.charges, dim=-1),
                                 E=F.softmax(pred.E, dim=-1), pos=pred.pos, node_mask=clean_data.node_mask, y=None,
                                 directed=self.directed)

        Qtb = self.noise_model.get_Qt_bar(z_t.t_int)
        Qsb = self.noise_model.get_Qt_bar(s_int)
        Qt = self.noise_model.get_Qt(t_int)

        # Compute distributions to compare with KL
        bs, n, d = clean_data.X.shape
        prob_true = diffusion_utils.posterior_distributions(clean_data=clean_data, noisy_data=z_t,
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb, is_directed=self.directed)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(clean_data=pred, noisy_data=z_t,
                                                            Qt=Qt, Qsb=Qsb, Qtb=Qtb, is_directed=self.directed)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true = diffusion_utils.mask_distributions(prob_true, node_mask)
        prob_pred = diffusion_utils.mask_distributions(prob_pred, node_mask)

        metrics = logger_metric(prob_pred, prob_true)
        return self.T * (metrics['XKl'] + metrics['EKl'] + metrics['ChargesKl'])

    def compute_val_loss(self, pred, z_t, clean_data, test):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        s_int = t_int - 1

        # Select test or validation metric
        logger_metric = self.test_metrics if test else self.val_metrics
        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(clean_data, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(clean_data, pred, z_t, s_int, node_mask, logger_metric)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t
        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        log_dict = {"kl prior": kl_prior.mean(),
                    "Estimator loss terms": loss_all_t.mean(),
                    "log_pn": log_pN.mean(),
                    'test_nll' if test else 'val_nll': nll}
        return nll, log_dict

    def compute_train_nll_loss(self, pred, z_t, clean_data):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           The method is designed for the training set.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        s_int = t_int - 1

        logger_metric = self.train_metrics
        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(clean_data, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(clean_data, pred, z_t, s_int, node_mask, logger_metric)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t
        # Update NLL metric object and return batch nll
        nll = self.train_nll(nlls)  # Average over the batch

        log_dict = {"train kl prior": kl_prior.mean(),
                    "Estimator loss terms": loss_all_t.mean(),
                    "log_pn": log_pN.mean(),
                    'train_nll': nll}
        return nll, log_dict

    @torch.no_grad()
    def sample_batch(self, n_nodes: list, number_chain_steps: int = 50, batch_id: int = 0, keep_chain: int = 0,
                     save_final: int = 0, test=True):
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, positions)
        """
        print(
            f"Sampling a batch with {len(n_nodes)} graphs. Saving {save_final} visualization and {keep_chain} full chains.")
        assert keep_chain >= 0
        assert save_final >= 0
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask, is_directed=self.directed)
        if not self.directed:
            assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        assert number_chain_steps < self.T

        n_max = z_T.X.size(1)
        chains = utils.PlaceHolder(X=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
                                   charges=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
                                   E=torch.zeros((number_chain_steps, keep_chain, n_max, n_max)),
                                   pos=torch.zeros((number_chain_steps, keep_chain, n_max, 3)),
                                   y=None,
                                   directed=self.directed)
        z_t = z_T
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T, 1 if test else self.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones((batch_size, 1), dtype=torch.long, device=z_t.X.device)

            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            # Save the first keep_chain graphs
            if (s_int * number_chain_steps) % self.T == 0:
                write_index = number_chain_steps - 1 - ((s_int * number_chain_steps) // self.T)
                discrete_z_s = z_s.collapse(self.dataset_infos.collapse_charges)
                chains.X[write_index] = discrete_z_s.X[:keep_chain]
                chains.charges[write_index] = discrete_z_s.charges[:keep_chain]
                chains.E[write_index] = discrete_z_s.E[:keep_chain]
                chains.pos[write_index] = discrete_z_s.pos[:keep_chain]

            z_t = z_s

        # Sample final data
        sampled = z_t.collapse(self.dataset_infos.collapse_charges)
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos

        chains.X[-1] = X[:keep_chain]  # Overwrite last frame with the resulting X, E
        chains.charges[-1] = charges[:keep_chain]
        chains.E[-1] = E[:keep_chain]
        chains.pos[-1] = pos[:keep_chain]

        non_molecular_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            # nodes, edges, pos, num_node_types
            non_molecular_list.append((atom_types, edge_types, conformer, len(self.dataset_infos.atom_decoder)))

        if keep_chain > 0:
            self.print('Batch sampled. Visualizing chains starts!')
            chains_path = os.path.join(os.getcwd(), f'chains/epoch{self.current_epoch}/',
                                       f'batch{batch_id}_GR{self.global_rank}')
            os.makedirs(chains_path, exist_ok=True)

            visualize_chains(chains_path, chains, num_nodes=n_nodes[:keep_chain], num_node_types=len(self.dataset_infos.atom_decoder))
        if save_final > 0:
            self.print(f'Visualizing {save_final} individual graphs...')

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path, f'graphs/epoch{self.current_epoch}_b{batch_id}/')
        _ = visualize(result_path, non_molecular_list, num_graphs_to_visualize=save_final)
        self.print("Non molecular Visualizing done.")
        return non_molecular_list

    def sample_zs_from_zt(self, z_t, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int, is_directed=self.directed)
        return z_s

    def sample_n_graphs(self, samples_to_generate: int, chains_to_save: int, samples_to_save: int, test: bool):
        if samples_to_generate <= 0:
            return []

        chains_left_to_save = chains_to_save

        samples = []
        # The first graphs are sampled without sorting the sizes, so that the visualizations are not biased
        first_sampling = min(samples_to_generate, max(samples_to_save, chains_to_save))
        factor = 0
        if first_sampling > 0:
            n_nodes = self.node_dist.sample_n(first_sampling, self.device)
            current_max_size = 0
            current_n_list = []
            for i, n in enumerate(n_nodes):
                n += factor
                potential_max_size = max(current_max_size, n)
                if self.cfg.dataset.adaptive_loader:
                    potential_ebs = effective_batch_size(potential_max_size, self.cfg.train.reference_batch_size,
                                                         sampling=True)
                else:
                    potential_ebs = int(1.8 * self.cfg.train.batch_size)  # No need to make a backward pass
                if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                    current_n_list.append(n)
                    current_max_size = potential_max_size
                else:
                    chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
                    samples.extend(self.sample_batch(n_nodes=current_n_list, batch_id=i,
                                                     save_final=len(current_n_list), keep_chain=chains_save,
                                                     number_chain_steps=self.number_chain_steps, test=test))
                    chains_left_to_save -= chains_save
                    current_n_list = [n]
                    current_max_size = n
            chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
            samples.extend(self.sample_batch(n_nodes=current_n_list, batch_id=i + 1,
                                             save_final=len(current_n_list), keep_chain=chains_save,
                                             number_chain_steps=self.number_chain_steps, test=test))
            if samples_to_generate - first_sampling <= 0:
                return samples

        # The remaining graphs are sampled in decreasing graph size
        n_nodes = self.node_dist.sample_n(samples_to_generate - first_sampling, self.device)

        if self.cfg.dataset.adaptive_loader:
            n_nodes = torch.sort(n_nodes, descending=True)[0]
        max_size = 0
        current_n_list = []
        for i, n in enumerate(n_nodes):
            max_size = max(max_size, n)
            potential_ebs = effective_batch_size(max_size, self.cfg.train.reference_batch_size, sampling=True) \
                if self.cfg.dataset.adaptive_loader else 1.8 * self.cfg.train.batch_size
            if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                current_n_list.append(n)
            else:
                samples.extend(
                    self.sample_batch(n_nodes=current_n_list, test=test, number_chain_steps=self.number_chain_steps))
                current_n_list = [n]
                max_size = n
        samples.extend(self.sample_batch(n_nodes=current_n_list, test=test, number_chain_steps=self.number_chain_steps))

        return samples

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def forward(self, z_t, extra_data):
        assert z_t.node_mask is not None
        model_input = z_t.copy()
        model_input.X = torch.cat((z_t.X, extra_data.X), dim=2).float()
        model_input.E = torch.cat((z_t.E, extra_data.E), dim=3).float()
        model_input.y = torch.hstack((z_t.y, extra_data.y, z_t.t)).float()  # y at least consists of the time step
        return self.model(model_input)

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch} finished: -- pos: {tle_log['train_epoch/pos_mse'] :.2f} -- "
                   f"X: {tle_log['train_epoch/x_CE'] :.2f} --"
                   f"degree: {tle_log['train_epoch/degree_kl'] :.2f} --"
                   f" charges: {tle_log['train_epoch/charges_CE']:.2f} --"
                   f" E: {tle_log['train_epoch/E_CE'] :.2f} --"
                   f" y: {tle_log['train_epoch/y_CE'] :.2f} -- {time.time() - self.start_epoch_time:.1f}s ")

        self.log_dict(tle_log, batch_size=self.BS)
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()
        # self.train_nll.reset()

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                      weight_decay=self.cfg.train.weight_decay)
        return optimizer
