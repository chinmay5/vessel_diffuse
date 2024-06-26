# Do not move these imports, the order seems to matter
import hydra
import omegaconf
import pathlib
import re
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import time
import os
import math
import numpy as np

import torch
import pytorch_lightning as pl
import wandb

import midi.analysis.visualization as visualizer
import midi.metrics.abstract_metrics as custom_metrics
from torch import nn
from torchmetrics import MeanSquaredError
from midi.analysis import non_molecular_visualization
from midi.diffusion.diffusion_utils import sum_except_batch
from midi.models.midi_transformer_model import EquivariantGraphTransformer
from midi.diffusion import diffusion_utils
from midi.metrics.abstract_metrics import SumExceptBatchMetric, NLL
from midi import utils
from midi.diffusion.extra_features import ExtraFeatures
from midi.datasets.adaptive_loader import effective_batch_size
from midi.utils import save_graphs

warnings.filterwarnings("ignore", category=PossibleUserWarning)


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = diffusion_utils.cosine_beta_schedule(timesteps)
        elif noise_schedule == 'custom':
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2  # (timesteps + 1, )

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = custom_metrics.NodeMSE()
        self.train_edge_mse = custom_metrics.EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y
        to_log = None
        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse, to_log

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log


class LocalNoiseDistClass(nn.Module):
    def __init__(self, T, cfg):
        super(LocalNoiseDistClass, self).__init__()
        self.T = T
        # We will sample the positions based on the pretrained diffusion model
        from bootstrap.diff_nodes import Diffusion, Simple_FC
        from environment_setup import PROJECT_ROOT_DIR

        # We will sample the positions based on the pretrained diffusion model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion_pos = Diffusion(input_size=3, device=device)
        diff_pos_model = Simple_FC(hidden_dim=cfg.model.diff_1_config['hidden_dim'],
                                   n_layers=cfg.model.diff_1_config['n_layers']).to(device)
        if cfg.dataset.name == "vessel":
            print("Vessap coordinate sampling")
            diff_pos_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                                   'checks_vessap_best_coord_diff_model.pth')))
        elif cfg.dataset.name == "cow":
            print("CoW coordinate sampling")
            diff_pos_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT_DIR, 'midi', 'models',
                                                                   'checks_cow_best_coord_diff_model.pth')))
        else:
            raise AttributeError(f"Invalid dataset choice: {cfg.dataset.name}")

        self.pos_denoiser = diff_pos_model
        self.diffusion_pos = diffusion_pos

    def sample_feature_noise_limit_dist(self, X_size, E_size, y_size, node_mask, directed):
        epsX = torch.randn(X_size).to(node_mask.device)
        epsE = torch.randn(E_size).to(node_mask.device)
        epsy = torch.randn(y_size).to(node_mask.device)

        float_mask = node_mask.float()
        epsX = epsX.type_as(float_mask).to(node_mask.device)
        epsE = epsE.type_as(float_mask).to(node_mask.device)
        epsy = epsy.type_as(float_mask).to(node_mask.device)
        if not directed:
            # Get upper triangular part of edge noise, without main diagonal
            upper_triangular_mask = torch.zeros_like(epsE)
            indices = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1)
            upper_triangular_mask[:, indices[0], indices[1], :] = 1

            epsE = epsE * upper_triangular_mask
            epsE = (epsE + torch.transpose(epsE, 1, 2))
        charges = torch.ones((epsX.size(0), epsX.size(1), 1), device=node_mask.device)

        # required_pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3)
        # print("Sampling coordinate locations")
        # pos = self.diffusion_pos.sample(model=self.pos_denoiser, pos=required_pos, node_mask=node_mask)
        # pos = pos * node_mask.unsqueeze(-1)
        # NOTE: This is the sampling process for normal MiDi
        # We can run the same with fixed node locations if that is needed
        pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3, device=node_mask.device)
        pos = pos * node_mask.unsqueeze(-1)
        pos = utils.remove_mean_with_mask(pos, node_mask)

        t_array = pos.new_ones((pos.shape[0], 1))
        t_int_array = self.T * t_array.long()
        return utils.PlaceHolder(X=epsX, E=epsE, y=epsy, charges=charges, pos=pos,
                                 node_mask=node_mask, directed=directed, t_int=t_int_array, t=t_array).mask()


class ContDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None

    def __init__(self, cfg, dataset_infos, train_smiles):
        super().__init__()
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.Xdim = input_dims.X
        self.Edim = input_dims.E
        self.ydim = input_dims.y
        self.Xdim_output = output_dims.X
        self.Edim_output = output_dims.E
        self.ydim_output = output_dims.y
        self.node_dist = nodes_dist

        self.dataset_infos = dataset_infos
        self.extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)
        self.output_dims = dataset_infos.output_dims
        # self.domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        # Train metrics
        self.train_loss = TrainLoss()
        self.train_metrics = TrainMolecularMetrics(dataset_infos)

        # Val Metrics
        self.val_nll = NLL()
        self.val_X_mse = custom_metrics.SumExceptBatchMSE()
        self.val_E_mse = custom_metrics.SumExceptBatchMSE()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = custom_metrics.SumExceptBatchMSE()

        # Computing the same loss on the val split just for sanity checks
        self.val_loss = TrainLoss()
        self.val_nll = NLL()

        # Test metrics
        self.test_nll = NLL()
        self.test_X_mse = custom_metrics.SumExceptBatchMSE()
        self.test_E_mse = custom_metrics.SumExceptBatchMSE()
        # self.test_y_mse = custom_metrics.SumExceptBatchMSE()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = custom_metrics.SumExceptBatchMSE()

        # Let us also put test loss for completeness
        # Again computing this term just for sanity check
        self.test_loss = TrainLoss()

        self.gamma = PredefinedNoiseSchedule(cfg.model.diffusion_noise_schedule, timesteps=self.T)
        self.save_hyperparameters(ignore=['train_metrics', 'dataset_infos'])
        self.is_molecular = False
        self.directed = cfg.dataset.get("is_directed", False)
        print(f"MODEL DIRECTED = {self.directed}")
        self.model = EquivariantGraphTransformer(input_dims=self.input_dims,
                                                 n_layers=cfg.model.n_layers,
                                                 hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                                 hidden_dims=cfg.model.hidden_dims,
                                                 output_dims=self.output_dims,
                                                 # Whether we generate directed graphs or not
                                                 is_directed=self.directed,
                                                 cfg=cfg)

        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = 10  # cfg.general.number_chain_steps
        self.local_denoiser = LocalNoiseDistClass(T=self.T, cfg=cfg)

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)

        noisy_data_whole = self.apply_noise(dense_data)
        # Converting the information into a format that can be easily processed by the model
        # {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
        #                       'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
        #                       'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        z_t = utils.PlaceHolder(pos=dense_data.pos, X=noisy_data_whole['X_t'], E=noisy_data_whole['E_t'],
                                t=noisy_data_whole['t'], t_int=noisy_data_whole['t_int'], y=noisy_data_whole['y_t'],
                                node_mask=dense_data.node_mask, directed=self.directed, charges=dense_data.charges)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        # masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y
        loss, tl_log_dict = self.train_loss(masked_pred_epsX=pred.X, masked_pred_epsE=pred.E, pred_y=pred.y,
                                            true_epsX=dense_data.X, true_epsE=dense_data.E, true_y=dense_data.y,
                                            log=i % self.log_every_steps == 0)

        tm_log_dict = self.train_metrics(masked_pred=pred, masked_true=dense_data,
                                         log=i % self.log_every_steps == 0)
        if tl_log_dict is not None:
            self.log_dict(tl_log_dict, batch_size=self.BS)
        if tm_log_dict is not None:
            self.log_dict(tm_log_dict, batch_size=self.BS)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                      weight_decay=self.cfg.train.weight_decay)
        return optimizer

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_mse: {tle_log['train_epoch/epoch_X_mse'] :.3f}"
                   f" -- E mse: {tle_log['train_epoch/epoch_E_mse'] :.3f} --"
                   f" y_mse: {tle_log['train_epoch/epoch_y_mse'] :.3f}"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")
        self.log_dict(tle_log, batch_size=self.BS)

        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def on_validation_epoch_start(self) -> None:
        # TODO: Include the extra metric that are used wrt continous variables
        self.val_nll.reset()
        self.val_X_mse.reset()
        self.val_E_mse.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()
        self.val_loss.reset()

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        noisy_data_whole = self.apply_noise(dense_data)
        # Converting the information into a format that can be easily processed by the model
        # {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
        #                       'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
        #                       'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        z_t = utils.PlaceHolder(pos=dense_data.pos, X=noisy_data_whole['X_t'], E=noisy_data_whole['E_t'],
                                t=noisy_data_whole['t'], t_int=noisy_data_whole['t_int'], y=noisy_data_whole['y_t'],
                                node_mask=dense_data.node_mask, directed=self.directed, charges=dense_data.charges)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        # Computing extra validation loss terms
        self.val_loss(masked_pred_epsX=pred.X, masked_pred_epsE=pred.E, pred_y=pred.y,
                      true_epsX=dense_data.X, true_epsE=dense_data.E, true_y=dense_data.y,
                      log=i % self.log_every_steps == 0)
        nll, log_dict = self.compute_val_loss(pred, noisy_data_whole, clean_data=dense_data, test=False)
        return {'loss': nll}, log_dict

        # TODO: check if compute val loss should be called on the normalized data or not
        # nll = self.compute_val_loss(pred, noisy_data, normalized_data.X, normalized_data.E, normalized_data.y,
        #                             node_mask, test=False)
        # return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_mse.compute(), self.val_E_mse.compute(),
                   self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_mse": metrics[1],
                       "val/E_mse": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4],
                       "val/y_logp": metrics[5]}, commit=False)

        print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type MSE {metrics[1] :.2f} -- ",
              f"Val Edge type MSE: {metrics[2] :.2f}",
              f"-- Val X Reconstruction loss {metrics[3] :.2f} -- Val E Reconstruction loss {metrics[4] :.2f}",
              f"-- Val y Reconstruction loss {metrics[5] : .2f}\n")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)
        if wandb.run:
            wandb.log(self.log_info(), commit=False)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

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
            print(f'Done on {self.local_rank}. Sampling took {time.time() - start:.2f} seconds\n')
            del samples
            torch.cuda.empty_cache()
        self.print(f"Val epoch {self.current_epoch} ends")

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_mse.reset()
        self.test_E_mse.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()
        self.test_loss.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        noisy_data_whole = self.apply_noise(dense_data)
        # Converting the information into a format that can be easily processed by the model
        # {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
        #                       'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
        #                       'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        z_t = utils.PlaceHolder(pos=dense_data.pos, X=noisy_data_whole['X_t'], E=noisy_data_whole['E_t'],
                                t=noisy_data_whole['t'], t_int=noisy_data_whole['t_int'], y=noisy_data_whole['y_t'],
                                node_mask=dense_data.node_mask, directed=self.directed, charges=dense_data.charges)
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        nll = self.compute_val_loss(pred, noisy_data_whole, clean_data=dense_data, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_mse.compute(), self.test_E_mse.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        log_dict = {"test/epoch_NLL": metrics[0],
                    "test/X_mse": metrics[1],
                    "test/E_mse": metrics[2],
                    "test/X_logp": metrics[3],
                    "test/E_logp": metrics[4],
                    "test/y_logp": metrics[5]}
        if wandb.run:
            wandb.log(log_dict, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type MSE {metrics[1] :.2f} -- ",
              f"Test Edge type MSE: {metrics[2] :.2f}",
              f"-- Test X Reconstruction loss {metrics[3] :.2f} -- Test E Reconstruction loss {metrics[4] :.2f}",
              f"-- Test y Reconstruction loss {metrics[5] : .2f}\n")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)
            wandb.log(self.log_info(), commit=False)

        print(f'Test loss: {test_nll :.4f}')
        print(f"Sampling start on GR{self.global_rank}")
        start = time.time()

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
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        X = clean_data.X
        ones = torch.ones((X.size(0), 1))
        ones = ones.type_as(X)
        gamma_T = self.gamma(ones)
        alpha_T = diffusion_utils.alpha(gamma_T, X.size())

        # Compute means.
        mu_T_X = alpha_T * X
        mu_T_E = alpha_T.unsqueeze(1) * clean_data.E
        mu_T_y = alpha_T.squeeze(1) * clean_data.y

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_X = diffusion_utils.sigma(gamma_T, mu_T_X.size())
        sigma_T_E = diffusion_utils.sigma(gamma_T, mu_T_E.size())
        sigma_T_y = diffusion_utils.sigma(gamma_T, mu_T_y.size())

        # Compute KL for h-part.
        kl_distance_X = diffusion_utils.gaussian_KL(mu_T_X, sigma_T_X)
        kl_distance_E = diffusion_utils.gaussian_KL(mu_T_E, sigma_T_E)
        kl_distance_y = diffusion_utils.gaussian_KL(mu_T_y, sigma_T_y)

        return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E) + sum_except_batch(kl_distance_y)

    def log_constants_p_y_given_z0(self, batch_size):
        """ Computes p(y|z0)= -0.5 ydim (log(2pi) + gamma(0)).
            sigma_y = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
            output size: (batch_size)
        """
        if self.ydim_output == 0:
            return 0.0

        zeros = torch.zeros((batch_size, 1))
        gamma_0 = self.gamma(zeros).squeeze(1)
        # Recall that
        return -0.5 * self.ydim * (gamma_0 + np.log(2 * np.pi))

    def reconstruction_logp(self, clean_data, data_0, gamma_0, eps, pred_0, node_mask, epsilon=1e-10, test=False):
        """ Reconstruction loss.
            output size: (1).
        """
        X, E, y = clean_data.X, clean_data.E, clean_data.y
        X_0, E_0, y_0 = data_0.values()

        # TODO: why don't we need the values of X and E?
        _, _, eps_y0 = eps.values()
        predy = pred_0.y

        # 1. Compute reconstruction loss for global, continuous features
        if test:
            # TODO: Remove the logging for y_logp since it just produces NANs.
            error_y = -0.5 * self.test_y_logp(predy, eps_y0) if eps_y0.numel() > 0 else 0
        else:
            error_y = -0.5 * self.val_y_logp(predy, eps_y0) if eps_y0.numel() > 0 else 0
        # The _constants_ depending on sigma_0 from the cross entropy term E_q(z0 | y) [log p(y | z0)].
        neg_log_constants = - self.log_constants_p_y_given_z0(y.shape[0])
        log_py = error_y + neg_log_constants

        # 2. Compute reconstruction loss for integer/categorical features on nodes and edges

        # Compute sigma_0 and rescale to the integer scale of the data_utils.
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=X_0.size())
        sigma_0_X = sigma_0  # * self.norm_values[0]
        # sigma_0_E = (sigma_0 * self.norm_values[1]).unsqueeze(-1)
        sigma_0_E = sigma_0.unsqueeze(-1)

        # Centered cat features around 1, since onehot encoded.
        E_0_centered = E_0 - 1
        X_0_centered = X_0 - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        log_pE_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((E_0_centered + 0.5) / sigma_0_E)
            - diffusion_utils.cdf_std_gaussian((E_0_centered - 0.5) / sigma_0_E)
            + epsilon)

        log_pX_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((X_0_centered + 0.5) / sigma_0_X)
            - diffusion_utils.cdf_std_gaussian((X_0_centered - 0.5) / sigma_0_X)
            + epsilon)

        # Normalize the distributions over the categories.
        norm_cst_E = torch.logsumexp(log_pE_proportional, dim=-1, keepdim=True)
        norm_cst_X = torch.logsumexp(log_pX_proportional, dim=-1, keepdim=True)

        log_probabilities_E = log_pE_proportional - norm_cst_E
        log_probabilities_X = log_pX_proportional - norm_cst_X

        # Select the log_prob of the current category using the one-hot representation.
        logps = utils.PlaceHolder(X=log_probabilities_X * X,
                                  E=log_probabilities_E * E,
                                  y=None, charges=clean_data.charges, pos=clean_data.pos).mask(node_mask)

        if test:
            log_pE = - self.test_E_logp(-logps.E)
            log_pX = - self.test_X_logp(-logps.X)
        else:
            log_pE = - self.val_E_logp(-logps.E)
            log_pX = - self.val_X_logp(-logps.X)
        return log_pE + log_pX + log_py

    def apply_noise(self, dense_data):
        """ Sample noise and apply it to the data. """
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1

        # Sample a timestep t.
        t_int = torch.randint(lowest_t, self.T + 1, size=(dense_data.X.size(0), 1))
        t_int = t_int.type_as(dense_data.X).float()  # (bs, 1)
        s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s_normalized = s_int / self.T
        t_normalized = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_normalized), dense_data.X.size())  # (bs, 1, 1),
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_normalized), dense_data.X.size())  # (bs, 1, 1)

        # Compute alpha_t and sigma_t from gamma, with correct size for X, E and z
        alpha_t = diffusion_utils.alpha(gamma_t, dense_data.X.size())  # (bs, 1, ..., 1), same n_dims than X
        sigma_t = diffusion_utils.sigma(gamma_t, dense_data.X.size())  # (bs, 1, ..., 1), same n_dims than X

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = diffusion_utils.sample_feature_noise(dense_data)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        X_t = alpha_t * dense_data.X + sigma_t * eps.X
        E_t = alpha_t.unsqueeze(1) * dense_data.E + sigma_t.unsqueeze(1) * eps.E
        y_t = alpha_t.squeeze(1) * dense_data.y + sigma_t.squeeze(1) * eps.y

        noisy_data = {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
                      'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
                      'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data_whole, clean_data, test=False):
        """ Computes an estimator for the variational lower bound, or the simple loss (MSE).
               pred: (batch_size, n, total_features)
               noisy_data: dict
               X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
               node_mask : (bs, n)
           Output: nll (size 1). """

        s = noisy_data_whole['s']
        gamma_s = noisy_data_whole['gamma_s']  # gamma_s.size() == X.size()
        gamma_t = noisy_data_whole['gamma_t']
        epsX = noisy_data_whole['epsX']
        epsE = noisy_data_whole['epsE']
        epsy = noisy_data_whole['epsy']
        X_t = noisy_data_whole['X_t']
        E_t = noisy_data_whole['E_t']
        y_t = noisy_data_whole['y_t']

        node_mask = clean_data.node_mask
        t_int = noisy_data_whole['t_int']
        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Normal(0, 1). Should be close to zero. Do not forget the prefactor
        kl_prior_without_prefactor = self.kl_prior(clean_data, node_mask)
        # We assume norm values to be (1, 1, 1). Thus, the terms can be skipped
        # delta_log_py = -self.ydim_output * np.log(self.norm_values[2])
        # delta_log_px = -self.Xdim_output * N * np.log(self.norm_values[0])
        # delta_log_pE = -self.Edim_output * 0.5 * N * (N - 1) * np.log(self.norm_values[1])
        # kl_prior = kl_prior_without_prefactor - delta_log_px - delta_log_py - delta_log_pE
        kl_prior = kl_prior_without_prefactor

        # 3. Diffusion loss

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization.
        SNR_weight = - (1 - diffusion_utils.SNR(gamma_s - gamma_t))
        sqrt_SNR_weight = torch.sqrt(SNR_weight)  # same n_dims than X
        # Compute the error.
        weighted_predX_diffusion = sqrt_SNR_weight * pred.X
        weighted_epsX_diffusion = sqrt_SNR_weight * epsX

        weighted_predE_diffusion = sqrt_SNR_weight.unsqueeze(1) * pred.E
        weighted_epsE_diffusion = sqrt_SNR_weight.unsqueeze(1) * epsE

        weighted_predy_diffusion = sqrt_SNR_weight.squeeze(1) * pred.y
        weighted_epsy_diffusion = sqrt_SNR_weight.squeeze(1) * epsy

        # Compute the MSE summed over channels
        if test:
            diffusion_error = (self.test_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.test_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion))
            # (self.test_y_mse(weighted_predy_diffusion,
            #                  weighted_epsy_diffusion) if weighted_epsy_diffusion.numel() > 0 else 0))
        else:
            diffusion_error = (self.val_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.val_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion))
            # (self.val_y_mse(weighted_predy_diffusion,
            #                 weighted_epsy_diffusion) if weighted_epsy_diffusion.numel() > 0 else 0))
        loss_all_t = 0.5 * self.T * diffusion_error  # t=0 is not included here.

        # 4. Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(s)  # bs, 1
        gamma_0 = diffusion_utils.inflate_batch_array(self.gamma(t_zeros), X_t.size())  # bs, 1, 1
        alpha_0 = diffusion_utils.alpha(gamma_0, X_t.size())  # bs, 1, 1
        sigma_0 = diffusion_utils.sigma(gamma_0, X_t.size())  # bs, 1, 1

        # Sample z_0 given X, E, y for timestep t, from q(z_t | X, E, y)
        # Converting the information into a format that can be easily processed by the model
        # {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
        #                       'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
        #                       'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        z_t = utils.PlaceHolder(pos=clean_data.pos, X=noisy_data_whole['X_t'], E=noisy_data_whole['E_t'],
                                t=noisy_data_whole['t'], t_int=noisy_data_whole['t_int'], y=noisy_data_whole['y_t'],
                                node_mask=clean_data.node_mask, directed=self.directed, charges=clean_data.charges)
        eps0 = diffusion_utils.sample_feature_noise(z_t)

        X_0 = alpha_0 * X_t + sigma_0 * eps0.X
        E_0 = alpha_0.unsqueeze(1) * E_t + sigma_0.unsqueeze(1) * eps0.E
        y_0 = alpha_0.squeeze(1) * y_t + sigma_0.squeeze(1) * eps0.y
        # TODO: Check if this is too restrictive
        # Since positions do not change
        noisy_data0 = utils.PlaceHolder(pos=clean_data.pos, X=X_0, E=E_0, y=y_0, t_int=t_zeros, directed=self.directed,
                                        charges=clean_data.charges, node_mask=clean_data.node_mask, t=t_zeros / self.T)
        extra_data = self.extra_features(noisy_data0)
        pred_0 = self.forward(noisy_data0, extra_data)

        loss_term_0 = - self.reconstruction_logp(clean_data=clean_data,
                                                 data_0={'X_0': X_0, 'E_0': E_0, 'y_0': y_0},
                                                 gamma_0=gamma_0,
                                                 eps={'eps_X0': eps0.X, 'eps_E0': eps0.E, 'eps_y0': eps0.y},
                                                 pred_0=pred_0,
                                                 node_mask=node_mask,
                                                 test=test)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t + loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        nll = self.test_nll(nlls) if test else self.val_nll(nlls)  # Average over the batch
        log_dict = {"kl prior": kl_prior.mean(),
                    "Estimator loss terms": loss_all_t.mean(),
                    "Loss term 0": loss_term_0,
                    "log_pn": log_pN.mean(),
                    'test_nll' if test else 'val_nll': nll}
        wandb.log(log_dict, commit=False)
        return nll, log_dict

    def forward(self, z_t, extra_data):
        assert z_t.node_mask is not None
        model_input = z_t.copy()
        model_input.X = torch.cat((z_t.X, extra_data.X), dim=2).float()
        model_input.E = torch.cat((z_t.E, extra_data.E), dim=3).float()
        model_input.y = torch.hstack((z_t.y, extra_data.y, z_t.t)).float()  # y at least consists of the time step
        return self.model(model_input)

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {'log_SNR_max': log_SNR_max.item(), 'log_SNR_min': log_SNR_min.item()}
        print("", info, "\n")

        return info

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
        n_nodes_max = node_mask.shape[1]

        z_T = self.local_denoiser.sample_feature_noise_limit_dist(X_size=(batch_size, n_nodes_max, self.Xdim_output),
                                                                  E_size=(
                                                                      batch_size, n_nodes_max, n_nodes_max,
                                                                      self.Edim_output),
                                                                  y_size=(batch_size, self.ydim_output),
                                                                  node_mask=node_mask, directed=self.directed)

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
            s_norm = s_array / self.T
            t_norm = (s_array + 1) / self.T
            z_s = self.sample_p_zs_given_zt(s=s_norm, t=t_norm, z_t=z_t)

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
        sampled = self.sample_discrete_graph_given_z0(z_t)
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos
        chains.X[-1] = X[:keep_chain]  # Overwrite last frame with the resulting X, E
        chains.charges[-1] = charges[:keep_chain]
        chains.E[-1] = E[:keep_chain]
        chains.pos[-1] = pos[:keep_chain]

        molecule_list = []
        non_molecular_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            if self.is_molecular:
                molecule_list.append(Molecule(atom_types=atom_types, charges=charge_vec,
                                              bond_types=edge_types, positions=conformer,
                                              atom_decoder=self.dataset_infos.atom_decoder))
            else:
                # nodes, edges, pos, num_node_types
                non_molecular_list.append((atom_types, edge_types, conformer, len(self.dataset_infos.atom_decoder)))

        # The visualization code block
        if self.is_molecular:
            # Visualize chains
            if keep_chain > 0:
                self.print('Batch sampled. Visualizing chains starts!')
                chains_path = os.path.join(os.getcwd(), f'chains/epoch{self.current_epoch}/',
                                           f'batch{batch_id}_GR{self.global_rank}')
                os.makedirs(chains_path, exist_ok=True)

                visualizer.visualize_chains(chains_path, chains,
                                            num_nodes=n_nodes[:keep_chain],
                                            atom_decoder=self.dataset_infos.atom_decoder)

            if save_final > 0:
                self.print(f'Visualizing {save_final} individual molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path, f'graphs/epoch{self.current_epoch}_b{batch_id}/')
            _ = visualizer.visualize(result_path, molecule_list, num_molecules_to_visualize=save_final)
            self.print("Visualizing done.")
            return molecule_list

        assert len(molecule_list) == 0, "Molecular graph getting visualized as non-molecular."
        # We can reuse a lot of things from above but still needs quite a bit of refactoring.
        if keep_chain > 0:
            self.print('Batch sampled. Visualizing chains starts!')
            chains_path = os.path.join(os.getcwd(), f'chains/epoch{self.current_epoch}/',
                                       f'batch{batch_id}_GR{self.global_rank}')
            os.makedirs(chains_path, exist_ok=True)

            non_molecular_visualization.visualize_chains(chains_path, chains,
                                                         num_nodes=n_nodes[:keep_chain],
                                                         num_node_types=len(self.dataset_infos.atom_decoder))
        if save_final > 0:
            self.print(f'Visualizing {save_final} individual graphs...')

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path, f'graphs/epoch{self.current_epoch}_b{batch_id}/')
        # _ = non_molecular_visualization.visualize(result_path, non_molecular_list, num_graphs_to_visualize=save_final)
        self.print("Non molecular Visualizing done.")
        return non_molecular_list

    def sample_discrete_graph_given_z0(self, z_0):
        """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
        to categorical values.
        """
        zeros = torch.zeros(size=(z_0.X.size(0), 1), device=z_0.X.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        t_int = torch.zeros(z_0.y.shape[0], 1).type_as(z_0.y)
        noisy_data = utils.PlaceHolder(pos=z_0.pos, X=z_0.X, E=z_0.E, y=z_0.y, charges=z_0.charges,
                                       t_int=t_int, t=t_int / self.T, directed=self.directed,
                                       node_mask=z_0.node_mask)
        extra_data = self.extra_features(noisy_data)
        eps0 = self.forward(noisy_data, extra_data)

        # Compute mu for p(zs | zt).
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (z_0.X - sigma_0 * eps0.X)
        pred_E = 1. / alpha_0.unsqueeze(1) * (z_0.E - sigma_0.unsqueeze(1) * eps0.E)
        pred_y = 1. / alpha_0.squeeze(1) * (z_0.y - sigma_0.squeeze(1) * eps0.y)

        sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, z_0)
        if not self.directed:
            assert (pred_E == torch.transpose(pred_E, 1, 2)).all()
            assert (sampled.E == torch.transpose(sampled.E, 1, 2)).all()

        # Converting the sampled distribution into the classes
        sampled = sampled.collapse(self.dataset_infos.collapse_charges)
        return sampled

    def sample_p_zs_given_zt(self, s, t, z_t):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       z_t.X.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=z_t.X.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=z_t.X.size())

        extra_data = self.extra_features(z_t)
        eps = self.forward(z_t, extra_data)

        # Compute mu for p(zs | zt).
        mu_X = z_t.X / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X
        mu_E = z_t.E / alpha_t_given_s.unsqueeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).unsqueeze(
            1) * eps.E
        mu_y = z_t.y / alpha_t_given_s.squeeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).squeeze(
            1) * eps.y

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        z_s = diffusion_utils.sample_normal(mu_X, mu_E, mu_y, sigma, z_t)

        return z_s

    def sample_n_graphs(self, samples_to_generate: int, chains_to_save: int, samples_to_save: int, test: bool):
        if samples_to_generate <= 0:
            return []

        chains_left_to_save = chains_to_save

        samples = []
        # The first graphs are sampled without sorting the sizes, so that the visualizations are not biased
        first_sampling = min(samples_to_generate, max(samples_to_save, chains_to_save))
        factor = 0
        if test:
            factor = int(self.cfg.train.get("node_increase", 0))
            print(f"Artificially increasing the number of sampled nodes by {factor=}")
        if first_sampling > 0:
            if self.cfg.model.get("restrict_num_nodes", False):
                n_nodes = torch.randint(10, 21, (first_sampling,), device=self.device)
            else:
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
        if self.cfg.model.get("restrict_num_nodes", False):
            n_nodes = torch.randint(10, 21, (samples_to_generate - first_sampling,), device=self.device)
        else:
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


def get_base_path(checkpoint_path):
    # Define the regex pattern
    pattern = '.*\d{2}-\d{2}-\d{2}-\w+-\w+-\w+'

    # Use re.search to find the folder name
    match = re.search(pattern, checkpoint_path)

    if match:
        folder_name = match.group(0)
        log_path = os.path.join(folder_name, "test_generated_samples_0")
        if os.path.exists(log_path):
            valid_test_folders = filter(lambda x: "test_generated_samples" in x, os.listdir(folder_name))
            max_idx = max(map(lambda x: int(x[x.rfind("_") + 1:]), valid_test_folders))
            log_path = os.path.join(folder_name, f"test_generated_samples_{max_idx + 1}")
        os.makedirs(log_path)
        return log_path
    else:
        raise AttributeError("Invalid checkpoint path")


def get_resume(cfg, dataset_infos, train_smiles, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    if train_smiles is not None:
        model = ContDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos,
                                                            train_smiles=train_smiles)
    else:
        model = ContDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos)
    cfg.general.gpus = gpus
    cfg.general.name = name
    # get the folder name so that we do not create a new folder for logging but use the existing one
    if test:
        new_path = get_base_path(checkpoint_path)
        print(f"Changing path to: {new_path=}")
        os.chdir(new_path)
    # We would update the general configurations of the model.
    print("Overriding cfg values from command line. Please see here")
    model.cfg.general = cfg.general
    model.cfg.train_one_epoch = cfg.train
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config_cont')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)
    if dataset_config.name == 'vessap':
        datamodule = VessapGraphDataModule(cfg)
        dataset_infos = VessapDatasetInfos(datamodule=datamodule, cfg=cfg)
        train_smiles = None
    elif dataset_config.name == 'cow':
        datamodule = CoWGraphDataModule(cfg)
        dataset_infos = CoWDatasetInfos(datamodule=datamodule, cfg=cfg)
        train_smiles = None

    if cfg.general.test_only:
        cfg, model = get_resume(cfg, dataset_infos, train_smiles, cfg.general.test_only, test=True)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        print("Resuming from {}".format(cfg.general.resume))
        cfg, model = get_resume(cfg, dataset_infos, train_smiles, cfg.general.resume, test=False)
    else:
        model = ContDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=train_smiles)

    callbacks = []
    # need to ignore metrics because otherwise ddp tries to sync them
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        # fix a name and keep overwriting
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # if cfg.general.name not in ['debug', 'test']:
        #     trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        for i in range(cfg.general.num_final_sampling):
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
