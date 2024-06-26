# Do not move these imports, the order seems to matter
import hydra
import omegaconf
import os
import pathlib
import pytorch_lightning as pl
import torch
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_scatter import scatter
from tqdm import tqdm

from midi import utils
from midi.analysis import non_molecular_visualization
from midi.datasets import vessap_dataset
from midi.diffusion_model import FullDenoisingDiffusion
from midi.utils import save_graphs

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_num_nodes_from_graph(data):
    ones = torch.ones(data.batch.size(0), dtype=torch.long, device=data.x.device)
    num_nodes_per_graph = scatter(ones, data.batch, reduce='add')
    return num_nodes_per_graph


class FullDenoisingDiagnosticDiffusion(FullDenoisingDiffusion):

    def __init__(self, cfg, dataset_infos, train_smiles):
        super(FullDenoisingDiagnosticDiffusion, self).__init__(cfg=cfg, dataset_infos=dataset_infos,
                                                               train_smiles=train_smiles)

    def on_test_epoch_start(self):
        pass

    # We would change this function a bit
    def test_step(self, data, i):
        # Get number of graphs in each elements
        num_nodes_per_graph = get_num_nodes_from_graph(data=data)
        dense_data = utils.to_dense(data, self.dataset_infos, is_directed=self.directed)
        z_T = self.noise_model.apply_noise(dense_data)
        number_chain_steps = 1
        keep_chain = 1
        save_final = 1
        batch_id = 1
        batch_size = len(num_nodes_per_graph)
        n_nodes = num_nodes_per_graph.long().to(self.device)
        # Now we want to reconstruct the data from this specific noise.
        # We can then check how the interpolation would look like.

        # z_T_prime = self.noise_model.sample_limit_dist(node_mask=dense_data.node_mask)
        if not self.directed:
            assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        assert number_chain_steps < self.T

        n_max = z_T.X.size(1)
        chains = utils.PlaceHolder(X=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
                                   E=torch.zeros((number_chain_steps, keep_chain, n_max, n_max)),
                                   pos=torch.zeros((number_chain_steps, keep_chain, n_max, 3)),
                                   y=None,
                                   directed=self.directed)
        # We need to modify z_T to indicate we are starting a reverse process on them
        t_array = z_T.pos.new_ones((z_T.pos.shape[0], 1))
        t_int_array = self.T * t_array.long()
        z_T.t_int = t_int_array
        z_T.t = t_array
        # Modification complete
        z_t = z_T
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        print("Sampling started")
        for s_int in tqdm(reversed(range(0, self.T, 1)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1), dtype=torch.long, device=z_t.X.device)

            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)

            # Save the first keep_chain graphs
            if (s_int * number_chain_steps) % self.T == 0:
                write_index = number_chain_steps - 1 - ((s_int * number_chain_steps) // self.T)
                discrete_z_s = z_s.collapse()
                chains.X[write_index] = discrete_z_s.X[:keep_chain]
                chains.E[write_index] = discrete_z_s.E[:keep_chain]
                chains.pos[write_index] = discrete_z_s.pos[:keep_chain]

            z_t = z_s
        print("Sampling Finished")
        # Sample final data
        sampled = z_t.collapse()
        X, E, y, pos = sampled.X, sampled.E, sampled.y, sampled.pos

        chains.X[-1] = X[:keep_chain]  # Overwrite last frame with the resulting X, E
        chains.E[-1] = E[:keep_chain]
        chains.pos[-1] = pos[:keep_chain]

        original_sample = []
        non_molecular_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types_orig, edge_types_orig, conformer_orig = dense_data.X[i, :n], \
                                                               dense_data.E[i, :n, :n], \
                                                               dense_data.pos[i, :n]
            # Converting values into one hot
            edge_types_orig = torch.argmax(edge_types_orig, dim=-1)

            atom_types = X[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            # nodes, edges, pos, num_node_types
            original_sample.append(
                (atom_types_orig, edge_types_orig, conformer_orig, len(self.dataset_infos.atom_decoder)))
            non_molecular_list.append((atom_types, edge_types, conformer, len(self.dataset_infos.atom_decoder)))

        if keep_chain > 0:
            self.print('Batch sampled. Visualizing chains starts!')
            chains_path = os.path.join(os.getcwd(), f'chains_interpolation/epoch{self.current_epoch}/',
                                       f'batch{batch_id}_GR{self.global_rank}')
            os.makedirs(chains_path, exist_ok=True)

            non_molecular_visualization.visualize_chains(chains_path, chains,
                                                         num_nodes=n_nodes[:keep_chain],
                                                         num_node_types=len(self.dataset_infos.atom_decoder))
        if save_final > 0:
            self.print(f'Visualizing {save_final} individual graphs...')

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path, f'graphs_interpolations/epoch{self.current_epoch}_b{batch_id}/')
        _ = non_molecular_visualization.visualize(result_path, non_molecular_list, num_graphs_to_visualize=save_final)
        self.print("Non molecular Visualizing done.")
        print("saving results")
        save_graphs(original_sample, filename='original_test_samples.pkl')
        save_graphs(non_molecular_list, filename='interpolated_test_samples.pkl')
        print("Done")

    def on_test_epoch_end(self) -> None:
        # This is a no-op case
        pass


def get_resume(cfg, dataset_infos, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model


@hydra.main(version_base='1.3', config_path='../../configs', config_name='config_interpolation')
def main(cfg: omegaconf.DictConfig):
    pl.seed_everything(cfg.train.seed)

    datamodule = vessel_dataset.VessapGraphDataModule(cfg)
    dataset_infos = vessel_dataset.VessapDatasetInfos(datamodule=datamodule, cfg=cfg)
    train_smiles = None

    cfg, _ = get_resume(cfg, dataset_infos, cfg.general.test_only, test=True)

    model = FullDenoisingDiagnosticDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=train_smiles)

    callbacks = []
    # need to ignore metrics because otherwise ddp tries to sync them
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    # if cfg.train.save_model:
    #     checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
    #                                           filename='{epoch}',
    #                                           monitor='val/epoch_NLL',
    #                                           save_top_k=5,
    #                                           mode='min',
    #                                           every_n_epochs=1)
    #     # fix a name and keep overwriting
    #     last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
    #     callbacks.append(checkpoint_callback)
    #     callbacks.append(last_ckpt_save)

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
