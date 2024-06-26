# Do not move these imports, the order seems to matter
import torch
import pytorch_lightning as pl

import os
import warnings
import pathlib

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from midi.datasets import vessap_dataset
from midi.diffusion_model import FullDenoisingDiffusion

warnings.filterwarnings("ignore", category=PossibleUserWarning)


class PretrainedDenoisingDiffusion(FullDenoisingDiffusion):
    def __init__(self, cfg, dataset_infos, train_smiles):
        super(PretrainedDenoisingDiffusion, self).__init__(cfg=cfg, dataset_infos=dataset_infos,
                                                           train_smiles=train_smiles)

    def load_pretrained_weights(self, pretrained_model_path):
        # Load weights from a pretrained model
        pretrained_dict = torch.load(pretrained_model_path)['state_dict']
        model_dict = self.state_dict()

        # Print layers that will be loaded and layers with shape mismatches
        print("Loading checkpoint")
        layers_to_load = {}
        matching_layers, mismatching_layers = [], []
        for name, param in pretrained_dict.items():
            if name in model_dict and param.shape == model_dict[name].shape:
                layers_to_load[name] = param
                matching_layers.append(name)
            else:
                mismatching_layers.append(name)
        print("\n")
        print("---------------")
        print("\n\n")
        print("Match")
        print("\n".join(matching_layers))
        print("---------------")
        print("\n\n")
        print("Mismatching layers")
        print("\n".join(mismatching_layers))
        # Load pretrained weights for matching layers
        model_dict.update(layers_to_load)
        self.load_state_dict(model_dict)

    def freeze_transformer_layers(self, verify_load=True):
        frozen_layers = []
        for name, param in self.named_children():
            # We let the last two layers to fine-tune
            if 'tf_layers' in name and not any(('10' in name, '11' in name)):
                param.requires_grad_(False)
                frozen_layers.append(name)
        # Let us just verify that things are in order.
        # This is a slow process and perhaps in the future, we can skip it.
        if verify_load:
            for layer in frozen_layers:
                loaded_module = self
                for attr in layer.split("."):
                    loaded_module = getattr(loaded_module, attr)
                # The module has been loaded and it should have requires_grad = False
                assert loaded_module.requires_grad == False, f"invalid result for {layer=}"

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.general.finetune_lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)


def get_resume(cfg, checkpoint_path, model):
    name = cfg.general.name + '_fine_tune'
    gpus = cfg.general.gpus
    model = model.load_pretrained_weights(checkpoint_path)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config_finetune')
def main(cfg: omegaconf.DictConfig):
    pl.seed_everything(cfg.train.seed)
    datamodule = vessel_dataset.VessapGraphDataModule(cfg)
    dataset_infos = vessel_dataset.VessapDatasetInfos(datamodule=datamodule, cfg=cfg)

    model = PretrainedDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=None)
    # Load the model weights
    cfg, _ = get_resume(cfg, cfg.general.test_only, model=model)
    # model.freeze_transformer_layers()

    callbacks = []
    # need to ignore metrics because otherwise ddp tries to sync them
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}-{val/epoch_NLL:.2f}',
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
                      max_epochs=cfg.general.finetune_epoch,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      )

    if cfg.model.do_train:
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
