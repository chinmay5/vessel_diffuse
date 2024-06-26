# Do not move these imports, the order seems to matter
import re

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

from midi.datasets.cow_dataset import CoWGraphDataModule, CoWDatasetInfos
from midi.datasets.vessap_dataset import VessapGraphDataModule, VessapDatasetInfos
from midi.diffusion_model import FullDenoisingDiffusion

warnings.filterwarnings("ignore", category=PossibleUserWarning)


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


def get_resume(cfg, dataset_infos, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos)
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
    model.cfg.train = cfg.train
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config_cow')
# @hydra.main(version_base='1.3', config_path='../configs', config_name='config_vessap')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)
    if dataset_config.name == 'vessel':
        datamodule = VessapGraphDataModule(cfg)
        dataset_infos = VessapDatasetInfos(datamodule=datamodule, cfg=cfg)
    elif dataset_config.name == 'cow':
        datamodule = CoWGraphDataModule(cfg)
        dataset_infos = CoWDatasetInfos(datamodule=datamodule, cfg=cfg)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        cfg, model = get_resume(cfg, dataset_infos, cfg.general.test_only, test=True)
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        print("Resuming from {}".format(cfg.general.resume))
        cfg, model = get_resume(cfg, dataset_infos, cfg.general.resume, test=False)
    else:
        model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos)

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
                      # accumulate_grad_batches=4
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
