import logging
import sys
import os
import shutil
from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Callback, seed_everything
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from src.common.utils import load_envs
import torch
torch.backends.cudnn.benchmark = True
# Set the cwd to the project root
os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """TODO: are these messing with multi-tpu training?""" 
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train.model_checkpoints:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, cfg=cfg, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    if cfg.eval_only:
        if cfg.model.weights is not None:
            print("Loading weights from ", cfg.model.weights)
            model = weights_update(model=model, checkpoint=torch.load(cfg.model.weights))

        trainer = pl.Trainer(
            logger=False,
            default_root_dir=hydra_dir,  # Path('./experiments/train'),
            # deterministic=cfg.train.deterministic,
            val_check_interval=cfg.logging.val_check_interval,
            log_every_n_steps=cfg.logging.log_every_n_steps,
            # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
            # auto_select_gpus=True,
            **cfg.train.pl_trainer,
        )

        hydra.utils.log.info(f"EVAL ONLY SELECTED. Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
        sys.exit()

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            name=cfg.data.datamodule.use_train_dataset + "__" + cfg.model.name,
            project=wandb_config.project,
            entity=wandb_config.entity,
            tags=cfg.core.tags,
            log_model=True,
        )
        hydra.utils.log.info(f"W&B is now watching <{wandb_config.watch.log}>!")
        wandb_logger.watch(
            model, log=wandb_config.watch.log, log_freq=wandb_config.watch.log_freq
        )

    hydra.utils.log.info(f"Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        #callbacks=callbacks,
        # deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        # progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        #auto_select_gpus=True,
        # benchmark=True,
        # accelerator='dp',
        #plugins=[DDPPlugin(find_unused_parameters=True)],
        **cfg.train.pl_trainer,
    )

    hydra.utils.log.info(f"Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info(f"Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    shutil.copytree(".hydra", Path(wandb_logger.experiment.dir) / "hydra")

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path="conf", config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
