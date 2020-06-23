"""
Primary testing script.
"""
import os
from argparse import ArgumentParser
from argparse import Namespace
import random
import time
from glob import glob
import shutil
import importlib
import plac

import numpy as np
import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import wandb

from main import str2num, hparams

# Setting seeds to ensure reproducibility. Setting CUDA to deterministic mode slows down
# the training.
SEED = 2334
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(hparams):
    """
    Main testing routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    Model = importlib.import_module(f"model.{hparams.model}").Model
    if hparams.model in ["unet"]:
        if hparams.out == 'fwi_global':
            ModelDataset = importlib.import_module(f"dataloader.fwi_global").ModelDataset
    elif hparams.model in ["exp0_m"]:
        if hparams.out == 'exp0':
            ModelDataset = importlib.import_module(f"dataloader.exp0").ModelDataset

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=hparams.checkpoint)

    model = Model(hparams)#.to(non_blocking=True)
    model.prepare_data(ModelDataset)

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    tb_logger = TensorBoardLogger(save_dir="logs/tb_logs/", name=name)
    tb_logger.experiment.add_graph(model, model.data[0][0].unsqueeze(0))
    wandb_logger = WandbLogger(
        name=hparams.comment if hparams.comment else time.ctime(),
        project=name,
        save_dir="logs",
    )
    if not hparams.test:
        wandb_logger.watch(model, log="all", log_freq=100)
    wandb_logger.log_hyperparams(model.hparams)
    for file in [
        i
        for s in [glob(x) for x in ["*.py", "dataloader/*.py", "model/*.py"]]
        for i in s
    ]:
        shutil.copy(file, wandb.run.dir)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        deterministic=False,
        gpus=hparams.gpus,
        checkpoint_callback=checkpoint_callback,
        logger=[wandb_logger, tb_logger],
    )
    # ------------------------
    # 3 START TESTING
    # ------------------------

    trainer.test(ckpt_path=hparams.checkpoint)

if __name__ == "__main__":

    # Converting dictionary to namespace
    hyperparams = Namespace(**plac.call(hparams, eager=False))
    print(hyperparams)

    # ---------------------
    # RUN TESTING
    # ---------------------

    main(hyperparams)
