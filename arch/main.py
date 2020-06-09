"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
from argparse import Namespace
import random
import time
from glob import glob
import shutil
import importlib

import numpy as np
import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import wandb

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
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    Model = importlib.import_module(f"model.{hparams.model}").Model
    if hparams.model in ["unet"]:
        ModelDataset = importlib.import_module(f"dataloader.fwi_global").ModelDataset

    name = hparams.model + "-" + hparams.out
    if hparams.test:
        name += "-test"

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=f"model/checkpoints/{name}/bestmodel",
        monitor="val_loss",
        verbose=True,
        save_top_k=2,
        save_weights_only=False,
        mode="auto",
        period=1,
        prefix=name + time.ctime(),
    )

    model = Model(hparams)
    model.prepare_data(ModelDataset)

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    tb_logger = TensorBoardLogger(save_dir="logs/tb_logs/", name=name)
    tb_logger.experiment.add_graph(model, next(iter(model.train_dataloader()))[0])
    wandb_logger = WandbLogger(
        name=hparams.comment if hparams.comment else time.ctime(), project=name, save_dir='logs'
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

    # ------------------------
    # INIT TRAINER
    # ------------------------

    if hparams.test:
        trainer = pl.Trainer(gpus=hparams.gpus, logger=[wandb_logger, tb_logger],)
        model.load_state_dict(torch.load(hparams.checkpoint)["state_dict"])
        model.eval()
        trainer.test(model)

    else:
        trainer = pl.Trainer(
            auto_lr_find=False,
            show_progress_bar=False,
            # Profiling the code to find bottlenecks
            # profiler=pl.profiler.AdvancedProfiler('profile'),
            max_epochs=hparams.epochs,
            # CUDA trick to speed up training after the first epoch
            benchmark=True,
            deterministic=False,
            # Sanity checks
            # fast_dev_run=False,
            # overfit_pct=0.01,
            gpus=hparams.gpus,
            precision=16 if hparams.use_16bit else 32,
            # Alternative method for 16-bit training
            # amp_level="O2",
            logger=[wandb_logger, tb_logger],
            checkpoint_callback=checkpoint_callback,
            # Using maximum GPU memory. NB: Learning rate should be adjusted according to
            # the batch size
            # auto_scale_batch_size='binsearch',
        )
    
        # ------------------------
        # LR FINDER
        # ------------------------

        # # Run learning rate finder
        # lr_finder = trainer.lr_find(model)

        # # Results can be found in
        # lr_finder.results

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()
    
        # ------------------------
        # BATCH SIZE SEARCH
        # ------------------------

        # # update hparams of the model
        # model.hparams.learning_rate = new_lr
    
        # # Invoke the batch size search using more sophisticated paramters.
        # new_batch_size = trainer.scale_batch_size(
        #     model, mode="binary", steps_per_trial=50, init_val=1, max_trials=10
        # )

        # # Override old batch size
        # model.hparams.batch_size = new_batch_size

        # ------------------------
        # 3 START TRAINING
        # ------------------------

        trainer.fit(model)

        # # Manual saving the last model state (non needed ideally)
        # torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------

    # these are project-wide arguments

    params = dict(
        #
        # U-Net config
        init_features=7,
        in_channels=8,
        #
        # General
        epochs=100,
        learning_rate=0.01,
        batch_size=1,
        split=0.2,
        use_16bit=True,
        gpus=1,
        optim="one_cycle",  # one_cycle, cosine
        #
        # Run specific
        model="unet",  # unet
        out="fwi_global",  # fwi_global
        forecast_dir='/nvme0/fwi-forecast',
        forcings_dir='/nvme1/fwi-forcings',
        thresh=10.4,
        comment="Learning rate 1 16bit",
        #
        # Test run
        test=False,
        checkpoint='',
    )

    hyperparams = Namespace(**params)

    # ---------------------
    # RUN TRAINING
    # ---------------------

    main(hyperparams)
