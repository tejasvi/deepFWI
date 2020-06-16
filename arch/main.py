"""
Primary training and evaluation script.
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
        if hparams.out == 'fwi_global':
            ModelDataset = importlib.import_module(f"dataloader.fwi_global").ModelDataset
    elif hparams.model in ["exp0_m"]:
        if hparams.out == 'exp0':
            ModelDataset = importlib.import_module(f"dataloader.exp0").ModelDataset

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

    model = Model(hparams).to(non_blocking=True)
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
            # benchmark=True,
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


def hparams(
    #
    # U-Net config
    init_features: ("Architecture complexity", "option") = 11,
    in_channels: ("Number of input channels", "option") = 8,
    #
    # General
    epochs: ("Number of training epochs", "option") = 100,
    learning_rate: ("Maximum learning rate", "option") = 0.001,
    loss: ("Loss function: mae or mse", "option") = "mse",
    batch_size: ("Batch size of the input", "option") = 1,
    split: ("Test split fraction", "option") = 0.2,
    use_16bit: ("Use 16-bit precision for training", "option") = True,
    gpus: ("Number of GPUs to use", "option") = 1,
    optim: ("Learning rate optimizer: one_cycle or cosine", "option") = "one_cycle",
    #
    # Run specific
    model: ("Model to use: unet or exp0_m", "option") = "exp0_m",
    out: ("Output data for training: fwi_global or exp0", "option") = "exp0",
    forecast_dir: (
        "Directory containing forecast data",
        "option",
    ) = "/nvme0/fwi-forecast",
    forcings_dir: (
        "Directory containing forcings data",
        "option",
    ) = "/nvme1/fwi-forcings",
    reanalysis_dir: (
        "Directory containing reanalysis data",
        "option",
    ) = "/nvme0/fwi-reanalysis",
    thresh: ("Threshold for accuracy: Half of output MAD", "option") = 9.4, # 10.4, 9.4
    comment: ("Used for logging", "option") = "None",
    #
    # Test run
    test: ("Use model for evaluation", "option") = False,
    checkpoint: ("Path to the test model checkpoint", "option") = "",
):
    """
    The project wide arguments. Run `python main.py -h` for usage details.

    Returns
    -------
    Dict
        Dictionary containing configuration options.
    """
    return locals()


if __name__ == "__main__":

    # Converting dictionary to namespace
    hyperparams = Namespace(**plac.call(hparams, eager=False))
    print(hyperparams)

    # ---------------------
    # RUN TRAINING
    # ---------------------

    main(hyperparams)
