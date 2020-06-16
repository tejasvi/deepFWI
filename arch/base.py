"""
Base model implementing helper methods.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
from glob import glob
import types

import xarray as xr
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Logging helpers
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
import wandb


class BaseModel(LightningModule):
    """
    The primary module containing all the training functionality. It is equivalent to
    PyTorch nn.Module in all aspects.

    Usage
    -----

    Passing hyperparameters:

        >>> div=3
            x=269//div
            y=183//div
            params = dict(
                in_width=x,
                in_length=y,
                in_depth=7,
                output_size=x*y,
                drop_prob=0.5,
                epochs=20,
                optimizer_name="adam",
                batch_size=1
            )
        >>> from argparse import Namespace
        >>> hparams = Namespace(**params)
        >>> model = Model(hparams)
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the
        model.
        """

        # init superclass
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.data_prepared = False
        self.aux = False

    def forward(self, x):
        """
        Forward pass
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        Called inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        return self.data.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Called inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.data.validation_step(self, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Called during manual invocation on test data."""

        return self.data.test_step(self, batch, batch_idx)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = np.mean([x["log"]["n_correct_pred_10"].item() for x in outputs])
        mean_error = np.mean([x["log"]["abs_error"].item() for x in outputs])
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_acc": val_acc,
            "abs_error": mean_error,
        }
        # wandb.log(tensorboard_logs)
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = np.mean([x["log"]["n_correct_pred_10"].item() for x in outputs])
        mean_error = np.mean([x["log"]["abs_error"].item() for x in outputs])
        tensorboard_logs = {
            "test_loss": avg_loss,
            "test_acc": test_acc,
            "abs_error": mean_error,
        }
        # wandb.log(tensorboard_logs)
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return optimizers and learning rate schedulers.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.optim == "cosine":
            scheduler = [
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=0, verbose=True, threshold=1e-1
                ),
            ]
        elif self.hparams.optim == "one_cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=len(self.train_data),
                epochs=self.hparams.epochs,
            )
        return [optimizer], [scheduler]

    def add_bias(self, bias):
        for w in reversed(self.state_dict().keys()):
            if "bias" in w:
                self.state_dict()[w].fill_(bias)
                break

    def prepare_data(self, ModelDataset=None, force=False):
        """
        Load and split the data for training and test.
        """
        if self.data_prepared and not force:
            pass
        elif ModelDataset:
            self.data = ModelDataset(
                forecast_dir=self.hparams.forecast_dir,
                forcings_dir=self.hparams.forcings_dir,
                reanalysis_dir=self.hparams.reanalysis_dir,
                hparams=self.hparams,
                out=self.hparams.out,
            )
            self.add_bias(self.data.out_mean)
            self.train_data, self.test_data = torch.utils.data.random_split(
                self.data,
                [len(self.data) * 8 // 10, len(self.data) - len(self.data) * 8 // 10,],
            )

            self.data_prepared = True
        else:
            raise TypeError(
                "ModelDataset must be passed manually during the first run."
            )

    def train_dataloader(self):
        log.info("Training data loader called.")
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        log.info("Validation data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # pin_memory=True,
        )

    def test_dataloader(self):
        log.info("Test data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # pin_memory=True,
        )
