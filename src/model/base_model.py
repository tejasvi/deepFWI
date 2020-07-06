"""
Base model implementing helper methods.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
from glob import glob
import types
import pickle
from collections import defaultdict

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

    def training_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack(
            [x["_log"]["_train_loss_unscaled"] for x in outputs]
        ).mean()
        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["train_loss"] = avg_loss
        return {
            "train_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["val_loss"] = avg_loss

        for n in range(self.data.n_output):
            tensorboard_logs[f"val_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["val_loss"] for x in outputs]]
            ).mean()
            tensorboard_logs[f"val_acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["n_correct_pred"] for x in outputs]]
            ).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["abs_error"] for x in outputs]]
            ).mean()

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["test_loss"] = avg_loss

        for n in range(self.data.n_output):
            tensorboard_logs[f"test_loss_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["test_loss"] for x in outputs]]
            ).mean()
            tensorboard_logs[f"test_acc_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["n_correct_pred"] for x in outputs]]
            ).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack(
                [d[str(n)] for d in [x["log"]["abs_error"] for x in outputs]]
            ).mean()

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
            if self.hparams.test_set:
                if hasattr(self.hparams, "eval"):
                    self.train_data = self.test_data = self.data
                else:
                    self.train_data = self.data
                    hparams = self.hparams
                    hparams.eval = True
                    self.test_data = ModelDataset(
                        forecast_dir=self.hparams.forecast_dir,
                        forcings_dir=self.hparams.forcings_dir,
                        reanalysis_dir=self.hparams.reanalysis_dir,
                        hparams=hparams,
                        out=self.hparams.out,
                    )
            else:
                self.train_data, self.test_data = torch.utils.data.random_split(
                    self.data,
                    [
                        len(self.data) * 8 // 10,
                        len(self.data) - len(self.data) * 8 // 10,
                    ],
                )
            if self.hparams.case_study and not self.hparams.test_set:
                assert (
                    max(self.test_data.indices) > 214
                ), "The data is outside the range of case study"
                self.test_data.indices = list(
                    set(self.test_data.indices) & set(range(214, 335))
                )

            # Saving list of test-set files
            if self.hparams.save_test_set:
                with open(self.hparams.save_test_set, "wb") as f:
                    pickle.dump(
                        [
                            sum(
                                [
                                    self.data.inp_files[i : i + 4]
                                    for i in self.test_data.indices
                                ],
                                [],
                            ),
                            [self.data.out_files[i] for i in self.test_data.indices],
                        ],
                        f,
                    )

            # Set flag to avoid resource intensive preparation during next call
            self.data_prepared = True

    def train_dataloader(self):
        log.info("Training data loader called.")
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            shuffle=True,
            pin_memory=True if self.hparams.gpus else False,
        )

    def val_dataloader(self):
        log.info("Validation data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            pin_memory=True if self.hparams.gpus else False,
        )

    def test_dataloader(self):
        log.info("Test data loader called.")
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=0 if self.hparams.dry_run else 8,
            pin_memory=True if self.hparams.gpus else False,
        )
