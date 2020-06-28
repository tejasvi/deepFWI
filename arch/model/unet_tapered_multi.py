"""
U-Net model tapered at the end for low res output.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
from glob import glob
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

from model.exp1_m import Model as BaseModel


class Model(BaseModel):
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
                learning_rate=0.001,
                root_dir='/root/net/',
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

        Parameters
        ----------
        hparams : Namespace
            It contains all the major hyperparameters altering the training in some manner.
        """

        # init superclass
        super().__init__(hparams)


    def training_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["_log"]["_train_loss_unscaled"] for x in outputs]).mean()
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
            tensorboard_logs[f"val_loss_{n}"] = torch.stack([d[n] for d in [x["log"]["val_loss"] for x in outputs]]).mean()
            tensorboard_logs[f"val_acc_{n}"] = torch.stack([d[n] for d in [x["log"]["n_correct_pred"] for x in outputs]]).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack([d[n] for d in [x["log"]["abs_error"] for x in outputs]]).mean()

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        mean_error = np.mean([x["log"]["abs_error"].item() for x in outputs])

        tensorboard_logs = defaultdict(dict)
        tensorboard_logs["test_loss"] = avg_loss

        for n in range(self.data.n_output):
            tensorboard_logs[f"val_acc_{n}"] = torch.stack([d[n] for d in [x["log"]["n_correct_pred"] for x in outputs]]).mean()
            tensorboard_logs[f"abs_error_{n}"] = torch.stack([d[n] for d in [x["log"]["abs_error"] for x in outputs]]).mean()

        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
        }
