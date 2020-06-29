"""
Modification in U-Net model for exp0.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
from glob import glob

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

from model.unet import Model as BaseModel


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
        out_channels = self.hparams.out_channels
        features = self.hparams.init_features

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=4, stride=4,
        )
