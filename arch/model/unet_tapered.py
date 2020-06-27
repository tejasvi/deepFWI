"""
U-Net model tapered at the end for low res output.
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
        delattr(self, 'upconv2')
        delattr(self, 'upconv1')


    def forward(self, x):
        """
        Forward pass
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        return self.conv(dec1)
