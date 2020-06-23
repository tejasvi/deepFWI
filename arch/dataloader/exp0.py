"""
Experiment 0 dataset class to be used with U-Net model.
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
from torch.nn import Sequential, MaxPool2d, ReLU, BatchNorm2d, Conv2d
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Logging helpers
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
import wandb

from dataloader.fwi_global import ModelDataset as BaseDataset


class ModelDataset(BaseDataset):
    """
    The dataset class responsible for loading the data and providing the samples for
    training.
    """

    def __init__(
        self,
        out_var=None,
        out_mean=None,
        forecast_dir=None,
        forcings_dir=None,
        reanalysis_dir=None,
        transform=None,
        hparams=None,
        **kwargs,
    ):

        self.hparams = hparams
        self.out_mean = out_mean
        self.out_var = out_var

        preprocess = lambda x: x.isel(time=slice(0, 1))

        inp_files = sorted(
            sorted(glob(f"{forcings_dir}/ECMWF_FO_20*.nc")),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) * 100
            + int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]),
        )[:12]
        inp_invalid = lambda x: not (
            1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]) <= 31
        )
        assert not (
            sum([inp_invalid(x) for x in out_files])
        ), "Invalid date format for input file(s). The dates should be formatted as YYMMDD."

        with xr.open_mfdataset(
            inp_files, preprocess=preprocess, engine="h5netcdf",  # parallel=True,
        ) as ds:
            self.input = ds.load()

        out_files = sorted(
            glob(f"{reanalysis_dir}/ECMWF_FWI_20*_1200_hr_fwi_e5.nc"),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x[-22:-20]) * 100 + int(x[-20:-18]),
        )[:3]
        out_invalid = lambda x: not (
            1 <= int(x[-22:-20]) <= 12 and 1 <= int(x[-20:-18]) <= 31
        )
        assert not (
            sum([out_invalid(x) for x in out_files])
        ), "Invalid date format for output file(s). The dates should be formatted as YYMMDD."

        with xr.open_mfdataset(
            out_files, preprocess=preprocess, engine="h5netcdf",  # parallel=True,
        ) as ds:
            self.output = ds.load()

        assert self.output.fwi.time.min(skipna=True) == self.input.rh.time.min(
            skipna=True
        )
        assert self.output.fwi.time.max(skipna=True) == self.input.rh.time.max(
            skipna=True
        )

        print(
            {
                "start_date": self.output.fwi.time.min(skipna=True),
                "end_date": self.output.fwi.time.max(skipna=True),
            }
        )

        assert len(self.input.time) == len(self.output.time)

        self.mask = ~torch.isnan(torch.from_numpy(self.output["fwi"][0].values))

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else 15.292629

        # Variance of output variable used to scale the training loss.
        self.out_var = (
            out_var
            if out_var
            else 18.819166
            if self.hparams.loss == "mae"
            else 621.65894
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Mean and standard deviation stats used to normalize the input data to
                # the mean of zero and standard deviation of one.
                transforms.Normalize(
                    (
                        72.47605,
                        279.96622,
                        2.4548044,
                        6.4765906,
                        72.47605,
                        279.96622,
                        2.4548044,
                        6.4765906,
                    ),
                    (
                        17.7426847,
                        21.2802498,
                        6.3852794,
                        3.69688883,
                        17.7426847,
                        21.2802498,
                        6.3852794,
                        3.69688883,
                    ),
                ),
            ]
        )

    def __getitem__(self, idx):
        """
        Internal method used by pytorch to fetch input and corresponding output tensors.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = np.stack(
            (
                self.input["rh"][idx],
                self.input["t2"][idx],
                self.input["tp"][idx],
                self.input["wspeed"][idx],
                self.input["rh"][idx + 1],
                self.input["t2"][idx + 1],
                self.input["tp"][idx + 1],
                self.input["wspeed"][idx + 1],
            ),
            axis=-1,
        )
        y = torch.from_numpy(self.output["fwi"][idx + 1].values).unsqueeze(0)

        if self.transform:
            X = self.transform(X)

        return X, y
