"""
The dataset class to be used with fwi-forcings and fwi-reanalysis data.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
from glob import glob
from collections import defaultdict
import pickle

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

from dataloader.base_loader import ModelDataset as BaseDataset


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

        super().__init__(
            out_var=out_var,
            out_mean=out_mean,
            forecast_dir=forecast_dir,
            forcings_dir=forcings_dir,
            reanalysis_dir=reanalysis_dir,
            transform=transform,
            hparams=hparams,
            **kwargs,
        )

        # Number of input and prediction days
        assert (
            not self.hparams.in_days > 0 and self.hparams.out_days > 0
        ), "The number of input and output days must be > 0."
        self.n_input = self.hparams.in_days
        self.n_output = self.hparams.out_days

        # Generate the list of all valid files in the specified directories
        get_inp_time = (
            lambda x: int(x.split("_20")[1][:2]) * 10000
            + int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) * 100
            + int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:])
        )
        inp_files = sorted(
            sorted(glob(f"{forcings_dir}/ECMWF_FO_20*.nc")),
            # Extracting the month and date from filenames to sort by time.
            key=get_inp_time,
        )
        get_out_time = (
            lambda x: int(x[-24:-22]) * 10000 + int(x[-22:-20]) * 100 + int(x[-20:-18])
        )
        out_files = sorted(
            glob(f"{reanalysis_dir}/ECMWF_FWI_20*_1200_hr_fwi_e5.nc"),
            # Extracting the month and date from filenames to sort by time.
            key=get_out_time,
        )

        # Loading list of test-set files
        if self.hparams.test_set:
            with open(self.hparams.test_set, "rb") as f:
                test_out = pickle.load(f)
                time_indices = set(map(get_inp_time, inp_files))
                inp_index = {
                    k: [x for x in inp_files if get_inp_time(x) == k]
                    for k in time_indices
                }
                test_inp = sum(
                    [inp_index[t] for f in test_out for t in (get_out_time(f),)], [],
                )

        # Handling the input and output files using test-set files
        if not self.hparams.dry_run and "test_inp" in locals():
            if hasattr(self.hparams, "eval"):
                inp_files = test_inp
                out_files = test_out
            else:
                inp_files = list(set(inp_files) - set(test_inp))
                out_files = list(set(out_files) - set(test_out))

        if self.hparams.dry_run:
            inp_files = inp_files[: 8 * (self.n_output + self.n_input)]
            out_files = out_files[: 2 * (self.n_output + self.n_input)]

        # Align the output files with the input files
        offset = len(out_files) - len(inp_files) // 4
        out_files = out_files[offset:] if offset > 0 else out_files

        # Checking for valid date format
        out_invalid = lambda x: not (
            1 <= int(x[-22:-20]) <= 12 and 1 <= int(x[-20:-18]) <= 31
        )
        assert not (
            sum([out_invalid(x) for x in out_files])
        ), "Invalid date format for output file(s). The dates should be formatted as YYMMDD."
        self.out_files = out_files

        inp_invalid = lambda x: not (
            1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]) <= 31
        )
        assert not (
            sum([inp_invalid(x) for x in inp_files])
        ), "Invalid date format for input file(s). The dates should be formatted as YYMMDD."
        self.inp_files = inp_files

        # Consider only ground truth and discard forecast values
        preprocess = lambda x: x.isel(time=slice(0, 1))

        with xr.open_mfdataset(
            inp_files,
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.input = ds.load()

        with xr.open_mfdataset(
            out_files,
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.output = ds.load()

        # Ensure timestamp matches for both the input and output
        assert self.output.fwi.time.min(skipna=True) == self.input.rh.time.min(
            skipna=True
        )
        assert self.output.fwi.time.max(skipna=True) == self.input.rh.time.max(
            skipna=True
        )
        assert len(self.input.time) == len(self.output.time)

        print(
            f"Start date: {self.output.fwi.time.min(skipna=True)}",
            f"\nEnd date: {self.output.fwi.time.max(skipna=True)}",
        )

        self.mask = (
            torch.nn.functional.max_pool2d(
                (
                    ~torch.from_numpy(
                        np.load(self.hparams.mask)
                        if self.hparams.mask
                        else ~np.isnan(self.output["fwi"][0].values)
                    )
                )
                .unsqueeze(0)
                .float(),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze()
            == 0
        ).cuda()

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else 15.292629

        # Variance of output variable used to scale the training loss.
        self.out_var = (
            out_var
            if out_var
            else 18.819166
            if self.hparams.loss == "mae"
            else 414.2136
            if self.hparams.mask
            else 621.65894
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Mean and standard deviation stats used to normalize the input data to
                # the mean of zero and standard deviation of one.
                transforms.Normalize(
                    [
                        x
                        for i in range(self.n_input)
                        for x in (72.47605, 279.96622, 2.4548044, 6.4765906,)
                    ],
                    [
                        x
                        for i in range(self.n_input)
                        for x in (17.7426847, 21.2802498, 6.3852794, 3.69688883,)
                    ],
                ),
            ]
        )

    def test_step(self, model, batch, batch_idx):
        """ Called during manual invocation on test data."""
        x, y_pre = batch
        y_hat_pre, _ = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y_pre[0][0])
        if self.hparams.case_study:
            mask = mask[355:480, 400:550]
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c]
                y_hat = y_hat_pre[b][c]
                if self.hparams.case_study:
                    y = y[355:480, 400:550][mask]
                    y_hat = y_hat[355:480, 400:550][mask]
                else:
                    y = y[mask]
                    y_hat = y_hat[mask]
                if self.hparams.clip_fwi:
                    y = y[(y_hat < 60) & (0.5 < y_hat)]
                    y_hat = y_hat[(y_hat < 60) & (0.5 < y_hat)]
                pre_loss = (
                    (y_hat - y).abs()
                    if model.hparams.loss == "mae"
                    else (y_hat - y) ** 2
                )
                loss = pre_loss.mean()
                assert loss == loss

                # Accuracy for a threshold
                n_correct_pred = (
                    ((y - y_hat).abs() < model.hparams.thresh).float().mean()
                )
                abs_error = (
                    (y - y_hat).abs().float().mean()
                    if model.hparams.loss == "mae"
                    else (y - y_hat).abs().float().mean()
                )

                tensorboard_logs["test_loss"][str(c)] = loss
                tensorboard_logs["n_correct_pred"][str(c)] = n_correct_pred
                tensorboard_logs["abs_error"][str(c)] = abs_error

        test_loss = torch.stack(list(tensorboard_logs["test_loss"].values())).mean()
        tensorboard_logs["_test_loss"] = test_loss

        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
        }
