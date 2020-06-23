"""
Dataset class for fwi-forcings and fwi-forecast.
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


class ModelDataset(Dataset):
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
            sorted(glob(f"{forcings_dir}/ECMWF_FO_2019*.nc")),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x.split("2019")[1].split("_1200_hr_")[0][:2]) * 100
            + int(x.split("2019")[1].split("_1200_hr_")[0][2:]),
        )[:736]
        with xr.open_mfdataset(
            inp_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.input = ds.load()

        out_files = sorted(
            glob(f"{forecast_dir}/ECMWF_FWI_2019*_1200_hr_fwi.nc"),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x[-19:-17]) * 100 + int(x[-17:-15]),
            )[:184]
        with xr.open_mfdataset(
            out_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.output = ds.load()

        assert len(self.input.time) == len(self.output.time)

        self.mask = ~torch.isnan(torch.from_numpy(self.output["fwi"][0].values))

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else 18.389227

        # Variance of output variable used to scale the training loss.
        self.out_var = out_var if out_var else 20.80943 if self.hparams.loss == 'mae' else 716.1736

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                
                # Mean and standard deviation stats used to normalize the input data to
                # the mean of zero and standard deviation of one.
                transforms.Normalize(
                    (72.03445, 281.2624, 2.4925985, 6.5504117,72.03445, 281.2624, 2.4925985, 6.5504117,),
                    (18.8233801, 21.9253515, 6.37190019, 3.73465273,18.8233801, 21.9253515, 6.37190019, 3.73465273,),
                ),
            ]
        )

    def __len__(self):
        return len(self.input.time) - 1

    def __getitem__(self, idx):
        """
        Internal method used by pytorch to fetch input and corresponding output tensors.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if 1 or idx % 2:
            X = np.stack(
                (
                    # self.input["rh"][idx][:, :2560],
                    # self.input["t2"][idx][:, :2560],
                    # self.input["tp"][idx][:, :2560],
                    # self.input["wspeed"][idx][:, :2560],
                    # self.input["rh"][idx+1][:, :2560],
                    # self.input["t2"][idx+1][:, :2560],
                    # self.input["tp"][idx+1][:, :2560],
                    # self.input["wspeed"][idx+1][:, :2560],self.input["rh"][idx][:, :2560],
                    self.input["rh"][idx],
                    self.input["t2"][idx],
                    self.input["tp"][idx],
                    self.input["wspeed"][idx],
                    self.input["rh"][idx+1],
                    self.input["t2"][idx+1],
                    self.input["tp"][idx+1],
                    self.input["wspeed"][idx+1],
                ),
                axis=-1,
            )
            y = torch.from_numpy(self.output["fwi"][idx+1].values).unsqueeze(0)
        else:
            X = np.stack(
                (
                    self.input["rh"][idx][:, 2560:],
                    self.input["t2"][idx][:, 2560:],
                    self.input["tp"][idx][:, 2560:],
                    self.input["wspeed"][idx][:, 2560:],
                    self.input["rh"][idx+1][:, 2560:],
                    self.input["t2"][idx+1][:, 2560:],
                    self.input["tp"][idx+1][:, 2560:],
                    self.input["wspeed"][idx+1][:, 2560:],
                ),
                axis=-1,
            )
            y = torch.from_numpy(self.output["fwi"][idx+1].values[:, 2560:]).unsqueeze(0)

        if self.transform:
            X = self.transform(X)

        return X, y

    def training_step(self, model, batch, batch_idx):
        """
        Called inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y_pre = batch
        y_hat_pre, aux_y_hat = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y_pre)
        if y_pre.shape != y_hat_pre.shape:
            breakpoint()
        y = y_pre[mask]
        y_hat = y_hat_pre[mask]
        pre_loss = (y_hat - y).abs() if model.hparams.loss == 'mae' else (y_hat - y) ** 2
        loss = pre_loss.mean()
        if model.aux:
            aux_y_hat = aux_y_hat[mask]
            aux_pre_loss = (y_hat - y).abs() if model.hparams.loss == 'mae' else (y_hat - y) ** 2
            loss += 0.3 * aux_pre_loss.mean()
        if loss != loss:
            breakpoint()
        tensorboard_logs = {"train_loss_unscaled": loss.item()}
        model.logger.log_metrics(tensorboard_logs)
        return {"loss": loss.true_divide(model.data.out_var), "log": tensorboard_logs}

    def validation_step(self, model, batch, batch_idx):
        """
        Called inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y_pre = batch
        y_hat_pre, aux_y_hat = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y_pre)
        if y_pre.shape != y_hat_pre.shape:
            breakpoint()
        y = y_pre[mask]
        y_hat = y_hat_pre[mask]
        pre_loss = (y_hat - y).abs() if model.hparams.loss == 'mae' else (y_hat - y) ** 2
        val_loss = pre_loss.mean()

        # Accuracy for multiple thresholds
        n_correct_pred = (((y - y_hat).abs() < model.hparams.thresh / 2)).float().mean()
        n_correct_pred_10 = (((y - y_hat).abs() < model.hparams.thresh)).float().mean()
        n_correct_pred_20 = (
            (((y - y_hat).abs() < model.hparams.thresh * 2)).float().mean()
        )
        abs_error = (y - y_hat).abs().float().mean() if model.hparams.loss == 'mae' else (y - y_hat).abs().float().mean()
        tensorboard_logs = {
            "val_loss": val_loss.item(),
            "n_correct_pred": n_correct_pred,
            "n_correct_pred_10": n_correct_pred_10,
            "n_correct_pred_20": n_correct_pred_20,
            "abs_error": abs_error,
        }
        model.logger.log_metrics(tensorboard_logs)
        return {
            "val_loss": val_loss,
            "log": tensorboard_logs,
        }

    def test_step(self, model, batch, batch_idx):
        """ Called during manual invocation on test data."""
        x, y = batch
        y_hat, _ = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y)
        y = y[mask]
        y_hat = y_hat[mask]
        pre_loss = (y_hat - y).abs() if model.hparams.loss == 'mae' else (y_hat - y) ** 2
        test_loss = pre_loss.mean()

        # Accuracy for multiple thresholds
        n_correct_pred = (((y - y_hat).abs() < model.hparams.thresh / 2)).float().mean()
        n_correct_pred_10 = (((y - y_hat).abs() < model.hparams.thresh)).float().mean()
        n_correct_pred_20 = (
            (((y - y_hat).abs() < model.hparams.thresh * 2)).float().mean()
        )
        abs_error = (y - y_hat).abs().float().mean() if model.hparams.loss == 'mae' else (y - y_hat).abs().float().mean()
        tensorboard_logs = {
            "test_loss": test_loss.item(),
            "n_correct_pred": n_correct_pred,
            "n_correct_pred_10": n_correct_pred_10,
            "n_correct_pred_20": n_correct_pred_20,
            "abs_error": abs_error,
        }
        model.logger.log_metrics(tensorboard_logs)
        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
        }
