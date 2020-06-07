"""
Phase 2 Dataset class to be used with U-Net and FCN-ResNet101 models.
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
import json
import glob

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
        inp_daily=None,
        inp_invar=None,
        output=None,
        out=None,
        mask=None,
        root_dir=None,
        transform=None,
        hparams=None,
        **kwargs,
    ):

        self.hparams = hparams

        file_list = glob(f'{forcings_dir}/ECMWF_FO_2019*.nc')
        files = sorted(sorted(glob(f'{forcings_dir}/ECMWF_FO_2019*.nc')), key=lambda x: int(x.split("2019")[1].split("_1200_hr_")[0][:2])*100 + int(x.split("2019")[1].split("_1200_hr_")[0][2:]))
        with xr.open_mfdataset(file_list) as ds:
            self.input = ds


        file_list = glob(f'{forecast_dir}/ECMWF_FWI_2019*.nc')
        

        if inp_invar:
            self.inp_invar = inp_invar
        else:
            with xr.open_dataset(root_dir + "/era5_invar.nc") as ds:
                self.inp_invar = np.stack(
                    [ds[var].values.squeeze() for var in ds.data_vars], axis=-1
                )

        if output:
            self.output = transforms.Compose([transforms.ToTensor(),])(
                output
            ).permute(1, 2, 0)
        else:
            with xr.open_dataset(root_dir + "/" + out + ".nc4") as ds:
                self.output = transforms.Compose([transforms.ToTensor(),])(
                    ds[next(iter(ds.data_vars))].values
                ).permute(1, 2, 0)

        if mask:
            self.mask = mask
        else:
            if out == "gfas_full":
                with xr.open_dataset(root_dir + "/mask.nc4") as ds:
                    self.mask = torch.from_numpy(ds.mask.values)
            elif out == "reanalysis_fwi_africa":
                self.mask = ~torch.isnan(self.output[0])

        self.out_mean = self.output[self.mask.expand_as(self.output)].mean()
        self.out_var = self.output[self.mask.expand_as(self.output)].var()

        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (
                        -5.7299817e-01,
                        6.2098390e-01,
                        3.5890472e01,
                        3.5890472e01,
                        7.0098579e-02,
                        9.1358015e-05,
                        1.6867602e-01,
                        9.4428211e-01,
                        3.6612868e-03,
                        3.9232330e00,
                    ),
                    (
                        2.9325078e00,
                        2.8127456e00,
                        9.5853851e01,
                        9.5853851e01,
                        2.6476136e-01,
                        5.3015084e-04,
                        0.33469415,
                        1.605381,
                        0.00712241,
                        7.042249,
                    ),
                ),
            ]
        )

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, idx):
        """
        Internal method used by pytorch to fetch input and corresponding output tensors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = np.concatenate((self.inp_daily[idx], self.inp_invar), axis=-1)
        y = self.output[idx].unsqueeze(0)

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
        y = y_pre[mask]; y_hat = y_hat_pre[mask]
        pre_loss = (y_hat - y) ** 2
        loss = pre_loss.mean()
        if model.aux:
            aux_y_hat = aux_y_hat[mask]
            aux_pre_loss = (aux_y_hat - y) ** 2
            loss += 0.3 * aux_pre_loss.mean()
        if loss != loss:
            breakpoint()
        tensorboard_logs = {"train_loss_unscaled": loss.item()}
        self.logger.experiment.log(tensorboard_logs)
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
        y = y_pre[mask]; y_hat = y_hat_pre[mask]
        pre_loss = (y_hat - y) ** 2
        val_loss = pre_loss.mean()

        # Accuracy for multiple thresholds
        n_correct_pred = (
            (((y - y_hat).abs() < model.hparams.thresh / 2)).float().mean()
        )
        n_correct_pred_10 = (
            (((y - y_hat).abs() < model.hparams.thresh)).float().mean()
        )
        n_correct_pred_20 = (
            (((y - y_hat).abs() < model.hparams.thresh * 2)).float().mean()
        )
        abs_error = ((y - y_hat).abs()).float().mean()
        tensorboard_logs = {
            "val_loss": val_loss.item(),
            "n_correct_pred": n_correct_pred,
            "n_correct_pred_10": n_correct_pred_10,
            "n_correct_pred_20": n_correct_pred_20,
            "abs_error": abs_error,
        }
        self.logger.experiment.log(tensorboard_logs)
        return {
            "val_loss": val_loss,
            "log": tensorboard_logs,
        }

    def test_step(self, model, batch, batch_idx):
        """ Called during manual invocation on test data."""
        x, y = batch
        y_hat, _ = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y)
        y = y[mask]; y_hat = y_hat[mask]
        pre_loss = (y_hat - y) ** 2
        test_loss = pre_loss.mean()

        # Accuracy for multiple thresholds
        n_correct_pred = (
            (((y - y_hat).abs() < model.hparams.thresh / 2)).float().mean()
        )
        n_correct_pred_10 = (
            (((y - y_hat).abs() < model.hparams.thresh)).float().mean()
        )
        n_correct_pred_20 = (
            (((y - y_hat).abs() < model.hparams.thresh * 2)).float().mean()
        )
        abs_error = ((y - y_hat).abs()).float().mean()
        tensorboard_logs = {
            "test_loss": test_loss.item(),
            "n_correct_pred": n_correct_pred,
            "n_correct_pred_10": n_correct_pred_10,
            "n_correct_pred_20": n_correct_pred_20,
            "abs_error": abs_error,
        }
        self.logger.experiment.log(tensorboard_logs)
        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
        }
