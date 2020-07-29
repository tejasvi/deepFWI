"""
Base Dataset class to work with fwi-forcings data.
"""
from collections import defaultdict
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import torch
from torch.utils.data import Dataset


class ModelDataset(Dataset):
    """
    The dataset class responsible for loading the data and providing the samples for \
training.

    :param Dataset: Base Dataset class to use with PyTorch models
    :type Dataset: torch.utils.data.Dataset
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
        """
        Constructor for the ModelDataset class

        :param out_var: Variance of the output variable, defaults to None
        :type out_var: float, optional
        :param out_mean: Mean of the output variable, defaults to None
        :type out_mean: float, optional
        :param forecast_dir: The directory containing the FWI-Forecast data, defaults \
to None
        :type forecast_dir: str, optional
        :param forcings_dir: The directory containing the FWI-Forcings data, defaults \
to None
        :type forcings_dir: str, optional
        :param reanalysis_dir: The directory containing the FWI-Reanalysis data, \
defaults to None
        :type reanalysis_dir: str, optional
        :param transform: Custom transform for the input variable, defaults to None
        :type transform: torch.transforms, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        self.hparams = hparams
        self.out_mean = out_mean
        self.out_var = out_var
        if self.hparams.binned:
            self.bin_intervals = self.hparams.binned

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else self.hparams.out_mean
        # Variance of output variable used to scale the training loss.
        self.out_var = out_var if out_var else self.hparams.out_var

    def __len__(self):
        """
        The internal method used to obtain the number of iteration samples.

        :return: The maximum possible iterations with the provided data.
        :rtype: int
        """
        return len(self.dates)

    def __getitem__(self, idx):
        """
        Internal method used by pytorch to fetch input and corresponding output tensors.

        :param idx: The index number of data sample.
        :type idx: int
        :return: Batch of data containing input and output tensors
        :rtype: tuple
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = np.stack(
            [
                self.input[v].sel(time=self.dates[idx] - np.timedelta64(i, "D")).values
                for i in range(self.n_input)
                for v in ["rh", "t2", "tp", "wspeed"]
            ],
            axis=-1,
        )
        y = torch.from_numpy(
            np.stack(
                [
                    self.output[list(self.output.data_vars)[0]]
                    .sel(time=self.dates[idx] + np.timedelta64(i, "D"))
                    .values
                    for i in range(self.n_output)
                ],
                axis=0,
            )
        )

        if self.transform:
            X = self.transform(X)

        return X, y

    def training_step(self, model, batch):
        """
        Called inside the training loop with the data from the training dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """

        # forward pass
        x, y_pre = batch
        y_hat_pre = model(x)
        mask = model.data.mask.expand_as(y_pre[0][0])
        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_to_zero:
                    y_hat = y_hat[y > self.hparams.round_to_zero]
                    y = y[y > self.hparams.round_to_zero]
                if self.hparams.boxcox:
                    y = torch.from_numpy(
                        boxcox(y.cpu(), lmbda=self.hparams.boxcox,)
                    ).to(y.device)
                pre_loss = (y_hat - y) ** 2
                loss = pre_loss.mean()
                assert loss == loss
                tensorboard_logs["train_loss_unscaled"][str(c)] = loss
        loss = torch.stack(
            list(tensorboard_logs["train_loss_unscaled"].values())
        ).mean()
        tensorboard_logs["_train_loss_unscaled"] = loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "loss": loss.true_divide(model.data.out_var * model.data.n_output),
            "_log": tensorboard_logs,
        }

    def validation_step(self, model, batch):
        """
        Called inside the validation loop with the data from the validation dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """

        # forward pass
        x, y_pre = batch
        y_hat_pre = model(x)
        mask = model.data.mask.expand_as(y_pre[0][0])
        assert y_pre.shape == y_hat_pre.shape
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_to_zero:
                    y_hat = y_hat[y > self.hparams.round_to_zero]
                    y = y[y > self.hparams.round_to_zero]
                if self.hparams.boxcox:
                    y = torch.from_numpy(
                        boxcox(y.cpu(), lmbda=self.hparams.boxcox,)
                    ).to(y.device)
                pre_loss = (y_hat - y) ** 2
                loss = pre_loss.mean()
                assert loss == loss

                # Accuracy for a threshold
                n_correct_pred = (
                    ((y - y_hat).abs() < model.hparams.thresh).float().mean()
                )
                abs_error = (y - y_hat).abs().float().mean()

                tensorboard_logs["val_loss"][str(c)] = loss
                tensorboard_logs["n_correct_pred"][str(c)] = n_correct_pred
                tensorboard_logs["abs_error"][str(c)] = abs_error

        val_loss = torch.stack(list(tensorboard_logs["val_loss"].values())).mean()
        tensorboard_logs["_val_loss"] = val_loss
        # model.logger.log_metrics(tensorboard_logs)
        return {
            "val_loss": val_loss,
            "log": tensorboard_logs,
        }

    def test_step(self, model, batch):
        """
        Called inside the testing loop with the data from the testing dataloader \
passed in as `batch`.

        :param model: The chosen model
        :type model: Model
        :param batch: Batch of input and ground truth variables
        :type batch: int
        :return: Loss and logs
        :rtype: dict
        """
        x, y_pre = batch
        y_hat_pre, _ = model(x) if model.aux else model(x), None
        mask = model.data.mask.expand_as(y_pre[0][0])
        tensorboard_logs = defaultdict(dict)
        for b in range(y_pre.shape[0]):
            for c in range(y_pre.shape[1]):
                y = y_pre[b][c][mask]
                y_hat = y_hat_pre[b][c][mask]
                if self.hparams.round_to_zero:
                    y_hat = y_hat[y > self.hparams.round_to_zero]
                    y = y[y > self.hparams.round_to_zero]
                if self.hparams.clip_output:
                    y = y[
                        (y_hat < self.hparams.clip_output[1])
                        & (self.hparams.clip_output[0] < y_hat)
                    ]
                    y_hat = y_hat[
                        (y_hat < self.hparams.clip_output[1])
                        & (self.hparams.clip_output[0] < y_hat)
                    ]
                if self.hparams.boxcox:
                    # Negative predictions give NaN after inverse-boxcox
                    y_hat[y_hat < 0] = 0
                    y_hat = torch.from_numpy(
                        inv_boxcox(y_hat.cpu().numpy(), self.hparams.boxcox)
                    ).cuda()
                if self.hparams.clip_fwi:
                    y = y[(y_hat < UPPER_BOUND_FWI) & (LOWER_BOUND_FWI < y_hat)]
                    y_hat = y_hat[(y_hat < UPPER_BOUND_FWI) & (LOWER_BOUND_FWI < y_hat)]

                if not y.numel():
                    return None

                pre_loss = (
                    (y_hat - y).abs()
                    if model.hparams.loss == "mae"
                    else (y_hat - y) ** 2
                )

                loss = lambda low, high: pre_loss[(y > low) & (y <= high)].mean()
                assert loss(y.min(), y.max()) == loss(y.min(), y.max())

                # Accuracy for a threshold
                n_correct_pred = (
                    lambda low, high: (
                        (y - y_hat)[(y > low) & (y <= high)].abs()
                        < model.hparams.thresh
                    )
                    .float()
                    .mean()
                )

                # Mean absolute error
                abs_error = (
                    lambda low, high: (y - y_hat)[(y > low) & (y <= high)]
                    .abs()
                    .float()
                    .mean()
                    if model.hparams.loss == "mae"
                    else (y - y_hat)[(y > low) & (y <= high)].abs().float().mean()
                )

                tensorboard_logs["test_loss"][str(c)] = loss(y.min(), y.max())
                tensorboard_logs["n_correct_pred"][str(c)] = n_correct_pred(
                    y.min(), y.max()
                )
                tensorboard_logs["abs_error"][str(c)] = abs_error(y.min(), y.max())

                # Inference on binned values
                if self.hparams.binned:
                    for i in range(len(self.bin_intervals) - 1):
                        low, high = (
                            self.bin_intervals[i],
                            self.bin_intervals[i + 1],
                        )
                        tensorboard_logs[f"test_loss_{low}_{high}"][str(c)] = loss(
                            low, high
                        )
                        tensorboard_logs[f"n_correct_pred_{low}_{high}"][
                            str(c)
                        ] = n_correct_pred(low, high)
                        tensorboard_logs[f"abs_error_{low}_{high}"][str(c)] = abs_error(
                            low, high
                        )
                    tensorboard_logs[f"test_loss_{self.bin_intervals[-1]}_max"][
                        str(c)
                    ] = loss(self.bin_intervals[-1], y.max())
                    tensorboard_logs[f"n_correct_pred_{self.bin_intervals[-1]}_max"][
                        str(c)
                    ] = n_correct_pred(self.bin_intervals[-1], y.max())
                    tensorboard_logs[f"abs_error_{self.bin_intervals[-1]}_max"][
                        str(c)
                    ] = abs_error(self.bin_intervals[-1], y.max())

        test_loss = torch.stack(list(tensorboard_logs["test_loss"].values())).mean()
        tensorboard_logs["_test_loss"] = test_loss

        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
        }
