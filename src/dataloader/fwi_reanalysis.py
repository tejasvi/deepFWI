"""
The dataset class to be used with fwi-forcings and fwi-reanalysis data.
"""
from glob import glob
import pickle

import xarray as xr
import numpy as np

import torch
import torchvision.transforms as transforms

from dataloader.base_loader import ModelDataset as BaseDataset
from pytorch_lightning import _logger as log


from data.fwi_reanalysis_stats import (
    REANALYSIS_FWI_MEAN,
    REANALYSIS_FWI_MAD,
    REANALYSIS_FWI_VAR,
    PROCESSED_REANALYSIS_FWI_VAR,
)
from data.forcing_stats import (
    FORCING_STD_TP,
    FORCING_STD_T2,
    FORCING_STD_WSPEED,
    FORCING_STD_RH,
    FORCING_MEAN_WSPEED,
    FORCING_MEAN_TP,
    FORCING_MEAN_T2,
    FORCING_MEAN_RH,
)


class ModelDataset(BaseDataset):
    """
    The dataset class responsible for loading the data and providing the samples for \
training.

    :param BaseDataset: Base Dataset class to inherit from
    :type BaseDataset: base_loader.BaseDataset
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
to defaults to None
        :type reanalysis_dir: str, optional
        :param transform: Custom transform for the input variable, defaults to None
        :type transform: torch.transforms, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

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
            self.hparams.in_days > 0 and self.hparams.out_days > 0
        ), "The number of input and output days must be > 0."
        self.n_input = self.hparams.in_days
        self.n_output = self.hparams.out_days
        self.hparams.thresh = REANALYSIS_FWI_MAD / 2

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
        assert not (sum([out_invalid(x) for x in out_files])), (
            "Invalid date format for output file(s)."
            "The dates should be formatted as YYMMDD."
        )
        self.out_files = out_files

        inp_invalid = lambda x: not (
            1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]) <= 31
        )
        assert not (sum([inp_invalid(x) for x in inp_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YYMMDD."
        )
        self.inp_files = inp_files

        # Consider only ground truth and discard forecast values
        preprocess = lambda x: x.isel(time=slice(0, 1))

        with xr.open_mfdataset(
            inp_files,
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
            coords="minimal",
            data_vars="minimal",
            compat="override",
        ) as ds:
            self.input = ds.sortby("time").load()

        with xr.open_mfdataset(
            out_files,
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
            coords="minimal",
            data_vars="minimal",
            compat="override",
        ) as ds:
            self.output = ds.sortby("time").load()

        # Ensure the data timestamp is ordered
        assert all(self.input.time.values[:-1] < self.input.time.values[1:])
        assert all(self.output.time.values[:-1] < self.output.time.values[1:])

        self.min_date = self.input.rh.time.min().values.astype("datetime64[D]")

        self.dates = []
            if all(
                [
                    t - np.timedelta(i, "D") in self.input.time.values
                    for i in range(self.inp_days)
                ]

        log.info(
            f"Start date: {self.output.fwi.time.min(skipna=True)}",
            f"\nEnd date: {self.output.fwi.time.max(skipna=True)}",
        )

        # Loading the mask for output variable if provided as generating from NaN mask
        self.mask = torch.from_numpy(
            np.load(self.hparams.mask)
            if self.hparams.mask
            else ~np.isnan(self.output["fwi"][0].values)
        ).cuda()

        # Mean of output variable used for bias-initialization.
        self.out_mean = out_mean if out_mean else REANALYSIS_FWI_MEAN

        # Variance of output variable used to scale the training loss.
        self.out_var = (
            out_var
            if out_var
            else REANALYSIS_FWI_MAD
            if self.hparams.loss == "mae"
            else PROCESSED_REANALYSIS_FWI_VAR
            if self.hparams.mask
            else REANALYSIS_FWI_VAR
        )

        # Input transforms including mean and std normalization
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    # Mean and standard deviation stats used to normalize the input data
                    # to the mean of zero and standard deviation of one.
                    transforms.Normalize(
                        [
                            x
                            for i in range(self.n_input)
                            for x in (
                                FORCING_MEAN_RH,
                                FORCING_MEAN_T2,
                                FORCING_MEAN_TP,
                                FORCING_MEAN_WSPEED,
                            )
                        ],
                        [
                            x
                            for i in range(self.n_input)
                            for x in (
                                FORCING_STD_RH,
                                FORCING_STD_T2,
                                FORCING_STD_TP,
                                FORCING_STD_WSPEED,
                            )
                        ],
                    ),
                ]
            )
        )
