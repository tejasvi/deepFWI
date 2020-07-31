"""
The dataset class to be used with fwi-forcings and fwi-reanalysis data.
"""
from glob import glob

import xarray as xr
import numpy as np

import torch

from dataloader.base_loader import ModelDataset as BaseDataset
from pytorch_lightning import _logger as log


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
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        super().__init__(
            out_var=out_var,
            out_mean=out_mean,
            forecast_dir=forecast_dir,
            forcings_dir=forcings_dir,
            reanalysis_dir=reanalysis_dir,
            hparams=hparams,
            **kwargs,
        )

        # Number of input and prediction days
        assert (
            self.hparams.in_days > 0 and self.hparams.out_days > 0
        ), "The number of input and output days must be > 0."
        self.hparams.thresh = self.hparams.out_mad / 2

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

        if self.hparams.dry_run:
            inp_files = inp_files[: 8 * (self.n_output + self.n_input)]
            out_files = out_files[: 2 * (self.n_output + self.n_input)]

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

        self.min_date = self.input.rh.time.min().values.astype("datetime64[D]")

        self.dates = []
        for t in self.input.time.values:
            if all(
                [
                    t - np.timedelta64(i, "D") in self.input.time.values
                    for i in range(self.hparams.in_days)
                ]
            ) and all(
                [
                    t + np.timedelta64(i, "D") in self.output.time.values
                    for i in range(self.hparams.out_days)
                ]
            ):
                self.dates.append(t)

        log.info(f"Start date: {self.dates[0]}\nEnd date: {self.dates[-1]}",)

        # Loading the mask for output variable if provided as generating from NaN mask
        self.mask = torch.from_numpy(
            np.load(self.hparams.mask)
            if self.hparams.mask
            else ~np.isnan(self.output["fwi"][0].values)
        ).to("cuda" if self.hparams.gpus else "cpu")
