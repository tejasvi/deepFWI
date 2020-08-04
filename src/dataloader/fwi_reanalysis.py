"""
The dataset class to be used with fwi-forcings and fwi-reanalysis data.
"""
from glob import glob

import xarray as xr
import numpy as np

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

        inp_files = glob(f"{forcings_dir}/ECMWF_FO_20*.nc")
        out_files = glob(f"{reanalysis_dir}/ECMWF_FWI_20*_1200_hr_fwi_e5.nc")

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
            self.input = ds.sortby("time")

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
            self.output = ds.sortby("time")

        if self.hparams.smos_input:
            self.smos_files = glob(f"{self.hparams.smos_dir}/20*_20*.as1.grib")

            with xr.open_mfdataset(
                self.smos_files,
                preprocess=lambda x: x.expand_dims("time"),
                engine="cfgrib",
                parallel=False,
            ) as ds:
                smos_input = ds

        # The t=0 dates
        self.dates = []
        for t in self.input.time.values:
            t = t.astype("datetime64[D]")
            if (
                # Date is within the range if specified
                (
                    not self.hparams.date_range
                    or self.hparams.date_range[0] <= t <= self.hparams.date_range[-1]
                )
                # Date is within the case-study range if specified
                and (
                    not self.hparams.case_study_dates
                    or min([r[0] <= t <= r[-1] for r in self.hparams.case_study_dates])
                )
                # Input data for preceding dates is available
                and all(
                    [
                        t - np.timedelta64(i, "D") in self.input.time.values
                        for i in range(self.hparams.in_days)
                    ]
                )
                # Output data for later dates is available
                and all(
                    [
                        t + np.timedelta64(i, "D") in self.output.time.values
                        for i in range(self.hparams.out_days)
                    ]
                )
            ):
                self.dates.append(t)
                if self.hparams.dry_run and len(self.dates) == 4:
                    break

        self.min_date = min(self.dates)

        # Required dates for operating on t=0 dates
        dates_spread = list(
            set(
                sum(
                    [
                        [
                            d + np.timedelta64(i - self.hparams.in_days, "D")
                            for i in range(
                                1, self.hparams.in_days + self.hparams.out_days
                            )
                        ]
                        for d in self.dates
                    ],
                    [],
                )
            )
        )

        # Load the data only for required dates
        self.input, self.output = (
            self.input.sel(time=dates_spread).load(),
            self.output.sel(time=dates_spread).load(),
        )

        log.info(f"Start date: {min(self.dates)}\nEnd date: {max(self.dates)}")
