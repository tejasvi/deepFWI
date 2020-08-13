"""
The dataset class to be used with fwi-forcings and fwi-forecast data.
"""
from glob import glob

import xarray as xr

from dataloader.base_loader import ModelDataset as BaseDataset


class ModelDataset(BaseDataset):
    """
    The dataset class responsible for loading the data and providing the samples for
    training.

    :param BaseDataset: Base Dataset class to inherit from
    :type BaseDataset: base_loader.BaseDataset
    """

    def __init__(
        self, dates, forecast_dir, hparams=None, **kwargs,
    ):
        """
        Constructor for the ModelDataset class

        :param dates: The t=0 dates
        :type dates: list
        :param forecast_dir: The directory containing the FWI-Forecast data, defaults to
            None
        :type forecast_dir: str, optional
        :param hparams: Holds configuration values, defaults to None
        :type hparams: Namespace, optional
        """

        super().__init__(
            forecast_dir=forecast_dir, hparams=hparams, **kwargs,
        )

        # Create new `lead` dimension to avoid duplication along `time` dimension
        preprocess = (
            lambda x: x.rename_dims(time="lead")
            .rename_vars(name_dict={"time": "lead"})
            .expand_dims("time")
            .assign_coords(time=("time", [x.time[0].values]))
            .assign_coords(lead=("lead", range(10)))
        )

        out_files = glob(f"{forecast_dir}/ECMWF_FWI_20*_1200_hr_fwi.nc")

        with xr.open_mfdataset(
            inp_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.input = ds.sortby("time").load()

        out_files = sorted(
            glob(f"{forecast_dir}/ECMWF_FWI_2019*_1200_hr_fwi.nc"),
            # Extracting the month and date from filenames to sort by time.
            key=lambda x: int(x[-19:-17]) * 100 + int(x[-17:-15]),
        )[:184]
        out_invalid = lambda x: not (
            1 <= int(x[-19:-17]) <= 12 and 1 <= int(x[-17:-15]) <= 31
        )
        # Checking for valid date format
        assert not (sum([out_invalid(x) for x in out_files])), (
            "Invalid date format for output file(s)."
            "The dates should be formatted as YYMMDD."
        )
        with xr.open_mfdataset(
            out_files, preprocess=preprocess, engine="h5netcdf"
        ) as ds:
            self.output = ds.sortby("time").load()

        # Ensure timestamp matches for both the input and output
        assert len(self.input.time) == len(self.output.time)

        self.min_date = self.input.rh.time.min().values

        # Loading the mask for output variable if provided as generating from NaN mask
        self.mask = ~torch.isnan(torch.from_numpy(self.output["fwi"][0].values))
