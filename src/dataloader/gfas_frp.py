"""
The dataset class to be used with fwi-forcings and gfas-frp data.
"""
from glob import glob
import pickle

import xarray as xr
import numpy as np

import torch
import torchvision.transforms as transforms

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
        frp_dir=None,
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
            frp_dir=None,
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
        out_time = lambda x: int(x[-8:-6]) * 100 + int(x[-5:-3])
        out_files_orig = sorted(
            sorted(glob(f"{frp_dir}/FRP_20??_??.nc")),
            # Extracting the year and month from filenames to sort by time.
            key=out_time,
        )

        if self.hparams.save_test_set:
            # Create out_files list of size same as inp_list
            time_indices = set(map(get_inp_time, inp_files))
            out_files = sorted(
                [
                    f
                    for f in out_files_orig
                    for t in time_indices
                    if t // 100 == out_time(f)
                ],
                key=out_time,
            )
        else:
            out_files = out_files_orig

        # Loading list of test-set files
        if self.hparams.test_set:
            with open(self.hparams.test_set, "rb") as f:
                _, test_inp, test_out = pickle.load(f)

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

        # Checking for valid date format
        inp_invalid = lambda x: not (
            1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][:2]) <= 12
            and 1 <= int(x.split("_20")[1][2:].split("_1200_hr_")[0][2:]) <= 31
        )
        assert not (sum([inp_invalid(x) for x in inp_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YYMMDD."
        )
        self.inp_files = inp_files

        out_invalid = lambda x: not (1 <= int(x[-5:-3]) <= 12)
        assert not (sum([out_invalid(x) for x in out_files])), (
            "Invalid date format for input file(s)."
            "The dates should be formatted as YY_MM."
        )
        self.out_files = out_files

        # Consider only ground truth and discard forecast values
        preprocess = lambda x: x.isel(time=slice(0, 1))

        with xr.open_mfdataset(
            # Remove duplicated file names
            set(inp_files),
            preprocess=preprocess,
            engine="h5netcdf",
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.input = ds.load()

        with xr.open_mfdataset(
            # Remove duplicated file names
            set(out_files),
            parallel=False if self.hparams.dry_run else True,
            combine="by_coords",
        ) as ds:
            self.output = ds.load()

        # Ensure timestamp matches for both the input and output
        assert self.output.frpfire.time.min(skipna=True) <= self.input.rh.time.max(
            skipna=True
        )
        assert self.output.frpfire.time.max(skipna=True) >= self.input.rh.time.min(
            skipna=True
        )

        self.min_date = self.input.rh.time.min().values

        print(
            f"Start date: {self.output.frpfire.time.min(skipna=True)}",
            f"\nEnd date: {self.output.frpfire.time.max(skipna=True)}",
        )

        self.mask = torch.from_numpy(np.load(self.hparams.mask)).cuda()

        # Mean of output variable used for bias-initialization.
        self.out_mean = (
            out_mean
            if out_mean
            else 0.0008449411
            if self.hparams.mask
            else 0.00021382833
        )

        # Variance of output variable used to scale the training loss.
        self.out_var = (
            out_var
            if out_var
            else 18.819166
            if self.hparams.loss == "mae"
            else 0.0051485044
            if self.hparams.mask
            else 0.0012904834002256393
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
                self.input[v].sel(time=self.min_date + np.timedelta64(idx + 1, "D"))
                for i in range(self.n_input)
                for v in ["rh", "t2", "tp", "wspeed"]
            ],
            axis=-1,
        )
        y = torch.from_numpy(
            np.stack(
                [
                    self.output["frpfire"]
                    .sel(
                        time=self.min_date
                        + np.timedelta64(idx + self.n_input - 1 + i, "D")
                    )
                    .values
                    for i in range(self.n_output)
                ],
                axis=0,
            )
        )

        if self.transform:
            X = self.transform(X)

        return X, y
