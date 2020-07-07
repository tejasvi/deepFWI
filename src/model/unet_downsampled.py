"""
The original U-Net model with a downsampling layer at the end to match with the
FWI-Reanalysis resolution.
"""
import torch.nn as nn

from model.unet import Model as BaseModel


class Model(BaseModel):
    """
    The primary module containing all the training functionality. It is equivalent to
    PyTorch nn.Module in all aspects.
    """

    def __init__(self, hparams):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the
        model.

        Parameters
        ----------
        hparams : Namespace
            It contains all the major hyperparameters altering the training in some
            manner.
        """

        # init superclass
        super().__init__(hparams)
        out_channels = self.hparams.out_days
        features = self.hparams.init_features

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=4, stride=4,
        )
