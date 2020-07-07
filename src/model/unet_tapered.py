"""
U-Net model tapered at the end for low res output.
"""
import torch
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
        features = self.hparams.init_features
        delattr(self, "upconv2")
        delattr(self, "upconv1")

        self.res32 = nn.Conv2d((features * 2) * 2, features * 2, kernel_size=1)
        self.res31 = nn.Conv2d((features * 2) * 2, features, kernel_size=1)
        self.res21 = nn.Conv2d(features * 2, features, kernel_size=1)

    def forward(self, x):
        """
        Forward pass
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.decoder2(dec3) + self.res32(dec3)
        dec1 = self.decoder1(dec2) + self.res21(dec2) + self.res31(dec3)
        return self.conv(dec1)
