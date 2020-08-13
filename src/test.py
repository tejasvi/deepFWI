"""
Primary inference and testing script. Run ``python3 test.py -h`` to see available
options.
"""
from argparse import Namespace
import random
import plac
import sys
import logging
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from IPython.display import clear_output

import torch

import pytorch_lightning as pl

sys.path.append("src")
from train import get_hparams, get_model  # noqa: E402

# Setting seeds to ensure reproducibility. Setting CUDA to deterministic mode slows down
# the training.
SEED = 2334
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(hparams, verbose=True):
    """
    Main testing routine specific for this project

    :param hparams: Namespace containing configuration values
    :type hparams: Namespace
    """
    # Set the evaluation flag
    hparams.eval = True

    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    model = get_model(hparams)
    if not hparams.benchmark:
        model.load_state_dict(torch.load(hparams.checkpoint_file)["state_dict"])
    model.eval()

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    trainer = pl.Trainer(gpus=hparams.gpus)  # , tb_logger],
        )
        wandb_logger.watch(model, log="all", log_freq=200)
        wandb_logger.log_hyperparams(model.hparams)
        for file in [
            i
            for s in [
                glob(x) for x in ["src/*.py", "src/dataloader/*.py", "src/model/*.py"]
            ]
            for i in s
        ]:
            shutil.copy(file, wandb.run.dir)

    trainer = pl.Trainer(
        gpus=hparams.gpus, logger=None if hparams.dry_run else [wandb_logger]
    )  # , tb_logger],

    # ------------------------
    # 3 START TESTING
    # ------------------------

    trainer.test(model)


if __name__ == "__main__":
    """
    Script entrypoint.
    """

    # Converting dictionary to namespace
    hyperparams = Namespace(**plac.call(get_hparams, eager=False))
    # Set the evaluation flag in hyperparamters
    hyperparams.eval = True
    # ---------------------
    # RUN TESTING
    # ---------------------

    main(hyperparams)
