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

    # ------------------------
    # 3 START TESTING
    # ------------------------

    # Temporary fix until next release of pytorch-lightning
    try:
        result = trainer.test(model, verbose=verbose)[0]
    except:
        result = trainer.test(model)[0]

    return result, model.hparams


def autolabel(rects, ax, width):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    :param rects: Bar containers
    :type rects: matplotlib.container.BarContainer
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=min(90 * width, 25),
            zorder=5,
        )


def single_day_plot(result, hparams, m, benchmark=None):
    """
    Plot mteric results for single day output.

    :param result: Model inference result dictionary
    :type result: dict
    :param hparams: Hyperparamters
    :type hparams: Namespace
    :param benchmark: Benchmark result dictionary, defaults to None
    :type benchmark: dict, optional
    """
    bin_range = hparams.binned
    bin_labels = [
        f"({bin_range[i]}, {bin_range[i+1]}]"
        for i in range(len(bin_range))
        if i < len(bin_range) - 1
    ]
    bin_labels.append(f"({bin_range[-1]}, inf)")

    xlab = "Prediction day"
    # The label locations
    x = np.arange(len(bin_labels))
    # The width of the bars
    width = 0.7 / (3 if benchmark else 2)
    title = f"{hparams.in_days} Day Input // 1 Day Prediction (Global)"
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ylabel = {
        "acc": "Accuracy",
        "mae": "Mean absolute error",
        "mse": "Mean squared error",
    }
    ax.set_ylabel(ylabel[m])
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)

    num_groups = 2 if benchmark else 1

    rect_list = []
    preds = [x[0] for x in result.values()]
    rect_list.append(
        ax.bar(
            x - width * num_groups / 2 + width * 1 / 2, preds, width, label="deepFWI"
        )
    )
    if benchmark:
        bench = [x[0] for x in benchmark.values()]
        rect_list.append(
            ax.bar(
                x - width * num_groups / 2 + width * 1 / 2 + width * 1,
                bench,
                width,
                label="FWI-Forecast",
            )
        )

    for rect in rect_list:
        autolabel(rect, ax, width)

    if benchmark:
        ax.legend()

    fig.tight_layout()
    plt.show()



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
