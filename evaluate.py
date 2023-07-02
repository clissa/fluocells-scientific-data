"""
This scripts assess predictions on given dataset. It takes 3 CLI arguments:

 - dataset (green, yellow or red): dataset to assess
 - seed: initialization for data split and network weights/training cycle
 - gpu_id: id of the gpu to use for training

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-28
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path
from tqdm.auto import tqdm

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).absolute()

sys.path.append(str(FLUOCELLS_PATH))

import os
import pandas as pd
import numpy as np
from skimage import measure
from fastai.vision.all import *

from fluocells.config import (
    REPO_PATH,
    DATA_PATH,
    DATA_PATH_g,
    DATA_PATH_y,
    DATA_PATH_r,
    METADATA,
    MODELS_PATH,
)
from fluocells.utils.data import post_process
from fluocells.models import cResUnet, c_resunet
from fluocells.utils.metrics import eval_prediction


torch.set_printoptions(precision=10)

import argparse

parser = argparse.ArgumentParser(description="Run a basic training pipeline")

# Add the dataset argument
parser.add_argument(
    "dataset",
    type=str,
    choices=["green", "yellow", "red"],
    help="Dataset to train on: green, yellow, or red",
)

# Add the experiment argument
parser.add_argument(
    "experiment",
    type=str,
    help="Name of the experiment folder. Needed for setup of input/output paths",
)

# Add the threshold argument
parser.add_argument(
    "--bin_thresh",
    type=float,
    default=0.5,
    help="Threshold for heatmap binarization (default: 0.5)",
)

# Add the hole_size argument
parser.add_argument(
    "--max_hole",
    type=int,
    default=50,
    help="Maximum hole size to fill (default: 50)",
)

# Add the min_size argument
parser.add_argument(
    "--min_size",
    type=int,
    default=200,
    help="Minimum allowed object size. Smaller objects are removed (default: 200)",
)

# Add the max_filt argument
parser.add_argument(
    "--max_dist",
    type=int,
    default=30,
    help="Max filter argument in ndimage.maximum_filter(default: 30)",
)

# Add the footprint argument
parser.add_argument(
    "--fp",
    type=int,
    default=40,
    help="Footprint argument in peak_local_maxi (default: 40)",
)

# Add the iou threshold argument
parser.add_argument(
    "--iou_thresh",
    type=float,
    default=0.5,
    help="Threshold used as IoU overlapping to determine true positives (default: 0.5)",
)

# Add the footprint argument
parser.add_argument(
    "--prox_thresh",
    type=int,
    default=40,
    help="Threshold used for centers matching to determine true positives (default: 40)",
)

# Add the device argument
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cpu",
    help="Device to use to get model prediction (affect dataloader and model) (default: 'cpu')",
)


def main(postproc_cfg):
    BS = 1
    DATASET = postproc_cfg.dataset
    EXP_NAME = postproc_cfg.experiment
    VAL_PCT = 0.2
    dataset_path = globals()[f"DATA_PATH_{DATASET[0]}"]

    # model params
    N_IN, N_OUT = 16, 2

    # optimizer params
    LOSS_FUNC, LOSS_NAME = (
        DiceLoss(axis=1, smooth=1e-06, reduction="mean", square_in_union=False),
        "Dice",
    )

    log_path = REPO_PATH / "logs" / EXP_NAME
    model_path = MODELS_PATH / EXP_NAME
    # results_path = REPO_PATH / "results" / EXP_NAME
    # results_path.mkdir(exist_ok=True, parents=True)

    trainval_path = dataset_path / "trainval" / "images"

    # read train/valid/test split dataframe
    trainval_fnames = [fn for fn in trainval_path.iterdir()]

    # augmentation
    tfms = [
        IntToFloatTensor(div_mask=255.0),  # need masks in [0, 1] format
    ]

    # splitter
    splitter = RandomSplitter(valid_pct=VAL_PCT)

    def label_func(p):
        return Path(str(p).replace("images", "ground_truths/masks"))

    # dataloader
    dls = SegmentationDataLoaders.from_label_func(
        DATA_PATH,
        fnames=trainval_fnames,
        label_func=label_func,
        bs=BS,
        splitter=splitter,
        batch_tfms=tfms,
        device=postproc_cfg.device,
    )

    # test
    test_path = dataset_path / "test" / "images"
    test_fnames = [fn for fn in test_path.iterdir()]

    print(f"Number of test images: {len(test_fnames)}")
    test_dl = dls.test_dl(test_fnames, with_labels=True)

    # learner
    arch = "c-ResUnet"
    # pretrained=True would load Morelli et al. 2021 weights. We add new pretrained weights after
    cresunet = c_resunet(
        arch=arch,
        n_features_start=N_IN,
        n_out=N_OUT,
        pretrained=False,  # this would load Morelli et al. 2022
    )

    learn = Learner(
        dls,
        model=cresunet,
        loss_func=LOSS_FUNC,
        metrics=[Dice(), JaccardCoeff(), foreground_acc],
        path=log_path,
        model_dir=model_path,
    )

    print(f"Loading pesi da: {model_path}")
    # load training pipeline instead
    learn.load(model_path / "model")  # model.pth
    # freeze to eval mode
    learn.eval()

    # compute metrics
    C = 1
    metrics_df = pd.DataFrame(
        {}, columns="TP_iou FP_iou FN_iou TP_prox FP_prox FN_prox".split(" ")
    )
    for i, b in enumerate(tqdm(test_dl)):
        image_name = test_dl.items[i].name
        img, mask = b
        heatmap = (
            learn.model(img).squeeze().permute(1, 2, 0)[:, :, C].detach().to("cpu")
        )

        # convert to matplotlib format
        img = img.squeeze().permute(1, 2, 0).to("cpu")
        thresh_image = np.squeeze(
            (heatmap.numpy() > postproc_cfg.bin_thresh).astype("uint8")
        )
        post_proc_mask = post_process(
            heatmap.numpy(),
            max_hole_size=postproc_cfg.max_hole,
            min_object_size=postproc_cfg.min_size,
            max_filter_size=postproc_cfg.max_dist,
            footprint=postproc_cfg.fp,
        )

        mask_label = measure.label(mask.squeeze())
        # pred_mask_label = measure.label(pred_mask)
        pred_mask_label = measure.label(post_proc_mask)

        TP, FP, FN = eval_prediction(
            mask_label, pred_mask_label, "iou", postproc_cfg.iou_thresh
        )
        metrics_df.loc[image_name, "TP_iou FP_iou FN_iou".split(" ")] = TP, FP, FN

        TP, FP, FN = eval_prediction(
            mask_label, pred_mask_label, "proximity", postproc_cfg.prox_thresh
        )
        metrics_df.loc[image_name, "TP_prox FP_prox FN_prox".split(" ")] = TP, FP, FN

    return metrics_df


if __name__ == "__main__":
    args = parser.parse_args()
    results_path = REPO_PATH / "results" / args.experiment
    results_path.mkdir(exist_ok=True, parents=True)
    metrics_df = main(args)
    metrics_df.to_csv(results_path / "metrics.csv")
