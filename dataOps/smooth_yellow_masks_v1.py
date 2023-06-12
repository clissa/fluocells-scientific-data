"""
This script applies a combination of erosion and dilation operations to get smoother contours in binary masks.
This is thought to get more regular shapes for semi-automatic masks in FNC yellow v1 (obtained from thresholding).

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-09
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute().parent

sys.path.append(str(FLUOCELLS_PATH))

from fluocells.config import REPO_PATH, DATA_PATH as oldpath
from fluocells.utils.data import _remove_noise_from_masks

import shutil
import pandas as pd
from skimage import io
from tqdm.auto import tqdm
import multiprocessing

DISK_SIZE = 3
BIN_THRESHOLD = 150
MAX_HOLE_SIZE = 25
MIN_OBJECT_SIZE = 40

DATA_PATH = REPO_PATH / "dataset_v1_7"
MASKS_PATH = DATA_PATH / "yellow" / "masks"
CLEANED_PATH = DATA_PATH / "yellow" / "ground_truths" / "cleaned_masks"
CLEANED_PATH.mkdir(exist_ok=True, parents=True)

meta_df = pd.read_excel(oldpath / "metadata_v1_6.xlsx", sheet_name="metadata")


# Define the function to process a single mask
def process_mask(mask_path):
    mask = io.imread(mask_path, as_gray=True)

    smoothing = False if mask_path.name in manually_segmented else True
    cleaned_mask = _remove_noise_from_masks(
        mask,
        bin_thresh=BIN_THRESHOLD,
        max_hole_size=MAX_HOLE_SIZE,
        min_object_size=MIN_OBJECT_SIZE,
        smoothing=smoothing,
        disk_size=DISK_SIZE,
    )

    meta_df = pd.read_excel(oldpath / "metadata_v1_6.xlsx", sheet_name="metadata")
    save_name = meta_df[meta_df.original_name == mask_path.name].image_name.values[0]

    io.imsave(CLEANED_PATH / save_name, cleaned_mask, check_contrast=False)


def _restore_dataset_partitions(mask_paths):
    for p in mask_paths:
        row = meta_df[meta_df.image_name == p.name]
        outpath = (
            MASKS_PATH.parent / row.partition.values[0] / "ground_truths/masks" / p.name
        )
        outpath.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(p, outpath)


manually_segmented = [
    "Mar31bS2C1R2_VLPAGr_200x_y.png",
    "Mar33bS1C4R2_DMl_200x_y.png",
    "Mar40S1C2R2_DMl_200x_o.png",
    "Mar41S3C1R1_DMl_200x_o.png",
    "MAR38S1C3R1_LHR_20_o.png",
    "MAR38S1C3R1_DML_20_o.png",
    "Mar42S2C2R2_DMr_200x_o.png",
    "Mar36bS1C6R2_DMl_200x_y.png",
    "MAR44S4C3R2_VLPAGL_20_o.png",
    "Mar40S1C2R2_DMr_200x_o.png",
    "Mar41S3C1R1_DMr_200x_o.png",
    "Mar33bS2C1R1_DMl_200x_y.png",
    "Mar43S1C5R3_DMr_200x_o.png",
    "Mar37S1C2R1_DMr_200x_o.png",
    "Mar40S3C4R2_VLPAGr_200x_o.png",
    "Mar43S2C3R1_VLPAGl_200x_o.png",
    "Mar37S1C2R1_DMl_200x_o.png",
    "Mar36bS1C6R2_DMr_200x_y.png",
    "MAR55S3C2R2_VLPAGL_20_o.png",
    "MAR39S2C2R2_DMR_200x_o.png",
    "Mar43S2C3R1_VLPAGr_200x_o.png",
    "MAR55S3C2R2_VLPAGR_20_o.png",
    "Mar31bS2C3R4_DMr_200x_y.png",
    "Mar32bS2C2R2_DMl_200x_y.png",
    "MAR39S2C2R2_DML_200x_o.png",
    "MAR52S2C1R3_LHL_20_o.png",
    "Mar42S2C4R2_VLPAGr_200x_o.png",
    "MAR55S1C5R3_DMR_20_o.png",
    "MAR38S1C3R1_DMR_20_o.png",
    "Mar33bS1C4R2_DMr_200x_y.png",
    "Mar41S3C3R3_VLPAGl_200x_o.png",
]


if __name__ == "__main__":
    # Create a pool of worker processes
    num_processes = 6
    pool = multiprocessing.Pool(num_processes)

    # Process the masks in parallel
    mask_paths = [*MASKS_PATH.iterdir()]

    for _ in tqdm(pool.imap_unordered(process_mask, mask_paths), total=len(mask_paths)):
        pass

    # Close the pool to release resources
    pool.close()
    pool.join()
    _restore_dataset_partitions([*CLEANED_PATH.iterdir()])
