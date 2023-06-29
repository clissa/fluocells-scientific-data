"""
This script transforms binary mask into several common annotation formats.

Supprorted formats include:
    - RLE encoding
    - Pascal VOC: polygon, bndbox, dots, count + metadata
    - COCO: segmentation, bbox, area, dots, count, iscrowd
    - VGG VIA: polygon

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-05
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute().parent

sys.path.append(str(FLUOCELLS_PATH))

import itertools
from skimage import io
from tqdm.auto import tqdm

from fluocells.config import DATA_PATH
from fluocells.utils.annotations import (
    binary_mask_to_rle,
    binary_mask_to_rle,
    save_rle_encoding,
    get_pascal_VOC_annotations,
    save_pascal_VOC_annotations,
    get_COCO_annotations,
    save_json_annotations,
    initialize_COCO_dict,
    get_VIA_annotations,
    initialize_VIA_dict,
)


import argparse

parser = argparse.ArgumentParser(description="Transform binary mask into several common annotation formats")

# Add the dataset argument
parser.add_argument(
    "--datasets",
    nargs="+",
    type=str,
    choices=["green", "yellow", "red"],
    default=["green", "yellow", "red"],
    help="Dataset(s) list. Option(s): green, yellow, red (default: green yellow red)",
)

# Add the split argument
parser.add_argument(
    "--splits",
    nargs="+",
    type=str,
    choices=["test", "trainval"],
    default=["test", "trainval"],
    help="Data split(s) list. Options: test, trainval (default: test trainval)",
)

args = parser.parse_args()

if __name__ == "__main__":
    datasets = args.datasets # ["green", "yellow", "red"]
    splits = args.splits # ["trainval", "test"]

    for dataset, split in tqdm(
        [*itertools.product(datasets, splits)],
        desc="Folders loop:",
        leave=True,
    ):
        mask_paths = DATA_PATH / f"{dataset}/{split}/ground_truths/masks"
        rle_path = mask_paths.parent / "rle"
        rle_path.mkdir(exist_ok=True, parents=True)
        voc_path = mask_paths.parent / "Pascal_VOC"
        voc_path.mkdir(exist_ok=True, parents=True)
        coco_path = mask_paths.parent / "COCO"
        coco_path.mkdir(exist_ok=True, parents=True)
        via_path = mask_paths.parent / "VIA"
        via_path.mkdir(exist_ok=True, parents=True)

        coco_dict = initialize_COCO_dict()
        via_dict = initialize_VIA_dict()
        for mask_path in tqdm(
            [*mask_paths.iterdir()], desc=f"{dataset}/{split} loop:", leave=False
        ):
            binary_mask = io.imread(mask_path, as_gray=True)
            rle_mask = binary_mask_to_rle(binary_mask)
            save_rle_encoding(rle_mask, rle_path / (mask_path.stem + ".pickle"))

            mask_relative_path = str(mask_path.relative_to(DATA_PATH))
            xml_tree = get_pascal_VOC_annotations(binary_mask, mask_relative_path)
            save_pascal_VOC_annotations(xml_tree, voc_path / (mask_path.stem + ".xml"))
            mask_coco_dict = get_COCO_annotations(binary_mask, mask_relative_path)
            coco_dict["images"].extend(mask_coco_dict["images"])
            coco_dict["annotations"].extend(mask_coco_dict["annotations"])
            mask_via_dict = get_VIA_annotations(binary_mask, mask_relative_path)
            via_dict.update(mask_via_dict)

        save_json_annotations(
            coco_dict, coco_path / f"annotations_{dataset}_{split}.json"
        )
        save_json_annotations(
            via_dict, via_path / f"annotations_{dataset}_{split}.json"
        )
