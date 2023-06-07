"""
This script transforms binary mask into several common annotation formats.

Supproted formats include:
    - RLE encoding
    - Pascal VOC: polygon, bndbox, dots, count + metadata
    - COCO: segmentation, bbox, area, dots, count, iscrowd

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
    get_pascal_voc_annotations,
    save_pascal_voc_annotations,
    get_coco_annotations,
    save_coco_annotations,
    initialize_coco_dict,
)


if __name__ == "__main__":
    datasets = ["green", "yellow", "red"]
    splits = ["trainval", "test"]

    for dataset, split in tqdm(
        [*itertools.product(datasets, splits)],
        desc="Folders loop:",
        leave=True,
    ):
        mask_paths = DATA_PATH / f"{dataset}/{split}/ground_truths/masks"
        rle_path = mask_paths.parent / "rle"
        rle_path.mkdir(exist_ok=True, parents=True)
        voc_path = mask_paths.parent / "pascal_voc"
        voc_path.mkdir(exist_ok=True, parents=True)
        coco_path = mask_paths.parent / "COCO"
        coco_path.mkdir(exist_ok=True, parents=True)

        coco_dict = initialize_coco_dict()
        for mask_path in tqdm(
            [*mask_paths.iterdir()], desc=f"{dataset}/{split} loop:", leave=False
        ):
            binary_mask = io.imread(mask_path, as_gray=True)
            rle_mask = binary_mask_to_rle(binary_mask)
            save_rle_encoding(rle_mask, rle_path / (mask_path.stem + ".pickle"))

            mask_relative_path = str(mask_path.relative_to(DATA_PATH))
            xml_tree = get_pascal_voc_annotations(binary_mask, mask_relative_path)
            save_pascal_voc_annotations(xml_tree, voc_path / (mask_path.stem + ".xml"))
            mask_coco_dict = get_coco_annotations(binary_mask, mask_relative_path)
            coco_dict["images"].extend(mask_coco_dict["images"])
            coco_dict["annotations"].extend(mask_coco_dict["annotations"])

        save_coco_annotations(
            coco_dict, coco_path / f"annotations_{dataset}_{split}.json"
        )