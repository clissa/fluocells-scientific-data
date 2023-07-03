"""
This script reads VIA annotations in csv format and generates binary masks.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-07
License: Apache License 2.0
"""


import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute().parent

sys.path.append(str(FLUOCELLS_PATH))

import pandas as pd

from fluocells.config import DATA_PATH, REPO_PATH, METADATA
from fluocells.utils.annotations import (
    apply_VIA_annotation_to_binary_mask,
    load_VIA_annotations,
)

import argparse

parser = argparse.ArgumentParser(description="Convert VIA annotations to binary masks")

# Add the dataset argument
parser.add_argument(
    "dataset",
    type=str,
    choices=["green", "yellow", "red"],
    help="Dataset to train on: green, yellow, or red",
)

args = parser.parse_args()

ANNOTATIONS_PATH = DATA_PATH / "annotations"
DATASET = args.dataset
MARKER = {"green": "c-FOS", "yellow": "CTb", "red": "Orx"}[DATASET]
ANNOTATION_ROUNDS = {
    "green": ["first", "second", "third"],
    "red": ["fourth"],  # ["first", "second", "third"],
    "yellow": ["first"],
}

if __name__ == "__main__":
    metadata_df = pd.read_excel(
        DATA_PATH / f"metadata_{METADATA['current_version']}.xlsx",
        sheet_name="metadata",
    )
    labeled_data = metadata_df.query("partition != 'unlabelled' and dataset==@DATASET")

    for annotation_round in ANNOTATION_ROUNDS[DATASET]:
        annotations_df = load_VIA_annotations(
            ANNOTATIONS_PATH / f"{MARKER}_{annotation_round}_round_reviewed.csv"
        )

        # annotation_tasks = annotations_df.iloc[:35].groupby("filename")
        annotation_tasks = annotations_df.groupby("filename")
        annotation_tasks.apply(
            apply_VIA_annotation_to_binary_mask, DATA_PATH, DATA_PATH, metadata_df
        )
