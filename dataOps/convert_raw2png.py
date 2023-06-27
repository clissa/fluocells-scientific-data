"""
This script converts raw fluorescence images from TIF and JPG format to png. Importantly, this maintains exif metadata, both in the files and as separate dumps in the metadata folder.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-05-31
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute().parent
sys.path.append(str(FLUOCELLS_PATH))

import multiprocessing
from tqdm.auto import tqdm
import pandas as pd
import pyexiv2

from fluocells.config import REPO_PATH, RAW_DATA_PATH, DATA_PATH
from fluocells.utils.data import get_all_paths, get_image_name_relative_path
from fluocells.utils.data import (
    get_exif_bytes,
    dump_exif_metadata,
    tif2png_with_metadata,
    jpg2png_with_metadata,
)


def get_corresponding_y_meta(fstem):
    y_orig_stem = fstem[:-2]
    name_condition = metadata_df.original_name.str.contains(y_orig_stem)
    dataset_condition = metadata_df.dataset == "yellow"
    corresponding_yellow = metadata_df.loc[(name_condition) & (dataset_condition)]
    return corresponding_yellow


def is_diff_partition(g_meta, y_meta):
    return g_meta.partition.values[0] != y_meta.partition.values[0]


def update_green_partition_(fstem, meta):
    updated = 0
    corresponding_yellow = get_corresponding_y_meta(fstem)
    return corresponding_yellow.partition.values[0]


def update_metadata(metadata_df, raw_image_paths):
    up = 0
    updated_metadata_df = metadata_df.copy()
    for i, p in enumerate(raw_image_paths):
        escaped_filename = p.stem.translate(
            str.maketrans({"(": "\(", ")": "\)"})
        )  # .replace('(', '\(').replace(')', '\)')
        image_metadata = metadata_df.loc[
            metadata_df.original_name.str.contains(escaped_filename)
        ]
        if (
            escaped_filename.endswith("_g") and image_metadata.double_marked.values[0]
        ):  # and (image_metadata.partition != 'unlabelled').values[0]:
            corresponding_y = get_corresponding_y_meta(escaped_filename[:-2])
            if is_diff_partition(image_metadata, corresponding_y):
                update = update_green_partition_(escaped_filename, image_metadata)
                updated_metadata_df.loc[
                    updated_metadata_df.original_name.str.split(".").apply(
                        lambda x: x[0]
                    )
                    == p.stem,
                    "partition",
                ] = update
                up += 1
    print(f"Updated {up} entries")
    return updated_metadata_df


def raw2png_with_metadata(p: Path, metadata_df: pd.DataFrame):
    # get outpath
    escaped_filename = p.stem.translate(str.maketrans({"(": "\(", ")": "\)"}))
    image_metadata = metadata_df.loc[
        metadata_df.original_name.str.contains(escaped_filename)
    ]

    rel_outpath = get_image_name_relative_path(image_metadata)
    outpath = DATA_PATH / rel_outpath
    outpath.mkdir(exist_ok=True, parents=True)
    outname = image_metadata.image_name.values[0]

    # conversion
    if p.suffix == ".TIF":
        tif2png_with_metadata(p, outpath, outname)
    else:
        jpg2png_with_metadata(p, outpath, outname)

    # dump metadata
    metadata_path = outpath.parent / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)

    with pyexiv2.Image(str(p)) as exif:
        metadata_exif = exif.read_exif()

    dump_exif_metadata(metadata_exif, metadata_path / (outname.split(".")[0] + ".txt"))


def parallel_helper(args):
    p, metadata_df = args
    raw2png_with_metadata(p, metadata_df)


def parallel_transform(paths_list, arg, num_processes):
    # Create a multiprocessing pool with the specified number of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Apply the function to each path in parallel
    with tqdm(
        total=len(paths_list), desc="Converting and moving raw images"
    ) as progress_bar:
        for _ in pool.imap_unordered(parallel_helper, [(p, arg) for p in paths_list]):
            progress_bar.update(1)

    # Close the pool to release resources
    pool.close()
    pool.join()


if __name__ == "__main__":
    metadata_df = pd.read_excel(DATA_PATH / "metadata_v1_5.xlsx", sheet_name="map")
    raw_image_paths, ignored = get_all_paths(RAW_DATA_PATH, exts=[".TIF", ".JPG"])

    metadata_df.loc[metadata_df.dataset == "c-FOS", "dataset"] = "green"
    metadata_df.loc[metadata_df.dataset == "Orx", "dataset"] = "red"
    metadata_df.loc[metadata_df.dataset == "CTb", "dataset"] = "yellow"
    updated_metadata_df = update_metadata(metadata_df, raw_image_paths)
    
    # first check that we have all images except 7 unrecoverable yellow
    # n_known_missing = 7
    # assert len(metadata_df) == (len(raw_image_paths)+n_known_missing), f"Found {len(raw_image_paths)+n_known_missing} raw images instead of the {len(metadata_df)} expected!"

    # print(metadata_df.dataset.value_counts())
    # for ds in ['green', 'yellow', 'red']:
    #     c = len([*(RAW_DATA_PATH / ds / 'images').iterdir()])
    #     print(ds, '\t', c)

    # for ds in metadata_df.dataset.unique():
    #     raw_imgs = [p.stem for p in (RAW_DATA_PATH / ds / 'images').iterdir()]
    #     expected_imgs = metadata_df.query("dataset == @ds").original_name.str.split('.').apply(lambda x: x[0]).values
    #     missing = set(expected_imgs).difference(raw_imgs)
    #     extra = set(raw_imgs).difference(expected_imgs)

    #     print(ds)
    #     print(f"Found {len(raw_imgs)}; Expected {len(expected_imgs)}")
    #     print(f"Missing {len(missing)}; Extra {len(extra)}")
    #     print(f"{missing=}\n\n{extra=}")

    # now move the images to the desired dataset structure
    parallel_transform(raw_image_paths, updated_metadata_df, 8)

    print(
        updated_metadata_df[["dataset", "partition"]].value_counts()
        - metadata_df[["dataset", "partition"]].value_counts()
    )

    updated_metadata_df.to_excel(
        DATA_PATH / "metadata_v1_6.xlsx", sheet_name="metadata", index=False
    )
