"""
The data module contains helpers to deal with fluocells data.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-05-31
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

import pandas as pd
import numpy as np

import tifffile
from PIL import Image, ExifTags
import piexif
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage import io, measure
from tqdm.auto import tqdm
from typing import List, Literal, Union

from fluocells.config import DATA_PATH, CURRENT_DATA_VERSION, DEBUG_PATH


N_POINTS: int = 40

MISSING = "unknown"


def get_exif_bytes(image_path: Path) -> bytes:
    exif_data = piexif.load(str(image_path))
    return piexif.dump(exif_data)


def tif2png_with_metadata(p: Path, outpath: Path, outname: Union[str, None] = None):
    with Image.open(p) as tiff_image:
        exif_bytes = get_exif_bytes(str(p))
        tiff_image.info = {"exif": exif_bytes}

        outpath.mkdir(exist_ok=True, parents=True)
        outname = outpath / outname if outname else outpath / (p.stem + ".png")
        # print(f'Saving converted file to: {outname}')
        tiff_image.save(outname, format="PNG")
    

def jpg2png_with_metadata(p: Path, outpath: Path, outname: Union[str, None] = None):
    with Image.open(p) as jpg_image:
        outpath.mkdir(exist_ok=True, parents=True)
        outname = outpath / outname if outname else outpath / (p.stem + ".png")
        # print(f'Saving converted file to: {outname}')
        jpg_image.save(outname, format="PNG")
        
        
def dump_tif_metadata(metadata, outpath):
    with open(outpath, "w") as file:
        header = "# metadata from tifffile.TiffFile().pages[0].tags\n"
        file.write(header)
        for key in metadata.keys():
            line = f"{metadata[key].name}\t{metadata[key].value}\n"
            file.write(line)


def dump_exif_metadata(metadata, outpath):
    with open(outpath, "w") as file:
        header = "# metadata from pyexiv2.Image().read_exif()\n"
        file.write(header)
        for key in metadata.keys():
            line = f"{key}\t{metadata[key]}\n"
            file.write(line)


def get_object_properties(mask: np.ndarray):
    labels = measure.label(mask, connectivity=1, return_num=False)
    regprops = measure.regionprops(labels)
    return labels, regprops


def save_objects(
    objs: List[measure._regionprops.RegionProperties],
    skimage_label: np.ndarray,
    save_path: Path,
) -> None:
    for idx, obj in enumerate(objs):
        blob = skimage_label[obj._slice].astype("uint8") * 255
        outpath = save_path / f"blob{idx}_a={int(obj.area)}.png"
        io.imsave(fname=outpath, arr=blob, check_contrast=False)


def check_noise_in_masks(masks_path: Path, debug_path: Path) -> None:
    """
    Read ground-truth masks and save separate images for each object detected with measure.label()
    :param masks_path: Pathlib.Path() to masks folder
    :param debug_path: Pathlib.Path() where to store single objects (created if don't exist)
    :return:
    """
    # from scipy import ndimage
    for p in tqdm([*masks_path.iterdir()]):
        # read tif image
        mask = io.imread(p, as_gray=True)

        # find objects
        label, objects = get_object_properties(mask)

        # initialize save_path
        save_path = debug_path / p.stem
        save_path.mkdir(exist_ok=True, parents=True)

        # save objects separately
        save_objects(objects, label, save_path)

    return


def check_mask_format_(mask: np.ndarray):
    if mask.max() == 1:
        mask = mask * 255
    if mask.dtype != np.uint8:
        mask = mask.astype("uint8")
    return mask


def clean_mask(
    mask: np.ndarray, bin_thresh: int, min_hole_size: int, min_object_size: int
):
    # binary
    mask[mask > bin_thresh] = 255
    mask[mask <= bin_thresh] = 0
    # fill holes
    mask = remove_small_holes(
        mask, area_threshold=min_hole_size
    )  # return array [False, True]
    mask = remove_small_objects(mask, min_size=min_object_size)
    return (mask * 255).astype("uint8")


def remove_noise_from_masks(
    mask_paths: List[Path],
    outpath: Path,
    bin_thresh: int,
    min_hole_size: int,
    min_object_size: int,
):
    for p in tqdm(mask_paths):
        mask = io.imread(p, as_gray=True)
        mask = check_mask_format_(mask)
        mask = clean_mask(mask, bin_thresh, min_hole_size, min_object_size)
        # fix filename and change format
        # outpath = outpath if outpath.name == "masks" else outpath / "masks"
        outpath.mkdir(parents=True, exist_ok=True)
        io.imsave(outpath / p.name, mask, check_contrast=False)
    return


def parse_animal(animal: str):
    animal = animal.lower()
    animal_type = "marvin" if animal.startswith("m") else "rat"
    animal_idx = animal[3:] if animal.startswith("m") else animal[2:]
    return animal_type, str(animal_idx)


def format_slice_id(slice_id):
    if slice_id == "u":
        return MISSING
    else:
        return slice_id


def parse_slice(slice: str):
    if len(slice) > 3:
        return (
            format_slice_id(slice[1]),
            format_slice_id(slice[3]),
            format_slice_id(slice[5]),
        )
    else:
        return MISSING, MISSING, MISSING


def parse_sample(sample: str):
    slice_start_idx = while_iter = -1
    char_list = ["S", "C", "R"]
    while slice_start_idx == -1:
        while_iter += 1
        slice_start_idx = sample.find(char_list[while_iter])
        if while_iter >= 2:
            return MISSING, MISSING, MISSING, MISSING, MISSING
    animal_data = sample[:slice_start_idx]
    slice_data = sample[slice_start_idx:]
    if slice_data.startswith("CR"):
        slice_data = "Cu" + slice_data[1:]
    if while_iter == 1:
        slice_data = char_list[0] + "u" + slice_data
    if while_iter == 2:
        slice_data = char_list[1] + "u" + slice_data
    parsed_animal, parsed_slice = parse_animal(animal_data), parse_slice(slice_data)
    return (
        parsed_animal[0],
        parsed_animal[1],
        parsed_slice[0],
        parsed_slice[1],
        parsed_slice[2],
    )


def parse_region(region: str):
    region = region.upper()
    if region.endswith("L") or region.endswith("R"):
        return region[:-1] + region[-1].lower()
    else:
        return region


def parse_zoom(zoom: str):
    return zoom if len(zoom) > 2 else zoom + "0x" if zoom else MISSING


def parse_filename(fn):
    parts = Path(fn).stem.split("_")
    if len(parts) == 2:
        animal_type, animal_idx, sample_idx, row, col = parse_sample(parts[0])
        region = zoom = MISSING
    elif len(parts) >= 3:
        animal_type, animal_idx, sample_idx, row, col = parse_sample(parts[0])
        region = parse_region(parts[1])
        zoom = parse_zoom(parts[2])
    else:
        animal_type = animal_idx = sample_idx = row = col = region = zoom = MISSING
    return animal_type, animal_idx, sample_idx, row, col, region, zoom


EXTENSIONS = [".png", ".jpg", ".TIFF", ".TIF"]


def get_all_paths(
    source_path: Path, mode: Literal[None, "images", "masks"] = "images", exts=EXTENSIONS
) -> tuple:
    """Recursevely search for files with desired extension inside source_path.

    Args:
        source_path (Path): parent data folder
        mode (Literal['images', 'masks'], optional): files folder. Either `images` or `masks`
        exts (list, optional): accepted file extensions. Defaults to [".png", ".jpg", ".TIFF", ".TIF"].

    Returns:
        tuple: tuple of found images and ignored file extensions
    """
    ignored_extensions = list()
    all_files_list = list()
    for p in source_path.iterdir():
        if p.is_dir():
            print(f"\n{p.name} is a directory! continue searching...")
            img_list, ignored = get_all_paths(p, mode=mode, exts=exts)
            all_files_list.extend(img_list)
            ignored_extensions.extend(ignored)
        elif (mode is not None) & (mode != p.parent.name):
            continue
        elif p.suffix in exts:
            all_files_list.append(p)
        else:
            ignored_extensions.append(p.suffix)
    return all_files_list, set(ignored_extensions)


def get_image_name_relative_path(
    image_name_row: pd.DataFrame, mode: Literal["images", "masks"] = "images"
) -> str:
    relative_path = "/".join(
            [image_name_row.dataset.values[0], image_name_row.partition.values[0], mode]
        )
    return relative_path


def compute_masks_stats(masks_paths: List[Path]) -> pd.DataFrame:
    """
    Read ground-truth masks and compute metrics for cell counts and shapes
    :param masks_path: masks folder
    :return:
    """
    # paths = [*masks_path.iterdir()]
    data = []
    for p in tqdm(masks_paths):
        mask = io.imread(p, as_gray=True)
        skimage_label, n_objs = measure.label(mask, connectivity=1, return_num=True)

        # add one row per object
        regions = measure.regionprops(skimage_label)
        for idx_obj, obj in enumerate(regions):
            data.append(
                [
                    p.name,
                    n_objs,
                    idx_obj,
                    obj.area,
                    obj.minor_axis_length,
                    obj.major_axis_length,
                    obj.equivalent_diameter,
                    obj.feret_diameter_max,
                    # obj.min_intensity, obj.mean_intensity, obj.max_intensity
                ]
            )
        # add empty row in case image has no objects
        if n_objs == 0:
            data.append(
                [
                    p.name,
                    int(n_objs),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    # None, None, None
                ]
            )
    stats_df = pd.DataFrame(
        data=data,
        columns=[
            "img_name",
            "n_cells",
            "cell_id",
            "area",
            "min_axis_length",
            "max_axis_length",
            "equivalent_diameter",
            "feret_diameter_max",
            # 'min_intensity', 'mean_intensity', 'max_intensity'
        ],
    )
    stats_df.round(4).to_csv(
        masks_paths[0].parent.parent.parent / "stats_df.csv", index=False
    )
    return stats_df
