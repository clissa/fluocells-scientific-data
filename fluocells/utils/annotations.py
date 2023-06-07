"""
This module contains utils to handle several annotations types in various formats including: Pascal VOC, COCO, RLE.

Each of the above formats implements: 
    - converter: from binary mask to the desired format
    - decoder: from format to binary mask
    - save/load: methods to dump or load annotations to disk
    - test: method to test implementation 

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-05
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

import json
import pickle
import numpy as np
from datetime import datetime
from pycocotools import mask as maskUtils
from pycocotools import coco as cocoUtils
import xml.etree.ElementTree as ET

import cv2
from matplotlib import pyplot as plt

from fluocells.config import DATA_PATH


N_POINTS = 50


# RLE
def binary_mask_to_rle(binary_mask):
    # Convert binary mask to RLE encoding
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded_mask = maskUtils.encode(binary_mask)

    # Return RLE encoding
    return encoded_mask


def rle_to_binary_mask(rle_encoding, img_height, img_width):
    # Convert RLE encoding to binary mask
    binary_mask = maskUtils.decode(rle_encoding)

    # Reshape and pad the binary mask to match the image dimensions
    binary_mask = np.reshape(binary_mask, (img_height, img_width), order="F")

    # Return binary mask
    return binary_mask


def save_rle_encoding(rle_encoding, save_path):
    # Save RLE encoding to a file
    with open(save_path, "wb") as file:
        pickle.dump(rle_encoding, file)


# TODO
def load_rle_encoding(rle_path):
    raise (NotImplementedError)


# POLYGON
def sample_contour_points(contour, max_points):
    # Sample points from the contour
    num_points = len(contour)
    indices = np.linspace(0, num_points - 1, max_points, dtype=np.int32)
    sampled_contour = contour[indices]
    return sampled_contour


def _get_object_contours(binary_mask, max_points=None):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sampled_contours = list(contours)
    for i, contour in enumerate(contours):
        contour = contour.squeeze(axis=1)

        if max_points is not None and len(contour) > max_points:
            contour = sample_contour_points(contour, max_points)
        sampled_contours[i] = contour
    return sampled_contours


def _get_polygon_from_contour(contour):
    # convert from np.int32 to int to avoid json serialization issues
    return [(int(point[0]), int(point[1])) for point in contour]


def binary_mask_to_polygon(binary_mask, max_points=None):
    # Convert binary mask to polygon annotation
    contours = _get_object_contours(binary_mask, max_points)
    polygons = []
    for contour in contours:
        polygon = _get_polygon_from_contour(contour)
        polygons.append(polygon)
    return polygons


def polygon_to_binary_mask(polygons, image_shape):
    # Convert polygons to binary mask
    binary_mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        contour = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(binary_mask, [contour], 1)
    return binary_mask


# BOUNDING BOXES
def binary_mask_to_bbox(binary_mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Extract bounding box for each contour
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))

    return bboxes


def bbox_to_binary_mask(boxes, image_shape):
    # Create a black image of the specified shape
    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    for bbox in boxes:
        # Extract bounding box coordinates
        x, y, w, h = bbox

        # Draw a filled rectangle on the binary mask
        binary_mask[y : y + h, x : x + w] = 1

    return binary_mask


# DOT ANNOTATIONS
def _get_centroid_from_contour(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def binary_mask_to_dots(binary_mask):
    contours = _get_object_contours(binary_mask)
    return [_get_centroid_from_contour(contour) for contour in contours]


def dots_to_binary_mask(dots, image_shape):
    # Create a black image of the specified shape
    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    # Set the value of dot locations to 1 in the binary mask
    for dot in dots:
        x, y = dot
        binary_mask[y, x] = 1

    return binary_mask


# COUNT
def binary_mask_to_count(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return len(contours)


# Pascal VOC format
def get_pascal_voc_annotations(binary_mask, mask_relative_path):
    # Convert binary mask to annotations
    polygons = binary_mask_to_polygon(binary_mask)
    bboxes = binary_mask_to_bbox(binary_mask)
    dots = binary_mask_to_dots(binary_mask)
    object_count = binary_mask_to_count(binary_mask)

    # Create Pascal VOC annotation XML structure
    split = mask_relative_path.split("/")[1]
    dataset_folder = mask_relative_path.split("/")[0]
    image_path = mask_relative_path.replace("ground_truths/masks", "images")
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "split").text = split
    ET.SubElement(annotation, "filename").text = mask_relative_path.split("/")[-1]
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "AMS_Acta"

    version = ET.SubElement(annotation, "version")
    version.text = "v1_6"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(binary_mask.shape[1])
    ET.SubElement(size, "height").text = str(binary_mask.shape[0])
    ET.SubElement(size, "depth").text = "1"

    segmented = ET.SubElement(annotation, "segmented").text = "Manually"

    # Add polygon annotations
    object_class = "nucleus" if dataset_folder == "green" else "citoplasm"
    for polygon in polygons:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = object_class
        # ET.SubElement(object_elem, "pose").text = "Unspecified"
        # ET.SubElement(object_elem, "truncated").text = "Unspecified"
        # ET.SubElement(object_elem, "difficult").text = "Unspecified"

        polygon_elem = ET.SubElement(object_elem, "polygon")
        for point in polygon:
            x, y = point
            point_elem = ET.SubElement(polygon_elem, "pt")
            ET.SubElement(point_elem, "x").text = str(x)
            ET.SubElement(point_elem, "y").text = str(y)

    # Add bounding box annotations
    for bbox in bboxes:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = object_class
        # ET.SubElement(object_elem, "pose").text = "Unspecified"
        # ET.SubElement(object_elem, "truncated").text = "Unspecified"
        # ET.SubElement(object_elem, "difficult").text = "Unspecified"

        bndbox_elem = ET.SubElement(object_elem, "bndbox")
        xmin, ymin, width, height = bbox
        ET.SubElement(bndbox_elem, "xmin").text = str(xmin)
        ET.SubElement(bndbox_elem, "ymin").text = str(ymin)
        ET.SubElement(bndbox_elem, "xmax").text = str(xmin + width)
        ET.SubElement(bndbox_elem, "ymax").text = str(ymin + height)

    # Add dot annotations
    for dot in dots:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = "object_class"
        # ET.SubElement(object_elem, "pose").text = "Unspecified"
        # ET.SubElement(object_elem, "truncated").text = "Unspecified"
        # ET.SubElement(object_elem, "difficult").text = "Unspecified"

        dot_elem = ET.SubElement(object_elem, "dot")
        ET.SubElement(dot_elem, "x").text = str(dot[0])
        ET.SubElement(dot_elem, "y").text = str(dot[1])

    # Add count annotation
    count_elem = ET.SubElement(annotation, "count")
    count_elem.text = str(object_count)

    # Create an ElementTree object from the annotation XML structure
    tree = ET.ElementTree(annotation)

    return tree


def save_pascal_voc_annotations(tree, outpath):
    # Save the Pascal VOC annotation to a file
    tree.write(outpath)


# COCO format
def initialize_coco_dict():
    categories = (
        [
            {
                "id": 0,
                "name": "bkgd",
                "supercategory": "background",
                "isthing": 0,
                "color": "#00000000",
            },
            {
                "id": 1,
                "name": "c-FOS",
                "supercategory": "nucleus",
                "isthing": 1,
                "color": "#c8b6ff",
            },
            {
                "id": 1,
                "name": "CTb",
                "supercategory": "citoplasm",
                "isthing": 1,
                "color": "#ffddd2",
            },
            {
                "id": 2,
                "name": "Orx",
                "supercategory": "citoplasm",
                "isthing": 1,
                "color": "#fff3b0",
            },
        ],
    )
    # Create COCO annotation structure
    coco_annotation = {
        "info": {
            "year": 2023,
            "version": "1.6",
            "description": "Fluorescent Neuronal Cells",
            "contributor": "Luca Clissa",
            "url": "<CHECK-AMS-ACTA-DOI>",  # TODO
            "date_created": datetime.today().strftime("%Y-%m-%d"),
        },
        "images": [],
        "annotations": [],
        "licenses": [
            {
                "id": 1,
                "name": "CC-BY-SA 4.0",
                "url": "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt",
            },
        ]
    }
    return coco_annotation

def get_coco_annotations(binary_mask, mask_relative_path):
    # Convert binary mask to annotations
    contours = _get_object_contours(binary_mask, max_points=N_POINTS)
    object_count = len(contours)

    split = mask_relative_path.split("/")[1]
    dataset_folder = mask_relative_path.split("/")[0]
    image_path = mask_relative_path.replace("ground_truths/masks", "images")
    filename = mask_relative_path.split("/")[-1]
    
    coco_annotation = initialize_coco_dict()

    # Create COCO image entry
    image_entry = {
        "id": filename.split(".")[0],  # Set your own image ID here
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
        "file_name": filename,
        "split": split,
        "path": image_path,
        "license": 1,  # Set your own license ID here
        # "date_captured": "2023-06-06" #TODO: get from metadata ?
    }
    coco_annotation["images"].append(image_entry)

    # Add annotations for each image
    object_class_id = {"green": 1, "yellow": 2, "red": 3}.get(dataset_folder, 0)
    annotation_entry = {
        # "id": filename.split(".")[0],  # Set your own object ID here
        "image_id": filename.split(".")[0],  # Set the corresponding image ID
        "category_id": object_class_id,
        "segmentation": [], # [x, y] points
        "bbox": [], # [xmin, ymin, width, height]
        "area": [],
        "dots": [], # xcenter, ycenter
        "count": object_count,
        "iscrowd": 0 # currently annotations are uninterrupted and non-overlapping
    }

    # Add annotations
    for contour in contours:
        
        # Add polygon annotations
        polygon = _get_polygon_from_contour(contour)
        annotation_entry["segmentation"].append(polygon)

        # Add bounding box annotations
        xmin, ymin, width, height = cv2.boundingRect(contour)
        annotation_entry["bbox"].append([xmin, ymin, width, height])

        # Add dots annotations 
        cX, cY = _get_centroid_from_contour(contour)    
        annotation_entry["dots"].append((cX, cY))

        # Add objects area
        annotation_entry["area"].append(cv2.contourArea(contour))

    coco_annotation["annotations"].append(annotation_entry)
    
    return coco_annotation


def save_coco_annotations(coco_dict, outpath):
    # Save the COCO annotation to a file
    with open(outpath, "w") as file:
        json.dump(coco_dict, file)
        
        
# TESTS
def test_rle(binary_mask):
    rle_encoding = binary_mask_to_rle(binary_mask)
    img_height, img_width = binary_mask.shape
    reco_binary_mask = rle_to_binary_mask(rle_encoding, img_height, img_width)
    assert np.array_equal(
        binary_mask, reco_binary_mask
    ), "Incorrect RLE encoding! Please fix the implementation"
    print("RLE conversion test passed.")


def test_polygon(binary_mask):
    # Test conversion from binary mask to polygon and back to binary mask
    polygons = binary_mask_to_polygon(binary_mask)
    restored_binary_mask = polygon_to_binary_mask(polygons, binary_mask.shape)

    assert np.array_equal(
        binary_mask, restored_binary_mask
    ), "Incorrect polygon encoding! Please fix the implementation"
    print("Polygon conversion test passed.")


def test_bbox(binary_mask):
    # Test conversion from binary mask to bounding box and back to binary mask
    bbox = binary_mask_to_bbox(binary_mask)
    restored_binary_mask = bbox_to_binary_mask(bbox, binary_mask.shape)

    assert np.array_equal(
        binary_mask, restored_binary_mask
    ), "Incorrect bbox encoding! Please fix the implementation"
    print("Bounding box conversion test passed.")


def test_dots(binary_mask):
    dots = binary_mask_to_dots(binary_mask)

    expected_output = np.array([(2, 0), (2, 3)])

    assert np.array_equal(
        dots, expected_output
    ), "Incorrect dot encoding! Please fix the implementation"
    print("Dot annotation conversion test passed.")


def test_count(binary_mask):
    n = binary_mask_to_count(binary_mask)
    expected_count = 2
    assert (
        n == expected_count
    ), "Incorrect count encoding! Please fix the implementation"
    print("Count retrieval test passed.")


if __name__ == "__main__":
    print("Running tests . . .")
    # Example usage: ENCODING
    binary_mask = np.array(
        [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    )
    test_rle(binary_mask)
    test_polygon(binary_mask)
    test_bbox(binary_mask)
    test_dots(binary_mask) #TODO: fix `ZeroDivisionError: float division by zero`
    test_count(binary_mask)
