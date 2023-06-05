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

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools import coco as cocoUtils
import xml.etree.ElementTree as ET

import cv2
from skimage.measure import regionprops, label
from matplotlib import pyplot as plt


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
    with open(save_path, "w") as file:
        file.write(rle_encoding)


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


def binary_mask_to_polygon(binary_mask, max_points=None):
    # Convert binary mask to polygon annotation
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        contour = contour.squeeze(axis=1)

        if max_points is not None and len(contour) > max_points:
            contour = sample_contour_points(contour, max_points)

        polygon = [(point[0], point[1]) for point in contour]
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
def binary_mask_to_dots(binary_mask):
    # Find contours in the binary mask: skimage seems more robust than cv2 for small objects
    labeled_mask = label(binary_mask.astype(np.uint8))
    region_properties = regionprops(labeled_mask)

    # Extract center coordinates for each object
    dots = []
    for prop in region_properties:
        centroid = prop.centroid
        cX = int(centroid[1])
        cY = int(centroid[0])
        dots.append((cX, cY))

    return dots


def dots_to_binary_mask(dots, image_shape):
    # Create a black image of the specified shape
    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    # Set the value of dot locations to 1 in the binary mask
    for dot in dots:
        x, y = dot
        binary_mask[y, x] = 1

    return binary_mask


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


if __name__ == "__main__":
    print("Running tests . . .")
    # Example usage: ENCODING
    binary_mask = np.array(
        [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    )
    test_rle(binary_mask)
    test_polygon(binary_mask)
    test_bbox(binary_mask)
    test_dots(binary_mask)
