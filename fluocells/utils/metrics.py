"""
The metrics module contains implementations of suggested metrics to assess model performance.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-06-29
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))


import numpy as np
from skimage.measure import label, regionprops


def iou_metrics(mask_labelled, pred_mask_labelled, iou_threshold):
    # Find the unique objects in the masks
    mask_objects = np.unique(mask_labelled)[1:]  # Ignore the background (0)
    pred_mask_objects = np.unique(pred_mask_labelled)[1:]

    # Initialize the counts
    TP = 0
    FN = 0
    FP = 0

    # Create dictionaries to store the maximum IoUs for each object
    mask_max_iou = {obj: 0 for obj in mask_objects}
    pred_mask_max_iou = {obj: 0 for obj in pred_mask_objects}

    # Calculate IoU for each pair of objects and update the dictionaries
    for mask_obj in mask_objects:
        for pred_mask_obj in pred_mask_objects:
            intersection = np.sum(
                (mask_labelled == mask_obj) & (pred_mask_labelled == pred_mask_obj)
            )
            union = np.sum(
                (mask_labelled == mask_obj) | (pred_mask_labelled == pred_mask_obj)
            )
            iou = intersection / union if union > 0 else 0
            if iou > mask_max_iou[mask_obj]:
                mask_max_iou[mask_obj] = iou
            if iou > pred_mask_max_iou[pred_mask_obj]:
                pred_mask_max_iou[pred_mask_obj] = iou

    # Count TP and FN based on the maximum IoUs for the mask objects
    for iou in mask_max_iou.values():
        if iou > iou_threshold:
            TP += 1  # True Positive
        else:
            FN += 1  # False Negative

    # Count FP based on the maximum IoUs for the pred_mask objects
    for iou in pred_mask_max_iou.values():
        if iou <= iou_threshold:
            FP += 1  # False Positive

    return TP, FP, FN


def proximity_metrics(mask_labelled, pred_mask_labelled, dist_threshold):
    # Initialize the counts
    TP = 0
    FN = 0
    FP = 0

    # Calculate the centroids for all objects
    mask_props = regionprops(mask_labelled)
    pred_mask_props = regionprops(pred_mask_labelled)
    mask_centroids = {prop.label: prop.centroid for prop in mask_props}
    pred_mask_centroids = {prop.label: prop.centroid for prop in pred_mask_props}

    # For each mask object, find the pred_mask object with the closest centroid
    for mask_obj, mask_centroid in mask_centroids.items():
        min_dist = np.inf
        for pred_mask_obj, pred_mask_centroid in pred_mask_centroids.items():
            dist = np.linalg.norm(
                np.array(mask_centroid) - np.array(pred_mask_centroid)
            )
            if dist < min_dist:
                min_dist = dist
        if min_dist <= dist_threshold:
            TP += 1  # True Positive
        else:
            FN += 1  # False Negative

    # For each pred_mask object, find the mask object with the closest centroid
    for pred_mask_obj, pred_mask_centroid in pred_mask_centroids.items():
        min_dist = np.inf
        for mask_obj, mask_centroid in mask_centroids.items():
            dist = np.linalg.norm(
                np.array(mask_centroid) - np.array(pred_mask_centroid)
            )
            if dist < min_dist:
                min_dist = dist
        if min_dist > dist_threshold:
            FP += 1  # False Positive

    return TP, FP, FN


def eval_prediction(mask, pred_mask, eval_type, TP_threshold):
    if eval_type == "iou":
        TP, FP, FN = iou_metrics(mask, pred_mask, TP_threshold)
    elif eval_type == "proximity":
        TP, FP, FN = proximity_metrics(mask, pred_mask, TP_threshold)
    else:
        raise ValueError(
            f"Unknown evaluation type '{eval_type}'. Please choose either 'iou' or 'proximity'."
        )
    return TP, FP, FN


def all_metrics(metrics, eval_type, eps=0.01):
    metrics["Target_count"] = (
        metrics["TP_iou"] + metrics["FN_prox"]
    )  # target don't depend on eval_type!
    metrics["Predicted_count"] = (
        metrics[f"TP_{eval_type}"] + metrics[f"FN_{eval_type}"]
    )  # predicted vary based on eval_type!
    # compute performance measure for the current quantile filter
    tot_tp_test = metrics[f"TP_{eval_type}"].sum()
    tot_fp_test = metrics[f"FP_{eval_type}"].sum()
    tot_fn_test = metrics[f"FN_{eval_type}"].sum()
    tot_abs_diff = abs(metrics["Target_count"] - metrics["Predicted_count"])
    tot_perc_diff = (metrics["Predicted_count"] - metrics["Target_count"]) / (
        metrics["Target_count"].replace(0, 1)
    )  # avoid ZeroDivisionError:
    accuracy = (tot_tp_test) / (tot_tp_test + tot_fp_test + tot_fn_test + eps)
    precision = (tot_tp_test) / (tot_tp_test + tot_fp_test + eps)
    recall = (tot_tp_test) / (tot_tp_test + tot_fn_test + eps)
    F1_score = 2 * precision * recall / (precision + recall)
    MAE = tot_abs_diff.mean()
    MedAE = tot_abs_diff.median()
    MPE = tot_perc_diff.mean()

    return (F1_score, accuracy, precision, recall, MAE, MedAE, MPE)
