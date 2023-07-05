"""
This script computes global performance metrics for segmentation, detection and counting tasks.
Results are stored in the form of `all_metrics.csv` and `all_metrics.tex` files.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-07-02
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path
import pandas as pd

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

from fluocells.config import REPO_PATH
from fluocells.utils.metrics import all_metrics


RESULTS_PATH = REPO_PATH / "results"

# edit here to select experiments to assess
EXPERIMENT_NAMES = ["green_12345", "red_98765", "yellow_54321"]
# experiment_paths = [p for p in RESULTS_PATH.iterdir() if p.is_dir()]
# experiment_paths.sort()
experiment_paths = [RESULTS_PATH / exp_name for exp_name in EXPERIMENT_NAMES]

columns_detection = pd.MultiIndex.from_product(
    [
        ["segmentation", "detection"],
        ["F1_score", "accuracy", "precision", "recall"],
    ],
    names=["eval_type", "metric_name"],
)
columns_counting = pd.MultiIndex.from_product(
    [
        ["counting"],
        ["MAE", "MedAE", "MPE"],
    ],
    names=["eval_type", "metric_name"],
)
all_columns = columns_detection.append(columns_counting)
results_df = pd.DataFrame(columns=all_columns)

for experiment_dir in experiment_paths:
    metrics_df = pd.read_csv(experiment_dir / "metrics.csv")
    iou_metrics = all_metrics(metrics_df, eval_type="iou")
    prox_metrics = all_metrics(metrics_df, eval_type="prox")
    results_df.loc[f"{experiment_dir.name}"] = iou_metrics[:4] + prox_metrics

print(results_df)

# get rid of accuracy
results_df = results_df.drop(("segmentation", "accuracy"), axis=1, errors="ignore")
results_df = results_df.drop(("detection", "accuracy"), axis=1, errors="ignore")
results_df.to_csv(RESULTS_PATH / "all_metrics.csv")
with open(RESULTS_PATH / "all_metrics.tex", "w") as f:
    f.write(results_df.style.format("{:.2f}").to_latex())
