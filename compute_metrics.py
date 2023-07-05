import sys
import inspect
from pathlib import Path
import pandas as pd
import argparse

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

from fluocells.config import REPO_PATH
from fluocells.utils.metrics import all_metrics


parser = argparse.ArgumentParser(
    description="Compute metrics for fluocells experiments."
)
parser.add_argument(
    "-e",
    "--experiments",
    nargs="+",
    default=["green_12345", "red_98765", "yellow_54321"],
    help='List of experiments to assess. Default: ["green_12345", "red_98765", "yellow_54321"]',
)


def main(args):
    RESULTS_PATH = REPO_PATH / "results"

    # experiment_paths = [p for p in RESULTS_PATH.iterdir() if p.is_dir()]
    # experiment_paths.sort()
    experiment_paths = [RESULTS_PATH / exp_name for exp_name in args.experiments]

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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
