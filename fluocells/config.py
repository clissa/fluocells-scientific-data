"""
This module contains project-wide settings and paths to data sources.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-05-31
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()  # type: ignore

sys.path.append(str(FLUOCELLS_PATH))
import fluocells as fluo


REPO_PATH = Path(fluo.__path__[0]).parent

TEST_PCT = 0.25
TRAINVAL_TEST_SEED = 10

# data
CURRENT_DATA_VERSION = "v1_6"
DATA_PATH = REPO_PATH / f"dataset_{CURRENT_DATA_VERSION}"

DATA_PATH_r = DATA_PATH / "red"
DATA_PATH_y = DATA_PATH / "yellow"
DATA_PATH_g = DATA_PATH / "green"

UNLABELLED_IMG_PATH_r = DATA_PATH_r / "unlabelled/images"
UNLABELLED_IMG_PATH_y = DATA_PATH_y / "unlabelled/images"
UNLABELLED_IMG_PATH_g = DATA_PATH_g / "unlabelled/images"