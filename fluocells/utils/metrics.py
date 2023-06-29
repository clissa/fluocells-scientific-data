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
