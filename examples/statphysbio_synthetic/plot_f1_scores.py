"""
Sweep through decoding parameters and calculate F1-score using known ground truth.

Shepherd 2024/12 - create script to run on simulation.
"""

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.PixelDecoder import PixelDecoder
from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from numpy.typing import ArrayLike
import json


if __name__ == "__main__":
    root_path = Path(r"/mnt/d/EQUIPEX/Data/2025012025_statphysbio_simulation/fixed/sim_acquisition")
    gt_path = Path(r"/mnt/d/EQUIPEX/Data/2025012025_statphysbio_simulation/fixed/GT_spots.csv")
    sweep_decode_params(root_path=root_path, gt_path=gt_path)