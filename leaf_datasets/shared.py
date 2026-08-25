"""
Provide shared paths, parameters, and preprocessing utilities for LEAF datasets.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import mne
import getpass
import json
import sys
import numpy as np
from pathlib import Path
import h5py
from dataclasses import dataclass
sys.path.append(str(Path(__file__).resolve().parent.parent))
from leaf_datasets.ElectrodeUnifier import ElectrodeUnifier
from load_config import load_paths

_paths = load_paths()
RAW_DATA_FOLDER = _paths['raw_data']
PRETRAIN_FOLDER = _paths['pretrain']
DATA_FOLDER = _paths['downstream']

@dataclass
class Param:
    lp = 0.1
    hp = 70
    MI_lp = 0.3
    MI_hp = 40
    resample = 200
    template = 'LEAF-ch65'

def pipeline(X, ch_names):
    assert X.ndim == 3, f'Expected 3D array (B, C, T), got {X.ndim}D'
    assert X.shape[0] != 0, f'Expected non-empty batch dimension, got {X.shape[0]}'
    assert X.shape[1] == len(ch_names), f'Expected {len(ch_names)} channels, got {X.shape[1]}'

    # Clip 0.1% and 99.9% percentiles, to remove 0.2% outliers
    r1, r2 = np.percentile(X, [0.1, 99.9]).astype(X.dtype)
    X = np.clip(X, r1, r2)  
    
    # Apply robust scaling
    q25, median, q75 = np.percentile(X, [25, 50, 75]).astype(X.dtype)
    iqr = q75 - q25
    X = (X - median) / iqr 

    X = ElectrodeUnifier(Param.template).map_to_template(X, ch_names)

    if X.max() > 12 or X.min() < -12:
        print(f'[WARN] check data range, min: {X.min():.3f} max: {X.max():.3f}')

    return X
