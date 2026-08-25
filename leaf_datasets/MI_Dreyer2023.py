"""
Load and validate the Dreyer2023 EEG dataset for held-out motor-imagery evaluation.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import h5py
import numpy as np

# https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Dreyer2023.py
NAME = 'MI_Dreyer2023'
TEXT_LABELS = ['MI/Left', 'MI/Right']  # UNVERIFIED: the delivered file carries no
                                       # label legend, so which of {0, 1} is left
                                       # is assumed from the other MI datasets.
SPLITS = ['train', 'val', 'test']
H5_PATH = '/media/public/LEAF/MI_Dreyer2023.h5'

# Already in template space — the file is stored on these 65 channels in this
# order, so no ElectrodeUnifier step is needed (or wanted) on load.
CH_NAMES = [
    'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'Fp1', 'Fp2',
    'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6',
    'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5',
    'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'PO4', 'PO8', 'O1', 'O2',
]

# 801 samples @ 200 Hz is 4.005 s — one sample past the 800 the other 4 s datasets
# use. Tokenizer.forward does x.unfold(-1, window_len, window_len) with
# window_len=100, so 801 yields the same 8 tokens as 800 and silently drops the
# trailing sample. Harmless, but it is why trialTime stays 4.
TRIAL_TIME = 4


def load_split(split, path=H5_PATH):
    """Read one split as (X, Y). Y is cast to uint8 to match the other builders."""
    with h5py.File(path, 'r', locking=False) as f:
        return f[split]['X'][:], f[split]['Y'][:].astype(np.uint8)


def verify(path=H5_PATH):
    """Re-check the delivered file against the conventions the repo assumes."""
    with h5py.File(path, 'r', locking=False) as f:
        assert sorted(f.keys()) == sorted(SPLITS), f'expected {SPLITS}, got {list(f.keys())}'
        for split in SPLITS:
            X, Y = f[split]['X'], f[split]['Y']
            assert X.ndim == 3, f'{split}/X: expected 3D, got {X.ndim}D'
            assert X.shape[1] == len(CH_NAMES), f'{split}/X: {X.shape[1]} channels != {len(CH_NAMES)}'
            assert X.shape[0] == Y.shape[0], f'{split}: {X.shape[0]} trials != {Y.shape[0]} labels'
            assert X.dtype == np.float32, f'{split}/X: expected float32, got {X.dtype}'

            # Stream the QC stats — train/X alone is ~2 GB.
            n_big, lo, hi, bad = 0, np.inf, -np.inf, False
            for i in range(0, X.shape[0], 500):
                blk = X[i:i + 500]
                n_big += int((np.abs(blk).max(axis=(1, 2)) > 12).sum())
                lo, hi = min(lo, blk.min()), max(hi, blk.max())
                bad = bad or not np.isfinite(blk).all()
            labels, counts = np.unique(Y[:], return_counts=True)
            print(f'{split:5s} X{X.shape} {X.dtype}  Y{Y.dtype} labels={dict(zip(labels.tolist(), counts.tolist()))}')
            print(f'      range=[{lo:.2f}, {hi:.2f}]  non-finite={bad}  '
                  f'trials over +/-12: {n_big}/{X.shape[0]} ({n_big / X.shape[0]:.1%})')


if __name__ == '__main__':
    verify()
