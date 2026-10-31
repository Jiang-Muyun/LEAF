"""
Load processed EEG datasets and construct the splits used by LEAF.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import h5py
import torch
import numpy as np
import torch.nn.functional as F

from leaf_datasets.shared import DATA_FOLDER

def load_by_index(ds, indices=None, concat=True):
    with h5py.File(DATA_FOLDER / f'{ds}.h5', 'r', locking=False) as f:
        subjects = list(f.keys())
        selected = subjects if indices is None else [subjects[i] for i in indices]
        X, Y = [], []
        for sub in selected:
            x, y = f[sub]['X'][:], f[sub]['Y'][:]
            X.append(x)
            Y.append(y)
        if concat:
            return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        else:
            return np.array(X), np.array(Y)
        
def load_predefined_split(ds):
    with h5py.File(DATA_FOLDER / f'{ds}.h5', 'r', locking=False) as f:
        trainX, trainY = f['trainX'][:], f['trainY'][:]
        valX, valY     = f['validX'][:], f['validY'][:]
        testX, testY   = f['testX'][:],  f['testY'][:]
    return trainX, trainY, valX, valY, testX, testY

def split_train_valid(trainvalX, trainvalY, val_ratio):
    assert val_ratio in [0.1, 0.2], "Only val_ratio = 0.1 or 0.2 is supported."
    assert len(trainvalX) == len(trainvalY)

    n_samples = len(trainvalX)
    step = int(1 / val_ratio)
    val_idx = np.arange(0, n_samples, step)
    train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

    trainX, valX = trainvalX[train_idx], trainvalX[val_idx]
    trainY, valY = trainvalY[train_idx], trainvalY[val_idx]
    return trainX, trainY, valX, valY

# HDF5 key order and subject boundaries match the published dataset splits.
SUBJECT_SPLITS = {
    'EMO_FACED': (100, 123), 'MI_BCIC_IV2a': (7, 9),
    'MI_OpenBMI': (42, 54), 'MI_BCIC_Upperlimb': (11, 15),
    'MI_ShanghaiU': (20, 25), 'MI_HighGamma': (10, 14),
    'MI_Cho2017': (40, 49), 'MI_Shin2017A': (22, 28),
    'MI_PhysioNet': (80, 109), 'Workload': (32, 36),
    'SSVEP_OpenBMI': (42, 54),
}
PREDEFINED_SPLITS = {
    'EMO_SEED_3_seg4', 'EMO_SEED_4_seg4', 'EMO_SEED_5_seg4',
    'EMO_SEED_7_seg4', 'ADHD_AliMotie',
}
SPEECH_TRAIN_END = 250
SPEECH_VALID_END = 300


def filter_classes(ds, x, y):
    if ds == 'MI_HighGamma':
        keep = y != 2
        return x[keep], y[keep]
    return x, y


def load_test_split(ds):
    """Read the standard test partition without allocating train/validation data."""
    if ds in SUBJECT_SPLITS:
        start, stop = SUBJECT_SPLITS[ds]
        return filter_classes(ds, *load_by_index(ds, range(start, stop)))
    with h5py.File(DATA_FOLDER / f'{ds}.h5', 'r', locking=False) as f:
        if ds in PREDEFINED_SPLITS:
            return f['testX'][:], f['testY'][:]
        if ds == 'CS_BCIC_Speech':
            subjects = list(f.keys())
            x = np.concatenate([f[sub]['X'][SPEECH_VALID_END:] for sub in subjects])
            y = np.concatenate([f[sub]['Y'][SPEECH_VALID_END:] for sub in subjects])
            return x, y
    raise ValueError(f'Unknown dataset {ds}')


def load_dataset(ds, val_ratio=0.2):
    trial_time = 10 if ds in {'EMO_FACED', 'ADHD_AliMotie'} else 4
    if ds in PREDEFINED_SPLITS:
        return (trial_time, *load_predefined_split(ds))
    if ds in SUBJECT_SPLITS:
        train_end, _ = SUBJECT_SPLITS[ds]
        x, y = filter_classes(ds, *load_by_index(ds, range(train_end)))
        train_x, train_y, val_x, val_y = split_train_valid(x, y, val_ratio)
    elif ds == 'CS_BCIC_Speech':
        x, y = load_by_index(ds, concat=False)
        train_x = x[:, :SPEECH_TRAIN_END].reshape(-1, *x.shape[2:])
        train_y = y[:, :SPEECH_TRAIN_END].reshape(-1)
        val_x = x[:, SPEECH_TRAIN_END:SPEECH_VALID_END].reshape(-1, *x.shape[2:])
        val_y = y[:, SPEECH_TRAIN_END:SPEECH_VALID_END].reshape(-1)
    else:
        raise ValueError(f'Unknown dataset {ds}')
    test_x, test_y = load_test_split(ds)
    return trial_time, train_x, train_y, val_x, val_y, test_x, test_y
