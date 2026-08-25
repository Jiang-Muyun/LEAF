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

def load_dataset(ds, val_ratio=0.2):
    if ds == 'EMO_FACED':
        trialTime = 10
        trainvalX, trainvalY = load_by_index(ds, range(0, 100))
        testX,  testY        = load_by_index(ds, range(100, 123))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds in ['EMO_SEED_3_seg4', 'EMO_SEED_4_seg4', 'EMO_SEED_5_seg4', 'EMO_SEED_7_seg4']:
        trialTime = 4
        trainX, trainY, valX, valY, testX, testY = load_predefined_split(ds)

    elif ds == 'MI_BCIC_IV2a':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 7))
        testX,  testY        = load_by_index(ds, range(7, 9))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_OpenBMI':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 42))
        testX,  testY        = load_by_index(ds, range(42, 54))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_BCIC_Upperlimb':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 11))
        testX,  testY        = load_by_index(ds, range(11, 15))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_ShanghaiU':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 20))
        testX,  testY        = load_by_index(ds, range(20, 25))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_HighGamma':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 10))
        testX,  testY        = load_by_index(ds, range(10, 14))
        trainvalX = trainvalX[trainvalY != 2]
        trainvalY = trainvalY[trainvalY != 2]
        testX = testX[testY != 2]
        testY = testY[testY != 2]
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_Cho2017':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 40))
        testX,  testY        = load_by_index(ds, range(40, 49))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_Shin2017A':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 22))
        testX,  testY        = load_by_index(ds, range(22, 28))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'MI_PhysioNet':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 80))
        testX,  testY        = load_by_index(ds, range(80, 109))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'CS_BCIC_Speech':
        trialTime = 4
        X, Y = load_by_index(ds, concat=False)
        trainX, trainY = X[:,    :250], Y[:,    :250]
        valX,   valY   = X[:, 250:300], Y[:, 250:300]
        testX, testY   = X[:, 300:],    Y[:, 300:]
        trainX = trainX.reshape(-1, trainX.shape[2], trainX.shape[3])
        trainY = trainY.reshape(-1)
        valX   = valX.reshape(-1,   valX.shape[2],   valX.shape[3])
        valY   = valY.reshape(-1)
        testX  = testX.reshape(-1,  testX.shape[2],  testX.shape[3])
        testY  = testY.reshape(-1)

    elif ds == 'Workload':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 32))
        testX    , testY     = load_by_index(ds, range(32, 36))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    elif ds == 'ADHD_AliMotie':
        trialTime = 10
        trainX, trainY, valX, valY, testX, testY = load_predefined_split(ds)

    elif ds == 'SSVEP_OpenBMI':
        trialTime = 4
        trainvalX, trainvalY = load_by_index(ds, range(0, 42))
        testX,  testY        = load_by_index(ds, range(42, 54))
        trainX, trainY, valX, valY = split_train_valid(trainvalX, trainvalY, val_ratio)

    else:
        raise ValueError(f'Unknown dataset {ds}')

    return trialTime, trainX, trainY, valX, valY, testX, testY
