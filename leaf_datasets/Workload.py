"""
Preprocess the EEG workload dataset for resting-versus-workload classification.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import os
import re
import glob
import h5py
import torch
import einops
import numpy as np
import multiprocessing as mp
import mne
from mne.io import read_raw_edf
from pathlib import Path
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = "Workload" 
SUBJECTS = [
    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '30', '31', '32', '33', '34', '35'
]
CH_NAMES = ['Fp1','Fp2','F3','F4','F7','F8','T3','T4','C3','C4','T5','T6','P3',
            'P4','O1','O2','Fz','Cz','Pz']
TEXT_LABELS = ['before', 'during']

def _load_one_file(sub, label):
    edf_path = Path(RAW_DATA_FOLDER) / f"{NAME}/Subject{sub}_{label}.edf"
    raw = read_raw_edf(edf_path, preload=True, verbose=False)
    raw.drop_channels(['ECG ECG'])
    raw.drop_channels(['EEG A2-A1'])
    new_names = {_old: _old[4:] for _old in raw.ch_names}
    mne.rename_channels(raw.info, new_names)
    raw.resample(Param.resample, verbose=False)
    raw.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
    data = raw.get_data().astype(np.float32)  # C x T
    return data

def proc_one(sub):
    X0 = _load_one_file(sub, 1)
    X1 = _load_one_file(sub, 2)
    X0 = einops.rearrange(torch.tensor(X0).unfold(1, Param.resample*4, Param.resample*4), 'C N T -> N C T').numpy()
    X1 = einops.rearrange(torch.tensor(X1).unfold(1, Param.resample*4, Param.resample*4), 'C N T -> N C T').numpy()
    Y0 = np.zeros((X0.shape[0]), dtype=np.uint8)
    Y1 = np.ones((X1.shape[0]), dtype=np.uint8)
    X, Y = np.concatenate([X0, X1], axis=0), np.concatenate([Y0, Y1], axis=0)
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__ == "__main__":
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
