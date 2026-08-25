"""
Preprocess the ShanghaiU EEG dataset for left-versus-right motor-imagery decoding.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import os
import mne
import h5py
import numpy as np
import scipy
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_ShanghaiU'
TEXT_LABELS = ['MI/Left', 'MI/Right']
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 
            'S21', 'S22', 'S23', 'S24', 'S25']
RAW_CH_NAMES = [
    'Fp1', 'Fp2','Fz', 'F3', 'F4', 'F7', 'F8', 
    'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4',
    'A1', 'A2',
    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 
    'PO3', 'PO4', 'Oz', 'O1', 'O2'
]
CH_NAMES = [
    'Fp1', 'Fp2','Fz', 'F3', 'F4', 'F7', 'F8', 
    'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4',
    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 
    'PO3', 'PO4', 'Oz', 'O1', 'O2'
]

def proc_one(sub):
    sess1 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/{sub}/d1.mat')
    sess2 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/{sub}/d2.mat')
    sess3 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/{sub}/d3.mat')
    sess4 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/{sub}/d4.mat')
    sess5 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/{sub}/d5.mat')
    epoch = np.concatenate([sess1['data'], sess2['data'], sess3['data'], sess4['data'], sess5['data']], axis=0)
    label = np.concatenate([sess1['labels'][0], sess2['labels'][0], sess3['labels'][0], sess4['labels'][0], sess5['labels'][0]], axis=0)
    info = mne.create_info(ch_names=RAW_CH_NAMES, sfreq=250, ch_types='eeg')
    epochs = mne.EpochsArray(epoch, info, verbose=False)
    epochs.drop_channels(['A1', 'A2'])
    epochs.resample(Param.resample, npad='auto', verbose=False)
    epochs.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    X = epochs.get_data().astype(np.float32)
    Y = label.astype(np.uint8) - 1
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__ == '__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
