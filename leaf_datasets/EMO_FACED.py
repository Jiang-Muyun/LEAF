"""
Preprocess the FACED EEG dataset for nine-class emotion recognition.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import torch
import einops
import mne
import numpy as np
import pickle
import h5py
from scipy import signal
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'EMO_FACED'
labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])
SUBJECTS = [f'{i:03d}' for i in range(0, 123)]
SEGMENT_LEN = 10
RAW_CH_NAMES = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
              'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'CP1', 'CP2', 'CP5',
              'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2']

CH_NAMES = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
              'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5',
              'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2']

def proc_one(sub):
    fn = f'{RAW_DATA_FOLDER}/{NAME}/Processed_data/sub{sub}.pkl'
    with open(fn, 'rb') as f:
        array = pickle.load(f)
    epo = mne.EpochsArray(array, mne.create_info(ch_names=RAW_CH_NAMES, sfreq=250, ch_types='eeg'), verbose=False)
    epo.drop_channels(['A1', 'A2'])
    epo.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
    epo.resample(Param.resample, npad='auto', verbose=False)
    eeg = epo.get_data().astype(np.float32)
    print(f'Processing {fn} with shape {eeg.shape}', len(labels))

    x = torch.tensor(eeg).unfold(2, Param.resample * SEGMENT_LEN, Param.resample * SEGMENT_LEN)
    B, C, N, T = x.shape
    X = einops.rearrange(x, 'B C N T -> (B N) C T').numpy()
    Y = torch.repeat_interleave(torch.tensor(labels), repeats=N).numpy().astype(np.uint8)

    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__ == '__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}_seg{SEGMENT_LEN}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
