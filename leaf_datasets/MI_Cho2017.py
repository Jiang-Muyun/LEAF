"""
Preprocess the Cho2017 EEG dataset for left-versus-right motor-imagery decoding.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import mne
import numpy as np
import h5py
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_Cho2017'
TEXT_LABELS = ['MI/Left', 'MI/Right']
# no s32, s46, s49
SUBJECTS = [
    's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 
    's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 
    's31', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41',
    's42', 's43', 's44', 's45', 's47', 's48', 's50', 's51', 's52'
]
CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]

def proc_one(subject):
    data = np.load(f'{RAW_DATA_FOLDER}/{NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data']/100, data['y_data'].astype(np.uint8)
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=data['fs'], ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.resample(Param.resample, npad='auto')
    epochs.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    x = epochs.get_data().astype(np.float32)
    x = np.pad(x, ((0, 0), (0, 0), (0, 200)), mode='edge')
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    x = pipeline(x, CH_NAMES)
    return subject, x, y

if __name__=='__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
