"""
Preprocess the BCI Competition IV 2a EEG dataset for four-class motor-imagery decoding.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import scipy.io
import numpy as np
import h5py
import mne
import multiprocessing as mp
from pathlib import Path
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_BCIC_IV2a'
TEXT_LABELS = ['MI/Left', 'MI/Right', 'MI/Foot', 'MI/Tongue']
SUBJECTS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
CH_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz','FC2', 'FC4', 
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1','Pz','P2','POz'
]

def proc_one_mat(data):
    num = len(data['data'][0])
    samplesX, samplesY = [], []
    for j in range(3, num):
        raw_data = data['data'][0, j][0, 0][0][:, :22]
        events = data['data'][0, j][0, 0][1][:, 0]
        labels = data['data'][0, j][0, 0][2][:, 0]
        length = raw_data.shape[0]
        events = events.tolist()
        events.append(length)
        # print(events)
        annos = []
        for i in range(len(events) - 1):
            annos.append((events[i], events[i + 1]))
        for i, (anno, label) in enumerate(zip(annos, labels)):
            sample = raw_data[anno[0]:anno[1]].transpose(1, 0)
            sample  = sample - np.mean(sample, axis=0, keepdims=True)
            sample = sample[:, 0 * 250:6 * 250]
            samplesX.append(sample)
            samplesY.append(label - 1)
    return np.array(samplesX), np.array(samplesY)

def proc_one(sub):
    XE, YE = proc_one_mat(scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}_mat/{sub}E.mat'))
    XT, YT = proc_one_mat(scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}_mat/{sub}T.mat'))
    X, Y = np.concatenate([XE, XT]), np.concatenate([YE, YT])
    epo = mne.EpochsArray(X, mne.create_info(ch_names=CH_NAMES, sfreq=250, ch_types='eeg'))
    epo.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    epo.resample(Param.resample, npad='auto', verbose=False)
    X = epo.get_data().astype(np.float32)[:,:, 200*2: 200*6]  # Keep only the last 4 seconds
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__=='__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
