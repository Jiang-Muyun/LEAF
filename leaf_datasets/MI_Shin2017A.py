"""
Preprocess the Shin2017A EEG dataset for left-versus-right motor-imagery decoding.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import os
import mne
import numpy as np
import scipy
import h5py
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

# https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bbci_eeg_fnirs.py#L192-L312
NAME = 'MI_Shin2017A'
TEXT_LABELS = ['MI/Left', 'MI/Right']
SUBJECTS = [
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 
    'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 
    'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29']
SUBJECTS.remove('S05')  # S05 is not available in the dataset
CH_NAMES = [
    'F7', 'AFF5h', 'F3', 'AFp1', 'AFp2', 'AFF6h', 'F4', 'F8', 'AFF1h', 
    'AFF2h', 'Cz', 'Pz', 'FCC5h', 'FCC3h', 'CCP5h', 'CCP3h', 'T7', 'P7',
    'P3', 'PPO1h', 'POO1', 'POO2', 'PPO2h', 'P4', 'FCC4h', 'FCC6h', 'CCP4h',
    'CCP6h', 'P8', 'T8'
]

def proc_one(subject):
    fname1 = f'{RAW_DATA_FOLDER}/{NAME}/subject {subject[1:]}/with occular artifact/cnt.mat'
    fname2 = f'{RAW_DATA_FOLDER}/{NAME}/subject {subject[1:]}/with occular artifact/mrk.mat'
    data = scipy.io.loadmat(fname1, squeeze_me=True, struct_as_record=False)["cnt"]
    mrk = scipy.io.loadmat(fname2, squeeze_me=True, struct_as_record=False)["mrk"]
    X, Y = [], []
    for ii in [0, 2, 4]:
        sfreq = 200
        ch_names = list(data[ii].clab)
        ch_types = ["eeg"] * 30 + ["eog"] * 2
        eeg = data[ii].x.T #* 1e-6
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(data=eeg, info=info, verbose=False)
        raw.drop_channels(['VEOG', 'HEOG'])
        raw.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
        raw.notch_filter(50, verbose=False)

        trig_offset = 0
        mkr_time = ((mrk[ii].time - 1) // 5) / sfreq
        mkr = mrk[ii].event.desc // 16 + trig_offset
        print(mkr_time.shape, mkr.shape)
        
        events = np.column_stack(((mkr_time * sfreq).astype(int), np.zeros_like(mkr), mkr)).astype(int)
        tmin = 0
        # tmax = 10
        tmax = 4
        epochs = mne.Epochs(raw, events=events, event_id=None, tmin=tmin, tmax=tmax, preload=True, verbose=False, baseline=None)
        epochs.resample(Param.resample, npad="auto")
        x = epochs.get_data()[:, : ,1:].astype(np.float32)
        y = mkr - 1
        X.append(x)
        Y.append(y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return subject, X, Y

if __name__=='__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.percentile(X, [0, 1, 10, 90, 99, 100]))
