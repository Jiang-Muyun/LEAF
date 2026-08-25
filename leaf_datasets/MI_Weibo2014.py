import mne
import numpy as np
import h5py
import multiprocessing as mp
from pathlib import Path
from leaf_datasets.shared import pipeline, Param

# https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Weibo2014.py#L64-L188
# The exported .npz already drops CB1/CB2 and the VEO/HEO ocular channels, so the
# 64-electrode montage above arrives here as the 60 channels listed in CH_NAMES.

# This dataset lives outside the shared raw_data mount and its .h5 is written to the
# LEAF root rather than configs/env.yaml's `downstream` folder, so both are set here.
SRC_FOLDER = Path('/media/datasets/EEG_Dataset')
DST_FOLDER = Path('/media/public/LEAF')

NAME = 'MI_Weibo2014'
TEXT_LABELS = ['MI/Left', 'MI/Right']
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']
CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
    'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
    'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
    'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2',
]

# Raw event codes: 1 left_hand, 2 right_hand, 3 hands, 4 feet,
# 5 left_hand_right_foot, 6 right_hand_left_foot, 7 rest.
# Only the unimanual trials are kept, to match the binary left/right protocol
# used by the other MI datasets.
KEEP_EVENTS = (1, 2)

def proc_one(subject):
    data = np.load(f'{SRC_FOLDER}/{NAME}/{subject}.npz', allow_pickle=True)

    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))

    fs = int(data['fs'])
    x, y = data['x_data'], data['y_data'].astype(np.uint8)
    mask = np.isin(y, KEEP_EVENTS)
    x, y = x[mask], y[mask] - 1
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))

    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0, verbose=False)
    epochs.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    if fs != Param.resample:
        epochs.resample(Param.resample, npad='auto', verbose=False)
    x = epochs.get_data().astype(np.float32)
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))
    x = pipeline(x, CH_NAMES)
    return subject, x, y

if __name__=='__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DST_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
