import os
import mne
import h5py
import numpy as np
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_HighGamma'
TEXT_LABELS = ['MI/Left', 'MI/Right', 'MI/Feet']
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
RAW_CH_NAMES = [
    'Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
    'M1','T7','C3','Cz','C4','T8','M2','CP5','CP1','CP2','CP6','P7','P3',
    'Pz','P4','P8','POz','O1','Oz','O2','AF7','AF3','AF4','AF8','F5','F1',
    'F2','F6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P5',
    'P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8',
    'FT9','FT10','TPP9h','TPP10h','PO9','PO10','P9','P10','AFF1','AFz','AFF2',
    'FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h',
    'CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','PPO1','PPO2','I1','Iz','I2','AFp3h',
    'AFp4h','AFF5h','AFF6h','FFT7h','FFC1h','FFC2h','FFT8h','FTT9h','FTT7h','FCC1h',
    'FCC2h','FTT8h','FTT10h','TTP7h','CCP1h','CCP2h','TTP8h','TPP7h','CPP1h','CPP2h',
    'TPP8h','PPO9h','PPO5h','PPO6h','PPO10h','POO9h','POO3h','POO4h','POO10h','OI1h','OI2h'
]

# remove the M1 and M2 channels
CH_NAMES = [
    'Fp1','Fp2','Fpz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
    'T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3',
    'Pz','P4','P8','POz','O1','Oz','O2','AF7','AF3','AF4','AF8','F5','F1',
    'F2','F6','FC3','FCz','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P5',
    'P1','P2','P6','PO5','PO3','PO4','PO6','FT7','FT8','TP7','TP8','PO7','PO8',
    'FT9','FT10','TPP9h','TPP10h','PO9','PO10','P9','P10','AFF1','AFz','AFF2',
    'FFC5h','FFC3h','FFC4h','FFC6h','FCC5h','FCC3h','FCC4h','FCC6h','CCP5h','CCP3h',
    'CCP4h','CCP6h','CPP5h','CPP3h','CPP4h','CPP6h','PPO1','PPO2','I1','Iz','I2','AFp3h',
    'AFp4h','AFF5h','AFF6h','FFT7h','FFC1h','FFC2h','FFT8h','FTT9h','FTT7h','FCC1h',
    'FCC2h','FTT8h','FTT10h','TTP7h','CCP1h','CCP2h','TTP8h','TPP7h','CPP1h','CPP2h',
    'TPP8h','PPO9h','PPO5h','PPO6h','PPO10h','POO9h','POO3h','POO4h','POO10h','OI1h','OI2h'
]

def proc_one(subject):
    data = np.load(f'{RAW_DATA_FOLDER}/{NAME}/{subject}.npz', allow_pickle=True)
    
    print(data['fs'], data['metadata'])
    print(data['x_data'].shape, data['y_data'].shape, np.unique(data['y_data'], return_counts=True))
    
    x, y = data['x_data'], data['y_data'].astype(np.uint8)-1
    # S01 (480, 126, 800) (480,) (array([0, 1, 2, 3], dtype=uint8), array([120, 120, 120, 120]))
    print(subject, x.shape, y.shape, np.unique(y, return_counts=True))

    # Keep only classes 0,1,2 (left, right, feet)
    mask = (y == 0) | (y == 1) | (y == 2)
    x = x[mask]
    y = y[mask]

    info = mne.create_info(ch_names=RAW_CH_NAMES, sfreq=data['fs'], ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.drop_channels(['M1', 'M2'])
    epochs.resample(Param.resample, npad='auto')
    epochs.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    x = epochs.get_data().astype(np.float32)
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
