"""
Preprocess the PhysioNet EEG Motor Movement/Imagery dataset for motor-imagery decoding.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import mne
import numpy as np
import multiprocessing as mp
import h5py
from functools import partial
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_PhysioNet'
TEXT_LABELS = ['MI/Left', 'MI/Right']
CH_NAMES = [
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3',
    'CP1','CPz','CP2','CP4','CP6','Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5',
    'F3','F1','Fz','F2','F4','F6','F8','FT7','FT8','T7','T8','T9','T10','TP7','TP8','P7','P5',
    'P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'
]
SUBJECTS = [
    'S001','S002','S003','S004','S005','S006','S007','S008','S009','S010',
    'S011','S012','S013','S014','S015','S016','S017','S018','S019','S020',
    'S021','S022','S023','S024','S025','S026','S027','S028','S029','S030',
    'S031','S032','S033','S034','S035','S036','S037','S038','S039','S040',
    'S041','S042','S043','S044','S045','S046','S047','S048','S049','S050',
    'S051','S052','S053','S054','S055','S056','S057','S058','S059','S060',
    'S061','S062','S063','S064','S065','S066','S067','S068','S069','S070',
    'S071','S072','S073','S074','S075','S076','S077','S078','S079','S080',
    'S081','S082','S083','S084','S085','S086','S087','S088','S089','S090',
    'S091','S092','S093','S094','S095','S096','S097','S098','S099','S100',
    'S101','S102','S103','S104','S105','S106','S107','S108','S109'
]

# In summary, the experimental runs were:
# Baseline, eyes open
# Baseline, eyes closed
# 3.  Task 1 (open and close left or right fist)
# 4.  Task 2 (imagine opening and closing left or right fist)
# 5.  Task 3 (open and close both fists or both feet)
# 6.  Task 4 (imagine opening and closing both fists or both feet)
# 7.  Task 1
# 8.  Task 2 (imagine opening and closing left or right fist)
# 9.  Task 3
# 10. Task 4
# 11. Task 1
# 12. Task 2 (imagine opening and closing left or right fist)
# 13. Task 3
# 14. Task 4
# Each annotation includes one of three codes (T0, T1, or T2):

# T0 corresponds to rest
# T1 corresponds to onset of motion (real or imagined) of
#    the left fist (in runs 3, 4, 7, 8, 11, and 12)
#    both fists    (in runs 5, 6, 9, 10, 13, and 14)
# T2 corresponds to onset of motion (real or imagined) of
#    the right fist (in runs 3, 4, 7, 8, 11, and 12)
#    both feet      (in runs 5, 6, 9, 10, 13, and 14)

# CBraMod setting
# CBraMod mixed motor imagery and motor execution tasks.
# run 04, 08, 12 are motor imagery tasks.
# run 06, 10, 14 are motor execution tasks.
# tasks = ['04', '06', '08', '10', '12', '14']

# ELIA setting, only motor imagery runs
# We only use motor imagery runs here.
tasks = ['04', '08',  '12'] 

def proc_one(sub, tmin=0.0, tmax=4.0):
    X, Y = [], []
    for run in range(3, 15):
        raw = mne.io.read_raw_edf(f"{RAW_DATA_FOLDER}/{NAME}/{sub}/{sub}R{run:02d}.edf", preload=True, verbose="error")
        if len(raw.info['bads']) > 0:
            raw.interpolate_bads(reset_bads=True, mode='accurate')
        raw.resample(Param.resample, npad='auto', verbose=False)
        raw.notch_filter(freqs=60, verbose=False)
        raw.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
        raw.rename_channels(lambda s: s.rstrip("."))
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True, verbose=False)
        y = epochs.events[:, 2].astype(np.uint8)
        print(sub, run, y)
        x = epochs.get_data(units='uV').astype(np.float32)
        x = x[:,:, :800-x.shape[2]]
        mask = (y == 2) | (y == 3)

        x = x[mask]
        y = y[mask] - 2

        X.append(x)
        Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__ == "__main__":
    with mp.Pool(64) as pool:
        res = pool.map(proc_one, SUBJECTS)

    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
