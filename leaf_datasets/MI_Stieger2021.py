import os
import mne
import numpy as np
import scipy
import multiprocessing as mp
import h5py
from pathlib import Path
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'MI_HeBin2021'
TEXT_LABELS = ['MI/Right', 'MI/Left', 'MI/Up', 'MI/Down']
SUBJECTS = [
    'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 
    'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 
    'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 
    'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 'S50', 'S51', 'S52', 
    'S53', 'S54', 'S55', 'S56', 'S57', 'S58', 'S59']

CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 
            'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
            'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 
            'CB1', 'O1', 'Oz', 'O2', 'CB2']

# Task number is used to identify the individual BCI tasks (1 = 'LR', 2 = 'UD', 3 = '2D'). 
# Target number is used to identify which target was presented to the participants (1 = right, 2 = left, 3 = up, 4 = down).
# This dataset is too large

# trials can be from 5 sec to 11 sec, first 2 sec is baseline
# average trial length: 9 sec
# there we give it 10 sec, and pad the rest with edge value
start, end = 0, 10

def proc_one_mat(fn):
    try:
        BCI = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=True)['BCI']
        trial_buf, label_buf = [], []
        for i in range(450):
            trial = BCI.data[i]
            targetnumber = BCI.TrialData[i].targetnumber
            if trial.shape[1] < end*1000:
                trial = np.pad(trial, ((0, 0), (0, end*1000-trial.shape[1])), 'constant', constant_values=0)
            else:
                trial = trial[:, :end*1000]
            trial_buf.append(trial)
            label_buf.append(targetnumber)
        trial_buf, label_buf = np.array(trial_buf), np.array(label_buf).astype(np.int32)
        epochs = mne.EpochsArray(trial_buf, mne.create_info(CH_NAMES, 1000, 'eeg'), verbose=False)
        epochs.resample(Param.resample, verbose=False)
        epochs.filter(Param.lp, Param.hp, verbose=False)
        x = epochs.get_data().astype(np.float32)
        clip_min, clip_max = np.percentile(x, [0.5, 99.5]).astype(np.float32)
        print(clip_min, clip_max)
        x = np.clip(x, clip_min, clip_max)
        y = label_buf.astype(np.uint8) - 1
        x = pipeline(x, CH_NAMES)
        x = np.clip(x, -15, 15)
        print(x.shape, y.shape, np.unique(y, return_counts=True), x.dtype)
        return x, y
    except Exception as e:
        print(f"Error in {fn}: {e}")
        return None

if __name__ == '__main__':
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as fo:
        for sub in SUBJECTS:
            file_list = []
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                fn = f'{RAW_DATA_FOLDER}/{NAME}/{sub}_Session_{i}.mat'
                if not os.path.exists(fn):
                    print(f'{fn} not exists')
                    continue
                file_list.append(fn)
            with mp.Pool(len(file_list)) as pool:
                res = pool.map(proc_one_mat, file_list)
            X = np.concatenate([r[0] for r in res if r is not None])
            Y = np.concatenate([r[1] for r in res if r is not None])
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

            fo.create_dataset(f'{sub}/X', data=X.astype(np.float32))
            fo.create_dataset(f'{sub}/Y', data=Y.astype(np.uint8))
