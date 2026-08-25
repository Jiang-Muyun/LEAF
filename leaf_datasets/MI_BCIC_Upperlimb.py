import mne
import numpy as np
import scipy
import multiprocessing as mp
import h5py
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

# https://brain.korea.ac.kr/bci2021/competition.php
# https://osf.io/etxp6?view_only=08e7108d89fd42bab2adbd6b98fb683d
NAME = 'MI_BCIC_Upperlimb'
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15']
CH_NAMES = [
    'Fp1', 'AF7', 'AF3', 'AFz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'FT7',
    'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TP7', 'CP5',
    'CP3', 'CP1', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO7', 'PO3',
    'POz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4',
    'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8',
    'P2', 'P4', 'P6', 'P8', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]
TEXT_LABELS = ['MI/Cylin', 'MI/Sphe', 'MI/Lumbrical']

def proc_one(sub):
    fname = f'{RAW_DATA_FOLDER}/{NAME}/Training set/sample{sub}.mat'
    data = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)['epo']
    x_1, y_1 = data.x, np.argmax(data.y, axis=0)
    # print(sub, sfreq, x_1.shape, y_1.shape, np.unique(y_1, return_counts=True))

    fname = f'{RAW_DATA_FOLDER}/{NAME}/Validation set/sample{sub}.mat'
    data = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)['epo']
    x_2, y_2 = data.x, np.argmax(data.y, axis=0)
    # print(sub, sfreq, x_2.shape, y_2.shape, np.unique(y_2, return_counts=True))
    x = np.concatenate([x_1, x_2], axis=2).transpose(2, 1, 0)[:,:,:-1]
    y = np.concatenate([y_1, y_2], axis=0)
    print(sub, x.shape, y.shape, np.unique(y, return_counts=True))
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=data.fs, ch_types='eeg')
    epochs = mne.EpochsArray(x, info, tmin=0)
    epochs.resample(Param.resample, npad='auto')
    epochs.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
    X = epochs.get_data().astype(np.float32)

    # A single trial lasted 10seconds (s) and contained three sub-stages, which were 3 s, 3 s, and 4 s long, respectively. 
    # The subjects perform motor imagery during the stage for 4 s after the visual cue was given
    X = X[:,:, Param.resample * 6:]
    Y = y.astype(np.uint8)

    print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
    X = pipeline(X, CH_NAMES)
    return sub, X, Y

if __name__ == '__main__':
    with mp.Pool(32) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
