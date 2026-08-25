import os
import mne
import numpy as np
import scipy
import multiprocessing as mp
import h5py
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'CS_BCIC_Speech'
TEXT_LABELS = ['CS/hello', 'CS/help-me', 'CS/stop', 'CS/thank-you', 'CS/yes']
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
CH_NAMES = [
    'Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 
    'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
    'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8'
]

def proc_one(sub):
    data_train = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/Training set/Data_Sample{sub}.mat')
    data_valid = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/{NAME}/Validation set/Data_Sample{sub}.mat')
    x_t, y_t = np.asarray(data_train['epo_train']['x'])[0][0], np.asarray(data_train['epo_train']['y'])[0][0].argmax(0)
    x_v, y_v = np.asarray(data_valid['epo_validation']['x'])[0][0], np.asarray(data_valid['epo_validation']['y'])[0][0].argmax(0)
    x_t = np.transpose(x_t, (2, 1, 0)).astype(np.float32)
    x_v = np.transpose(x_v, (2, 1, 0)).astype(np.float32)
    x, y = np.concatenate((x_t, x_v), axis=0), np.concatenate((y_t, y_v), axis=0).astype(np.uint8)
    x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
    print(sub, x.shape, y.shape)
    x = pipeline(x, CH_NAMES)
    return sub, x, y

if __name__ == '__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
