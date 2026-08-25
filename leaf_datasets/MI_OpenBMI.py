import os
import sys
import mne
import numpy as np
import scipy
import multiprocessing as mp
from functools import partial
import h5py
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME    = 'MI_OpenBMI'
TEXT_LABELS = ['MI/Right', 'MI/Left']
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 
    'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 
    'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 
    'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 
    'AF3', 'AF4', 'AF8', 'PO3', 'PO4'
]
SUBJECTS = [
    's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 
    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 
    's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 
    's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 
    's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48', 's49', 's50',
    's51', 's52', 's53', 's54',
]

def proc_one(sub, task):
    sess1 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/MI_OpenBMI/DB_mat/session1/{sub}/EEG_{task}.mat')
    sess2 = scipy.io.loadmat(f'{RAW_DATA_FOLDER}/MI_OpenBMI/DB_mat/session2/{sub}/EEG_{task}.mat')
    d1, l1 = np.transpose(sess1[f'EEG_{task}_train']['smt'][0,0], (1, 2, 0)), sess1[f'EEG_{task}_train']['y_dec'][0,0][0]
    d2, l2 = np.transpose(sess1[f'EEG_{task}_test']['smt'][0,0], (1, 2, 0)),  sess1[f'EEG_{task}_test']['y_dec'][0,0][0]
    d3, l3 = np.transpose(sess2[f'EEG_{task}_train']['smt'][0,0], (1, 2, 0)), sess2[f'EEG_{task}_train']['y_dec'][0,0][0]
    d4, l4 = np.transpose(sess2[f'EEG_{task}_test']['smt'][0,0], (1, 2, 0)),  sess2[f'EEG_{task}_test']['y_dec'][0,0][0]
    print(sub, d1.shape, l1.shape, d2.shape, l2.shape, d3.shape, l3.shape, d4.shape, l4.shape)
    epoch = np.concatenate([d1, d2, d3, d4], axis=0)
    label = np.concatenate([l1, l2, l3, l4], axis=0)
    epo = mne.EpochsArray(epoch, mne.create_info(ch_names=CH_NAMES, sfreq=1000, ch_types='eeg'), verbose=False)
    epo.resample(Param.resample, npad='auto', verbose=False)
    epo.filter(l_freq=Param.MI_lp, h_freq=Param.MI_hp, verbose=False)
    x = epo.get_data().astype(np.float32)
    y = label - 1
    x = pipeline(x, CH_NAMES)
    return sub, x, y

if __name__ == '__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(partial(proc_one, task='MI'), SUBJECTS)

    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))
