"""
Preprocess the SEED EEG dataset for three-class emotion recognition.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import torch
import einops
import glob
import mne
import numpy as np
import scipy
import h5py
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'EMO_SEED_3'
SUBJECTS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
TEXT_LABELS = ['EMO/Negative', 'EMO/Neutral', 'EMO/Positive']
CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'Oz', 'O2', 'CB2']
SEGMENT_LEN = 4
RAW_TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

def proc_one(sub):
    matfn = sorted(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/eeg_raw_data/{sub}_*.mat'))
    session1 = scipy.io.loadmat(matfn[0], squeeze_me=True)
    session2 = scipy.io.loadmat(matfn[1], squeeze_me=True)
    session3 = scipy.io.loadmat(matfn[2], squeeze_me=True)
    info = mne.create_info(ch_names=CH_NAMES, sfreq=200, ch_types='eeg')
    subShortName = [k for k in session1.keys() if isinstance(k,str) and '_eeg' in k][0].split('_')[0]
    trainX, trainY = [], []
    validX, validY = [], []
    testX, testY = [], []

    for trialID, trialLabel in enumerate(RAW_TRIAL_LABELS):
        data = np.concatenate([
            session1[f'{subShortName}_eeg{trialID+1}'], 
            session2[f'{subShortName}_eeg{trialID+1}'], 
            session3[f'{subShortName}_eeg{trialID+1}']
        ], axis=1)
        raw = mne.io.RawArray(data, info, verbose=False)
        if raw.info['sfreq'] != Param.resample:
            raw.resample(Param.resample, npad='auto', verbose=False)
        raw.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
        raw.notch_filter(freqs=50, notch_widths=2, verbose=False)
        x = raw.get_data(units="uV").astype(np.float32) 
        x = einops.rearrange(torch.tensor(x).unfold(1, Param.resample*SEGMENT_LEN, Param.resample*SEGMENT_LEN), 'C N T -> N C T').numpy()
        y = np.array(trialLabel+1).repeat(x.shape[0]).astype(np.uint8)

        if trialID < 9:
            trainX.append(x)
            trainY.append(y)
        elif trialID < 12:
            validX.append(x)
            validY.append(y)
        elif trialID < 15:
            testX.append(x)
            testY.append(y)

        print(f'Subject {sub}, Session {trialLabel+1}, Label {y.shape}, {x.shape}', y[0])

    trainX, trainY = np.concatenate(trainX), np.concatenate(trainY)
    validX, validY = np.concatenate(validX), np.concatenate(validY)
    testX, testY = np.concatenate(testX), np.concatenate(testY)
    trainX = pipeline(trainX, CH_NAMES)
    validX = pipeline(validX, CH_NAMES)
    testX = pipeline(testX, CH_NAMES)
    return sub, trainX, trainY, validX, validY, testX, testY

if __name__ == '__main__':
    with mp.Pool(len(SUBJECTS)) as pool:
        res = pool.map(proc_one, SUBJECTS)

    trainX, trainY = [], []
    validX, validY = [], []
    testX, testY = [], []
    for sub, _trainX, _trainY, _validX, _validY, _testX, _testY in res:
        trainX.append(_trainX)
        trainY.append(_trainY)
        validX.append(_validX)
        validY.append(_validY)
        testX.append(_testX)
        testY.append(_testY)

    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)
    validX = np.concatenate(validX)
    validY = np.concatenate(validY)
    testX = np.concatenate(testX)
    testY = np.concatenate(testY)
    
    print(trainX.shape, trainY.shape, np.unique(trainY, return_counts=True))
    print(validX.shape, validY.shape, np.unique(validY, return_counts=True))
    print(testX.shape, testY.shape, np.unique(testY, return_counts=True))

    with h5py.File(f'{DATA_FOLDER}/{NAME}_seg{SEGMENT_LEN}_v2.h5', 'w') as f:
        # f.create_dataset('trainX', data=trainX)
        f.create_dataset('trainX', data=np.clip(trainX, -10, 10))
        f.create_dataset('trainY', data=trainY)
        f.create_dataset('validX', data=np.clip(validX, -10, 10))
        f.create_dataset('validY', data=validY)
        f.create_dataset('testX', data=np.clip(testX, -10, 10))
        f.create_dataset('testY', data=testY)
