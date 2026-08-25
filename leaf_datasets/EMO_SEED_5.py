import torch
import einops
import glob
import mne
import numpy as np
import h5py
import multiprocessing as mp
import warnings
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = 'EMO_SEED_5'
SUBJECTS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
TEXT_LABELS = ['EMO/Disgust', 'EMO/Fear', 'EMO/Sad', 'EMO/Neutral', 'EMO/Happy']

CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
            'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
            'CB1', 'O1', 'Oz', 'O2', 'CB2']
SEGMENT_LEN = 4
session_labels ={
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

time_stamp = {
    1: {
        'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
    },
    2: {
        'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
    },  
    3: {
        'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
        'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    },
}

def proc_one(sub):
    trainX, trainY = [], []
    validX, validY = [], []
    testX, testY = [], []
    for session in [1, 2, 3]:
        fn = list(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/EEG_raw/{sub}_{session}*.cnt'))[0]
        print(f'Processing subject {sub}, session {session}, file {fn}')
        raw = mne.io.read_raw_cnt(fn, preload=True, verbose=False)
        # print(raw.info)
        raw.drop_channels(['VEO', 'HEO', 'M1', 'M2'])
        raw.resample(Param.resample, verbose=False)
        raw.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
        raw.notch_filter(freqs=50, notch_widths=2, verbose=False)
        time_stamp_start = time_stamp[session]['start']
        time_stamp_end = time_stamp[session]['end']
        data = raw.get_data(units="uV")

        for i in range(1, 16):
            trial = data[:, time_stamp_start[i-1]*Param.resample : time_stamp_end[i-1]*Param.resample]
            # trial = einops.rearrange(torch.tensor(trial).unfold(1, Param.resample*10, Param.resample*10), 'C N T -> N C T').numpy()
            trial = einops.rearrange(torch.tensor(trial).unfold(1, Param.resample*SEGMENT_LEN, Param.resample*SEGMENT_LEN), 'C N T -> N C T').numpy()
            print(i, trial.shape)
            y = np.array(session_labels[session][i-1]).repeat(trial.shape[0]).astype(np.uint8)
            if i <= 5:
                trainX.append(trial); trainY.append(y)
            elif i <= 10:
                validX.append(trial); validY.append(y)
            elif i <= 15:
                testX.append(trial); testY.append(y)

            print(i, len(trainX), len(validX), len(testX))

    trainX, trainY = np.concatenate(trainX), np.concatenate(trainY)
    validX, validY = np.concatenate(validX), np.concatenate(validY)
    testX, testY = np.concatenate(testX), np.concatenate(testY)
    trainX = pipeline(trainX, CH_NAMES)
    validX = pipeline(validX, CH_NAMES)
    testX = pipeline(testX, CH_NAMES)
    return sub, trainX, trainY, validX, validY, testX, testY

if __name__ == '__main__':
    # sub, trainX, trainY, validX, validY, testX, testY = proc_one(SUBJECTS[0])
    # print(sub, trainX.shape, trainY.shape, validX.shape, validY.shape, testX.shape, testY.shape)
    # print(np.unique(trainY, return_counts=True))
    # print(np.unique(validY, return_counts=True))
    # print(np.unique(testY, return_counts=True))

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
