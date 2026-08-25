"""
Preprocess the SEED-VII EEG dataset for seven-class emotion recognition.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import os
import glob
import csv
import datetime
import pickle
import mne
import h5py
import numpy as np
import torch
import einops
import multiprocessing as mp
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

NAME = "EMO_SEED_7"  
SUBJECTS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
TEXT_LABELS = ['EMO/Happy', 'EMO/Surprise', 'EMO/Neutral', 'EMO/Sad', 'EMO/Disgust', 'EMO/Fear', 'EMO/Anger']
CH_NAMES = [
    "Fp1","Fpz","Fp2","AF3","AF4","F7","F5","F3","F1","Fz","F2","F4","F6","F8","FT7",
    "FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","T7","C5","C3","C1","Cz","C2","C4",
    "C6","T8","TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8","P7","P5","P3","P1",
    "Pz","P2","P4","P6","P8","PO7","PO5","PO3","POz","PO4","PO6","PO8","CB1","O1","Oz","O2","CB2",
]
SEGMENT_LEN = 4
video_order = (
    ["happy","neutral","disgust","sad","anger","anger","sad","disgust","neutral","happy"] * 2 +  
    ["anger","sad","fear","neutral","surprise","surprise","neutral","fear","sad","anger"] * 2 + 
    ["happy","surprise","disgust","fear","anger","anger","fear","disgust","surprise","happy"] * 2 + 
    ["disgust","sad","fear","surprise","happy","happy","surprise","fear","sad","disgust"] * 2
)
label_dict = {
    "happy": 0,
    "surprise": 1,
    "neutral": 2,
    "sad": 3,
    "disgust": 4,
    "fear": 5,
    "anger": 6,
}
LABELS_PER_SESSION = [ label_dict[v] for v in video_order ]

def load_and_prepare(raw_path):
    raw = mne.io.read_raw_cnt(raw_path, preload=True, verbose=False)
    raw.drop_channels(['M1', 'M2', 'ECG', 'HEO', 'VEO'])
    raw.resample(Param.resample, npad="auto", verbose=False)
    raw.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
    raw.notch_filter(freqs=50, notch_widths=2, verbose=False)

    try:
        events, _ = mne.events_from_annotations(raw, verbose=False)
        t = events[:, 0].tolist()
    except Exception:
        t = []

    sfreq = raw.info["sfreq"]
    assert sfreq == 200
    if os.path.basename(raw_path) == "14_20221015_1.cnt":
        t = []
        start = datetime.datetime.strptime("14:25:34", "%H:%M:%S")
        with open(f"{RAW_DATA_FOLDER}/{NAME}/save_info/14_20221015_1_trigger_info.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                end = datetime.datetime.strptime(row[1].split(" ")[-1], "%H:%M:%S.%f")
                t.append(int(round((end.timestamp() - start.timestamp()) * sfreq)))
    elif os.path.basename(raw_path) == "9_20221111_3.cnt":
        t = []
        start = datetime.datetime.strptime("14:01:27", "%H:%M:%S")
        with open(f"{RAW_DATA_FOLDER}/{NAME}/save_info/9_20221111_3_trigger_info.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                end = datetime.datetime.strptime(row[1].split(" ")[-1], "%H:%M:%S.%f")
                t.append(int(round((end.timestamp() - start.timestamp()) * sfreq)))

    data_uV = raw.get_data(units="uV").astype(np.float32)  # C x T
    return data_uV, t

def proc_one(sub):
    subject_files = [
        list(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/EEG_raw/{sub}_*_1.cnt'))[0],
        list(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/EEG_raw/{sub}_*_2.cnt'))[0],
        list(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/EEG_raw/{sub}_*_3.cnt'))[0],
        list(glob.glob(f'{RAW_DATA_FOLDER}/{NAME}/EEG_raw/{sub}_*_4.cnt'))[0],
    ]
    trainX, trainY = [], []
    validX, validY = [], []
    testX, testY = [], []
    for sess_id, raw_path in enumerate(subject_files):
        data, t = load_and_prepare(raw_path)
        print(f"[{sub}] {len(t)} timestamps found in session {sess_id}")

        for i in range(20):
            start_samp = t[2 * i]
            end_samp = t[2 * i + 1]
            x = data[:, start_samp:end_samp]  # C x T
            x = einops.rearrange(torch.tensor(x).unfold(1, Param.resample * SEGMENT_LEN, Param.resample * SEGMENT_LEN), "C N T -> N C T").numpy()
            y = np.array(LABELS_PER_SESSION[sess_id*20 + i]).repeat(x.shape[0]).astype(np.uint8)

            if i < 10:
                trainX.append(x); trainY.append(y)
            elif i < 15:
                validX.append(x); validY.append(y)
            else:
                testX.append(x); testY.append(y)

    trainX, trainY = np.concatenate(trainX), np.concatenate(trainY)
    validX, validY = np.concatenate(validX), np.concatenate(validY)
    testX, testY = np.concatenate(testX), np.concatenate(testY)
    trainX = pipeline(trainX, CH_NAMES)
    validX = pipeline(validX, CH_NAMES)
    testX = pipeline(testX, CH_NAMES)
    return sub, trainX, trainY, validX, validY, testX, testY

if __name__ == "__main__":
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
