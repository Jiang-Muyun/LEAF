import torch
import einops
import glob
import mne
import numpy as np
import os
import pickle
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
from leaf_datasets.shared import RAW_DATA_FOLDER, DATA_FOLDER, pipeline, Param

CH_NAMES = [
    'Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1',
    'Cz','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3','POz',
    'PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

def proc_h5(fn, ignore_keys=[]):
    buf = []
    info = mne.create_info(ch_names=CH_NAMES, sfreq=200, ch_types='eeg')
    with h5py.File(fn, 'r') as fin:
        for key in fin.keys():
            if key in ignore_keys:
                print(f'Skipping {key}...')
                continue
            X = fin[key][()] / 1000
            raw = mne.io.RawArray(X, info)
            raw.filter(l_freq=Param.lp, h_freq=Param.hp, verbose=False)
            X = raw.get_data().astype(np.float32)
            X = torch.from_numpy(X).unfold(-1, Param.resample*10, Param.resample*10).permute(1,0,2).numpy()
            X = pipeline(X, CH_NAMES)
            buf.append((key, X))
    return buf

def write_h5(dst, data_rows):
    with h5py.File(dst, 'w') as fo:
        for sub, X in data_rows:
            fo.create_dataset(f'{sub}/X', data=X)
            fo.create_dataset(f'{sub}/Y', data=np.zeros(X.shape[0], dtype=np.uint8))
            print(sub, X.shape)

if __name__ == '__main__':
    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_3.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-3.h5',
                                                           ignore_keys=['dujingcheng_1027', 'zhujiayi_0709']))

    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_4.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-4.h5',
                                                           ignore_keys=['huan_20151012', 'liyu_20160406']))
    
    # write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_5.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-5.h5'))
    # write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_7.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-7.h5'))

    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_French.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-french.h5'))

    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_Neg.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-neg.h5'))
    
    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_Sleep2.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-sleep2.h5'))

    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_German.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-german.h5', 
                                                    ignore_keys=['christoph_20161025', 'jannik_20161118']))

    write_h5(f'{DATA_FOLDER}/EMO_SEED_Pretrain_Sleep_Emo3.h5', proc_h5(f'{RAW_DATA_FOLDER}/EMO_SEED_Pretrain/seed-sleep-emo3.h5',
                                                    ignore_keys=['chechaohui-emotion-20161203', 'yinjun-emotion-20160930']))
