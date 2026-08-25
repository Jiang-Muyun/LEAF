"""
Map dataset-specific EEG channels to the 65-channel LEAF montage.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import mne
import json
import warnings
import numpy as np
from pathlib import Path

class ElectrodeUnifier:
    """
        Class for unifying EEG channel configurations to a predefined template
    """
    INTERP_WARN_FRAC = 0.5  # warn if more than this fraction of the template needs interpolation

    def __init__(self, template):
        self.template = template

        if template == 'LEAF-ch65':
            wd = Path(__file__).resolve().parent.parent / 'configs' / template  # configs/<template>/
            with open(wd / f'{template}.json', 'r') as f:
                config = json.load(f)
            montage = mne.channels.read_custom_montage(wd / f'{template}.bvef')
            info = mne.create_info(config['Electrodes'], sfreq=200, ch_types='eeg', verbose=False)
            info.set_montage(montage, verbose=False)

        else:
            raise ValueError(f'Unknown channel template: {template}')

        assert set(config['Policy']) == set(config['Electrodes']), \
            'Policy keys must match Electrodes'
        self.info = info
        self.policy = config['Policy']
        self.channels = config['Electrodes']
        self.reverse_lookup = {}
        for k, v in self.policy.items():
            for ch in v:
                self.reverse_lookup[ch] = k

    def map_trial(self, src_X, src_ch_names):

        newX = np.zeros((len(self.channels), src_X.shape[1]), dtype=np.float32)
        occurrence = np.zeros(len(self.channels), dtype=np.int32)
        for x, ch in zip(src_X, src_ch_names):
            idx = self.channels.index(self.reverse_lookup[ch])
            newX[idx] += x
            occurrence[idx] += 1

        zero_channels = []
        for idx, occ in enumerate(occurrence):
            if occ == 0:
                zero_channels.append(self.channels[idx])
            else:
                newX[idx] /= occ

        return zero_channels, newX

    def map_to_template(self, X, src_ch_names, mode='fast'):

        assert isinstance(X, np.ndarray), f'Expected np.ndarray, got {type(X)}'
        assert len(X.shape) == 3, f'Expected 3D array, got {len(X.shape)}D'
        assert len(src_ch_names) == X.shape[1], f'{len(src_ch_names)} != {X.shape[1]}'

        X = X.astype(np.float32)
        _newX = []
        for x in X:
            missing_channels, newX = self.map_trial(x, src_ch_names)
            _newX.append(newX)
        newX = np.array(_newX)

        if len(missing_channels) > 0:
            frac = len(missing_channels) / len(self.channels)
            msg = (f'Interpolating {len(missing_channels)}/{len(self.channels)} '
                   f'missing channels: {missing_channels}')
            if frac > self.INTERP_WARN_FRAC:
                msg += (f' — {frac:.0%} of the template is missing, '
                        f'spline interpolation may be unreliable')
            warnings.warn(msg)
            epoch = mne.EpochsArray(newX, self.info, events=None, tmin=0, verbose=False)
            epoch.info['bads'] = missing_channels
            epoch.interpolate_bads(reset_bads=True, mode=mode)
            newX = epoch.get_data().astype(np.float32)

        return newX


if __name__ == '__main__':
    unifier = ElectrodeUnifier('LEAF-ch65')

    seq_len = 1000
    channels = [
        'Fz', 'FC3', 'FC1', 'FCz','FC2', 'FC4', 
        'C5','C3','C1','Cz','C2','C4','C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1','Pz','P2','POz'
    ]
    n_channels = len(channels)
    sample = np.random.randn(20, n_channels, seq_len).astype(np.float32)

    mapped = unifier.map_to_template(sample, channels)
    print(mapped.shape)
