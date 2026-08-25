# LEAF

**Language-EEG Aligned Foundation Model for Brain-Computer Interfaces**

[Project page](https://leaf-bci.github.io/) · [Paper](https://arxiv.org/abs/2509.24302) · [Documentation](https://leaf-bci.github.io/docs/) · [Checkpoints](https://github.com/Jiang-Muyun/LEAF/releases)

LEAF aligns EEG representations with natural-language instructions and label semantics. It combines self-supervised EEG pretraining with instruction tuning, enabling direct inference across multiple BCI tasks using text embeddings as class prototypes.

<p align="center">
  <img src="assets/leaf-overview.png" alt="Overview of LEAF and language-aligned EEG modeling" width="100%">
</p>
<p align="center"><em>LEAF connects diverse EEG decoding tasks through a shared language-aligned representation space.</em></p>

<p align="center">
  <img src="assets/leaf-architecture.png" alt="LEAF model architecture" width="100%">
</p>
<p align="center"><em>The framework combines self-supervised EEG pretraining with instruction-conditioned semantic alignment.</em></p>

## Quick start

Configure the raw, pretraining, and downstream dataset paths in `configs/env.yaml`. Dataset preprocessing scripts are provided in `leaf_datasets/`.

Download the released checkpoints into `checkpoints/`, then run direct inference:

```bash
wget -c -P checkpoints https://github.com/Jiang-Muyun/LEAF/releases/download/v1.0/leaf-v1.0-pretrain.ckpt
wget -c -P checkpoints https://github.com/Jiang-Muyun/LEAF/releases/download/v1.0/leaf-v1.0-instruct-mpnet-base.ckpt
```

Alternatively, download both files with `bash download_checkpoints.sh`.

```bash
python c_inference.py \
  --config configs/LEAF_mpnet.yaml \
  --ckpt leaf-v1.0-instruct-mpnet-base.ckpt \
  --gpu 0 --bs 512 --level 0,1,2
```

To run the training stages:

```bash
# Self-supervised EEG pretraining
python a_pretrain.py --config configs/LEAF_mpnet.yaml --bs 256 --gpu 0,1

# Instruction tuning from the released pretrained Tower
python b_instruct_tuning.py \
  --config configs/LEAF_mpnet.yaml \
  --pretrain_ckpt leaf-v1.0-pretrain.ckpt \
  --bs 256 --gpu 0,1 --seed 0
```

See the [documentation](https://leaf-bci.github.io/docs/) for dataset preparation, training settings, and evaluation details.

## Apply LEAF to a new EEG dataset

Assume a new dataset contains 4-second EEG trials recorded from 32 channels at 250 Hz. The raw array has shape `(N, 32, 1000)`, where `N` is the number of trials.

### 1. Preprocess and align the channels

```python
import mne
import numpy as np

from leaf_datasets.shared import pipeline

channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
    'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4',
    'P8', 'POz', 'O1', 'Oz', 'O2', 'AF3', 'AF4', 'FCz',
]

# Replace this random array with your epoched EEG data.
raw_eeg = np.random.randn(2, 32, 1000).astype(np.float32)

info = mne.create_info(channels, sfreq=250, ch_types='eeg')
epochs = mne.EpochsArray(raw_eeg, info, verbose=False)
epochs.filter(l_freq=0.1, h_freq=70, method='iir', verbose=False)
epochs.resample(200, verbose=False)

eeg = pipeline(epochs.get_data().astype(np.float32), channels)
print(eeg.shape)  # (2, 65, 800)
```

`pipeline` clips extreme values, applies robust scaling with the median and interquartile range, maps the supplied channel names to the LEAF 65-channel montage, and interpolates missing template channels. Channel names must match an electrode or alias in `configs/LEAF-ch65/LEAF-ch65.json`; add an alias there if a dataset uses a different name.

### 2. Input length and padding

LEAF uses a sampling rate of 200 Hz and accepts up to 2,000 samples, or 10 seconds, with the default configuration. During pretraining and instruction tuning, shorter trials are zero-padded to 10 seconds and longer trials are cropped. The padding or crop position is randomized for training samples; validation samples are padded on the right or cropped from the beginning.

Padding to 10 seconds is not required for inference. The 4-second example above contains 800 samples and can be passed to LEAF directly. The model divides the signal into non-overlapping 100-sample windows, producing eight EEG tokens in this case. For recordings longer than 10 seconds, crop the signal or divide it into segments before inference.

### 3. Extract LEAF embeddings

```python
import torch

from LEAF import LEAF
from init_text_embeddings import load_embeddings
from load_config import build_model_config, load_yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = build_model_config(load_yaml('configs/LEAF_mpnet.yaml'))

model = LEAF(config)
state = torch.load(
    'checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt',
    map_location='cpu',
    weights_only=True,
)
model.load_state_dict(state, strict=True)
model.to(device).eval()

text_embeddings = load_embeddings(config.text_emb_model_name)
instruction = torch.tensor(
    text_embeddings['This is an emotion recognition task'],
    dtype=torch.float32,
    device=device,
)

x = torch.tensor(eeg, dtype=torch.float32, device=device)
instruction = instruction.unsqueeze(0).expand(x.shape[0], -1)

with torch.inference_mode():
    eeg_embeddings, _ = model(x, instruction)

print(eeg_embeddings.shape)  # (2, 768) with MPNet
```

### 4. Compare EEG with label text

LEAF and the text prototypes are L2-normalized, so their dot product is cosine similarity. The following example compares every EEG trial with `Happy` and `Sad`:

```python
labels = ['Happy', 'Sad']
prototypes = torch.tensor(
    np.stack([text_embeddings[label] for label in labels]),
    dtype=torch.float32,
    device=device,
)

similarities = eeg_embeddings @ prototypes.T
distances = 1 - similarities
predictions = [labels[index] for index in similarities.argmax(dim=1).tolist()]

print(similarities.shape)  # (2, 2)
print(distances.shape)     # (2, 2)
print(predictions)
```

Replace the instruction and labels with the task of interest. If new text is not already cached, add it to `configs/tasks.yaml` and run `python init_text_embeddings.py mpnet-base` before inference.

## Citation

```bibtex
@article{jiang2026leaf,
  title={LEAF: Language-EEG Aligned Foundation Model for Brain-Computer Interfaces},
  author={Jiang, Muyun and Zhang, Shuailei and Yang, Zhenjie and Wu, Mengjun and Jiang, Weibang and Liu, Chenyu and Guo, Zhiwei and Zhang, Wei and Liu, Rui and Zhang, Shangen and Li, Yong and Ding, Yi and Guan, Cuntai},
  journal={arXiv preprint arXiv:2509.24302},
  year={2026}
}
```
