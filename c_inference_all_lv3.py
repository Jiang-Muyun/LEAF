"""Direct inference for all 17 LEAF datasets at instruction levels 0, 1, 2.

Uses c_inference's instruction embeddings and prototype scoring, with the same
seeded instruction choices. Only the test portions of the standard splits are
loaded. Writes one row per dataset with all three levels side by side, updating
the CSV after each dataset. Balanced accuracy and kappa use four decimal places.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""
import argparse
import csv
import random
from pathlib import Path

import h5py
import numpy as np
import torch

import c_inference as inference
from leaf_datasets.shared import DATA_FOLDER
from load_config import DEFAULT_CONFIG

# Subject indices follow load_datasets.load_dataset, including HDF5 key order.
TEST_SUBJECTS = {
    'MI_OpenBMI': (42, 54), 'MI_BCIC_IV2a': (7, 9),
    'MI_ShanghaiU': (20, 25), 'MI_BCIC_Upperlimb': (11, 15),
    'MI_HighGamma': (10, 14), 'MI_Cho2017': (40, 49),
    'MI_Shin2017A': (22, 28), 'MI_PhysioNet': (80, 109),
    'EMO_FACED': (100, 123), 'SSVEP_OpenBMI': (42, 54),
    'Workload': (32, 36),
}
LEVEL_NAMES = ['Default', 'Task', 'Task+Target']
FIELDS = ['dataset', 'n_samples'] + [
    f'level{level}_{metric}'
    for level in range(3) for metric in ('bAcc', 'kappa')
]


def load_test_split(ds):
    """Read exactly the test samples selected by load_dataset, without train/val."""
    with h5py.File(DATA_FOLDER / f'{ds}.h5', 'r', locking=False) as f:
        if ds.startswith('EMO_SEED_') or ds == 'ADHD_AliMotie':
            return f['testX'][:], f['testY'][:]
        subjects = list(f.keys())
        if ds == 'CS_BCIC_Speech':
            xs = [f[s]['X'][300:] for s in subjects]
            ys = [f[s]['Y'][300:] for s in subjects]
        else:
            start, stop = TEST_SUBJECTS[ds]
            selected = [subjects[i] for i in range(start, stop)]
            xs = [f[s]['X'][:] for s in selected]
            ys = [f[s]['Y'][:] for s in selected]
    x, y = np.concatenate(xs), np.concatenate(ys)
    if ds == 'MI_HighGamma':
        keep = y != 2
        x, y = x[keep], y[keep]
    return x, y


def select_instructions(seed):
    rng = random.Random(seed)
    selected = {}
    # Match c_inference's level-first random choice order even though data is
    # loaded dataset-first here to avoid reading every test split three times.
    for level in range(3):
        for ds in inference.Targets:
            family = ds
            for prefix, name in [('MI', 'Motor'), ('EMO', 'Emotion'),
                                 ('CS', 'Speech'), ('SSVEP', 'SSVEP')]:
                if ds.startswith(prefix):
                    family = name
                    break
            selected[ds, level] = rng.choice(
                inference.Instructions['Default' if level == 0 else family])
    return selected


def write_report(path, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['dataset'], {})[row['level']] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for ds, levels in grouped.items():
            result = dict(dataset=ds, n_samples=levels[0]['n_samples'])
            for level in range(3):
                for metric in ('bAcc', 'kappa'):
                    result[f'level{level}_{metric}'] = f"{levels[level][metric]:.4f}"
            writer.writerow(result)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, default=inference.CHECKPOINT_DIR / 'leaf-v1.0-instruct-mpnet-base.ckpt')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--gpu', default='0', help='GPU index, or -1 for CPU')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.bs <= 0:
        p.error('--bs must be positive')
    ckpt = args.ckpt.resolve()
    if not ckpt.is_file():
        ckpt = inference.CHECKPOINT_DIR / args.ckpt
    if not ckpt.is_file():
        p.error(f'Checkpoint not found: {args.ckpt}')
    missing = [ds for ds in inference.Targets if not (DATA_FOLDER / f'{ds}.h5').is_file()]
    if missing:
        p.error(f'Missing datasets in {DATA_FOLDER}: {missing}')
    output = args.output or ckpt.parent / 'direct_inference' / f'{ckpt.stem}_per_dataset.csv'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    inference.device = 'cpu' if args.gpu == '-1' else f'cuda:{args.gpu}'
    config = inference.build_model_config(inference.load_yaml(args.config))
    _, _, inference.text2emb = inference.build_prototype(inference.Targets, config.text_emb_model_name)
    inference.model = inference.LEAF(config)
    inference.model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
    inference.model.to(inference.device).eval()
    selected = select_instructions(args.seed)
    rows = []
    print(f'Checkpoint: {ckpt}\nOutput: {output}', flush=True)
    for ds in inference.Targets:
        x, y = load_test_split(ds)
        if len(y) == 0 or y.min() < 0 or y.max() >= len(inference.Targets[ds]):
            raise ValueError(f'Invalid test labels for {ds}')
        prototypes = torch.tensor(np.array([inference.text2emb[c] for c in inference.Targets[ds]]))
        for level in range(3):
            instruction = selected[ds, level]
            embedding = inference.make_instruct_emb(ds, instruction, level)
            bacc, kappa, n = inference.eval_subset(x, y, embedding, prototypes, args.bs)
            rows.append(dict(dataset=ds, level=level, instruction_type=LEVEL_NAMES[level],
                bAcc=bacc, kappa=kappa, n_samples=n, instruction=instruction,
                seed=args.seed, checkpoint=str(ckpt)))
            print(f'[{len(rows)}/51] {ds} level={level}: bAcc={bacc:.4f}, kappa={kappa:.4f}, n={n}', flush=True)
        write_report(output, rows)
        del x, y
    print(f'Saved {len(inference.Targets)} datasets (3 levels each) to {output}', flush=True)


if __name__ == '__main__':
    main()
