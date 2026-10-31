"""Evaluate a frozen instruction-tuned LEAF checkpoint on standard test splits.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing
and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from LEAF import LEAF
from load_config import DEFAULT_CONFIG, build_model_config, load_instructions, load_targets, load_yaml
from load_datasets import load_test_split

ROOT = Path(__file__).resolve().parent
LEVEL_NAMES = {0: 'Default', 1: 'Task', 2: 'Task+Target'}


def resolve_file(path, fallback):
    for candidate in (path, ROOT / path, fallback / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'File not found: {path} (also searched {fallback})')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--seed', type=int, default=42, help='Seed for reproducible instruction selection')
    p.add_argument('--level', default='0,1,2', help='Comma-separated instruction levels: 0=Default, 1=Task, 2=Task+Target')
    p.add_argument('--ds', help='Comma-separated dataset names; default: all datasets in configs/tasks.yaml')
    p.add_argument('--output', type=Path, help='CSV destination; must not already exist')
    p.add_argument('--dim', help='Model dimension override; must match the checkpoint')
    p.add_argument('--emb', help='Text embedding model override; must match the checkpoint')
    args = p.parse_args()
    if args.bs <= 0:
        p.error('--bs must be positive')
    if args.gpu != '-1' and not args.gpu.isdigit():
        p.error('--gpu must be one nonnegative GPU index, or -1 for CPU')
    try:
        args.levels = list(dict.fromkeys(int(v) for v in args.level.split(',')))
    except ValueError:
        p.error('--level must contain comma-separated integers from 0,1,2')
    if not args.levels or any(v not in LEVEL_NAMES for v in args.levels):
        p.error('--level must contain only 0,1,2')
    return args


def load_model(config, checkpoint, device):
    model = LEAF(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_text_embeddings(config):
    name = config.text_emb_model_name
    texts = (ROOT / 'text_embeddings' / f'{name}.txt').read_text().splitlines()
    vectors = np.load(ROOT / 'text_embeddings' / f'{name}.npy', allow_pickle=False).astype(np.float32)
    if vectors.shape != (len(texts), config.text_emb_model_dim) or len(set(texts)) != len(texts):
        raise ValueError(f'Invalid text embedding cache: {name}')
    if not np.isfinite(vectors).all() or not np.allclose((vectors ** 2).sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f'Text embeddings must be finite and L2-normalized: {name}')
    return dict(zip(texts, vectors))


def select_instructions(targets, instructions, seed):
    """Select in legacy level-first order, independently of requested subsets."""
    rng = random.Random(seed)
    selected = {}
    for level in LEVEL_NAMES:
        for ds in targets:
            family = next((name for prefix, name in (
                ('MI', 'Motor'), ('EMO', 'Emotion'), ('CS', 'Speech'), ('SSVEP', 'SSVEP'),
            ) if ds.startswith(prefix)), ds)
            selected[ds, level] = rng.choice(instructions['Default' if level == 0 else family])
    return selected


def build_instruction_embedding(text, labels, level, text_embeddings, device):
    embedding = text_embeddings[text].copy()
    if level == 2:
        mean = np.mean([text_embeddings[label] for label in labels], axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            raise ValueError('Candidate-label mean has zero norm')
        embedding = (embedding + mean / norm) / 2.0
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError('Blended instruction has zero norm')
        embedding = embedding / norm
    return torch.as_tensor(embedding, device=device)


@torch.inference_mode()
def evaluate_dataset(model, x, y, instruction, prototypes, device, batch_size):
    """Score each batch on CPU; retain predictions rather than all embeddings."""
    if x.ndim != 3 or y.shape != (len(x),) or len(y) == 0:
        raise ValueError('Expected nonempty EEG (N, C, T) and labels (N,)')
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= len(prototypes):
        raise ValueError('Labels must index the ordered dataset prototypes')
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    predictions = []
    for (batch,) in loader:
        embedding, _ = model(batch.to(device), instruction.unsqueeze(0).expand(len(batch), -1))
        logits = embedding.cpu() @ prototypes.T
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite cosine logits')
        predictions.append(logits.argmax(dim=1).numpy())
    prediction = np.concatenate(predictions)
    return dict(bAcc=float(balanced_accuracy_score(y, prediction)),
                kappa=float(cohen_kappa_score(y, prediction)), n_samples=len(y))


def write_report(path, rows, levels):
    fields = ['dataset', 'n_samples'] + [f'level{level}_{metric}' for level in levels for metric in ('bAcc', 'kappa')]
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f'{value:.4f}' if key not in ('dataset', 'n_samples') else value
                             for key, value in row.items()})
        if rows:
            mean = {key: f'{np.mean([row[key] for row in rows]):.4f}' for key in fields[2:]}
            writer.writerow(dict(dataset='MEAN', n_samples='', **mean))
    temporary.replace(path)


def write_metadata(path, metadata):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(path)


def main():
    args = parse_args()
    targets = load_targets()
    datasets = list(targets) if args.ds is None else list(dict.fromkeys(v.strip() for v in args.ds.split(',')))
    unknown = [ds for ds in datasets if ds not in targets]
    if unknown:
        raise ValueError(f'Unknown datasets: {unknown}')
    checkpoint = resolve_file(args.ckpt, ROOT / 'checkpoints')
    config_path = resolve_file(args.config, ROOT / 'configs')
    config = build_model_config(load_yaml(config_path), args.dim, emb=args.emb)
    output = (args.output or checkpoint.parent / 'direct_inference' / f'{checkpoint.stem}_per_dataset.csv').resolve()
    metadata_path = output.with_suffix('.json')
    if output.suffix.lower() != '.csv':
        raise ValueError('--output must have a .csv extension')
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f'Results already exist: {output} or {metadata_path}; choose a new --output')
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    text_embeddings = load_text_embeddings(config)
    selected = select_instructions(targets, load_instructions(), args.seed)
    for ds in datasets:
        for text in [*targets[ds], *(selected[ds, level] for level in args.levels)]:
            if text not in text_embeddings:
                raise ValueError(f'Missing cached text embedding: {text!r}')
    model = load_model(config, checkpoint, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint=str(checkpoint), config=str(config_path), model_config=config.to_dict(),
                    seed=args.seed, batch_size=args.bs, device=str(device), levels=args.levels,
                    datasets=datasets, completed_datasets=[], status='running',
                    labels={ds: targets[ds] for ds in datasets},
                    instructions={ds: {str(level): selected[ds, level] for level in args.levels} for ds in datasets})
    write_metadata(metadata_path, metadata)
    rows = []
    for ds in datasets:
        x, y = load_test_split(ds)
        prototypes = torch.from_numpy(np.stack([text_embeddings[label] for label in targets[ds]]))
        row = dict(dataset=ds, n_samples=len(y))
        for level in args.levels:
            instruction = build_instruction_embedding(selected[ds, level], targets[ds], level, text_embeddings, device)
            metrics = evaluate_dataset(model, x, y, instruction, prototypes, device, args.bs)
            for metric in ('bAcc', 'kappa'):
                row[f'level{level}_{metric}'] = metrics[metric]
            print(f'{ds} level={level}: bAcc={metrics["bAcc"]:.4f}, kappa={metrics["kappa"]:.4f}, n={len(y)}', flush=True)
        rows.append(row)
        write_report(output, rows, args.levels)
        metadata['completed_datasets'].append(ds)
        write_metadata(metadata_path, metadata)
        del x, y
    metadata['status'] = 'complete'
    write_metadata(metadata_path, metadata)
    print(f'Saved {len(rows)} datasets to {output}')


if __name__ == '__main__':
    main()
