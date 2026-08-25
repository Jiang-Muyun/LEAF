import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

from utils import green
from load_config import load_yaml, build_model_config, load_targets, load_instructions
from load_datasets import load_dataset
from LEAF import LEAF
from b_instruct_tuning import build_prototype

Targets = load_targets()
Instructions = load_instructions()
CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints'

def make_instruct_emb(ds, instruct_text, level):
    """Build instruction embedding for a dataset at the given detail level.
    level 0: Default instruction only (no task, no target blending)
    level 1: Task-specific instruction only (no target blending)
    level 2: Task-specific instruction + target label blending
    """
    emb = text2emb[instruct_text].copy()

    if level >= 2:
        label_emb = np.mean([text2emb[label] for label in Targets[ds]], axis=0)
        label_emb = label_emb / np.linalg.norm(label_emb, ord=2)
        emb = (emb + label_emb) / 2.0
        emb = emb / np.linalg.norm(emb, ord=2)

    return torch.tensor(emb).to(device)


def eval_subset(X, Y, instruct_emb_t, prototypes_cpu, batch_size):
    """Evaluate a single subset (subject or full dataset) in two steps:
      1. forward all test data through the model on the GPU, copy embeddings to CPU
      2. score against the class prototypes (matmul + argmax + metrics) on the CPU
    Returns (bAcc, kappa, n_samples).
    """
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(Y, dtype=torch.long)),
        batch_size=batch_size, shuffle=False,
    )

    # ---- Step 1: GPU forward -> collect all embeddings on CPU ----
    embs, labels = [], []
    for x_mb, y_mb in loader:
        x_mb = x_mb.to(device)
        with torch.no_grad():
            emb, _ = model(x_mb, instruct_emb_t.unsqueeze(0).expand(x_mb.shape[0], -1))
        embs.append(emb.cpu())
        labels.append(y_mb)
    embs = torch.cat(embs)                  # (N, text_dim), on CPU
    labels = torch.cat(labels).numpy()

    # ---- Step 2: CPU scoring against the class prototypes ----
    preds = (embs @ prototypes_cpu.T).argmax(dim=1).numpy()
    bAcc = balanced_accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    return bAcc, kappa, len(labels)


def eval_dataset(ds, instruct_text, level, batch_size):
    """Evaluate a dataset on its pooled test split. Returns (overall_bAcc, overall_kappa, n_samples)."""
    instruct_emb_t = make_instruct_emb(ds, instruct_text, level)
    prototypes_cpu = torch.tensor(np.array([text2emb[c] for c in Targets[ds]]))  # CPU: used in step 2

    *_, testX, testY = load_dataset(ds)
    overall_bAcc, overall_kappa, n_samples = eval_subset(
        testX, testY, instruct_emb_t, prototypes_cpu, batch_size
    )

    print(f"  {overall_bAcc:.4f}, {overall_kappa:.4f}  {ds}  ({n_samples} samples)")
    return overall_bAcc, overall_kappa, n_samples


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1,2,3')
    p.add_argument('--bs', type=int, default=256)
    p.add_argument(
        '--ckpt', type=Path, required=True,
        help=(
            'relative path to an instruction-tuning checkpoint; a bare filename '
            'is also searched for under checkpoints/'
        ),
    )
    p.add_argument('--dim', type=str, default=None, help='override model dims (default: from --config YAML)')
    p.add_argument('--emb', type=str, default=None, help='override embedding model (default: from --config YAML)')
    p.add_argument('--level', type=str, default='0,1,2',
                   help='Instruction detail levels (comma-separated): 0=Default only, 1=Task only, 2=Task+Target')
    p.add_argument('--seed', type=int, default=42, help='instruction-tuning seed (default: 42)')
    p.add_argument('--config', type=str, default='configs/LEAF_mpnet.yaml')
    args = p.parse_args()

    if args.bs <= 0:
        p.error('--bs must be positive')
    if args.ckpt.is_absolute():
        p.error('--ckpt must be a relative path')

    random.seed(args.seed)
    device = 'cpu' if args.gpu == '-1' else f'cuda:{args.gpu.split(",")[0]}'

    cfg = load_yaml(args.config)
    config = build_model_config(cfg, args.dim, emb=args.emb)

    ckpt = args.ckpt.resolve()
    fallback_ckpt = (
        CHECKPOINT_DIR.parent / args.ckpt
        if args.ckpt.parts[:1] == (CHECKPOINT_DIR.name,)
        else CHECKPOINT_DIR / args.ckpt
    ).resolve()
    if not ckpt.is_file():
        ckpt = fallback_ckpt

    if not ckpt.is_file():
        p.error(
            f'checkpoint not found: {args.ckpt} '
            f'(also checked {fallback_ckpt})'
        )
    report_dir = ckpt.parent / 'direct_inference'

    prototypes, key2idx, text2emb = build_prototype(Targets, config.text_emb_model_name)

    model = LEAF(config)
    print(f">> Load checkpoint: {green(ckpt)}")
    try:
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # --- Evaluate all datasets for each level ---
    levels = [int(l) for l in args.level.split(',')]
    report_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        csv_path = report_dir / f'{ckpt.stem}_level{level}.csv'
        if csv_path.exists():
            print(f">> {csv_path} already exists, skipping level {level}")
            continue

        print(f">> Instruction level: {level} ({'Default' if level == 0 else 'Task' if level == 1 else 'Task+Target'})")

        results = {}       # ds -> (overall_bAcc, overall_kappa, n_samples)
        for ds in Targets.keys():
            if level == 0:
                instruct_text = random.choice(Instructions['Default'])
            elif ds.startswith('MI'):
                instruct_text = random.choice(Instructions['Motor'])
            elif ds.startswith('EMO'):
                instruct_text = random.choice(Instructions['Emotion'])
            elif ds.startswith('CS'):
                instruct_text = random.choice(Instructions['Speech'])
            elif ds.startswith('SSVEP'):
                instruct_text = random.choice(Instructions['SSVEP'])
            else:
                instruct_text = random.choice(Instructions[ds])

            overall_bAcc, overall_kappa, n_samples = eval_dataset(
                ds, instruct_text, level, args.bs
            )
            results[ds] = (overall_bAcc, overall_kappa, n_samples)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'bAcc', 'kappa', 'n_samples'])

            for ds, (overall_bAcc, overall_kappa, n_samples) in results.items():
                writer.writerow([ds, f'{overall_bAcc:.4f}', f'{overall_kappa:.4f}', n_samples])

            mean_bAcc = np.mean([v[0] for v in results.values()])
            mean_kappa = np.mean([v[1] for v in results.values()])
            writer.writerow(['MEAN', f'{mean_bAcc:.4f}', f'{mean_kappa:.4f}', ''])

        print(f"\n>> Level {level} — Mean bAcc: {mean_bAcc:.4f}, Mean kappa: {mean_kappa:.4f}")
        print(f">> CSV report saved: {green(csv_path)}")
