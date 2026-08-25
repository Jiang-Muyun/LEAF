"""Export direct-inference embeddings for a global UMAP visualization.

This is the current-codebase port of Code_v1/LEAF_big_inference.py.  It runs the
pooled test split of every configured downstream dataset through one LEAF
checkpoint and stores the trial embeddings together with ground-truth and
predicted labels.  Keeping the labels makes it possible to plot actual class
clusters rather than only coloring points by their source dataset.

Run from the LEAF directory, for example:

    python export_test_embeddings.py --gpu 0 --ckpt 100 --level 2
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from LEAF import LEAF
from b_instruct_tuning import build_prototype
from load_config import build_model_config, load_instructions, load_targets, load_yaml
from load_datasets import load_dataset
from utils import green, seed_all


ROOT = Path(__file__).resolve().parent
TARGETS = load_targets()
INSTRUCTIONS = load_instructions()


def instruction_family(dataset: str) -> str:
    if dataset.startswith('MI'):
        return 'Motor'
    if dataset.startswith('EMO'):
        return 'Emotion'
    if dataset.startswith('CS'):
        return 'Speech'
    if dataset.startswith('SSVEP'):
        return 'SSVEP'
    return dataset


def make_instruction_embedding(dataset, level, text2emb, rng):
    family = 'Default' if level == 0 else instruction_family(dataset)
    instruction = rng.choice(INSTRUCTIONS[family])
    embedding = text2emb[instruction].copy()

    if level >= 2:
        label_embedding = np.mean(
            [text2emb[label] for label in TARGETS[dataset]], axis=0
        )
        label_embedding /= np.linalg.norm(label_embedding, ord=2)
        embedding = (embedding + label_embedding) / 2.0
        embedding /= np.linalg.norm(embedding, ord=2)

    return instruction, torch.tensor(embedding)


def parse_datasets(value):
    if value is None:
        return list(TARGETS)
    requested = [item.strip() for item in value.split(',') if item.strip()]
    unknown = sorted(set(requested) - set(TARGETS))
    if unknown:
        raise ValueError(f'Unknown dataset(s): {unknown}')
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='CUDA device index, or -1 for CPU')
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--ckpt', type=int, default=100)
    parser.add_argument('--dim', default=None, help='override model dimensions from YAML')
    parser.add_argument('--emb', default=None, help='override text embedding model from YAML')
    parser.add_argument('--level', type=int, default=2, choices=(0, 1, 2))
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'LEAF_mpnet.yaml'))
    parser.add_argument('--datasets', default=None, help='optional comma-separated subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.gpu != '-1' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available; pass --gpu -1 for CPU')
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu.split(",")[0]}')

    seed_all(args.seed)
    datasets = parse_datasets(args.datasets)

    cfg = load_yaml(args.config)
    config = build_model_config(cfg, args.dim, emb=args.emb)
    fingerprint = (
        f'w{config.window_len}-c{config.dim_cnn}-t{config.dim_token}'
        f'-lay{config.num_layers}'
    )
    it_fingerprint = (
        f'q{config.num_q}-qlay{config.num_qformer_layers}'
        f'-{config.text_emb_model_name}-{args.seed}'
    )
    checkpoint = (
        ROOT / 'ckpt' / fingerprint / it_fingerprint / f'it{args.ckpt:03d}.ckpt'
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else checkpoint.parent / 'global_inference' / f'level{args.level}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, text2emb = build_prototype(TARGETS, config.text_emb_model_name)
    model = LEAF(config)
    print(f'>> Load checkpoint: {green(checkpoint)}')
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device).eval()

    for dataset in datasets:
        output_path = output_dir / f'{dataset}.npz'
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as saved:
                y_true = saved['y_true']
                y_pred = saved['y_pred']
            bacc = balanced_accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            print(f'>> {output_path.name} exists; skipping ({len(y_true)} trials)')
            continue

        # Make instruction selection stable per dataset, including after a resumed run.
        dataset_rng = random.Random(f'{args.seed}:{args.level}:{dataset}')
        instruction, instruction_embedding = make_instruction_embedding(
            dataset, args.level, text2emb, dataset_rng
        )
        instruction_embedding = instruction_embedding.to(device)
        prototypes = torch.tensor(
            np.asarray([text2emb[label] for label in TARGETS[dataset]]),
            device=device,
        )

        *_, test_x, test_y = load_dataset(dataset)
        # The 10-second FACED trials use substantially more activation memory than
        # the 4-second datasets.  Large batches can trigger a CUDA illegal access
        # in the convolution kernel on the current PyTorch/CUDA stack.
        effective_batch_size = min(args.bs, 64) if test_x.shape[-1] > 1000 else args.bs
        if effective_batch_size != args.bs:
            print(
                f'  {dataset}: reducing batch size from {args.bs} to '
                f'{effective_batch_size} for {test_x.shape[-1]}-sample trials'
            )
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(test_x),
                torch.as_tensor(test_y, dtype=torch.long),
            ),
            batch_size=effective_batch_size,
            shuffle=False,
            pin_memory=device.type == 'cuda',
        )

        embedding_batches = []
        prediction_batches = []
        for x_batch, _ in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            instruction_batch = instruction_embedding.unsqueeze(0).expand(
                x_batch.shape[0], -1
            )
            with torch.inference_mode():
                embeddings, _ = model(x_batch, instruction_batch)
                predictions = (embeddings @ prototypes.T).argmax(dim=1)
            embedding_batches.append(embeddings.float().cpu().numpy())
            prediction_batches.append(predictions.cpu().numpy())

        embeddings = np.concatenate(embedding_batches).astype(np.float32, copy=False)
        y_true = np.asarray(test_y, dtype=np.int64)
        y_pred = np.concatenate(prediction_batches).astype(np.int64, copy=False)
        bacc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        np.savez(
            output_path,
            embeddings=embeddings,
            y_true=y_true,
            y_pred=y_pred,
            label_names=np.asarray(TARGETS[dataset]),
            dataset=np.asarray(dataset),
            instruction=np.asarray(instruction),
            level=np.asarray(args.level, dtype=np.int64),
            checkpoint=np.asarray(str(checkpoint)),
        )
        print(
            f'  {dataset}: {embeddings.shape}, bAcc={bacc:.4f}, '
            f'kappa={kappa:.4f} -> {output_path}'
        )

        del test_x, test_y, embeddings, y_true, y_pred

    # Rebuild the manifest from every completed export in the directory.  This
    # keeps summary.csv complete when a resumed invocation targets only a subset.
    summary = []
    for dataset in TARGETS:
        output_path = output_dir / f'{dataset}.npz'
        if not output_path.exists():
            continue
        with np.load(output_path, allow_pickle=False) as saved:
            y_true = saved['y_true']
            y_pred = saved['y_pred']
        summary.append((
            dataset,
            len(y_true),
            balanced_accuracy_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred),
        ))

    summary_path = output_dir / 'summary.csv'
    with summary_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'n_samples', 'bAcc', 'kappa'])
        for dataset, n_samples, bacc, kappa in summary:
            writer.writerow([dataset, n_samples, f'{bacc:.6f}', f'{kappa:.6f}'])
        if summary:
            writer.writerow([
                'MEAN',
                sum(item[1] for item in summary),
                f'{np.mean([item[2] for item in summary]):.6f}',
                f'{np.mean([item[3] for item in summary]):.6f}',
            ])

    print(f'>> Embeddings: {green(output_dir)}')
    print(f'>> Summary: {green(summary_path)}')


if __name__ == '__main__':
    main()
