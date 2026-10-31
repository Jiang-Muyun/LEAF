"""Fine-tune the EEG tower with a dataset-specific FlattenClassifier MLP.

The Q-Former is not instantiated. Every tower tensor is restored strictly from
an instruction-tuned LEAF checkpoint; only the dataset-specific MLP is newly
initialized. The best checkpoint is selected by validation BAcc (loss breaks
ties), then the test set is evaluated once.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing
and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from LEAF import Tower, FlattenClassifier
from d_downstream_qformer_ce import ROOT, resolve_file, make_loader, evaluate
from load_config import build_model_config, build_train_config, load_targets, load_yaml
from load_datasets import load_dataset
from utils import cosine_annealing_lr, make_lr_lambda, seed_all


def load_tower(config, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not any(key.startswith('qformer.') for key in state):
        raise ValueError('MLP fine-tuning requires a full instruction-tuned LEAF checkpoint')
    unexpected = [key for key in state if not key.startswith(('tower.', 'qformer.'))]
    if unexpected:
        raise ValueError(f'Unexpected checkpoint keys: {unexpected}')
    tower = Tower(config)
    tower.load_state_dict({key.removeprefix('tower.'): value for key, value in state.items()
                           if key.startswith('tower.')}, strict=True)
    return tower


class MLPClassifier(nn.Module):
    def __init__(self, tower, n_classes, input_samples):
        super().__init__()
        config = tower.config
        if not config.window_len <= input_samples <= config.max_seq_len or input_samples % config.window_len:
            raise ValueError('Trial length must be within model limits and divisible by window_len')
        self.tower = tower
        # Original flattened concatenated branch tokens and MLP head.
        n_tokens = 2 * (input_samples // config.window_len)
        self.classifier = FlattenClassifier(config.dim_token, n_tokens, n_classes)

    def forward(self, x):
        return self.classifier(self.tower(x))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--ds', choices=list(load_targets()), default='MI_BCIC_IV2a')
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dim', help='Architecture override; must match checkpoint')
    p.add_argument('--emb', help='Model config override; no text embeddings are used')
    p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], default='bf16-mixed')
    p.add_argument('--outdir', type=Path, help='New output directory; never overwrite results')
    args = p.parse_args()
    if args.bs <= 0 or args.epochs <= 0:
        p.error('--bs and --epochs must be positive')
    if args.gpu != '-1' and not args.gpu.isdigit():
        p.error('--gpu must be one nonnegative GPU index, or -1 for CPU')
    return args


def main():
    args = parse_args()
    checkpoint = resolve_file(args.ckpt, ROOT / 'checkpoints')
    cfg = load_yaml(resolve_file(args.config, ROOT))
    config = build_model_config(cfg, args.dim, emb=args.emb)
    train = build_train_config(cfg, 'downstream')
    epochs = args.epochs if args.epochs is not None else train.max_epochs
    if epochs <= 0 or not 0 <= train.warmup_epochs <= epochs:
        raise ValueError('Require 0 <= warmup_epochs <= epochs and epochs > 0')
    precision = args.precision or train.precision
    if precision not in ('32-true', 'bf16-mixed'):
        raise ValueError(f'Unsupported precision: {precision}')
    device = torch.device('cpu' if args.gpu == '-1' else f'cuda:{args.gpu}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if precision == 'bf16-mixed' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support bf16; use --precision 32-true')
    use_bf16 = device.type == 'cuda' and precision == 'bf16-mixed'
    outdir = (args.outdir or checkpoint.parent / 'mlp_finetuning' / checkpoint.stem
              / args.ds / f'seed{args.seed}').resolve()
    if outdir.exists():
        raise FileExistsError(f'Output directory already exists: {outdir}; choose a new --outdir')

    seed_all(args.seed)
    # Use the same convolution settings as Q-Former fine-tuning.
    torch.backends.cudnn.benchmark = False
    labels = load_targets()[args.ds]
    _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(args.ds)
    input_samples = train_x.shape[-1]
    if any(x.shape[-1] != input_samples for x in (val_x, test_x)):
        raise ValueError('All splits must have the same trial length for the flatten MLP')
    model = MLPClassifier(load_tower(config, checkpoint), len(labels), input_samples).to(device)
    loaders = [make_loader(x, y, config, len(labels), args.bs, training)
               for x, y, training in ((train_x, train_y, True), (val_x, val_y, False),
                                       (test_x, test_y, False))]
    train_loader, val_loader, test_loader = loaders
    fast = list(model.tower.tokenizer.parameters()) + list(model.classifier.parameters())
    fast_ids = {id(param) for param in fast}
    slow = [param for param in model.parameters() if id(param) not in fast_ids]
    optimizer = torch.optim.AdamW([
        {'params': fast, 'lr': train.lr_base * train.lr_scale_fast_params},
        {'params': slow, 'lr': train.lr_base * train.lr_scale_slow_params},
    ], weight_decay=train.weight_decay)
    schedule = cosine_annealing_lr(1.0, 0.1, epochs, len(train_loader), train.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(schedule))

    outdir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'source_checkpoint': str(checkpoint), 'dataset': args.ds, 'labels': labels,
        'head': 'mlp', 'input_samples': input_samples,
        'classifier': str(model.classifier),
        'seed': args.seed, 'epochs': epochs, 'precision': 'bf16-mixed' if use_bf16 else '32-true',
        'batch_size': args.bs, 'device': str(device),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'split_sizes': dict(zip(['train', 'validation', 'test'],
                               [len(loader.dataset) for loader in loaders])),
        'model_config': config.to_dict(), 'training_config': train.to_dict(),
        'selection': 'maximum validation BAcc; ties resolved by minimum validation loss',
    }
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    best_key, best_epoch = (-math.inf, -math.inf), 0
    history = {'trainLoss': [], **{f'val{k}': [] for k in ['Loss', 'BAcc', 'Auroc', 'wF1', 'Kappa']}}
    print(f'Loaded EEG tower from {checkpoint}; Q-Former removed; new FlattenClassifier MLP.')
    print(f'{args.ds}: class order={labels}; input samples={input_samples}')
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
        history['trainLoss'].append(loss_sum / len(train_loader.dataset))
        validation = evaluate(model, val_loader, device)
        for key, value in validation.items():
            history[f'val{key}'].append(value)
        key = (validation['BAcc'], -validation['Loss'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            temporary = outdir / 'best.ckpt.tmp'
            torch.save(model.state_dict(), temporary)
            temporary.replace(outdir / 'best.ckpt')
        np.savez(outdir / 'metrics.npz', **history, bestEpoch=best_epoch)
        print(f'{epoch}/{epochs} train loss={history["trainLoss"][-1]:.4f} '
              f'val BAcc={validation["BAcc"]:.4f} val Kappa={validation["Kappa"]:.4f}')

    model.load_state_dict(torch.load(outdir / 'best.ckpt', map_location=device, weights_only=True), strict=True)
    test = evaluate(model, test_loader, device)
    np.savez(outdir / 'metrics.npz', **history, bestEpoch=best_epoch,
             **{f'test{key}': value for key, value in test.items()})
    metadata.update(best_epoch=best_epoch, test=test)
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Best epoch={best_epoch}; test BAcc={test["BAcc"]:.4f}, Kappa={test["Kappa"]:.4f}')
    print(f'Saved tower + MLP checkpoint and metrics to {outdir}')


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        # Exit status 86 lets callers identify a CUDA out-of-memory failure.
        # Other failures (including host-memory failures) must not trigger it.
        if 'cuda' not in str(exc).lower():
            raise
        traceback.print_exc()
        print('CUDA OOM: this attempt failed; exiting with status 86.', file=sys.stderr)
        sys.exit(86)
