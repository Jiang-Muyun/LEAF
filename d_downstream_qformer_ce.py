"""Fine-tune the complete LEAF model using frozen text-prototype classification.

No classifier is added. Every model weight, including
the Q-Former, must load strictly from an instruction-tuned LEAF checkpoint.
Logits are cosine similarities (equivalently negative cosine distances,
up to a class-independent constant).

Example, from LEAF/:
    python d_downstream_qformer_ce.py --ds MI_OpenBMI \
        --ckpt leaf-v1.0-instruct-mpnet-base.ckpt --gpu 0

The best checkpoint is selected by validation balanced accuracy, with validation
loss breaking ties. The test set is evaluated only after that selection.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing
and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score,
)

from LEAF import LEAF
from load_config import (
    build_model_config, build_train_config, load_instructions, load_targets,
    load_yaml,
)
from load_datasets import load_dataset
from utils import cosine_annealing_lr, make_lr_lambda, seed_all


ROOT = Path(__file__).resolve().parent


def resolve_file(path, fallback_root):
    for candidate in (path, fallback_root / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'File not found: {path} (also checked {fallback_root / path})')


def normalized(vector, name):
    vector = torch.as_tensor(vector, dtype=torch.float32).detach().clone()
    norms = vector.norm(dim=-1, keepdim=True)
    if not torch.isfinite(vector).all() or (norms <= 1e-12).any():
        raise ValueError(f'{name} must contain finite, nonzero embeddings')
    return vector / norms


def prepare_text(config, dataset, instruction_text=None, embedding_dir=None):
    """Keep prototype row i aligned with numeric dataset label i; never sort labels."""
    labels = load_targets()[dataset]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f'{dataset}: expected at least two distinct target labels')
    directory = Path(embedding_dir) if embedding_dir is not None else ROOT / 'text_embeddings'
    name = config.text_emb_model_name
    texts = (directory / f'{name}.txt').read_text().splitlines()
    vectors = np.load(directory / f'{name}.npy', allow_pickle=False)
    if vectors.shape != (len(texts), config.text_emb_model_dim) or len(set(texts)) != len(texts):
        raise ValueError(f'Invalid text embedding cache for {name}')
    cache = dict(zip(texts, vectors))
    instructions = load_instructions()
    if instruction_text is None:
        family = next((family for prefix, family in (
            ('MI_', 'Motor'), ('EMO_', 'Emotion'), ('CS_', 'Speech'),
            ('SSVEP_', 'SSVEP'),
        ) if dataset.startswith(prefix)), dataset)
        instruction_text = instructions[family][0]
    missing = [text for text in [*labels, instruction_text] if text not in cache]
    if missing:
        raise ValueError(f'Missing cached text embeddings: {missing}. Run init_text_embeddings.py {name}.')
    prototypes = normalized(np.stack([cache[label] for label in labels]), 'prototypes')
    instruction = normalized(cache[instruction_text], 'instruction')
    # Condition on the task and candidate classes, never the trial's true label.
    target_mean = normalized(prototypes.mean(dim=0), 'mean prototype')
    instruction = normalized(instruction + target_mean, 'blended instruction')
    return labels, instruction_text, instruction, prototypes


class PrototypeClassifier(nn.Module):
    """Parameter-free cosine readout around an existing, fully restored LEAF."""

    def __init__(self, leaf, instruction, prototypes):
        super().__init__()
        dim = leaf.config.text_emb_model_dim
        if instruction.shape != (dim,) or prototypes.ndim != 2 or prototypes.shape[1] != dim:
            raise ValueError('Instruction/prototype dimensions do not match the LEAF configuration')
        self.leaf = leaf
        self.register_buffer('instruction', normalized(instruction, 'instruction'))
        self.register_buffer('prototypes', normalized(prototypes, 'prototypes'))

    def forward(self, x):
        embedding, _ = self.leaf(x, self.instruction.unsqueeze(0).expand(x.size(0), -1))
        # Explicit fp32 normalization also keeps the readout stable under bf16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.normalize(embedding.float(), dim=-1) @ self.prototypes.T


def load_classifier(config, checkpoint, instruction, prototypes):
    leaf = LEAF(config)
    # A Tower-only or partially matching checkpoint must fail, rather than leave
    # any randomly initialized Q-Former weights in the finetuning model.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    leaf.load_state_dict(state, strict=True)
    return PrototypeClassifier(leaf, instruction, prototypes)


def make_loader(x, y, config, n_classes, batch_size, training):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    if x.ndim != 3 or x.shape[1] != config.n_channels or len(x) == 0:
        raise ValueError(f'Expected nonempty (N, {config.n_channels}, T) EEG, got {x.shape}')
    if not config.window_len <= x.shape[2] <= config.max_seq_len:
        raise ValueError(f'EEG length {x.shape[2]} is outside the configured input limits')
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Labels must be a one-dimensional integer array matching the EEG trials')
    if not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError('Every split must contain labels 0..K-1 in configs/tasks.yaml order')
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite evaluation logits')
    probability = logits.softmax(dim=1).numpy()
    truth, prediction = labels.numpy(), logits.argmax(dim=1).numpy()
    auc = (roc_auc_score(truth, probability[:, 1]) if probability.shape[1] == 2
           else roc_auc_score(truth, probability, multi_class='ovr', average='macro'))
    return {
        'Loss': F.cross_entropy(logits, labels).item(),
        'BAcc': float(balanced_accuracy_score(truth, prediction)),
        'Auroc': float(auc),
        'wF1': float(f1_score(truth, prediction, average='weighted')),
        'Kappa': float(cohen_kappa_score(truth, prediction)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/LEAF_mpnet.yaml')
    p.add_argument('--ckpt', type=Path, required=True, help='Full instruction-tuned LEAF state dict; also searched in LEAF/checkpoints/')
    p.add_argument('--ds', choices=list(load_targets()), default='MI_BCIC_IV2a')
    p.add_argument('--gpu', default='0', help='One GPU index, or -1 for CPU')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--epochs', type=int, help='Override downstream.max_epochs')
    p.add_argument('--seed', type=int, default=42, help='Finetuning seed; checkpoint is selected explicitly by --ckpt')
    p.add_argument('--dim', help='Architecture override; must match the checkpoint')
    p.add_argument('--emb', help='Text embedding model override; must match the checkpoint')
    p.add_argument('--instrct', help='Override instruction with an exact cached string')
    p.add_argument('--precision', choices=['32-true', 'bf16-mixed'], help='Override downstream.precision; CPU always uses fp32')
    p.add_argument('--embedding-dir', type=Path, default=ROOT / 'text_embeddings')
    p.add_argument('--outdir', type=Path, help='New output directory; existing results are never overwritten')
    args = p.parse_args()
    if args.bs <= 0 or (args.epochs is not None and args.epochs <= 0):
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
    outdir = (args.outdir or checkpoint.parent / 'qformer_finetuning' / checkpoint.stem
              / args.ds / f'seed{args.seed}').resolve()
    if outdir.exists():
        raise FileExistsError(f'Output directory already exists: {outdir}; choose a new --outdir')

    seed_all(args.seed)
    # Avoid convolution autotuning after repeated CUDA illegal-address failures.
    # Keep this local: seed_all is also used by unrelated training entry points.
    torch.backends.cudnn.benchmark = False
    labels, text, instruction, prototypes = prepare_text(
        config, args.ds, args.instrct, args.embedding_dir,
    )
    model = load_classifier(config, checkpoint, instruction, prototypes).to(device)
    _, train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(args.ds)
    loaders = [make_loader(x, y, config, len(labels), args.bs, training)
               for x, y, training in ((train_x, train_y, True), (val_x, val_y, False),
                                       (test_x, test_y, False))]
    train_loader, val_loader, test_loader = loaders
    fast = list(model.leaf.tower.tokenizer.parameters())
    fast_ids = {id(param) for param in fast}
    slow = [param for param in model.leaf.parameters() if id(param) not in fast_ids]
    optimizer = torch.optim.AdamW([
        {'params': fast, 'lr': train.lr_base * train.lr_scale_fast_params},
        {'params': slow, 'lr': train.lr_base * train.lr_scale_slow_params},
    ], weight_decay=train.weight_decay)
    schedule = cosine_annealing_lr(1.0, 0.1, epochs, len(train_loader), train.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(schedule))

    outdir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'source_checkpoint': str(checkpoint), 'dataset': args.ds, 'labels': labels,
        'instruction': text, 'conditioning': 'task+targets',
        'seed': args.seed, 'epochs': epochs, 'precision': 'bf16-mixed' if use_bf16 else '32-true',
        'batch_size': args.bs, 'device': str(device),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'split_sizes': dict(zip(['train', 'validation', 'test'],
                               [len(loader.dataset) for loader in loaders])),
        'model_config': config.to_dict(), 'training_config': train.to_dict(),
        'selection': 'maximum validation BAcc; ties resolved by minimum validation loss',
    }
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez(outdir / 'text_readout.npz', labels=np.asarray(labels),
             instruction=model.instruction.cpu().numpy(), prototypes=model.prototypes.cpu().numpy())
    best_key, best_epoch = (-math.inf, -math.inf), 0
    history = {'trainLoss': [], **{f'val{k}': [] for k in ['Loss', 'BAcc', 'Auroc', 'wF1', 'Kappa']}}
    print(f'Loaded all LEAF weights from {checkpoint}; no new trainable parameters.')
    print(f'{args.ds}: class order={labels}; instruction={text!r}')
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
            torch.save(model.leaf.state_dict(), temporary)
            temporary.replace(outdir / 'best.ckpt')
        np.savez(outdir / 'metrics.npz', **history, bestEpoch=best_epoch)
        print(f'{epoch}/{epochs} train loss={history["trainLoss"][-1]:.4f} '
              f'val BAcc={validation["BAcc"]:.4f} val Kappa={validation["Kappa"]:.4f}')

    model.leaf.load_state_dict(torch.load(outdir / 'best.ckpt', map_location=device, weights_only=True), strict=True)
    test = evaluate(model, test_loader, device)
    np.savez(outdir / 'metrics.npz', **history, bestEpoch=best_epoch,
             **{f'test{key}': value for key, value in test.items()})
    metadata.update(best_epoch=best_epoch, test=test)
    (outdir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Best epoch={best_epoch}; test BAcc={test["BAcc"]:.4f}, Kappa={test["Kappa"]:.4f}')
    print(f'Saved checkpoint, fixed text readout, and metrics to {outdir}')


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
