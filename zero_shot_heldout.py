"""Zero-shot LEAF evaluation on held-out Dreyer2023 and Weibo2014 MI data.

Neither dataset is part of ``configs/tasks.yaml`` or instruction tuning. The
trained model is kept frozen and predictions are obtained only by matching each
EEG embedding against the cached ``Left`` and ``Right`` text prototypes.

The default run evaluates the epoch-100 MPNet and Qwen3 checkpoints at all three
instruction levels and writes one combined CSV under the architecture checkpoint
root:

    python3 zero_shot_heldout.py

Dreyer's delivered HDF5 file has no label legend. Its mapping is therefore the
repository's existing assumption: 0 = Left and 1 = Right. Weibo's mapping is
known from its raw event codes (1 = left hand and 2 = right hand, then minus 1).
"""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from init_text_embeddings import load_embeddings
from LEAF import LEAF
from load_config import build_model_config, load_yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = ROOT / "ckpt" / "w100-c64-t256-lay12"
DEFAULT_DREYER_H5 = Path("/media/public/LEAF/MI_Dreyer2023.h5")
DEFAULT_WEIBO_H5 = Path("/media/public/LEAF/MI_Weibo2014.h5")

TARGETS = ("Left", "Right")
DEFAULT_INSTRUCTION = "None"
MOTOR_INSTRUCTION = "This is a motor imagery decoding task"

MODEL_CONFIGS = {
    "MPNet-base": ROOT / "configs" / "LEAF_mpnet.yaml",
    "Qwen3-Embedding-4B": ROOT / "configs" / "LEAF_qwen3.yaml",
}


@dataclass
class HeldoutDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    label_mapping: str


def validate_dataset(dataset: HeldoutDataset) -> None:
    if dataset.x.ndim != 3 or dataset.x.shape[1] != 65:
        raise ValueError(f"{dataset.name}: expected X shaped (N, 65, T), got {dataset.x.shape}")
    if len(dataset.x) != len(dataset.y):
        raise ValueError(f"{dataset.name}: X/Y length mismatch")
    labels = np.unique(dataset.y)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(f"{dataset.name}: expected labels [0, 1], got {labels.tolist()}")
    if not np.isfinite(dataset.x).all():
        raise ValueError(f"{dataset.name}: input contains NaN or infinity")


def load_dreyer(path: Path, split: str) -> HeldoutDataset:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r", locking=False) as handle:
        x = handle[split]["X"][:].astype(np.float32, copy=False)
        y = handle[split]["Y"][:].astype(np.int64, copy=False)

    # The delivered trials contain 801 samples. Match the 800-sample convention
    # used by all other four-second datasets instead of relying on unfold() to
    # silently discard the final sample.
    x = np.ascontiguousarray(x[:, :, :800])
    dataset = HeldoutDataset(
        name="MI_Dreyer2023",
        x=x,
        y=y,
        label_mapping="assumed 0=Left;1=Right (HDF5 has no label legend)",
    )
    validate_dataset(dataset)
    return dataset


def load_weibo(path: Path, excluded_subjects: set[str]) -> HeldoutDataset:
    if not path.exists():
        raise FileNotFoundError(path)

    xs, ys = [], []
    with h5py.File(path, "r", locking=False) as handle:
        for subject in sorted(handle.keys()):
            if subject in excluded_subjects:
                print(f"  {subject}: excluded")
                continue
            xs.append(handle[subject]["X"][:].astype(np.float32, copy=False))
            ys.append(handle[subject]["Y"][:].astype(np.int64, copy=False))

    if not xs:
        raise ValueError("MI_Weibo2014: no subjects remain after exclusion")
    dataset = HeldoutDataset(
        name="MI_Weibo2014",
        x=np.ascontiguousarray(np.concatenate(xs)),
        y=np.concatenate(ys),
        label_mapping="verified 0=Left;1=Right",
    )
    validate_dataset(dataset)
    return dataset


def instruction_embedding(
    text_to_embedding: dict[str, np.ndarray], level: int
) -> tuple[str, np.ndarray]:
    if level == 0:
        text = DEFAULT_INSTRUCTION
    elif level in (1, 2):
        text = MOTOR_INSTRUCTION
    else:
        raise ValueError(f"Unsupported instruction level: {level}")

    embedding = text_to_embedding[text].copy()
    if level == 2:
        target_mean = np.mean([text_to_embedding[label] for label in TARGETS], axis=0)
        target_mean /= np.linalg.norm(target_mean)
        embedding = (embedding + target_mean) / 2.0
        embedding /= np.linalg.norm(embedding)
    return text, embedding.astype(np.float32, copy=False)


@torch.inference_mode()
def predict(
    model: LEAF,
    x: np.ndarray,
    instruction: np.ndarray,
    prototypes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    instruction_tensor = torch.from_numpy(instruction).to(device)
    prototype_tensor = torch.from_numpy(prototypes).to(device)
    logits = []

    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        embedding, _ = model(
            batch,
            instruction_tensor.unsqueeze(0).expand(len(batch), -1),
        )
        logits.append((embedding @ prototype_tensor.T).cpu().numpy())

    scores = np.concatenate(logits)
    return scores.argmax(axis=1), scores


def calculate_metrics(y: np.ndarray, prediction: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    # The score difference is monotonic with the two-class softmax probability
    # and avoids unnecessary exponentiation.
    right_score = scores[:, 1] - scores[:, 0]
    return {
        "bAcc": balanced_accuracy_score(y, prediction),
        "kappa": cohen_kappa_score(y, prediction),
        "weighted_f1": f1_score(y, prediction, average="weighted"),
        "AUROC": roc_auc_score(y, right_score),
        "true_left": int((y == 0).sum()),
        "true_right": int((y == 1).sum()),
        "pred_left": int((prediction == 0).sum()),
        "pred_right": int((prediction == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def parse_levels(value: str) -> list[int]:
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not levels or any(level not in (0, 1, 2) for level in levels):
        raise argparse.ArgumentTypeError("levels must be a comma-separated subset of 0,1,2")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="GPU index, or -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="instruction-tuning seed")
    parser.add_argument("--levels", type=parse_levels, default=parse_levels("0,1,2"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help="model checkpoints to evaluate (default: all)",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dreyer-h5", type=Path, default=DEFAULT_DREYER_H5)
    parser.add_argument("--dreyer-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--weibo-h5", type=Path, default=DEFAULT_WEIBO_H5)
    parser.add_argument(
        "--exclude-weibo-subjects",
        default="",
        help="Optional comma-separated subject IDs such as S03; empty includes everyone",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT / "zero_shot_heldout_it100.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    excluded = {item.strip().upper() for item in args.exclude_weibo_subjects.split(",") if item.strip()}
    datasets = [
        load_dreyer(args.dreyer_h5, args.dreyer_split),
        load_weibo(args.weibo_h5, excluded),
    ]
    for dataset in datasets:
        counts = np.bincount(dataset.y, minlength=2)
        print(f">> {dataset.name}: X{dataset.x.shape}, Left={counts[0]}, Right={counts[1]}")

    device = torch.device("cpu" if args.gpu == "-1" else f"cuda:{args.gpu.split(',')[0]}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False

    checkpoint_root = args.checkpoint_root.resolve()
    rows: list[dict[str, object]] = []

    for display_name in args.models:
        config_path = MODEL_CONFIGS[display_name]
        config = build_model_config(load_yaml(config_path))
        embedding_name = config.text_emb_model_name
        run_name = f"q{config.num_q}-qlay{config.num_qformer_layers}-{embedding_name}-{args.seed}"
        checkpoint = checkpoint_root / run_name / f"it{args.checkpoint:03d}.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        text_to_embedding = load_embeddings(embedding_name)
        prototypes = np.stack([text_to_embedding[label] for label in TARGETS]).astype(np.float32)

        model = LEAF(config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        print(f"\n>> {display_name}: {checkpoint}")

        for level in args.levels:
            instruction_text, instruction = instruction_embedding(text_to_embedding, level)
            for dataset in datasets:
                prediction, scores = predict(
                    model,
                    dataset.x,
                    instruction,
                    prototypes,
                    device,
                    args.batch_size,
                )
                metrics = calculate_metrics(dataset.y, prediction, scores)
                print(
                    f"  level={level} {dataset.name}: "
                    f"bAcc={metrics['bAcc']:.4f}, kappa={metrics['kappa']:.4f}, "
                    f"AUROC={metrics['AUROC']:.4f}, "
                    f"pred=[{metrics['pred_left']}, {metrics['pred_right']}]"
                )
                rows.append(
                    {
                        "model": display_name,
                        "embedding": embedding_name,
                        "checkpoint": args.checkpoint,
                        "dataset": dataset.name,
                        "split": args.dreyer_split if dataset.name == "MI_Dreyer2023" else "all_subjects",
                        "level": level,
                        "instruction": instruction_text,
                        "n_samples": len(dataset.y),
                        "bAcc": f"{metrics['bAcc']:.4f}",
                        "kappa": f"{metrics['kappa']:.4f}",
                        "weighted_f1": f"{metrics['weighted_f1']:.4f}",
                        "AUROC": f"{metrics['AUROC']:.4f}",
                        "true_left": metrics["true_left"],
                        "true_right": metrics["true_right"],
                        "pred_left": metrics["pred_left"],
                        "pred_right": metrics["pred_right"],
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tp": metrics["tp"],
                        "label_mapping": dataset.label_mapping,
                    }
                )

        model.cpu()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
