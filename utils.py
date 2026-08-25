import os
import csv
import h5py
import numpy as np
import json
import glob
import time
from pathlib import Path
import random
import torch
torch.set_num_threads(16)

def bold(x):       return '\033[1m'  + str(x) + '\033[0m'
def dim(x):        return '\033[2m'  + str(x) + '\033[0m'
def italicized(x): return '\033[3m'  + str(x) + '\033[0m'
def underline(x):  return '\033[4m'  + str(x) + '\033[0m'
def blink(x):      return '\033[5m'  + str(x) + '\033[0m'
def inverse(x):    return '\033[7m'  + str(x) + '\033[0m'
def gray(x):       return '\033[90m' + str(x) + '\033[0m'
def red(x):        return '\033[91m' + str(x) + '\033[0m'
def green(x):      return '\033[92m' + str(x) + '\033[0m'
def yellow(x):     return '\033[93m' + str(x) + '\033[0m'
def blue(x):       return '\033[94m' + str(x) + '\033[0m'
def magenta(x):    return '\033[95m' + str(x) + '\033[0m'
def cyan(x):       return '\033[96m' + str(x) + '\033[0m'
def white(x):      return '\033[97m' + str(x) + '\033[0m'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

def cosine_annealing_lr(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def make_lr_lambda(schedule):
    """Adapt a finite per-step schedule to PyTorch's LambdaLR convention.

    LambdaLR evaluates its callable with step=0 during initialization, so step
    must index the schedule directly. Clamping the terminal index keeps the
    callable valid for the final scheduler.step() after the last optimizer step.
    """
    if len(schedule) == 0:
        raise ValueError('learning-rate schedule must not be empty')
    return lambda step: float(schedule[min(step, len(schedule) - 1)])

def constant_lr(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters, endpoint=False)
    else:
        warmup_schedule = np.array([], dtype=float)

    const_iters = total_iters - warmup_iters
    constant_schedule = np.full(const_iters, base_value, dtype=float)

    schedule = np.concatenate((warmup_schedule, constant_schedule))
    assert len(schedule) == total_iters, "Schedule length mismatch."
    return schedule

def save_model_safely(to_save, save_ckpt):
    while True:
        try:
            torch.save(to_save.state_dict(), save_ckpt)
            print(f"Checkpoint saved to {save_ckpt}")
            break
        except Exception as e:
            print(f"Error saving checkpoint {save_ckpt}: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

def print_trainable_params(model, detail=False):
    n, p = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param = param.numel()
            if detail:
                print(f"Layer {name} is trainable, parameters: {param/1000:.3f} K")
            n += 1
            p += param
    print(f"[Trainable] > Layers: {n}")
    print(f"[Trainable] > Parameters: {p/1000/1000:.3f} M")
