import argparse
import random
from pathlib import Path

import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy

from utils import seed_all, cosine_annealing_lr, make_lr_lambda, yellow, save_model_safely
from load_config import load_yaml, build_model_config, build_train_config
from LEAF import Tower
from leaf_datasets.shared import PRETRAIN_FOLDER

CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints'
CHECKPOINT_EPOCHS = frozenset({5, 10})

class PretrainDataset(Dataset):
    def __init__(self, dataPath, trialKeys, training, pad_to):
        self.dataPath = Path(dataPath)
        self.trialKeys = trialKeys 
        self.training = training
        self.pad_to = pad_to
        self.fp_cache = {}

    def __len__(self):
        return len(self.trialKeys)

    def load(self, fn_npy, trial):
        if fn_npy not in self.fp_cache:
            self.fp_cache[fn_npy] = np.load(self.dataPath / fn_npy, mmap_mode='r')
        return np.asarray(self.fp_cache[fn_npy][trial]).copy()

    def pad(self, sample):
        # padding if needed
        if sample.shape[1] < self.pad_to:
            need_pad = self.pad_to - sample.shape[1]
            if self.training:
                left_pad = random.randint(0, need_pad)
                right_pad = need_pad - left_pad
            else:
                left_pad = 0
                right_pad = need_pad
            sample = np.pad(sample, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)

        elif sample.shape[1] > self.pad_to:
            if self.training:
                start = random.randint(0, sample.shape[1] - self.pad_to)
            else:
                start = 0
            sample = sample[:, start:start + self.pad_to]
        
        return np.ascontiguousarray(sample)

    def __getitem__(self, idx):
        name, trial = self.trialKeys[idx].split('/')
        X = self.pad(self.load(f'{name}.npy', int(trial)))
        return X.astype(np.float32)
    
def split_pretrain_trainval(PRETRAIN_FOLDER, val_ratio=0.1):
    assert val_ratio in [0.1, 0.2], "Only val_ratio = 0.1 or 0.2 is supported."
    interval = int(1 / val_ratio)
    file_list = sorted(PRETRAIN_FOLDER.glob('*.npy'))
    print(f'Found {len(file_list)} files in {PRETRAIN_FOLDER}', file_list[0])

    counter, train_samples, val_samples = 0, [], []
    for fname in file_list:
        n = np.load(fname, mmap_mode='r').shape[0]
        for i in range(n):
            sample = f'{fname.stem}/{i:06d}'  # e.g., 'CARE_000/000001'
            if counter % interval == interval - 1:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
            counter += 1

    return train_samples, val_samples

class Pretrain_Tower_Module(pl.LightningModule):
    def __init__(self, config, niter_per_ep, lr_base, max_epochs, warmup_epochs, checkpoint_dir):
        super().__init__()
        self.config = config
        self.lr_base = lr_base
        self.valLoss = []
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = Tower(config)
        self.lr_list = cosine_annealing_lr(1, 0.1, max_epochs, niter_per_ep, warmup_epochs=warmup_epochs)

    def configure_optimizers(self):
        # self.optimizer = optim.AdamW(self.parameters(), lr=self.lr_base, weight_decay=1e-4)
        fastLR_params = list(self.model.tokenizer.parameters())
        fast_ids = set(id(p) for p in fastLR_params)
        slowLR_params = [p for p in self.model.parameters() if id(p) not in fast_ids]
        self.optimizer = optim.AdamW([
            {"params": fastLR_params, "lr": self.lr_base},
            {"params": slowLR_params, "lr": self.lr_base * 0.1},
        ])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, make_lr_lambda(self.lr_list))
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x = batch
        loss1, loss2 = self.model.loss(x)
        return loss1 + loss2

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        loss1, loss2 = self.model.loss(x)
        self.log("loss1", loss1, on_epoch=True, sync_dist=True)
        self.log("loss2", loss2, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or self.global_rank != 0:
            return
        loss1 = self.trainer.callback_metrics['loss1'].item()
        loss2 = self.trainer.callback_metrics['loss2'].item()
        self.valLoss.append([loss1, loss2])
        epoch = self.current_epoch + 1
        print(f'{epoch} {loss1:.4f} {loss2:.4f}')
        if epoch in CHECKPOINT_EPOCHS:
            checkpoint = self.checkpoint_dir / f'leaf-pretrain-epoch-{epoch:02d}.ckpt'
            print(yellow(f'Saving epoch {epoch} pretrained Tower -> {checkpoint}'))
            save_model_safely(self.model, checkpoint)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default="0")
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--dim', type=str, default=None, help='override model dims (default: from --config YAML)')
    p.add_argument('--config', type=str, default='configs/LEAF_mpnet.yaml')
    p.add_argument('--epochs', type=int, default=10, help='number of pretraining epochs (default: 10)')
    p.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    args = p.parse_args()

    if args.epochs <= 0:
        p.error('--epochs must be positive')

    # Seed before Tower construction so model initialization is reproducible.
    seed_all(args.seed)

    gpus = [int(_) for _ in args.gpu.split(',')]

    cfg = load_yaml(args.config)
    train_config = build_train_config(cfg, 'pretrain')
    config = build_model_config(cfg, args.dim)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CHECKPOINT_DIR / 'leaf-pretrain-loss.npy'

    if log_file.exists():
        print(f'Found existing log: {log_file}. Skipping training.')
        exit()
    
    trainTrials, valTrials = split_pretrain_trainval(PRETRAIN_FOLDER)
    print(f'Train Trials: {len(trainTrials)}', trainTrials[0])
    print(f'Val Trials: {len(valTrials)}', valTrials[0])

    trainDataset = PretrainDataset(PRETRAIN_FOLDER, trainTrials, training=True, pad_to=config.max_seq_len)
    validDataset = PretrainDataset(PRETRAIN_FOLDER, valTrials, training=False,  pad_to=config.max_seq_len)
    trainLoader = DataLoader(trainDataset, args.bs, shuffle=True,  num_workers=16, pin_memory=True, drop_last=True)
    validLoader = DataLoader(validDataset, args.bs, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    # Under DDP, Lightning injects a DistributedSampler so each process runs
    # len(trainLoader) // num_gpus optimizer steps per epoch. The LR schedule
    # advances once per process step, so size it to the per-process count.
    niter_per_epoch = len(trainLoader) // len(gpus)
    pl_model = Pretrain_Tower_Module(
        config,
        niter_per_epoch,
        train_config.lr_base,
        args.epochs,
        train_config.warmup_epochs,
        CHECKPOINT_DIR,
    )

    strategy = DDPStrategy(find_unused_parameters=False) if len(gpus) > 1 else 'auto'
    trainer = pl.Trainer(
        strategy=strategy,
        accelerator='gpu',
        devices=gpus,
        max_epochs=args.epochs,
        precision=train_config.precision, 
        enable_progress_bar=True, 
        enable_checkpointing=False,
        benchmark=True, 
        deterministic=False, 
        enable_model_summary=False, 
        logger=False,
        check_val_every_n_epoch=1
    )

    trainer.fit(pl_model, train_dataloaders=trainLoader, val_dataloaders=validLoader)
    if trainer.global_rank == 0:
        np.save(log_file, np.array(pl_model.valLoss), allow_pickle=False)
