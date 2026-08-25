import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
from utils import seed_all, cosine_annealing_lr, make_lr_lambda, yellow, green
from load_config import load_yaml, build_model_config, build_train_config
from load_datasets import load_dataset
from init_text_embeddings import load_embeddings
from LEAF import LEAF, FlattenClassifier

def eval_model(model, loader, device, n_classes, instrct_emb=None):
    model.eval()
    with torch.no_grad():
        Logits, Labels = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if instrct_emb is None:
                logits = model(x)
            else:
                logits = model(x, instrct_emb)
            Logits.append(logits.cpu())
            Labels.append(y.cpu())
        Logits, Labels = torch.cat(Logits, dim=0), torch.cat(Labels, dim=0)
        Prob = F.softmax(Logits, dim=1).numpy()
        y_true = Labels.numpy()
        y_pred = Logits.argmax(dim=1).numpy()
    
    bAcc = balanced_accuracy_score(y_true, y_pred)
    wF1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    if n_classes == 2:
        auroc = roc_auc_score(y_true, Prob[:, 1])
    else:
        auroc = roc_auc_score(y_true, Prob, multi_class="ovr", average="macro")
    return bAcc, auroc, wF1, kappa

class DownStream(nn.Module):
    def __init__(self, config, n_classes, trialTime):
        super().__init__()
        self.leaf = LEAF(config)
        self.classifier = FlattenClassifier(config.dim_token, 2*(trialTime*config.sfreq//config.window_len), n_classes)

    def forward(self, x, instrct_emb):
        emb, tokens = self.leaf(x, instrct_emb.unsqueeze(0).expand(x.size(0), -1))
        return self.classifier(tokens)
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default="0")
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--dim', type=str, default=None, help='override model dims (default: from --config YAML)')
    p.add_argument('--ds', type=str, default='MI_BCIC_IV2a')
    p.add_argument('--itEpo', type=int, default=100)
    p.add_argument('--emb', type=str, default=None, help='override embedding model (default: from --config YAML)')
    p.add_argument('--instrct', type=str, default='None')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', type=str, default='configs/LEAF_mpnet.yaml')
    args = p.parse_args()

    device = 'cpu' if args.gpu == '-1' else f'cuda:{args.gpu.split(",")[0]}'

    cfg = load_yaml(args.config)
    train_config = build_train_config(cfg, 'downstream')
    config = build_model_config(cfg, args.dim, emb=args.emb)
    emb_name = config.text_emb_model_name
    instrct_emb = torch.tensor(load_embeddings(emb_name)[args.instrct]).to(device)

    fingerprint = f'w{config.window_len}-c{config.dim_cnn}-t{config.dim_token}-lay{config.num_layers}'
    it_fingerprint = f'q{config.num_q}-qlay{config.num_qformer_layers}-{emb_name}-{args.seed}'

    wd = Path(f'ckpt/{fingerprint}/{it_fingerprint}/{args.itEpo}-{args.seed}')
    wd.mkdir(parents=True, exist_ok=True)
    log_file = wd / f"{args.ds}.npz"
    if log_file.exists():
        print(yellow(f">> Log file exists: {log_file}"))
        exit(0)
    ckpt = wd.parent / f'it{args.itEpo:03d}.ckpt'

    trialTime, trainX, trainY, valX, valY, testX, testY = load_dataset(args.ds)
    n_classes = np.unique(trainY).shape[0]

    seed_all(args.seed)
    trainDataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY, dtype=torch.long))
    valDataset   = TensorDataset(torch.tensor(valX),   torch.tensor(valY,   dtype=torch.long))
    testDataset  = TensorDataset(torch.tensor(testX),  torch.tensor(testY,  dtype=torch.long))

    trainLoader = DataLoader(trainDataset, batch_size=args.bs, shuffle=True)
    valLoader   = DataLoader(valDataset,   batch_size=args.bs, shuffle=False)
    testLoader  = DataLoader(testDataset,  batch_size=args.bs, shuffle=False)

    model = DownStream(config, n_classes, trialTime).to(device)
    print(f">> Load checkpoint: {green(ckpt)}")
    model.leaf.load_state_dict(torch.load(ckpt, map_location=device), strict=True)

    fastLR_params = list(model.leaf.tower.tokenizer.parameters()) + list(model.classifier.parameters())
    fast_ids = set(id(p) for p in fastLR_params)
    slowLR_params = [p for p in model.parameters() if id(p) not in fast_ids]
    optimizer = optim.AdamW([
        {"params": fastLR_params, "lr": train_config.lr_base * train_config.lr_scale_fast_params},
        {"params": slowLR_params, "lr": train_config.lr_base * train_config.lr_scale_slow_params},
    ], weight_decay=train_config.weight_decay)
    lr_list = cosine_annealing_lr(1.0, 0.1, train_config.max_epochs, len(trainLoader), warmup_epochs=train_config.warmup_epochs)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(lr_list))

    metrics = {
        "valBAcc": [],"valAuroc": [],"valwF1": [],"valKappa": [],
        "testBAcc": [], "testAuroc": [], "testwF1": [], "testKappa": []
    }

    criterion = nn.CrossEntropyLoss()
    # bf16-mixed needs no GradScaler (bf16 has fp32's exponent range); cross-entropy
    # runs in fp32 under autocast, so the loss stays numerically stable.
    for epoch in range(train_config.max_epochs):
        model.train()
        for x, y in trainLoader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                logits = model(x, instrct_emb)
                loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            scheduler.step()

        valBAcc, valAuroc, valwF1, valKappa = eval_model(model, valLoader, device, n_classes, instrct_emb=instrct_emb)
        metrics['valBAcc'].append(valBAcc)
        metrics['valAuroc'].append(valAuroc)
        metrics['valwF1'].append(valwF1)
        metrics['valKappa'].append(valKappa)

        testBAcc, testAuroc, testwF1, testKappa = eval_model(model, testLoader, device, n_classes, instrct_emb=instrct_emb)
        metrics['testBAcc'].append(testBAcc)
        metrics['testAuroc'].append(testAuroc)
        metrics['testwF1'].append(testwF1)
        metrics['testKappa'].append(testKappa)

        print(f"{epoch+1}/{train_config.max_epochs}  Test bAcc: {testBAcc:.4f} Kappa: {testKappa:.4f} {args.ds}")

    np.savez(log_file, **metrics)
