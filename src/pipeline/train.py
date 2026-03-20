"""
学習パイプライン: Mamba3 モデルの学習・評価・バックテスト
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.components.mamba3_model import Mamba3Forecaster
from src.components.dataset import SectorLeadLagDataset
from src.components import backtest
from src.config import settings

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


class DirectionalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        dir_loss = 1.0 - (torch.sign(pred) * torch.sign(target)).mean()
        return (1 - self.alpha) * mse_loss + self.alpha * dir_loss


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE, clip_grad=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for us_seq, jp_seq, target in loader:
        pred = model(us_seq.to(device), jp_seq.to(device))
        loss = criterion(pred, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for us_seq, jp_seq, target in loader:
        us_seq, jp_seq, target = us_seq.to(device), jp_seq.to(device), target.to(device)
        pred = model(us_seq, jp_seq)
        total_loss += criterion(pred, target).item()
        correct += (torch.sign(pred) == torch.sign(target)).float().sum().item()
        total += target.numel()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def generate_signals(model, dataset: SectorLeadLagDataset, device=DEVICE, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dates = dataset.get_dates()
    cols = [f"jp_{i}" for i in range(dataset.n_jp)]
    preds = []
    for us_seq, jp_seq, _ in loader:
        preds.append(model(us_seq.to(device), jp_seq.to(device)).cpu().numpy())
    return pd.DataFrame(np.concatenate(preds), index=dates[:sum(len(p) for p in preds)], columns=cols)


def backtest_signals(signals, jp_oc_ret, q=0.3):
    signals.columns = list(jp_oc_ret.columns)[:signals.shape[1]]
    common = signals.index.intersection(jp_oc_ret.index)
    weights = backtest.construct_long_short_weights(signals.loc[common], q=q)
    return backtest.compute_strategy_returns(weights, jp_oc_ret.loc[common])


def compute_metrics(daily_returns):
    r = daily_returns.dropna()
    if len(r) < 10:
        return {"AR": 0.0, "RISK": 0.0, "RR": 0.0, "MDD": 0.0}
    return {
        "AR": round(backtest.annualized_return(r) * 100, 2),
        "RISK": round(backtest.annualized_risk(r) * 100, 2),
        "RR": round(backtest.risk_return_ratio(r), 2),
        "MDD": round(abs(backtest.max_drawdown(r)) * 100, 2),
    }


def train_and_evaluate(hparams, train_ds, val_ds, test_ds, jp_oc_ret, save_dir=None):
    if save_dir is None:
        save_dir = settings.MODELS_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    d_model = hparams.get("d_model", 64)
    d_state = hparams.get("d_state", 32)
    n_layers = hparams.get("n_layers", 2)
    expand = hparams.get("expand", 2)
    headdim = hparams.get("headdim", 16)
    dropout = hparams.get("dropout", 0.1)
    lr = hparams.get("lr", 1e-3)
    weight_decay = hparams.get("weight_decay", 1e-4)
    epochs = hparams.get("epochs", 50)
    batch_size = hparams.get("batch_size", 32)
    loss_alpha = hparams.get("loss_alpha", 0.3)
    q = hparams.get("q", 0.3)
    patience = hparams.get("patience", 10)
    seq_len = hparams.get("seq_len", 60)

    model = Mamba3Forecaster(
        n_us=train_ds.n_us, n_jp=train_ds.n_jp,
        d_model=d_model, d_state=d_state, n_layers=n_layers,
        expand=expand, headdim=headdim, dropout=dropout, seq_len=seq_len,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}, device: {DEVICE}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = DirectionalLoss(alpha=loss_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_epoch, no_improve = float("inf"), 0, 0
    model_path = f"{save_dir}/best_model.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, DEVICE)
        val_loss, val_dir = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} dir={val_dir:.3f} (best@{best_epoch})")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} (best@{best_epoch})")
            break

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    model.load_state_dict(torch.load(model_path, weights_only=True))

    val_metrics = compute_metrics(backtest_signals(generate_signals(model, val_ds, DEVICE), jp_oc_ret, q))
    test_metrics = compute_metrics(backtest_signals(generate_signals(model, test_ds, DEVICE), jp_oc_ret, q))
    _, val_dir_final = evaluate(model, val_dl, criterion, DEVICE)
    _, test_dir_final = evaluate(model, test_dl, criterion, DEVICE)

    return {
        **{f"hp_{k}": v for k, v in hparams.items()},
        "n_params": n_params, "best_epoch": best_epoch, "train_time_s": round(elapsed, 1),
        "val_loss": round(best_val_loss, 6), "val_dir_acc": round(val_dir_final, 4),
        "val_AR": val_metrics["AR"], "val_RISK": val_metrics["RISK"],
        "val_RR": val_metrics["RR"], "val_MDD": val_metrics["MDD"],
        "test_dir_acc": round(test_dir_final, 4),
        "test_AR": test_metrics["AR"], "test_RISK": test_metrics["RISK"],
        "test_RR": test_metrics["RR"], "test_MDD": test_metrics["MDD"],
    }
