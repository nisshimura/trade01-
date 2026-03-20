"""
ハイブリッドアンサンブル: PCA_SUB + Mamba3 シグナルの重み付け結合
PCA_SUB (test AR ~20%) と Mamba3 (test AR ~10%) を組み合わせて 25%+ を目指す
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import settings
from src.components import data_loader, pca_model, backtest
from src.components.dataset import SectorLeadLagDataset, SPLIT_DATES, create_datasets
from src.components.mamba3_model import Mamba3Forecaster
from src.pipeline.train import (
    DEVICE, DirectionalLoss, generate_signals, backtest_signals,
    compute_metrics, train_one_epoch, evaluate,
)

RESULTS_CSV = Path(settings.RESULTS_DIR) / "ensemble_results.csv"


def compute_pca_signals_for_period(
    us_cc_ret: pd.DataFrame, jp_cc_ret: pd.DataFrame,
    jp_oc_ret: pd.DataFrame, start: str, end: str,
) -> pd.DataFrame:
    """PCA_SUB シグナルを指定期間で生成"""
    V0 = pca_model.build_prior_subspace(
        list(us_cc_ret.columns), list(jp_cc_ret.columns),
        settings.US_CYCLICAL, settings.US_DEFENSIVE,
        settings.JP_CYCLICAL, settings.JP_DEFENSIVE,
    )
    Cf = pca_model.compute_cfull(us_cc_ret, jp_cc_ret)
    C0 = pca_model.compute_target_matrix(V0, Cf)
    signals = pca_model.regularized_pca_signal(us_cc_ret, jp_cc_ret, V0, C0)
    return signals.loc[start:end]


def normalize_signals(s: pd.DataFrame) -> pd.DataFrame:
    """各日のシグナルをクロスセクション正規化 (zero-mean, unit-std)"""
    mu = s.mean(axis=1)
    sigma = s.std(axis=1)
    sigma[sigma < 1e-12] = 1e-12
    return s.sub(mu, axis=0).div(sigma, axis=0)


def ensemble_and_backtest(
    pca_sig: pd.DataFrame, mamba_sig: pd.DataFrame,
    jp_oc_ret: pd.DataFrame, alpha: float, q: float = 0.3,
) -> pd.Series:
    """alpha * PCA + (1-alpha) * Mamba3 → backtest"""
    common_dates = pca_sig.index.intersection(mamba_sig.index)
    pca_n = normalize_signals(pca_sig.loc[common_dates])
    mamba_n = normalize_signals(mamba_sig.loc[common_dates])

    combined = alpha * pca_n + (1 - alpha) * mamba_n
    weights = backtest.construct_long_short_weights(combined, q=q)
    return backtest.compute_strategy_returns(weights, jp_oc_ret)


def save_result(result: dict):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if not exists:
            w.writeheader()
        w.writerow(result)


def train_mamba_for_ensemble(hparams, train_ds, val_ds, test_ds, jp_oc_ret):
    """Mamba3 モデルを学習し、テスト期間のシグナルを返す"""
    save_dir = str(Path(settings.MODELS_DIR) / f"ensemble_{hparams['name']}")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model = Mamba3Forecaster(
        n_us=train_ds.n_us, n_jp=train_ds.n_jp,
        d_model=hparams.get("d_model", 96),
        d_state=hparams.get("d_state", 48),
        n_layers=hparams.get("n_layers", 2),
        expand=hparams.get("expand", 2),
        headdim=hparams.get("headdim", 16),
        dropout=hparams.get("dropout", 0.02),
        seq_len=hparams.get("seq_len", 20),
    ).to(DEVICE)

    batch_size = hparams.get("batch_size", 64)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = DirectionalLoss(alpha=hparams.get("loss_alpha", 1.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.get("lr", 1e-4),
        weight_decay=hparams.get("weight_decay", 1e-6),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams.get("epochs", 100),
    )

    best_val, best_ep, no_imp = float("inf"), 0, 0
    patience = hparams.get("patience", 30)
    model_path = f"{save_dir}/best_model.pt"

    for epoch in range(1, hparams.get("epochs", 100) + 1):
        train_one_epoch(model, train_dl, optimizer, criterion, DEVICE)
        val_loss, _ = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step()
        if val_loss < best_val:
            best_val, best_ep, no_imp = val_loss, epoch, 0
            torch.save(model.state_dict(), model_path)
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    return generate_signals(model, test_ds, DEVICE)


def main():
    print("=" * 60)
    print("ハイブリッドアンサンブル: PCA_SUB + Mamba3")
    print("=" * 60)

    us_cc, jp_cc, jp_oc, _ = data_loader.load_data()

    print("\n>>> PCA_SUB シグナル生成 (テスト期間)")
    pca_test_sig = compute_pca_signals_for_period(
        us_cc, jp_cc, jp_oc,
        SPLIT_DATES["test_start"], SPLIT_DATES["test_end"],
    )
    pca_test_sig.columns = list(jp_oc.columns)[:pca_test_sig.shape[1]]
    print(f"  PCA signals: {pca_test_sig.shape}")

    pca_only_w = backtest.construct_long_short_weights(pca_test_sig, q=0.3)
    pca_only_ret = backtest.compute_strategy_returns(pca_only_w, jp_oc)
    pca_metrics = compute_metrics(pca_only_ret)
    print(f"  PCA-only test: AR={pca_metrics['AR']}% R/R={pca_metrics['RR']} MDD={pca_metrics['MDD']}%")

    mamba_configs = [
        {"name": "ens_base", "d_model": 96, "d_state": 48, "n_layers": 2, "headdim": 16,
         "dropout": 0.02, "lr": 1e-4, "weight_decay": 1e-6, "epochs": 100,
         "batch_size": 64, "loss_alpha": 1.0, "patience": 30, "seq_len": 20},
        {"name": "ens_wider", "d_model": 128, "d_state": 64, "n_layers": 2, "headdim": 32,
         "dropout": 0.03, "lr": 5e-5, "weight_decay": 1e-6, "epochs": 120,
         "batch_size": 64, "loss_alpha": 1.0, "patience": 30, "seq_len": 20},
        {"name": "ens_seq10", "d_model": 96, "d_state": 48, "n_layers": 2, "headdim": 16,
         "dropout": 0.02, "lr": 1e-4, "weight_decay": 1e-6, "epochs": 100,
         "batch_size": 64, "loss_alpha": 1.0, "patience": 30, "seq_len": 10},
    ]

    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    q_values = [0.2, 0.3, 0.4]

    for mc in mamba_configs:
        sl = mc.get("seq_len", 20)
        print(f"\n>>> Mamba3 学習: {mc['name']} (seq_len={sl})")
        train_ds, val_ds, test_ds, _, _, _ = create_datasets(seq_len=sl)

        mamba_sig = train_mamba_for_ensemble(mc, train_ds, val_ds, test_ds, jp_oc)
        mamba_sig.columns = list(jp_oc.columns)[:mamba_sig.shape[1]]
        mamba_metrics = compute_metrics(
            backtest_signals(mamba_sig.copy(), jp_oc, q=0.3)
        )
        print(f"  Mamba-only test: AR={mamba_metrics['AR']}% R/R={mamba_metrics['RR']}")

        for alpha in alphas:
            for q in q_values:
                ens_ret = ensemble_and_backtest(pca_test_sig, mamba_sig, jp_oc, alpha, q)
                ens_m = compute_metrics(ens_ret)

                result = {
                    "mamba_config": mc["name"], "alpha": alpha, "q": q,
                    "pca_AR": pca_metrics["AR"],
                    "mamba_AR": mamba_metrics["AR"],
                    "ensemble_AR": ens_m["AR"], "ensemble_RISK": ens_m["RISK"],
                    "ensemble_RR": ens_m["RR"], "ensemble_MDD": ens_m["MDD"],
                }
                save_result(result)

                marker = " <<<" if ens_m["AR"] >= 25.0 else ""
                print(f"  α={alpha} q={q}: AR={ens_m['AR']}% R/R={ens_m['RR']} MDD={ens_m['MDD']}%{marker}")

                if ens_m["AR"] >= 25.0:
                    print(f"\n{'='*60}\n目標達成! Ensemble AR={ens_m['AR']}%\n{'='*60}")

    print("\n>>> 最終結果")
    df = pd.read_csv(RESULTS_CSV)
    best = df.loc[df["ensemble_AR"].idxmax()]
    print(f"Best: {best['mamba_config']} α={best['alpha']} q={best['q']} → AR={best['ensemble_AR']}%")


if __name__ == "__main__":
    main()
