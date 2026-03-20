"""
性能改善サイクル: ハイパーパラメータを段階的に調整し、年率25%を目標
結果は results/results.csv に行ごとに記録（1行 = 1サイクル = 1モデル）
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.components.dataset import create_datasets
from src.pipeline.train import train_and_evaluate
from src.config import settings

RESULTS_CSV = Path(settings.RESULTS_DIR) / "results.csv"
TARGET_AR = 25.0

HPARAM_SCHEDULE = [
    {"name": "c1_baseline", "d_model": 64, "d_state": 32, "n_layers": 2, "expand": 2, "headdim": 16,
     "dropout": 0.05, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 40, "batch_size": 64, "loss_alpha": 0.3,
     "q": 0.3, "patience": 10, "seq_len": 60},
    {"name": "c2_directional", "d_model": 64, "d_state": 32, "n_layers": 2, "expand": 2, "headdim": 16,
     "dropout": 0.05, "lr": 5e-4, "weight_decay": 5e-5, "epochs": 50, "batch_size": 64, "loss_alpha": 0.7,
     "q": 0.3, "patience": 12, "seq_len": 60},
    {"name": "c3_wider", "d_model": 128, "d_state": 64, "n_layers": 2, "expand": 2, "headdim": 32,
     "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4, "epochs": 50, "batch_size": 64, "loss_alpha": 0.5,
     "q": 0.3, "patience": 12, "seq_len": 60},
    {"name": "c4_deeper", "d_model": 128, "d_state": 64, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.1, "lr": 3e-4, "weight_decay": 5e-5, "epochs": 60, "batch_size": 32, "loss_alpha": 0.6,
     "q": 0.3, "patience": 15, "seq_len": 60},
    {"name": "c5_high_dir", "d_model": 128, "d_state": 64, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.08, "lr": 2e-4, "weight_decay": 1e-5, "epochs": 60, "batch_size": 32, "loss_alpha": 0.8,
     "q": 0.3, "patience": 15, "seq_len": 60},
    {"name": "c6_q20", "d_model": 128, "d_state": 64, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.08, "lr": 2e-4, "weight_decay": 1e-5, "epochs": 60, "batch_size": 32, "loss_alpha": 0.6,
     "q": 0.2, "patience": 15, "seq_len": 60},
    {"name": "c7_big", "d_model": 192, "d_state": 96, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.1, "lr": 2e-4, "weight_decay": 5e-5, "epochs": 80, "batch_size": 32, "loss_alpha": 0.6,
     "q": 0.3, "patience": 20, "seq_len": 60},
    {"name": "c8_pure_dir", "d_model": 192, "d_state": 96, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.05, "lr": 1e-4, "weight_decay": 1e-5, "epochs": 80, "batch_size": 32, "loss_alpha": 0.9,
     "q": 0.3, "patience": 20, "seq_len": 60},
    # --- Phase 2: c8 insights - pure directional, low lr, low dropout ---
    {"name": "c9_full_dir", "d_model": 192, "d_state": 96, "n_layers": 3, "expand": 2, "headdim": 32,
     "dropout": 0.03, "lr": 5e-5, "weight_decay": 1e-5, "epochs": 100, "batch_size": 32, "loss_alpha": 1.0,
     "q": 0.3, "patience": 25, "seq_len": 60},
    {"name": "c10_short_seq", "d_model": 128, "d_state": 64, "n_layers": 2, "expand": 2, "headdim": 32,
     "dropout": 0.03, "lr": 1e-4, "weight_decay": 1e-5, "epochs": 80, "batch_size": 64, "loss_alpha": 0.95,
     "q": 0.3, "patience": 20, "seq_len": 30},
    {"name": "c11_small_dir_q40", "d_model": 64, "d_state": 32, "n_layers": 2, "expand": 2, "headdim": 16,
     "dropout": 0.02, "lr": 5e-5, "weight_decay": 1e-6, "epochs": 100, "batch_size": 64, "loss_alpha": 0.95,
     "q": 0.4, "patience": 25, "seq_len": 60},
    {"name": "c12_wide_shallow", "d_model": 256, "d_state": 64, "n_layers": 1, "expand": 2, "headdim": 32,
     "dropout": 0.05, "lr": 1e-4, "weight_decay": 1e-5, "epochs": 80, "batch_size": 32, "loss_alpha": 0.95,
     "q": 0.3, "patience": 20, "seq_len": 60},
    {"name": "c13_short20_dir", "d_model": 96, "d_state": 48, "n_layers": 2, "expand": 2, "headdim": 16,
     "dropout": 0.02, "lr": 1e-4, "weight_decay": 1e-6, "epochs": 100, "batch_size": 64, "loss_alpha": 1.0,
     "q": 0.3, "patience": 25, "seq_len": 20},
    {"name": "c14_tiny_overfit", "d_model": 32, "d_state": 16, "n_layers": 1, "expand": 2, "headdim": 16,
     "dropout": 0.0, "lr": 2e-4, "weight_decay": 0, "epochs": 200, "batch_size": 128, "loss_alpha": 1.0,
     "q": 0.3, "patience": 40, "seq_len": 60},
    {"name": "c15_mid_q20_dir", "d_model": 128, "d_state": 64, "n_layers": 2, "expand": 2, "headdim": 32,
     "dropout": 0.03, "lr": 5e-5, "weight_decay": 1e-6, "epochs": 100, "batch_size": 32, "loss_alpha": 1.0,
     "q": 0.2, "patience": 25, "seq_len": 40},
]


def load_existing_results():
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def save_result(result: dict):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result.keys())
        if not exists:
            w.writeheader()
        w.writerow(result)


def main():
    print("=" * 60)
    print("性能改善サイクル: Mamba3 Lead-Lag Forecaster")
    print(f"目標: test AR >= {TARGET_AR}%")
    print("=" * 60)

    existing = load_existing_results()
    completed = set()
    if not existing.empty and "hp_name" in existing.columns:
        completed = set(existing["hp_name"].values)
        best = existing["test_AR"].max() if "test_AR" in existing.columns else 0
        print(f"既存結果: {len(existing)} サイクル, best test AR = {best}%")
        if best >= TARGET_AR:
            print(f"既に目標達成済み (test AR = {best}%) !")
            return

    remaining = [h for h in HPARAM_SCHEDULE if h["name"] not in completed]
    if not remaining:
        print("全スケジュール完了。HPARAM_SCHEDULE を追加してください。")
        return

    first_sl = remaining[0].get("seq_len", 60)
    print(f"\n>>> データ読み込み (seq_len={first_sl})")
    train_ds, val_ds, test_ds, us_cc, jp_cc, jp_oc = create_datasets(seq_len=first_sl)
    cur_sl = first_sl

    for ci, hp in enumerate(remaining):
        num = len(completed) + ci + 1
        name = hp["name"]
        print(f"\n{'='*60}\nサイクル {num}: {name}\n{'='*60}")
        print(f"  Params: {hp}")

        sl = hp.get("seq_len", 60)
        if sl != cur_sl:
            train_ds, val_ds, test_ds, us_cc, jp_cc, jp_oc = create_datasets(seq_len=sl)
            cur_sl = sl

        result = train_and_evaluate(
            hparams=hp, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            jp_oc_ret=jp_oc,
            save_dir=str(Path(settings.MODELS_DIR) / f"cycle_{num}_{name}"),
        )
        result["cycle"] = num
        save_result(result)

        print(f"\n  --- サイクル {num} 結果 ---")
        print(f"  Val:  AR={result['val_AR']}% R/R={result['val_RR']} MDD={result['val_MDD']}%")
        print(f"  Test: AR={result['test_AR']}% R/R={result['test_RR']} MDD={result['test_MDD']}%")

        if result["test_AR"] >= TARGET_AR:
            print(f"\n{'='*60}\n目標達成! test AR = {result['test_AR']}% >= {TARGET_AR}%\n{'='*60}")
            return

    print(f"\n全 {len(HPARAM_SCHEDULE)} サイクル完了。")
    final = pd.read_csv(RESULTS_CSV)
    best_row = final.loc[final["test_AR"].idxmax()]
    print(f"Best test AR: {best_row['test_AR']}% (cycle {best_row.get('cycle', '?')})")
    if best_row["test_AR"] < TARGET_AR:
        print(f"目標 {TARGET_AR}% は未達。追加探索が必要です。")


if __name__ == "__main__":
    main()
