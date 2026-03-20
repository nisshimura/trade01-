"""
メインスクリプト: 部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略
Nakagawa et al. (2026) SIG-FIN-036-13

実行: uv run python -m src.main
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src.config import settings
from src.components import data_loader, pca_model, backtest


def main():
    print("=" * 60)
    print("部分空間正則化付きPCA 日米業種リードラグ投資戦略")
    print("Nakagawa et al. (2026) SIG-FIN-036-13")
    print(f"データソース: {settings.DATA_SOURCE} (Yahoo Finance 不使用)")
    print("=" * 60)

    print("\n>>> Step 1: データ読み込み")
    t0 = time.time()
    us_cc_ret, jp_cc_ret, jp_oc_ret, _ = data_loader.load_data()
    print(f"    所要時間: {time.time() - t0:.1f}s")

    us_tickers = list(us_cc_ret.columns)
    jp_tickers = list(jp_cc_ret.columns)

    print("\n>>> Step 2: 事前部分空間 V0 構築")
    V0 = pca_model.build_prior_subspace(
        us_tickers, jp_tickers,
        settings.US_CYCLICAL, settings.US_DEFENSIVE,
        settings.JP_CYCLICAL, settings.JP_DEFENSIVE,
    )
    print(f"    V0 shape: {V0.shape}")
    print(f"    直交性: {np.allclose(V0.T @ V0, np.eye(V0.shape[1]))}")

    print("\n>>> Step 3: C_full → C0")
    C_full = pca_model.compute_cfull(us_cc_ret, jp_cc_ret)
    C0 = pca_model.compute_target_matrix(V0, C_full)

    eval_start = settings.CFULL_END

    print("\n>>> Step 4: シグナル計算")
    t0 = time.time()
    sig_pca_sub = pca_model.regularized_pca_signal(us_cc_ret, jp_cc_ret, V0, C0)
    print(f"    PCA_SUB: {sig_pca_sub.dropna(how='all').shape[0]} 日 ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sig_pca_plain = pca_model.plain_pca_signal(us_cc_ret, jp_cc_ret)
    print(f"    PCA_PLAIN: ({time.time() - t0:.1f}s)")

    sig_mom = pca_model.momentum_signal(jp_cc_ret)

    print("\n>>> Step 5-6: ポートフォリオ → 戦略リターン")
    strategies = {
        "MOM": backtest.compute_strategy_returns(
            backtest.construct_long_short_weights(sig_mom), jp_oc_ret),
        "PCA_PLAIN": backtest.compute_strategy_returns(
            backtest.construct_long_short_weights(sig_pca_plain), jp_oc_ret),
        "PCA_SUB": backtest.compute_strategy_returns(
            backtest.construct_long_short_weights(sig_pca_sub), jp_oc_ret),
        "DOUBLE": backtest.compute_strategy_returns(
            backtest.construct_double_sort_weights(sig_mom, sig_pca_sub), jp_oc_ret),
    }

    print(f"\n>>> Step 7: パフォーマンス (評価: {eval_start}~)")
    perf = backtest.compute_performance_table(strategies, eval_start=eval_start)
    print("\n" + perf.to_string())

    print("\n--- 論文 Table 2 (参考) ---")
    print("MOM:5.63/10.59/0.53/16.97  PCA_PLAIN:6.24/9.94/0.62/23.65  PCA_SUB:23.79/10.70/2.22/9.58  DOUBLE:18.86/11.16/1.69/12.10")

    print("\n>>> Step 8: 累積リターンプロット")
    cum = backtest.compute_cumulative_returns(strategies, eval_start=eval_start)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"MOM": "#888", "PCA_PLAIN": "#4ECDC4", "PCA_SUB": "#E63946", "DOUBLE": "#457B9D"}
    for c in cum.columns:
        ax.plot(cum.index, cum[c], label=c, color=colors.get(c), linewidth=1.5)
    ax.set_title("Cumulative Returns: Lead-Lag Strategies (JP sectors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out = settings.RESULTS_DIR + "/cumulative_returns.png"
    fig.savefig(out, dpi=150)
    print(f"    保存: {out}")
    print("\n完了")


if __name__ == "__main__":
    main()
