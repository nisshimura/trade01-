"""
バックテストエンジン: ロング・ショートポートフォリオ構築と評価指標
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings


def construct_long_short_weights(signals: pd.DataFrame, q: float = settings.QUANTILE_Q) -> pd.DataFrame:
    weights_arr = np.zeros((len(signals), len(signals.columns)))
    col_idx = {c: i for i, c in enumerate(signals.columns)}

    for t_idx in range(len(signals)):
        valid = signals.iloc[t_idx].dropna()
        if len(valid) < 3:
            continue
        n_q = max(1, int(np.floor(len(valid) * q)))
        s = valid.sort_values()
        for t in s.index[-n_q:]:
            weights_arr[t_idx, col_idx[t]] = 1.0 / n_q
        for t in s.index[:n_q]:
            weights_arr[t_idx, col_idx[t]] = -1.0 / n_q

    return pd.DataFrame(weights_arr, index=signals.index, columns=signals.columns)


def construct_double_sort_weights(signal_mom: pd.DataFrame, signal_pca: pd.DataFrame) -> pd.DataFrame:
    cols = signal_mom.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    w = np.zeros((len(signal_mom), len(cols)))

    for t_idx in range(len(signal_mom)):
        mr = signal_mom.iloc[t_idx].dropna()
        pr = signal_pca.iloc[t_idx].dropna()
        common = mr.index.intersection(pr.index)
        if len(common) < 4:
            continue
        mm, pm = mr[common].median(), pr[common].median()
        long_set = common[(mr[common] >= mm) & (pr[common] >= pm)]
        short_set = common[(mr[common] < mm) & (pr[common] < pm)]
        for t in long_set:
            w[t_idx, col_idx[t]] = 1.0 / max(len(long_set), 1)
        for t in short_set:
            w[t_idx, col_idx[t]] = -1.0 / max(len(short_set), 1)

    return pd.DataFrame(w, index=signal_mom.index, columns=cols)


def compute_strategy_returns(weights: pd.DataFrame, jp_oc_ret: pd.DataFrame) -> pd.Series:
    sw = weights.shift(1)
    common_dates = sw.index.intersection(jp_oc_ret.index)
    common_cols = sw.columns.intersection(jp_oc_ret.columns)
    ret = (sw.loc[common_dates, common_cols].fillna(0) * jp_oc_ret.loc[common_dates, common_cols].fillna(0)).sum(axis=1)
    ret.name = "strategy_return"
    return ret


def annualized_return(daily_returns: pd.Series) -> float:
    return daily_returns.mean() * 252


def annualized_risk(daily_returns: pd.Series) -> float:
    return daily_returns.std(ddof=1) * np.sqrt(252)


def risk_return_ratio(daily_returns: pd.Series) -> float:
    risk = annualized_risk(daily_returns)
    return annualized_return(daily_returns) / risk if risk > 1e-12 else 0.0


def max_drawdown(daily_returns: pd.Series) -> float:
    cum = (1 + daily_returns).cumprod()
    return (cum / cum.cummax() - 1).min()


def compute_performance_table(strategy_returns: dict[str, pd.Series], eval_start: str | None = None) -> pd.DataFrame:
    rows = []
    for name, ret in strategy_returns.items():
        r = ret.loc[eval_start:].dropna() if eval_start else ret.dropna()
        if len(r) == 0:
            continue
        rows.append({
            "Strategy": name,
            "AR (%)": round(annualized_return(r) * 100, 2),
            "RISK (%)": round(annualized_risk(r) * 100, 2),
            "R/R": round(risk_return_ratio(r), 2),
            "MDD (%)": round(abs(max_drawdown(r)) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def compute_cumulative_returns(strategy_returns: dict[str, pd.Series], eval_start: str | None = None) -> pd.DataFrame:
    out = {}
    for name, ret in strategy_returns.items():
        r = ret.loc[eval_start:].dropna() if eval_start else ret.dropna()
        out[name] = (1 + r).cumprod()
    return pd.DataFrame(out)
