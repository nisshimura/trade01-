"""
部分空間正則化付き PCA による日米業種リードラグ・シグナル
Nakagawa et al. (2026) SIG-FIN-036-13
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from src.config import settings


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    n, k = vectors.shape
    Q = np.zeros_like(vectors, dtype=float)
    for j in range(k):
        v = vectors[:, j].astype(float)
        for i in range(j):
            v -= np.dot(Q[:, i], v) * Q[:, i]
        norm = np.linalg.norm(v)
        Q[:, j] = v / norm if norm > 1e-12 else 0.0
    return Q


def build_prior_subspace(
    us_tickers: list[str], jp_tickers: list[str],
    us_cyclical: list[str], us_defensive: list[str],
    jp_cyclical: list[str], jp_defensive: list[str],
) -> np.ndarray:
    n_us, n_jp = len(us_tickers), len(jp_tickers)
    n_total = n_us + n_jp

    v1_raw = np.ones(n_total)
    v2_raw = np.zeros(n_total)
    v2_raw[:n_us] = 1.0
    v2_raw[n_us:] = -1.0

    v3_raw = np.zeros(n_total)
    for i, t in enumerate(us_tickers + jp_tickers):
        if t in us_cyclical or t in jp_cyclical:
            v3_raw[i] = 1.0
        elif t in us_defensive or t in jp_defensive:
            v3_raw[i] = -1.0

    return gram_schmidt(np.column_stack([v1_raw, v2_raw, v3_raw]))


def compute_target_matrix(V0: np.ndarray, C_full: np.ndarray) -> np.ndarray:
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T
    delta = np.diag(C0_raw).copy()
    delta[delta < 1e-12] = 1e-12
    d_inv = 1.0 / np.sqrt(delta)
    C0 = C0_raw * np.outer(d_inv, d_inv)
    np.fill_diagonal(C0, 1.0)
    return C0


def compute_rolling_correlation(Z: np.ndarray) -> np.ndarray:
    L = Z.shape[0]
    C = Z.T @ Z / L
    d = np.sqrt(np.diag(C))
    d[d < 1e-12] = 1e-12
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def regularized_pca_signal(
    us_cc_ret: pd.DataFrame, jp_cc_ret: pd.DataFrame,
    V0: np.ndarray, C0: np.ndarray,
    lam: float = settings.LAMBDA_REG,
    K: int = settings.K_COMPONENTS,
    L: int = settings.WINDOW_LENGTH,
) -> pd.DataFrame:
    us_tickers = list(us_cc_ret.columns)
    jp_tickers = list(jp_cc_ret.columns)
    n_us = len(us_tickers)
    dates = us_cc_ret.index

    joint = pd.concat([us_cc_ret, jp_cc_ret], axis=1)
    joint.columns = us_tickers + jp_tickers
    jf = joint.fillna(0.0)

    signals_arr = np.full((len(dates), len(jp_tickers)), np.nan)

    for t_idx in range(L, len(dates)):
        wv = jf.iloc[t_idx - L : t_idx].values
        mu = wv.mean(axis=0)
        sigma = wv.std(axis=0, ddof=0)
        sigma[sigma < 1e-12] = 1e-12
        Z = (wv - mu) / sigma

        C_reg = (1 - lam) * compute_rolling_correlation(Z) + lam * C0
        np.fill_diagonal(C_reg, 1.0)
        if np.any(~np.isfinite(C_reg)):
            continue

        evals, evecs = eigh(C_reg)
        V_K = evecs[:, np.argsort(evals)[::-1][:K]]

        us_today = jf.iloc[t_idx, :n_us].values
        z_us = (us_today - mu[:n_us]) / sigma[:n_us]
        if np.any(~np.isfinite(z_us)):
            continue

        signals_arr[t_idx] = V_K[n_us:, :] @ (V_K[:n_us, :].T @ z_us)

    return pd.DataFrame(signals_arr, index=dates, columns=jp_tickers)


def momentum_signal(jp_cc_ret: pd.DataFrame, L: int = settings.WINDOW_LENGTH) -> pd.DataFrame:
    return jp_cc_ret.rolling(window=L, min_periods=L).mean()


def plain_pca_signal(
    us_cc_ret: pd.DataFrame, jp_cc_ret: pd.DataFrame,
    K: int = settings.K_COMPONENTS, L: int = settings.WINDOW_LENGTH,
) -> pd.DataFrame:
    us_tickers = list(us_cc_ret.columns)
    jp_tickers = list(jp_cc_ret.columns)
    n_us = len(us_tickers)
    dates = us_cc_ret.index

    joint = pd.concat([us_cc_ret, jp_cc_ret], axis=1)
    joint.columns = us_tickers + jp_tickers
    jf = joint.fillna(0.0)

    signals_arr = np.full((len(dates), len(jp_tickers)), np.nan)

    for t_idx in range(L, len(dates)):
        wv = jf.iloc[t_idx - L : t_idx].values
        mu = wv.mean(axis=0)
        sigma = wv.std(axis=0, ddof=0)
        sigma[sigma < 1e-12] = 1e-12
        Z = (wv - mu) / sigma
        C_t = compute_rolling_correlation(Z)
        if np.any(~np.isfinite(C_t)):
            continue

        evals, evecs = eigh(C_t)
        V_K = evecs[:, np.argsort(evals)[::-1][:K]]

        us_today = jf.iloc[t_idx, :n_us].values
        z_us = (us_today - mu[:n_us]) / sigma[:n_us]
        if np.any(~np.isfinite(z_us)):
            continue

        signals_arr[t_idx] = V_K[n_us:, :] @ (V_K[:n_us, :].T @ z_us)

    return pd.DataFrame(signals_arr, index=dates, columns=jp_tickers)


def compute_cfull(
    us_cc_ret: pd.DataFrame, jp_cc_ret: pd.DataFrame,
    start: str = settings.CFULL_START, end: str = settings.CFULL_END,
) -> np.ndarray:
    joint = pd.concat([us_cc_ret, jp_cc_ret], axis=1).loc[start:end].fillna(0.0)
    mu = joint.mean(axis=0).values.copy()
    sigma = joint.std(axis=0, ddof=0).values.copy()
    sigma[sigma < 1e-12] = 1e-12
    return compute_rolling_correlation((joint.values - mu) / sigma)
