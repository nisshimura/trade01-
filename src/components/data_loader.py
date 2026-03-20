"""
データローダー: Stooq をデフォルトデータソースとして使用
Yahoo Finance は使用しない（価格データの不整合、分割/配当調整の矛盾、TZ問題のため）
"""

from __future__ import annotations

import io
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import settings

STOOQ_BASE = "https://stooq.com/q/d/l/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _stooq_ticker_us(ticker: str) -> str:
    return f"{ticker}.US"


def _stooq_ticker_jp(ticker: str) -> str:
    return f"{ticker}.JP"


def _date_to_stooq(date_str: str) -> str:
    return date_str.replace("-", "")


def fetch_single_stooq(stooq_ticker: str, start: str, end: str) -> pd.DataFrame | None:
    params = {
        "s": stooq_ticker.lower(),
        "d1": _date_to_stooq(start),
        "d2": _date_to_stooq(end),
        "i": "d",
    }
    try:
        resp = requests.get(STOOQ_BASE, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content = resp.text
        if "No data" in content or len(content.strip().split("\n")) < 2:
            print(f"  [WARN] {stooq_ticker}: データが空またはNo data")
            return None
        df = pd.read_csv(io.StringIO(content), parse_dates=["Date"], index_col="Date")
        df = df.sort_index()
        if df.empty:
            print(f"  [WARN] {stooq_ticker}: パース後データが空")
            return None
        return df
    except Exception as e:
        print(f"  [ERROR] {stooq_ticker}: {e}")
        return None


def fetch_all_stooq(
    us_tickers: list[str],
    jp_tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"stooq_{start}_{end}.pkl"
        if cache_file.exists():
            print(f"キャッシュから読み込み: {cache_file}")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            return cached["us"], cached["jp"]

    us_data: dict[str, pd.DataFrame] = {}
    jp_data: dict[str, pd.DataFrame] = {}

    print("=== 米国 ETF データ取得 (Stooq) ===")
    for t in us_tickers:
        stooq_t = _stooq_ticker_us(t)
        print(f"  取得中: {stooq_t}")
        df = fetch_single_stooq(stooq_t, start, end)
        if df is not None:
            us_data[t] = df
        time.sleep(1.0)

    print("=== 日本 ETF データ取得 (Stooq) ===")
    for t in jp_tickers:
        stooq_t = _stooq_ticker_jp(t)
        print(f"  取得中: {stooq_t}")
        df = fetch_single_stooq(stooq_t, start, end)
        if df is not None:
            jp_data[t] = df
        time.sleep(1.0)

    if cache_dir:
        with open(cache_file, "wb") as f:
            pickle.dump({"us": us_data, "jp": jp_data}, f)
        print(f"キャッシュ保存: {cache_file}")

    return us_data, jp_data


def build_price_panel(data: dict[str, pd.DataFrame], price_col: str = "Close") -> pd.DataFrame:
    frames = {t: df[price_col] for t, df in data.items() if price_col in df.columns}
    panel = pd.DataFrame(frames)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def build_ohlc_panels(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_price_panel(data, "Open"), build_price_panel(data, "Close")


def compute_cc_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change()


def compute_oc_returns(open_panel: pd.DataFrame, close_panel: pd.DataFrame) -> pd.DataFrame:
    return close_panel / open_panel - 1


def align_trading_days(
    us_close: pd.DataFrame, jp_open: pd.DataFrame, jp_close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = us_close.index.intersection(jp_close.index).sort_values()
    return us_close.loc[common], jp_open.loc[common], jp_close.loc[common]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    us_data, jp_data = fetch_all_stooq(
        settings.US_TICKERS, settings.JP_TICKERS,
        settings.SAMPLE_START, settings.SAMPLE_END,
        cache_dir=settings.CACHE_DIR,
    )

    missing_us = set(settings.US_TICKERS) - set(us_data.keys())
    missing_jp = set(settings.JP_TICKERS) - set(jp_data.keys())
    if missing_us:
        print(f"[WARN] 米国: 取得失敗 {missing_us}")
    if missing_jp:
        print(f"[WARN] 日本: 取得失敗 {missing_jp}")

    us_close = build_price_panel(us_data, "Close")
    jp_open_p, jp_close_p = build_ohlc_panels(jp_data)
    us_close, jp_open_p, jp_close_p = align_trading_days(us_close, jp_open_p, jp_close_p)

    us_cc_ret = compute_cc_returns(us_close)
    jp_cc_ret = compute_cc_returns(jp_close_p)
    jp_oc_ret = compute_oc_returns(jp_open_p, jp_close_p)

    us_cc_ret = us_cc_ret.dropna(how="all")
    jp_cc_ret = jp_cc_ret.loc[us_cc_ret.index]
    jp_oc_ret = jp_oc_ret.loc[us_cc_ret.index]

    print(f"\n=== データ概要 ===")
    print(f"共通取引日数: {len(us_cc_ret)}, 米国: {us_cc_ret.shape[1]}, 日本: {jp_cc_ret.shape[1]}")
    print(f"期間: {us_cc_ret.index[0].date()} ~ {us_cc_ret.index[-1].date()}")

    return us_cc_ret, jp_cc_ret, jp_oc_ret, jp_close_p
