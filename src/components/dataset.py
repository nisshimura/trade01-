"""
時系列データセット: Mamba3 学習用
train / val / test 分割（金融時系列ルール: 未来リーク防止）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.components import data_loader

SPLIT_DATES = {
    "train_start": "2010-03-01", "train_end": "2019-12-31",
    "val_start": "2020-01-01", "val_end": "2021-12-31",
    "test_start": "2022-01-01", "test_end": "2025-12-31",
}


class SectorLeadLagDataset(Dataset):
    def __init__(self, us_cc_ret, jp_cc_ret, jp_oc_ret, start_date, end_date, seq_len=60):
        self.seq_len = seq_len
        mask = (us_cc_ret.index >= start_date) & (us_cc_ret.index <= end_date)
        dates_in_range = us_cc_ret.index[mask]

        all_dates = us_cc_ret.index
        start_idx = all_dates.get_loc(dates_in_range[0])
        need_from = max(0, start_idx - seq_len)
        expanded_mask = (us_cc_ret.index >= all_dates[need_from]) & (us_cc_ret.index <= end_date)

        self.us_vals = us_cc_ret.loc[expanded_mask].fillna(0.0).values.astype(np.float32)
        self.jp_cc_vals = jp_cc_ret.loc[expanded_mask].fillna(0.0).values.astype(np.float32)
        self.jp_oc_vals = jp_oc_ret.loc[expanded_mask].fillna(0.0).values.astype(np.float32)
        self.dates = us_cc_ret.loc[expanded_mask].index
        self.offset = start_idx - need_from

        self.valid_indices = [i for i in range(self.offset, len(self.dates) - 1) if i >= seq_len]
        self.n_us = self.us_vals.shape[1]
        self.n_jp = self.jp_cc_vals.shape[1]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        return (
            torch.from_numpy(self.us_vals[t - self.seq_len : t]),
            torch.from_numpy(self.jp_cc_vals[t - self.seq_len : t]),
            torch.from_numpy(self.jp_oc_vals[t]),
        )

    def get_dates(self):
        return [self.dates[i] for i in self.valid_indices]


def create_datasets(seq_len=60):
    us_cc_ret, jp_cc_ret, jp_oc_ret, _ = data_loader.load_data()
    datasets = {}
    for split in ("train", "val", "test"):
        ds = SectorLeadLagDataset(
            us_cc_ret, jp_cc_ret, jp_oc_ret,
            SPLIT_DATES[f"{split}_start"], SPLIT_DATES[f"{split}_end"], seq_len=seq_len,
        )
        datasets[split] = ds
        print(f"{split.capitalize():5s}: {len(ds)} samples ({SPLIT_DATES[f'{split}_start']} ~ {SPLIT_DATES[f'{split}_end']})")
    return datasets["train"], datasets["val"], datasets["test"], us_cc_ret, jp_cc_ret, jp_oc_ret
