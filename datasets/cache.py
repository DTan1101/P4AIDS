"""Lưu/nạp dữ liệu đã cắt cửa sổ + chia tập, dùng chung giữa mọi script ``tools/``."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd


def save_prepared_data(path: str, X: np.ndarray, y: np.ndarray, end_time: pd.DatetimeIndex,
                        train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, X=X, y=y, end_time=end_time.values.astype("datetime64[ns]"),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )


def load_prepared_data(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã chuẩn bị tại '{path}'. Chạy 'python tools/prepare_data.py' trước."
        )
    npz = np.load(path, allow_pickle=False)
    return {
        "X": npz["X"], "y": npz["y"],
        "end_time": pd.DatetimeIndex(npz["end_time"]),
        "train_idx": npz["train_idx"], "val_idx": npz["val_idx"], "test_idx": npz["test_idx"],
    }
