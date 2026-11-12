"""Dataset PyTorch cho nhánh Deep Learning: value/mask/delta (Mục 4.6.1 báo cáo)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def forward_fill(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Điền giá trị thiếu bằng forward-fill theo thời gian; đầu cửa sổ còn thiếu
    (chưa có quan trắc nào) được điền 0. Dùng cho GRU-Fill (Mục 4.6.2).
    """
    n_samples, window, n_features = X.shape
    out = np.where(mask, X, np.nan).astype(np.float32)
    for t in range(1, window):
        missing = ~mask[:, t, :]
        out[:, t, :] = np.where(missing, out[:, t - 1, :], out[:, t, :])
    return np.nan_to_num(out, nan=0.0)


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """Delta kiểu GRU-D (Che et al., 2018): thời gian (số bước) từ lần quan trắc
    hợp lệ gần nhất, với bước thời gian đều nhau (1 giờ)."""
    n_samples, window, n_features = mask.shape
    delta = np.zeros((n_samples, window, n_features), dtype=np.float32)
    for t in range(1, window):
        prev_observed = mask[:, t - 1, :]
        delta[:, t, :] = np.where(prev_observed, 1.0, 1.0 + delta[:, t - 1, :])
    return delta


def compute_channel_stats(X_filled: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std theo kênh, ước lượng **chỉ trên tập train** (chống rò rỉ, Mục 2.2)."""
    train_values = X_filled[train_idx]
    mean = train_values.mean(axis=(0, 1))
    std = train_values.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X_filled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X_filled - mean) / std).astype(np.float32)


class WeatherWindowDataset(Dataset):
    """Trả về ``(value, mask, delta, label)`` -- value đã forward-fill + chuẩn hoá."""

    def __init__(self, value: np.ndarray, mask: np.ndarray, delta: np.ndarray,
                 y: Optional[np.ndarray] = None) -> None:
        self.value = torch.from_numpy(value.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))
        self.delta = torch.from_numpy(delta.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64)) if y is not None else None

    def __len__(self) -> int:
        return self.value.shape[0]

    def __getitem__(self, index: int):
        if self.y is not None:
            return self.value[index], self.mask[index], self.delta[index], self.y[index]
        return self.value[index], self.mask[index], self.delta[index]


def prepare_dl_arrays(X: np.ndarray, mask: np.ndarray, train_idx: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline đầy đủ: forward-fill -> chuẩn hoá (stats từ train) -> delta.

    Trả về ``(value_norm, mask_float, delta, mean, std)``.
    """
    filled = forward_fill(X, mask)
    mean, std = compute_channel_stats(filled, train_idx)
    value_norm = normalize(filled, mean, std)
    delta = compute_delta(mask)
    return value_norm, mask.astype(np.float32), delta, mean, std
