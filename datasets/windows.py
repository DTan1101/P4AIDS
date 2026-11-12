"""Cắt cửa sổ trượt 24 giờ -> nhãn Rain/Snow tại t+1 (Mục 1.2, 4.2 của báo cáo)."""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from datasets.szeged import FEATURE_COLUMNS

logger = logging.getLogger("p4aids")


class WindowedData(NamedTuple):
    """Kết quả cắt cửa sổ.

    ``X`` có NaN tại các vị trí thiếu dữ liệu *tự nhiên* (xem
    :func:`datasets.szeged.clean`) -- đây là phần thiếu dữ liệu thật của bộ dữ
    liệu, độc lập với mặt nạ thiếu dữ liệu *giả lập* ở Mục 4.3.
    """

    X: np.ndarray            # (N, window, n_features), float32, có thể chứa NaN
    y: np.ndarray            # (N,), int64, nhãn tại t+1
    end_time: pd.DatetimeIndex  # (N,), timestamp của nhãn (t+1) cho mỗi cửa sổ


def build_windows(df: pd.DataFrame, window: int = 24,
                   feature_cols: Sequence[str] = FEATURE_COLUMNS) -> WindowedData:
    """Cắt cửa sổ trượt bước 1 giờ trên lưới giờ liên tục đã làm sạch.

    Với mỗi vị trí ``i`` (``i >= window-1`` và ``i+1 < len(df)``): đặc trưng là
    ``df.iloc[i-window+1 : i+1]``, nhãn là ``df.iloc[i+1]["label"]``. Cửa sổ bị
    loại nếu nhãn tại t+1 không hợp lệ (NaN); đặc trưng vẫn được giữ kể cả khi
    chứa NaN do thiếu dữ liệu tự nhiên.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame phải được sắp xếp theo thời gian tăng dần trước khi cắt cửa sổ.")

    feature_values = df[list(feature_cols)].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()
    n_rows = len(df)

    windows, window_labels, end_times = [], [], []
    for i in range(window - 1, n_rows - 1):
        label = labels[i + 1]
        if np.isnan(label):
            continue
        windows.append(feature_values[i - window + 1: i + 1])
        window_labels.append(int(label))
        end_times.append(df.index[i + 1])

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(window_labels, dtype=np.int64)
    end_time = pd.DatetimeIndex(end_times)
    logger.info(
        "Cắt cửa sổ: %d mẫu (window=%d, %d đặc trưng), %d bị loại vì thiếu nhãn t+1.",
        len(y), window, len(feature_cols), (n_rows - window) - len(y),
    )
    return WindowedData(X=X, y=y, end_time=end_time)
