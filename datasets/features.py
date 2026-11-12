"""Trích xuất đặc trưng cho nhánh Classic ML (Mục 4.4.1 báo cáo).

- :func:`lite_stat_temp` -- đặc trưng thống kê/nhiệt độ rút gọn tự thiết kế,
  luôn khả dụng (chỉ cần numpy).
- :func:`mask_delta_features` -- đặc trưng từ mặt nạ thiếu dữ liệu (tỷ lệ
  thiếu, độ dài đoạn thiếu dài nhất, độ "cũ" của quan trắc gần nhất).
- :func:`catch22_features` / :func:`tsfresh_features` -- bộ đặc trưng mở rộng,
  tuỳ chọn (Mục 4.4.1), đòi hỏi cài thêm ``pycatch22``/``tsfresh``. Hai hàm này
  CHƯA được kiểm thử với các thư viện thật (không cài trong lần refactor này vì
  chúng nặng và chỉ là bổ sung tuỳ chọn) -- hãy kiểm tra lại API khi bật dùng.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np

from datasets.szeged import FEATURE_COLUMNS

STAT_NAMES: Tuple[str, ...] = ("mean", "std", "min", "max", "slope")
TEMPERATURE_CHANNEL = 0  # vị trí của "Temperature (C)" trong FEATURE_COLUMNS


def _nan_slope(values: np.ndarray) -> float:
    """Hệ số góc hồi quy tuyến tính theo thời gian, chỉ dùng các điểm hợp lệ."""
    t = np.arange(values.shape[0], dtype=np.float64)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.nan
    t_valid, v_valid = t[valid], values[valid].astype(np.float64)
    t_centered = t_valid - t_valid.mean()
    denom = np.sum(t_centered ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum(t_centered * (v_valid - v_valid.mean())) / denom)


def lite_stat_temp(X: np.ndarray,
                    feature_cols: Sequence[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, List[str]]:
    """Đặc trưng thống kê rút gọn cho một batch cửa sổ ``X`` (N, window, n_feat).

    Với mỗi biến: mean/std/min/max/slope (bỏ qua NaN). Riêng nhiệt độ có thêm
    một đặc trưng: số lần cửa sổ "cắt" ngưỡng 0 độ C (đổi dấu so với 0), vì đây
    là dải nhiệt độ chồng lấn Rain/Snow (Mục 2.2).

    Trả về ``(features, names)`` với ``features`` shape ``(N, n_feat*5 + 1)``.
    """
    n_samples, _, n_features = X.shape
    names: List[str] = []
    for col in feature_cols:
        names.extend(f"{col}__{stat}" for stat in STAT_NAMES)
    names.append("Temperature (C)__zero_crossings")

    out = np.full((n_samples, len(names)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        # all-NaN slice (ví dụ mất dữ liệu kiểu block xoá trọn một biến) sinh
        # RuntimeWarning dự kiến từ nanmean/nanstd/nanmin/nanmax -- bỏ qua.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(n_samples):
            sample = X[i]  # (window, n_feat)
            col = 0
            for feat in range(n_features):
                series = sample[:, feat]
                out[i, col] = np.nanmean(series)
                out[i, col + 1] = np.nanstd(series)
                out[i, col + 2] = np.nanmin(series) if np.isfinite(series).any() else np.nan
                out[i, col + 3] = np.nanmax(series) if np.isfinite(series).any() else np.nan
                out[i, col + 4] = _nan_slope(series)
                col += 5

            temp_series = sample[:, TEMPERATURE_CHANNEL]
            valid_temp = temp_series[~np.isnan(temp_series)]
            if valid_temp.size >= 2:
                signs = np.sign(valid_temp - 0.0)
                out[i, -1] = float(np.sum(np.diff(signs) != 0))
            else:
                out[i, -1] = np.nan

    return out, names


def _longest_false_run(valid: np.ndarray) -> int:
    """Độ dài đoạn ``False`` (thiếu) dài nhất trong một mảng bool 1D."""
    longest = current = 0
    for value in valid:
        if not value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def mask_delta_features(mask: np.ndarray,
                         feature_cols: Sequence[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, List[str]]:
    """Đặc trưng từ mặt nạ thiếu dữ liệu (N, window, n_feat) -> bool.

    Với mỗi biến: tỷ lệ thiếu, độ dài đoạn thiếu dài nhất (chuẩn hoá theo
    window), và "độ cũ" (delta) của quan trắc hợp lệ gần nhất tính đến cuối
    cửa sổ (chuẩn hoá theo window; = 1.0 nếu biến đó thiếu toàn bộ cửa sổ).
    """
    n_samples, window, n_features = mask.shape
    names: List[str] = []
    for col in feature_cols:
        names.extend([f"{col}__missing_rate", f"{col}__longest_gap", f"{col}__last_delta"])

    out = np.zeros((n_samples, len(names)), dtype=np.float64)
    for i in range(n_samples):
        col = 0
        for feat in range(n_features):
            series_mask = mask[i, :, feat]
            out[i, col] = 1.0 - series_mask.mean()
            out[i, col + 1] = _longest_false_run(series_mask) / window

            observed_steps = np.flatnonzero(series_mask)
            if observed_steps.size == 0:
                out[i, col + 2] = 1.0
            else:
                out[i, col + 2] = (window - 1 - observed_steps[-1]) / window
            col += 3

    return out, names


def catch22_features(X: np.ndarray,
                      feature_cols: Sequence[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, List[str]]:
    """Đặc trưng catch22 (Lubba et al., 2019) cho mỗi biến. Tuỳ chọn, cần ``pycatch22``."""
    try:
        import pycatch22
    except ImportError as exc:
        raise RuntimeError(
            "catch22_features() cần thư viện 'pycatch22': pip install pycatch22"
        ) from exc

    n_samples, _, n_features = X.shape
    names: List[str] = []
    rows: List[np.ndarray] = []
    for i in range(n_samples):
        row_values: List[float] = []
        row_names: List[str] = []
        for feat in range(n_features):
            series = X[i, :, feat]
            series_filled = np.where(np.isnan(series), np.nanmean(series), series)
            if np.isnan(series_filled).all():
                series_filled = np.zeros_like(series_filled)
            result = pycatch22.catch22_all(series_filled.tolist())
            row_values.extend(result["values"])
            if i == 0:
                row_names.extend(f"{feature_cols[feat]}__{name}" for name in result["names"])
        rows.append(np.asarray(row_values, dtype=np.float64))
        if i == 0:
            names = row_names
    return np.stack(rows), names


def tsfresh_features(X: np.ndarray,
                      feature_cols: Sequence[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, List[str]]:
    """Đặc trưng tsfresh (Christ et al., 2018) cho mỗi biến. Tuỳ chọn, cần ``tsfresh``."""
    try:
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError as exc:
        raise RuntimeError(
            "tsfresh_features() cần thư viện 'tsfresh': pip install tsfresh"
        ) from exc

    n_samples, window, n_features = X.shape
    frames = []
    for feat in range(n_features):
        ids = np.repeat(np.arange(n_samples), window)
        times = np.tile(np.arange(window), n_samples)
        values = X[:, :, feat].reshape(-1)
        frames.append(pd.DataFrame({"id": ids, "time": times, feature_cols[feat]: values}))

    long_df = frames[0]
    for extra in frames[1:]:
        long_df = long_df.merge(extra, on=["id", "time"])

    extracted = extract_features(
        long_df, column_id="id", column_sort="time",
        default_fc_parameters=MinimalFCParameters(), disable_progressbar=True,
    )
    extracted = extracted.reindex(range(n_samples))
    return extracted.to_numpy(dtype=np.float64), list(extracted.columns)
