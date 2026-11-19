"""Kiểm tra datasets/windows.py: build_windows (Mục 1.2/4.2 báo cáo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasets.windows import build_windows


def _make_df(n_hours: int, labels=None, nan_at=None):
    """DataFrame giờ liên tục; cột feature = chỉ số giờ (để kiểm tra alignment dễ dàng)."""
    index = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = np.arange(n_hours, dtype=np.float32)
    df = pd.DataFrame({
        "Temperature (C)": values, "Humidity": values, "Wind Speed (km/h)": values,
        "Pressure (millibars)": values, "Visibility (km)": values,
    }, index=index)
    df["label"] = (np.arange(n_hours) % 2) if labels is None else labels
    df["label"] = df["label"].astype(float)
    if nan_at is not None:
        df.loc[df.index[nan_at], "label"] = np.nan
    return df


def test_build_windows_shape():
    df = _make_df(30)
    result = build_windows(df, window=24)
    # n_rows=30, window=24 -> i chạy từ 23..28 (i+1 < 30) => 6 cửa sổ.
    assert result.X.shape == (6, 24, 5)
    assert result.y.shape == (6,)
    assert len(result.end_time) == 6


def test_build_windows_label_is_t_plus_1_not_window_end():
    """Nhãn phải lấy từ GIỜ NGAY SAU cửa sổ (t+1), không phải giờ cuối cửa sổ (t)."""
    df = _make_df(26)
    df["label"] = 0.0
    df.iloc[24, df.columns.get_loc("label")] = 1.0  # giờ thứ 25 (index 24) -> nhãn 1
    result = build_windows(df, window=24)
    # Cửa sổ đầu tiên: features = giờ 0..23, nhãn = giờ 24 -> phải là 1.
    assert result.y[0] == 1
    assert result.end_time[0] == df.index[24]


def test_build_windows_feature_content_matches_source_rows():
    df = _make_df(26)
    result = build_windows(df, window=24)
    # Cửa sổ đầu tiên phải đúng là giờ 0..23 của cột Temperature.
    np.testing.assert_array_equal(result.X[0, :, 0], np.arange(24, dtype=np.float32))


def test_build_windows_drops_windows_with_missing_t1_label():
    df = _make_df(26, nan_at=24)  # nhãn tại giờ t+1 của cửa sổ đầu tiên bị thiếu
    result = build_windows(df, window=24)
    # Có 2 cửa sổ khả dĩ (i=23,24); cửa sổ ứng với nhãn tại giờ 24 phải bị loại.
    assert df.index[24] not in result.end_time


def test_build_windows_keeps_nan_features_natural_missingness():
    df = _make_df(26)
    df.iloc[5, df.columns.get_loc("Temperature (C)")] = np.nan
    result = build_windows(df, window=24)
    assert np.isnan(result.X[0, 5, 0])


def test_build_windows_rejects_non_monotonic_index():
    df = _make_df(26)
    shuffled = df.iloc[::-1]
    with pytest.raises(ValueError):
        build_windows(shuffled, window=24)
