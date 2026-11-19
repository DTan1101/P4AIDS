"""Kiểm tra datasets/features.py: lite_stat_temp, mask_delta_features (Mục 4.4.1)."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.features import lite_stat_temp, mask_delta_features


def test_lite_stat_temp_known_values_no_missing():
    # 1 cửa sổ, 1 biến (Temperature) tăng dần 0..23 -> mean/std/min/max/slope biết trước.
    window = np.arange(24, dtype=np.float32)
    X = np.zeros((1, 24, 5), dtype=np.float32)
    X[0, :, 0] = window

    feats, names = lite_stat_temp(X)
    idx = {name: i for i, name in enumerate(names)}

    assert feats[0, idx["Temperature (C)__mean"]] == pytest.approx(window.mean())
    assert feats[0, idx["Temperature (C)__min"]] == 0.0
    assert feats[0, idx["Temperature (C)__max"]] == 23.0
    # Chuỗi tuyến tính hoàn hảo bước 1 -> slope phải đúng bằng 1.0.
    assert feats[0, idx["Temperature (C)__slope"]] == pytest.approx(1.0)


def test_lite_stat_temp_zero_crossings_counts_sign_changes():
    # Nhiệt độ đổi dấu quanh 0 đúng 2 lần: -1,-1,1,1,-1,-1 (padding các giờ còn lại bằng -1).
    X = np.full((1, 24, 5), -1.0, dtype=np.float32)
    X[0, :6, 0] = [-1, -1, 1, 1, -1, -1]
    feats, names = lite_stat_temp(X)
    idx = names.index("Temperature (C)__zero_crossings")
    assert feats[0, idx] == 2


def test_lite_stat_temp_all_nan_channel_gives_nan_not_crash():
    X = np.full((1, 24, 5), np.nan, dtype=np.float32)
    feats, names = lite_stat_temp(X)
    assert np.isnan(feats).all()
    assert feats.shape == (1, len(names))


def test_lite_stat_temp_output_shape():
    X = np.zeros((10, 24, 5), dtype=np.float32)
    feats, names = lite_stat_temp(X)
    # 5 biến x 5 thống kê (mean/std/min/max/slope) + 1 zero_crossings = 26.
    assert feats.shape == (10, 26)
    assert len(names) == 26


def test_mask_delta_features_missing_rate_matches_mask():
    mask = np.ones((1, 24, 5), dtype=bool)
    mask[0, :12, 1] = False  # kênh 1: thiếu đúng nửa cửa sổ
    feats, names = mask_delta_features(mask)
    idx = names.index("Humidity__missing_rate")
    assert feats[0, idx] == pytest.approx(0.5)


def test_mask_delta_features_longest_gap():
    mask = np.ones((1, 24, 5), dtype=bool)
    mask[0, 2:7, 0] = False  # đoạn thiếu dài 5 liên tiếp
    feats, names = mask_delta_features(mask)
    idx = names.index("Temperature (C)__longest_gap")
    assert feats[0, idx] == pytest.approx(5 / 24)


def test_mask_delta_features_last_delta_when_channel_fully_missing():
    mask = np.ones((1, 24, 5), dtype=bool)
    mask[0, :, 2] = False  # biến thứ 2 (Wind Speed) thiếu toàn bộ cửa sổ
    feats, names = mask_delta_features(mask)
    idx = names.index("Wind Speed (km/h)__last_delta")
    assert feats[0, idx] == 1.0


def test_mask_delta_features_last_delta_when_last_step_observed():
    mask = np.ones((1, 24, 5), dtype=bool)
    mask[0, -1, 3] = True  # bước cuối cùng có quan trắc -> delta phải = 0
    feats, names = mask_delta_features(mask)
    idx = names.index("Pressure (millibars)__last_delta")
    assert feats[0, idx] == 0.0
