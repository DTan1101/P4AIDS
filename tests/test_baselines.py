"""Kiểm tra evaluation/baselines.py (Mục 5.2 báo cáo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import (
    fit_temperature_threshold, majority_predict, mean_channel,
    persistence_predict, temperature_threshold_predict,
)


def test_majority_predict_returns_most_frequent_class():
    y_train = np.array([0, 0, 0, 1, 1])
    preds = majority_predict(y_train, n=4)
    assert (preds == 0).all()
    assert len(preds) == 4


def test_majority_predict_tie_breaks_to_a_valid_class():
    y_train = np.array([0, 1])
    preds = majority_predict(y_train, n=1)
    assert preds[0] in (0, 1)


def test_persistence_predict_looks_up_label_at_t():
    index = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame({"label": [0.0, 1.0, 0.0, 1.0, 0.0]}, index=index)
    # Dự đoán cho nhãn tại t+1 = giờ thứ 3 (index[3]) phải lấy nhãn tại giờ thứ 2 (index[2]) = 0.0.
    end_time = pd.DatetimeIndex([index[3]])
    preds = persistence_predict(df, end_time)
    assert preds[0] == 0


def test_persistence_predict_falls_back_to_majority_for_unknown_hour():
    index = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame({"label": [1.0, 1.0, 1.0, 1.0, 0.0]}, index=index)
    # Giờ không tồn tại trong df -> fallback về lớp đa số (1.0 ở đây).
    missing_hour = pd.DatetimeIndex([pd.Timestamp("2030-01-01")])
    preds = persistence_predict(df, missing_hour)
    assert preds[0] == 1


def test_fit_temperature_threshold_finds_near_perfect_separator():
    rng = np.random.default_rng(0)
    rain_temp = rng.normal(loc=10, scale=1, size=200)   # Rain: nhiệt độ cao
    snow_temp = rng.normal(loc=-10, scale=1, size=200)  # Snow: nhiệt độ thấp
    mean_temperature = np.concatenate([rain_temp, snow_temp])
    y_train = np.concatenate([np.zeros(200), np.ones(200)]).astype(np.int64)

    threshold = fit_temperature_threshold(mean_temperature, y_train)
    preds = temperature_threshold_predict(mean_temperature, threshold)
    accuracy = (preds == y_train).mean()
    assert accuracy > 0.95


def test_temperature_threshold_predict_direction():
    # Dưới ngưỡng -> Snow (1); trên ngưỡng -> Rain (0), theo đúng Mục 5.2.
    preds = temperature_threshold_predict(np.array([-5.0, 5.0]), threshold=0.0)
    assert list(preds) == [1, 0]


def test_mean_channel_ignores_nan():
    X = np.array([[[1.0], [np.nan], [3.0]]])  # (1 sample, 3 timesteps, 1 channel)
    result = mean_channel(X, channel=0)
    assert result[0] == pytest.approx(2.0)
