"""Kiểm tra datasets/szeged.py: load_raw + clean (Mục 2.3 báo cáo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasets.szeged import FEATURE_COLUMNS, clean, load_raw


def _make_raw_csv(path, rows):
    """Tạo CSV thô với đúng 12 cột gốc của Szeged, từ danh sách dict ``rows``."""
    base = dict(
        Summary="Clear", **{"Apparent Temperature (C)": 10.0, "Wind Bearing (degrees)": 180.0,
                             "Loud Cover": 0.0, "Daily Summary": "x"},
    )
    records = []
    for row in rows:
        record = dict(base)
        record.update(row)
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def raw_csv(tmp_path):
    """8 giờ liên tục (00h-07h) với các lỗi chất lượng đã biết trước, cộng một
    khoảng hở (09h, bỏ qua 08h) để kiểm tra reindex về lưới giờ liên tục.
    """
    hours = [f"2020-01-01 0{h}:00:00.000 +0000" for h in range(8)] + \
            ["2020-01-01 09:00:00.000 +0000"]
    rows = []
    for i, ts in enumerate(hours):
        rows.append({
            "Formatted Date": ts,
            "Precip Type": "rain",
            "Temperature (C)": float(i),
            "Humidity": 0.5,
            "Wind Speed (km/h)": 10.0,
            "Visibility (km)": 10.0,
            "Pressure (millibars)": 1000.0,
        })
    # Lỗi 1: Pressure == 0 tại giờ 02h (index 2) -> phải thành NaN.
    rows[2]["Pressure (millibars)"] = 0.0
    # Lỗi 2: timestamp trùng giờ 03h -> dòng trùng phải bị loại (giữ bản ghi đầu).
    duplicate = dict(rows[3])
    duplicate["Temperature (C)"] = 999.0  # giá trị khác để phát hiện nếu dòng sai bị giữ lại
    rows.insert(4, duplicate)
    # Lỗi 3: nhãn không hợp lệ tại giờ 05h -> dòng phải bị loại khỏi kết quả có nhãn.
    rows[6]["Precip Type"] = "unknown"

    path = tmp_path / "weatherHistory.csv"
    return _make_raw_csv(path, rows)


def test_load_raw_sorts_and_parses_timestamp(raw_csv):
    df = load_raw(str(raw_csv))
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].dt.tz is None  # naive UTC, xem comment trong szeged.py


def test_clean_drops_loud_cover(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    assert "Loud Cover" not in df.columns


def test_clean_pressure_zero_becomes_nan(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    ts = pd.Timestamp("2020-01-01 02:00:00")
    assert np.isnan(df.loc[ts, "Pressure (millibars)"])


def test_clean_deduplicates_keeping_first(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    ts = pd.Timestamp("2020-01-01 03:00:00")
    # Chỉ một dòng cho timestamp này, và phải là bản ghi ĐẦU TIÊN (Temperature=3.0),
    # không phải bản ghi trùng được chèn sau (999.0).
    assert df.loc[ts, "Temperature (C)"] == 3.0


def test_clean_drops_invalid_label(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    # rows[6] sau khi đã insert bản ghi trùng ở index 4 ứng với giờ 05:00 gốc.
    ts = pd.Timestamp("2020-01-01 05:00:00")
    assert np.isnan(df.loc[ts, "label"])


def test_clean_reindexes_to_continuous_hourly_grid(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    # 00h..09h liên tục = 10 hàng, kể cả giờ 08h bị hở trong dữ liệu gốc.
    expected_index = pd.date_range("2020-01-01 00:00:00", "2020-01-01 09:00:00", freq="h")
    pd.testing.assert_index_equal(df.index, expected_index)
    gap_ts = pd.Timestamp("2020-01-01 08:00:00")
    assert np.isnan(df.loc[gap_ts, "Temperature (C)"])
    assert np.isnan(df.loc[gap_ts, "label"])


def test_clean_keeps_only_feature_and_label_columns(raw_csv):
    df = clean(load_raw(str(raw_csv)))
    assert list(df.columns) == list(FEATURE_COLUMNS) + ["label"]
