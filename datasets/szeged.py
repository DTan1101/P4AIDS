"""Nạp và làm sạch bộ dữ liệu thời tiết Szeged (2006-2016).

Tương ứng Mục 2.1 (nguồn dữ liệu) và 2.3 (vấn đề chất lượng dữ liệu) của báo cáo.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("p4aids")

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "archive", "weatherHistory.csv")

# 5 biến quan trắc dùng làm đầu vào, theo đúng Mục 1.2 của báo cáo.
FEATURE_COLUMNS: Sequence[str] = (
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Pressure (millibars)",
    "Visibility (km)",
)

LABEL_COLUMN = "Precip Type"
LABEL_MAP = {"rain": 0, "snow": 1}
CLASS_NAMES = ("Rain", "Snow")


def load_raw(path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Đọc CSV gốc, parse timestamp và sắp xếp theo thời gian tăng dần."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu Szeged tại '{path}'. Tải weatherHistory.csv vào đó."
        )
    df = pd.read_csv(path)
    # utc=True chuẩn hoá các offset (+0200/+0100...) về cùng một mốc UTC, sau đó
    # bỏ tzinfo (tz_localize(None)) để toàn bộ pipeline dùng timestamp "naive UTC"
    # thống nhất -- tránh lệch kiểu tz-aware/naive khi cache qua .npz (xem datasets/cache.py).
    df["timestamp"] = pd.to_datetime(df["Formatted Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Đã đọc %d dòng dữ liệu thô từ '%s'.", len(df), path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu theo các vấn đề chất lượng đã khảo sát ở Mục 2.3.

    - Loại cột ``Loud Cover`` (hằng số, không mang thông tin).
    - ``Pressure == 0`` là giá trị không hợp lý về vật lý -> coi là thiếu (NaN).
    - Timestamp trùng: giữ bản ghi đầu tiên.
    - Chỉ giữ các bản ghi có nhãn Rain/Snow hợp lệ (loại bỏ ~0.5% thiếu nhãn).
    - Reindex về lưới giờ liên tục (phát hiện được đúng 1 khoảng hở 4 giờ trong
      toàn bộ 2006-2016); giờ bị hở trở thành hàng NaN tự nhiên, tách biệt khỏi
      phần mất dữ liệu được giả lập ở Mục 4.3.
    """
    df = df.drop(columns=["Loud Cover"], errors="ignore")

    pressure_col = "Pressure (millibars)"
    zero_pressure = df[pressure_col] == 0
    if zero_pressure.any():
        logger.info("Pressure == 0: %d dòng -> chuyển thành NaN.", int(zero_pressure.sum()))
        df.loc[zero_pressure, pressure_col] = np.nan

    before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    if before != len(df):
        logger.info("Timestamp trùng: loại %d dòng (giữ bản ghi đầu tiên).", before - len(df))

    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.lower()
    valid_label = df[LABEL_COLUMN].isin(LABEL_MAP)
    if not valid_label.all():
        logger.info(
            "Nhãn %s không hợp lệ/thiếu: loại %d dòng.",
            LABEL_COLUMN, int((~valid_label).sum()),
        )
    df = df.loc[valid_label].set_index("timestamp").sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = full_index.difference(df.index)
    if len(missing_hours) > 0:
        logger.info("Lưới giờ có %d giờ bị hở tự nhiên (không có bản ghi gốc).", len(missing_hours))
    df = df.reindex(full_index)

    label_numeric = df[LABEL_COLUMN].map(LABEL_MAP)
    df["label"] = label_numeric

    keep_cols = list(FEATURE_COLUMNS) + ["label"]
    df = df[keep_cols]
    logger.info("Sau làm sạch: %d dòng (%d có nhãn hợp lệ).", len(df), int(df["label"].notna().sum()))
    return df


def load_clean(path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Tiện ích gộp :func:`load_raw` + :func:`clean`."""
    return clean(load_raw(path))
