"""Chia dữ liệu theo thời gian: train/val/test theo năm và fold rolling-origin.

Tương ứng Mục 2.2 (chia theo năm, chống rò rỉ) và Mục 5.1 (giao thức chia dữ liệu).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("p4aids")


@dataclass
class YearSplit:
    """Chỉ số (không phải dữ liệu) của ba tập, theo năm của ``end_time``."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_years: Tuple[int, int]
    val_years: Tuple[int, int]
    test_years: Tuple[int, int]


def _year_mask(end_time: pd.DatetimeIndex, years: Tuple[int, int]) -> np.ndarray:
    start_year, stop_year = years
    return (end_time.year >= start_year) & (end_time.year <= stop_year)


def year_split(end_time: pd.DatetimeIndex,
                train_years: Tuple[int, int] = (2006, 2013),
                val_years: Tuple[int, int] = (2014, 2014),
                test_years: Tuple[int, int] = (2015, 2016)) -> YearSplit:
    """Chia theo năm, mô phỏng kịch bản dự báo tương lai thực tế (Mục 2.2).

    Mặc định: train 2006-2013, validation 2014, test 2015-2016 (tập test cuối
    cùng, cố định, không dùng cho bất kỳ quyết định chọn mô hình/siêu tham số
    nào -- xem Mục 5.1).
    """
    train_idx = np.flatnonzero(_year_mask(end_time, train_years))
    val_idx = np.flatnonzero(_year_mask(end_time, val_years))
    test_idx = np.flatnonzero(_year_mask(end_time, test_years))

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        logger.warning(
            "Một trong các tập train/val/test rỗng: train=%d val=%d test=%d.",
            len(train_idx), len(val_idx), len(test_idx),
        )
    logger.info(
        "Chia theo năm: train=%d (%d-%d), val=%d (%d), test=%d (%d-%d).",
        len(train_idx), *train_years, len(val_idx), val_years[0],
        len(test_idx), *test_years,
    )
    return YearSplit(train_idx, val_idx, test_idx, train_years, val_years, test_years)


def rolling_origin_folds(end_time: pd.DatetimeIndex,
                           train_start_year: int = 2006,
                           first_val_year: int = 2010,
                           last_val_year: int = 2013) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fold rolling-origin cho việc chọn mô hình/siêu tham số (Mục 5.1).

    Fold ``k`` huấn luyện trên ``[train_start_year, val_year-1]`` và validate
    trên năm ``val_year``, với ``val_year`` chạy từ ``first_val_year`` đến
    ``last_val_year``. Không bao giờ đụng tới các năm test cố định
    (2015-2016 theo :func:`year_split`), vì ``last_val_year`` mặc định dừng ở
    2013.
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for val_year in range(first_val_year, last_val_year + 1):
        train_idx = np.flatnonzero(_year_mask(end_time, (train_start_year, val_year - 1)))
        val_idx = np.flatnonzero(_year_mask(end_time, (val_year, val_year)))
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning("Fold val_year=%d rỗng (train=%d, val=%d), bỏ qua.",
                            val_year, len(train_idx), len(val_idx))
            continue
        folds.append((train_idx, val_idx))
    logger.info("Sinh %d fold rolling-origin (val_year %d..%d).",
                len(folds), first_val_year, last_val_year)
    return folds
