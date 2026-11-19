"""Kiểm tra datasets/splits.py: year_split, rolling_origin_folds (Mục 2.2/5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datasets.splits import rolling_origin_folds, year_split


def _end_time(years):
    """DatetimeIndex rải đều qua danh sách năm, vài mốc/năm."""
    timestamps = []
    for year in years:
        timestamps.extend(pd.date_range(f"{year}-01-01", periods=5, freq="30D"))
    return pd.DatetimeIndex(timestamps)


def test_year_split_assigns_every_index_to_exactly_one_split():
    end_time = _end_time([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
    split = year_split(end_time, train_years=(2006, 2013), val_years=(2014, 2014), test_years=(2015, 2016))

    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert len(all_idx) == len(end_time)
    assert len(set(all_idx.tolist())) == len(end_time)  # không trùng lặp


def test_year_split_respects_year_boundaries():
    end_time = _end_time([2013, 2014, 2015])
    split = year_split(end_time, train_years=(2006, 2013), val_years=(2014, 2014), test_years=(2015, 2016))

    assert (end_time[split.train_idx].year <= 2013).all()
    assert (end_time[split.val_idx].year == 2014).all()
    assert (end_time[split.test_idx].year >= 2015).all()


def test_rolling_origin_folds_train_before_val_year():
    end_time = _end_time([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])
    folds = rolling_origin_folds(end_time, train_start_year=2006, first_val_year=2010, last_val_year=2012)

    assert len(folds) == 3
    for val_year, (train_idx, val_idx) in zip(range(2010, 2013), folds):
        assert (end_time[train_idx].year < val_year).all()
        assert (end_time[val_idx].year == val_year).all()


def test_rolling_origin_folds_never_touch_test_years():
    """Mặc định last_val_year=2013 phải luôn dừng trước năm test 2015-2016 (Mục 5.1)."""
    end_time = _end_time([2006, 2010, 2013, 2014, 2015, 2016])
    folds = rolling_origin_folds(end_time)
    for train_idx, val_idx in folds:
        assert (end_time[train_idx].year < 2015).all()
        assert (end_time[val_idx].year < 2015).all()
