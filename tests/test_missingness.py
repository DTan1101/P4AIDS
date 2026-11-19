"""Kiểm tra datasets/missingness.py: giả lập point/subsequence/block (Mục 4.3 báo cáo)."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.missingness import (
    ALL_CONFIGS, KINDS, RATES, MissingnessConfig, apply_mask,
    generate_mask, generate_or_load_mask,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 24, 5)).astype(np.float32)


@pytest.fixture
def X_with_natural_nan():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 24, 5)).astype(np.float32)
    # Đánh dấu thiếu tự nhiên tại các vị trí cố định, biết trước, để kiểm tra
    # mặt nạ giả lập không bao giờ "hiện lại" các vị trí này.
    X[0, 0, 0] = np.nan
    X[0, 5, 2] = np.nan
    X[10, :, 3] = np.nan  # toàn bộ một biến bị thiếu tự nhiên trong một cửa sổ
    return X


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rate", RATES)
def test_generate_mask_hits_target_rate_closely(X, kind, rate):
    config = MissingnessConfig(kind=kind, rate=rate)
    mask = generate_mask(X, config, seed=42)
    actual_rate = 1.0 - mask.mean()
    # 200 cửa sổ x 120 điểm/cửa sổ -> sai số làm tròn mỗi cửa sổ tối đa 1/120,
    # cho phép dung sai 1% là đủ chặt để bắt lỗi logic nhưng không bắt lỗi làm tròn.
    assert abs(actual_rate - rate) < 0.01


def test_generate_mask_never_unhides_natural_nan(X_with_natural_nan, ):
    for config in ALL_CONFIGS:
        mask = generate_mask(X_with_natural_nan, config, seed=7)
        natural_missing = np.isnan(X_with_natural_nan)
        # Mọi vị trí thiếu tự nhiên phải vẫn là "thiếu" (mask=False) sau khi áp
        # thêm mất dữ liệu giả lập -- không được "hồi phục" dữ liệu không tồn tại.
        assert not mask[natural_missing].any(), f"config={config.key} làm hiện lại NaN tự nhiên"


def test_generate_mask_is_deterministic_given_same_seed(X):
    config = MissingnessConfig("point", 0.3)
    mask_a = generate_mask(X, config, seed=123)
    mask_b = generate_mask(X, config, seed=123)
    np.testing.assert_array_equal(mask_a, mask_b)


def test_generate_mask_different_seeds_give_different_masks(X):
    config = MissingnessConfig("point", 0.3)
    mask_a = generate_mask(X, config, seed=1)
    mask_b = generate_mask(X, config, seed=2)
    assert not np.array_equal(mask_a, mask_b)


def test_point_missingness_is_scattered_not_structured(X):
    """Kiểu point phải rải rác độc lập theo từng điểm, không tạo thành đoạn dài."""
    config = MissingnessConfig("point", 0.3)
    mask = generate_mask(X, config, seed=42)
    # Với p=0.3 độc lập, xác suất một đoạn liên tiếp dài >= 10 trong 24 bước là cực nhỏ;
    # kiểu block/subsequence với rate 0.3 thường tạo ít nhất một đoạn như vậy.
    max_run = 0
    for sample in mask:
        for channel in range(sample.shape[1]):
            series = sample[:, channel]
            run = 0
            for v in series:
                run = run + 1 if not v else 0
                max_run = max(max_run, run)
    assert max_run < 10


def test_block_missingness_can_wipe_entire_channel(X):
    """Kiểu block phải có khả năng xoá trọn một biến trong một cửa sổ (Mục 4.3)."""
    config = MissingnessConfig("block", 0.5)
    mask = generate_mask(X, config, seed=42)
    any_fully_missing_channel = any(
        (~mask[i, :, c]).all() for i in range(mask.shape[0]) for c in range(mask.shape[2])
    )
    assert any_fully_missing_channel


def test_apply_mask_sets_false_positions_to_nan(X):
    config = MissingnessConfig("point", 0.3)
    mask = generate_mask(X, config, seed=42)
    X_masked = apply_mask(X, mask)
    assert np.isnan(X_masked[~mask]).all()
    np.testing.assert_array_equal(X_masked[mask], X[mask])


def test_missingness_config_rejects_invalid_kind_and_rate():
    with pytest.raises(ValueError):
        MissingnessConfig(kind="invalid", rate=0.1)
    with pytest.raises(ValueError):
        MissingnessConfig(kind="point", rate=1.5)


def test_generate_or_load_mask_cache_round_trip(X, tmp_path):
    config = MissingnessConfig("point", 0.1)
    mask_a = generate_or_load_mask(X, config, seed=42, cache_dir=str(tmp_path))
    cached_files = list(tmp_path.glob("*.npz"))
    assert len(cached_files) == 1

    mask_b = generate_or_load_mask(X, config, seed=42, cache_dir=str(tmp_path))
    np.testing.assert_array_equal(mask_a, mask_b)
    # Lần nạp thứ hai phải lấy từ cache, không ghi thêm file mới.
    assert len(list(tmp_path.glob("*.npz"))) == 1
