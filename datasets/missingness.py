"""Giả lập cảm biến mất dữ liệu: point / subsequence / block (Mục 4.3 báo cáo).

Mặt nạ được sinh **một lần** cho mỗi cấu hình ``(kind, rate, seed)`` và lưu cache
ra đĩa, để mọi nhánh phương pháp (Classic ML, ROCKET, Deep Learning) dùng đúng
cùng một bộ mặt nạ -- điều kiện bắt buộc để so sánh công bằng.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger("p4aids")

DEFAULT_CACHE_DIR = os.path.join("outputs", "masks")
KINDS: Tuple[str, ...] = ("point", "subsequence", "block")
RATES: Tuple[float, ...] = (0.10, 0.30, 0.50)


@dataclass(frozen=True)
class MissingnessConfig:
    """Một cấu hình mất dữ liệu: kiểu + tỷ lệ (tính trên tổng số điểm/cửa sổ)."""

    kind: str
    rate: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind phải thuộc {KINDS}, nhận '{self.kind}'.")
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate phải trong [0, 1], nhận {self.rate}.")

    @property
    def key(self) -> str:
        return f"{self.kind}_{round(self.rate * 100):02d}"


#: Cấu hình "0% thiếu" dùng cho Mục 6.1 -- kind không quan trọng vì rate=0 không
#: ẩn thêm gì cả, mask chỉ còn phản ánh phần thiếu tự nhiên của dữ liệu.
FULL_DATA_CONFIG = MissingnessConfig(kind="point", rate=0.0)

ALL_CONFIGS: Tuple[MissingnessConfig, ...] = tuple(
    MissingnessConfig(kind=kind, rate=rate) for kind in KINDS for rate in RATES
)


def _budget(observed_sample: np.ndarray, rate: float) -> int:
    """Số điểm cần ẩn thêm để tổng tỷ lệ thiếu (tự nhiên + giả lập) đạt ``rate``."""
    total = observed_sample.size
    target_total_missing = int(round(rate * total))
    already_missing = int((~observed_sample).sum())
    return max(0, target_total_missing - already_missing)


def _apply_point(observed: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Ẩn ``budget`` điểm độc lập, chọn ngẫu nhiên trong các điểm đang quan sát được."""
    mask = observed.copy()
    flat = mask.ravel()
    observed_idx = np.flatnonzero(flat)
    n_take = min(budget, observed_idx.size)
    if n_take > 0:
        chosen = rng.choice(observed_idx, size=n_take, replace=False)
        flat[chosen] = False
    return mask


def _apply_subsequence(observed: np.ndarray, budget: int, rng: np.random.Generator,
                        max_run_frac: float = 0.5, max_iters: int = 500) -> np.ndarray:
    """Ẩn nhiều đoạn liên tiếp ngắn/vừa, mỗi đoạn trên một biến (mô phỏng mất kết nối tạm thời)."""
    window, n_features = observed.shape
    mask = observed.copy()
    max_len = max(1, int(round(window * max_run_frac)))
    remaining = budget

    for _ in range(max_iters):
        if remaining <= 0:
            break
        feat = int(rng.integers(0, n_features))
        length = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(0, window))
        end = min(window, start + length)
        segment = mask[start:end, feat]
        droppable = np.flatnonzero(segment)
        n_take = min(droppable.size, remaining)
        if n_take == 0:
            continue
        segment[droppable[:n_take]] = False
        mask[start:end, feat] = segment
        remaining -= n_take
    return mask


def _apply_block(observed: np.ndarray, budget: int, rng: np.random.Generator,
                  min_len_frac: float = 0.5, max_iters: int = 50) -> np.ndarray:
    """Ẩn toàn bộ một/vài biến trong một khoảng thời gian dài (mô phỏng một cảm biến hỏng)."""
    window, n_features = observed.shape
    mask = observed.copy()
    min_len = max(1, int(round(window * min_len_frac)))
    remaining = budget

    for _ in range(max_iters):
        if remaining <= 0:
            break
        n_feats_affected = int(rng.integers(1, n_features + 1))
        feats = rng.choice(n_features, size=n_feats_affected, replace=False)
        length = int(rng.integers(min_len, window + 1))
        start = int(rng.integers(0, max(1, window - length + 1)))
        end = start + length
        for feat in feats:
            if remaining <= 0:
                break
            segment = mask[start:end, feat]
            droppable = np.flatnonzero(segment)
            n_take = min(droppable.size, remaining)
            if n_take == 0:
                continue
            segment[droppable[:n_take]] = False
            mask[start:end, feat] = segment
            remaining -= n_take
    return mask


_APPLIERS: Dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "point": _apply_point,
    "subsequence": _apply_subsequence,
    "block": _apply_block,
}


def generate_mask(X: np.ndarray, config: MissingnessConfig, seed: int) -> np.ndarray:
    """Sinh mặt nạ (N, window, n_features) cho toàn bộ batch cửa sổ ``X``.

    ``X`` có thể chứa NaN (thiếu dữ liệu tự nhiên, xem :mod:`datasets.szeged`).
    Mặt nạ trả về là bool, ``True`` = quan sát được, đã tính cả phần tự nhiên
    lẫn phần ẩn thêm giả lập theo ``config``.
    """
    apply_fn = _APPLIERS[config.kind]
    observed_natural = ~np.isnan(X)
    rng = np.random.default_rng(seed)
    masks = np.empty_like(observed_natural, dtype=bool)
    for i in range(X.shape[0]):
        budget = _budget(observed_natural[i], config.rate)
        masks[i] = apply_fn(observed_natural[i], budget, rng)

    actual_rate = 1.0 - masks.mean()
    logger.info("Mặt nạ '%s' (seed=%d): tỷ lệ thiếu thực tế %.4f (mục tiêu %.2f).",
                config.key, seed, actual_rate, config.rate)
    return masks


def mask_cache_path(config: MissingnessConfig, seed: int, n_samples: int,
                     cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"mask_{config.key}_seed{seed}_n{n_samples}.npz")


def generate_or_load_mask(X: np.ndarray, config: MissingnessConfig, seed: int,
                           cache_dir: str = DEFAULT_CACHE_DIR, force: bool = False) -> np.ndarray:
    """Giống :func:`generate_mask` nhưng cache ra đĩa để tái dùng giữa các nhánh."""
    path = mask_cache_path(config, seed, X.shape[0], cache_dir)
    if os.path.isfile(path) and not force:
        cached = np.load(path)["mask"]
        if cached.shape == X.shape:
            logger.info("Dùng mặt nạ cache: %s", path)
            return cached
        logger.warning("Cache '%s' lệch shape (%s != %s), sinh lại.", path, cached.shape, X.shape)

    mask = generate_mask(X, config, seed)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, mask=mask)
    logger.info("Đã lưu mặt nạ vào cache: %s", path)
    return mask


def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Trả về bản sao của ``X`` với các vị trí ``mask == False`` thay bằng NaN."""
    return np.where(mask, X, np.nan).astype(X.dtype)


def load_masks_for_configs(X: np.ndarray, configs, seed: int,
                            cache_dir: str = DEFAULT_CACHE_DIR) -> Dict[MissingnessConfig, np.ndarray]:
    """Nạp (từ cache, không sinh mới) mặt nạ cho một danh sách cấu hình.

    Dùng trong ``tools/run_*.py`` sau khi ``tools/build_masks.py`` đã chạy; báo
    lỗi rõ ràng nếu thiếu cache, thay vì âm thầm sinh lại (có thể lệch seed).
    """
    masks: Dict[MissingnessConfig, np.ndarray] = {}
    for config in configs:
        path = mask_cache_path(config, seed, X.shape[0], cache_dir)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Thiếu cache mặt nạ '{path}'. Chạy 'python tools/build_masks.py' trước."
            )
        masks[config] = np.load(path)["mask"]
    return masks
