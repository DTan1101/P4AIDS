"""Trainer cho nhánh ROCKET: chạy MiniRocket/MultiRocket qua các cấu hình mất dữ liệu (Mục 4.5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from datasets.features import mask_delta_features
from datasets.missingness import MissingnessConfig, apply_mask
from evaluation.metrics import compute_all_metrics
from models.rocket import RocketClassifier

logger = logging.getLogger("p4aids")


@dataclass
class RocketRunResult:
    config: MissingnessConfig
    variant: str
    use_mask_delta: bool
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)


def run_rocket(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
               train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
               config: MissingnessConfig, variant: str = "mini",
               use_mask_delta: bool = True, seed: int = 42, n_jobs: int = 2,
               max_fit_samples: int = 2500, transform_batch_size: int = 2000,
               n_kernels: Optional[int] = None) -> RocketRunResult:
    Xm = apply_mask(X, mask)
    mask_feats, _ = mask_delta_features(mask) if use_mask_delta else (None, None)

    def _slice(idx: np.ndarray):
        md = mask_feats[idx] if mask_feats is not None else None
        return Xm[idx], md

    x_train, md_train = _slice(train_idx)
    x_val, md_val = _slice(val_idx)
    x_test, md_test = _slice(test_idx)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    clf = RocketClassifier(variant=variant, random_state=seed, n_jobs=n_jobs,
                            max_fit_samples=max_fit_samples, transform_batch_size=transform_batch_size,
                            n_kernels=n_kernels)
    clf.fit(x_train, y_train, mask_delta=md_train)

    val_preds = clf.predict(x_val, md_val)
    val_proba = clf.predict_proba_positive(x_val, md_val)
    test_preds = clf.predict(x_test, md_test)
    test_proba = clf.predict_proba_positive(x_test, md_test)

    return RocketRunResult(
        config=config, variant=variant, use_mask_delta=use_mask_delta,
        val_metrics=compute_all_metrics(y_val, val_preds, val_proba),
        test_metrics=compute_all_metrics(y_test, test_preds, test_proba),
    )


def run_rocket_sweep(X: np.ndarray, y: np.ndarray, masks: Dict[MissingnessConfig, np.ndarray],
                      train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                      variant: str = "mini", use_mask_delta: bool = True,
                      seed: int = 42, n_jobs: int = 2,
                      max_fit_samples: int = 2500, transform_batch_size: int = 2000,
                      n_kernels: Optional[int] = None) -> List[RocketRunResult]:
    results = []
    for config, mask in masks.items():
        logger.info("ROCKET | %s | variant=%s | mask_delta=%s", config.key, variant, use_mask_delta)
        results.append(run_rocket(X, y, mask, train_idx, val_idx, test_idx, config,
                                   variant, use_mask_delta, seed, n_jobs, max_fit_samples,
                                   transform_batch_size, n_kernels))
    return results
