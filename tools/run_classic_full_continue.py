"""Tiếp tục các ablation chưa hoàn tất mà không chạy lại matrix chính."""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.cache import load_prepared_data
from datasets.missingness import ALL_CONFIGS, FULL_DATA_CONFIG
from tools.run_classic_full import (
    run_exists,
    run_one,
    write_run,
    write_summary,
)
from utils.utils import setup_logger


def load_runs(output_dir: str):
    import glob
    import json
    from dataclasses import fields
    from tools.run_classic_full import ClassicFullRun

    runs = []
    for path in glob.glob(os.path.join(output_dir, "*.json")):
        if os.path.basename(path) == "environment.json":
            continue
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if "model" not in data:
            continue
        runs.append(ClassicFullRun(**{
            f.name: data.get(f.name)
            for f in fields(ClassicFullRun)
            if f.name in data
        }))
    return runs


def main() -> int:
    setup_logger()
    logger = logging.getLogger("p4aids")
    data = load_prepared_data("outputs/data/windows.npz")
    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    output_dir = "outputs/results/classic_full"

    masks = {FULL_DATA_CONFIG: ~np.isnan(X)}
    for config in ALL_CONFIGS:
        path = os.path.join(
            "outputs", "masks",
            f"mask_{config.key}_seed123_n{X.shape[0]}.npz",
        )
        masks[config] = np.load(path)["mask"]

    runs = load_runs(output_dir)

    configs = [
        FULL_DATA_CONFIG,
        next(c for c in ALL_CONFIGS if c.kind == "point" and c.rate == 0.5),
        next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5),
    ]
    feature_sets = ("lite_mask_delta", "raw_lag_mask_delta",
                    "catch22_mask_delta", "tsfresh_mask_delta")
    for config in configs:
        for feature_set in feature_sets:
            run = run_one(
                "xgboost", feature_set, "native", "scale_pos_weight",
                X, y, masks[config], train_idx, val_idx, test_idx,
                config.kind, config.rate, seed=42,
            )
            if not run_exists(output_dir, run):
                logger.info("Ablation | %s | %s", feature_set, config.key)
                runs.append(run)
                write_run(output_dir, run)

    missforest_configs = [
        FULL_DATA_CONFIG,
        next(c for c in ALL_CONFIGS if c.kind == "point" and c.rate == 0.3),
        next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5),
    ]
    for config in missforest_configs:
        run = run_one(
            "xgboost", "lite_mask_delta", "missforest", "scale_pos_weight",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42,
        )
        if not run_exists(output_dir, run):
            logger.info("MissForest | %s", config.key)
            runs.append(run)
            write_run(output_dir, run)

    advanced_configs = [
        next(c for c in ALL_CONFIGS if c.kind == "point" and c.rate == 0.3),
        next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5),
    ]
    for config in advanced_configs:
        smote = run_one(
            "xgboost", "lite_mask_delta", "median_mask", "smote",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42,
        )
        if not run_exists(output_dir, smote):
            logger.info("SMOTE | %s", config.key)
            runs.append(smote)
            write_run(output_dir, smote)

        tuned = run_one(
            "xgboost", "lite_mask_delta", "native", "scale_pos_weight",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42, tune=True,
        )
        if not run_exists(output_dir, tuned):
            logger.info("XGBoost tuning | %s", config.key)
            runs.append(tuned)
            write_run(output_dir, tuned)

    write_summary(output_dir, runs)
    logger.info("Hoàn tất tiếp tục; tổng run trong summary=%d", len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
