"""Chạy các baseline (Mục 5.2): lớp đa số, persistence, ngưỡng nhiệt độ, logreg, k-NN.

    python tools/run_baselines.py --config configs/rain_snow_data.yaml
    python tools/run_baselines.py --missing-kind point --missing-rate 0.1
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.cache import load_prepared_data  # noqa: E402
from datasets.features import lite_stat_temp  # noqa: E402
from datasets.missingness import apply_mask  # noqa: E402
from datasets.szeged import DEFAULT_CSV_PATH, clean, load_raw  # noqa: E402
from evaluation.baselines import (  # noqa: E402
    build_knn_baseline, build_logreg_baseline, fit_temperature_threshold,
    majority_predict, mean_channel, persistence_predict, temperature_threshold_predict,
)
from evaluation.metrics import compute_all_metrics  # noqa: E402
from options.experiment_options import parse_experiment_options, resolve_masks  # noqa: E402
from utils.utils import ensure_dir, save_json, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_data.yaml"
ARG_TO_CONFIG = {}


def run_one_config(X, y, df, end_time, train_idx, val_idx, test_idx, mask) -> dict:
    Xm = apply_mask(X, mask)
    feats, _ = lite_stat_temp(Xm)

    results = {}
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    maj_val = majority_predict(y_train, len(y_val))
    maj_test = majority_predict(y_train, len(y_test))
    results["majority"] = {"val": compute_all_metrics(y_val, maj_val),
                            "test": compute_all_metrics(y_test, maj_test)}

    persist_val = persistence_predict(df, end_time[val_idx])
    persist_test = persistence_predict(df, end_time[test_idx])
    results["persistence"] = {"val": compute_all_metrics(y_val, persist_val),
                               "test": compute_all_metrics(y_test, persist_test)}

    mean_temp_train = mean_channel(Xm[train_idx], 0)
    mean_temp_val = mean_channel(Xm[val_idx], 0)
    mean_temp_test = mean_channel(Xm[test_idx], 0)
    threshold = fit_temperature_threshold(mean_temp_train, y_train)
    thr_val = temperature_threshold_predict(mean_temp_val, threshold)
    thr_test = temperature_threshold_predict(mean_temp_test, threshold)
    results["temperature_threshold"] = {
        "threshold": threshold,
        "val": compute_all_metrics(y_val, thr_val), "test": compute_all_metrics(y_test, thr_test),
    }

    logreg = build_logreg_baseline().fit(feats[train_idx], y_train)
    results["logistic_regression"] = {
        "val": compute_all_metrics(y_val, logreg.predict(feats[val_idx])),
        "test": compute_all_metrics(y_test, logreg.predict(feats[test_idx])),
    }

    knn = build_knn_baseline().fit(feats[train_idx], y_train)
    results["knn"] = {
        "val": compute_all_metrics(y_val, knn.predict(feats[val_idx])),
        "test": compute_all_metrics(y_test, knn.predict(feats[test_idx])),
    }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg, args = parse_experiment_options(
        argv, __doc__, DEFAULT_CONFIG, ARG_TO_CONFIG,
    )
    setup_logger()
    logger = logging.getLogger("p4aids")

    data = load_prepared_data(cfg.get("data", {}).get("output_path", "outputs/data/windows.npz"))
    X, y, end_time = data["X"], data["y"], data["end_time"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    df = clean(load_raw(cfg.get("data", {}).get("csv_path", DEFAULT_CSV_PATH)))

    masks = resolve_masks(X, args, cfg.get("missing", {}))

    output_dir = "outputs/results/baselines"
    ensure_dir(output_dir)
    for config, mask in masks.items():
        logger.info("Baselines | %s", config.key)
        results = run_one_config(X, y, df, end_time, train_idx, val_idx, test_idx, mask)
        save_json(results, os.path.join(output_dir, f"baselines_{config.key}.json"))
        for name, r in results.items():
            logger.info("  %-22s test_macro_f1=%.4f", name, r["test"]["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
