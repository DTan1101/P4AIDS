"""Chạy nhánh Classic ML (Mục 4.4): feature engineering + impute + XGBoost (+ Optuna).

    python tools/run_classic.py --config configs/rain_snow_classic.yaml
    python tools/run_classic.py --missing-kind point --missing-rate 0.1
    python tools/run_classic.py --opts classic.impute_strategy=missforest classic.tune=true
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.cache import load_prepared_data  # noqa: E402
from options.experiment_options import parse_experiment_options, resolve_masks  # noqa: E402
from trainers.classic_trainer import run_classic_sweep  # noqa: E402
from utils.utils import ensure_dir, save_json, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_classic.yaml"
ARG_TO_CONFIG = {
    "impute_strategy": "classic.impute_strategy",
    "imbalance_strategy": "classic.imbalance_strategy",
    "tune": "classic.tune",
}


def add_arguments(parser) -> None:
    parser.add_argument("--impute-strategy", type=str, default=None,
                         choices=["xgboost_native", "median_mask", "missforest"])
    parser.add_argument("--imbalance-strategy", type=str, default=None,
                         choices=["none", "scale_pos_weight", "smote"])
    parser.add_argument("--tune", action="store_true", default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg, args = parse_experiment_options(argv, __doc__, DEFAULT_CONFIG, ARG_TO_CONFIG, add_arguments)
    setup_logger()
    logger = logging.getLogger("p4aids")

    data = load_prepared_data(cfg["data"].get("output_path", "outputs/data/windows.npz"))
    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    masks = resolve_masks(X, args, cfg.get("missing", {}))

    classic_cfg = cfg["classic"]
    results = run_classic_sweep(
        X, y, masks, train_idx, val_idx, test_idx,
        impute_strategy=classic_cfg.get("impute_strategy", "xgboost_native"),
        imbalance_strategy=classic_cfg.get("imbalance_strategy", "scale_pos_weight"),
        tune=bool(classic_cfg.get("tune", False)),
        n_trials=int(classic_cfg.get("n_trials", 30)),
        seed=int(cfg["experiment"].get("seed", 42)),
    )

    output_dir = cfg["experiment"].get("output_dir", "outputs/results/classic")
    ensure_dir(output_dir)
    for result in results:
        payload = dataclasses.asdict(result)
        save_json(payload, os.path.join(output_dir, f"classic_{result.config.key}.json"))
        logger.info("Classic ML | %s -> test_macro_f1=%.4f test_pr_auc=%.4f",
                    result.config.key, result.test_metrics["macro_f1"], result.test_metrics["pr_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
