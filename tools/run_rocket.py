"""Chạy nhánh ROCKET (Mục 4.5): nội suy + MiniRocket/MultiRocket + RidgeClassifierCV.

    python tools/run_rocket.py --config configs/rain_snow_rocket.yaml
    python tools/run_rocket.py --missing-kind block --missing-rate 0.5 --opts rocket.variant=multi
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
from trainers.rocket_trainer import run_rocket_sweep  # noqa: E402
from utils.utils import ensure_dir, save_json, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_rocket.yaml"
ARG_TO_CONFIG = {"variant": "rocket.variant", "use_mask_delta": "rocket.use_mask_delta"}


def add_arguments(parser) -> None:
    parser.add_argument("--variant", type=str, default=None, choices=["mini", "multi"])
    parser.add_argument("--no-mask-delta", dest="use_mask_delta", action="store_false", default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg, args = parse_experiment_options(argv, __doc__, DEFAULT_CONFIG, ARG_TO_CONFIG, add_arguments)
    setup_logger()
    logger = logging.getLogger("p4aids")

    data = load_prepared_data(cfg["data"].get("output_path", "outputs/data/windows.npz"))
    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    masks = resolve_masks(X, args, cfg.get("missing", {}))

    rocket_cfg = cfg["rocket"]
    results = run_rocket_sweep(
        X, y, masks, train_idx, val_idx, test_idx,
        variant=rocket_cfg.get("variant", "mini"),
        use_mask_delta=bool(rocket_cfg.get("use_mask_delta", True)),
        seed=int(cfg["experiment"].get("seed", 42)),
        n_jobs=int(rocket_cfg.get("n_jobs", 2)),
        max_fit_samples=int(rocket_cfg.get("max_fit_samples", 2500)),
        transform_batch_size=int(rocket_cfg.get("transform_batch_size", 2000)),
        n_kernels=rocket_cfg.get("n_kernels"),
    )

    output_dir = cfg["experiment"].get("output_dir", "outputs/results/rocket")
    ensure_dir(output_dir)
    for result in results:
        save_json(dataclasses.asdict(result), os.path.join(output_dir, f"rocket_{result.config.key}.json"))
        logger.info("ROCKET | %s -> test_macro_f1=%.4f test_pr_auc=%.4f",
                    result.config.key, result.test_metrics["macro_f1"], result.test_metrics["pr_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
