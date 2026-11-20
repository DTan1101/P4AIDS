"""Chạy nhánh Deep Learning (Mục 4.6): GRU-Fill/GRU+mask-delta/GRU-D (tự cài) hoặc
BRITS/SAITS/GRU-D (PyPOTS), nhiều seed (Mục 4.6.3).

    python tools/run_deep.py --config configs/rain_snow_deep.yaml
    python tools/run_deep.py --missing-kind block --missing-rate 0.5 --opts deep.model=brits
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
from trainers.deep_trainer import CUSTOM_MODEL_NAMES, PYPOTS_MODEL_NAMES, run_deep_multiseed  # noqa: E402
from utils.utils import ensure_dir, save_json, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_deep.yaml"
ALL_MODEL_NAMES = CUSTOM_MODEL_NAMES + PYPOTS_MODEL_NAMES
ARG_TO_CONFIG = {
    "model": "deep.model", "hidden_size": "deep.hidden_size", "epochs": "deep.epochs",
    "patience": "deep.patience", "batch_size": "deep.batch_size", "lr": "deep.lr",
}


def add_arguments(parser) -> None:
    parser.add_argument("--model", type=str, default=None, choices=list(ALL_MODEL_NAMES))
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg, args = parse_experiment_options(argv, __doc__, DEFAULT_CONFIG, ARG_TO_CONFIG, add_arguments)
    setup_logger()
    logger = logging.getLogger("p4aids")

    data = load_prepared_data(cfg["data"].get("prepared_path", cfg["data"].get("output_path", "outputs/data/windows.npz")))
    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    masks = resolve_masks(X, args, cfg.get("missing", {}))

    deep_cfg = cfg["deep"]
    model_name = deep_cfg.get("model", "gru_fill")
    seeds = [int(args.seed)] if args.seed is not None else cfg["experiment"].get("seeds", [int(cfg["experiment"].get("seed", 42))])
    patience = deep_cfg.get("patience", 10)
    kwargs = dict(
        hidden_size=int(deep_cfg.get("hidden_size", 64)), epochs=int(deep_cfg.get("epochs", 50)),
        patience=None if patience is None else int(patience),
        batch_size=int(deep_cfg.get("batch_size", 128)),
        device=deep_cfg.get("device", "auto"),
    )
    kwargs["lr"] = float(deep_cfg.get("lr", 1e-3))
    if model_name in PYPOTS_MODEL_NAMES:
        kwargs.update(deep_cfg.get("pypots", {}))

    output_dir = cfg["experiment"].get("output_dir", "outputs/results/deep")
    ensure_dir(output_dir)
    for config, mask in masks.items():
        logger.info("Deep Learning | %s | model=%s | seeds=%s", config.key, model_name, seeds)
        results = run_deep_multiseed(seeds, X, y, mask, train_idx, val_idx, test_idx,
                                      config, model_name, **kwargs)
        payload = [dataclasses.asdict(r) for r in results]
        save_json(payload, os.path.join(output_dir, f"deep_{model_name}_{config.key}.json"))
        for r in results:
            logger.info("  seed=%d -> test_macro_f1=%.4f test_pr_auc=%.4f",
                        r.seed, r.test_metrics["macro_f1"], r.test_metrics.get("pr_auc", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
