"""Run ROCKET experiments on shared P4AIDS data."""

import dataclasses
import hashlib
import json
import logging
import platform
import re
import sys

from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from datasets.cache import load_prepared_data
from datasets.missingness import FULL_DATA_CONFIG
from options.experiment_options import (
    parse_experiment_options, resolve_masks
)
from trainers.rocket_trainer import CHANNELS, run_rocket
from utils.utils import setup_logger


DEFAULT_CONFIG = "configs/rain_snow_rocket.yaml"

ARG_TO_CONFIG = {
    "variant": "rocket.variant",
    "use_mask_delta": "rocket.use_mask_delta"
}


def add_arguments(parser):

    parser.add_argument(
        "--variant",
        choices=["mini", "multi"],
        default=None
    )

    parser.add_argument(
        "--no-mask-delta",
        dest="use_mask_delta",
        action="store_false",
        default=None
    )

    parser.add_argument(
        "--full-only",
        action="store_true"
    )

    parser.add_argument(
        "--tag",
        default="main"
    )

    parser.add_argument(
        "--no-weight",
        action="store_true"
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        choices=list(CHANNELS)
    )

    parser.add_argument(
        "--imputer",
        choices=["linear", "ffill"]
    )

    parser.add_argument(
        "--kernels",
        type=int
    )


def fingerprint(value):

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def write_json(path, record):

    tmp = path.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    tmp.replace(path)


def main(argv=None):

    cfg, args = parse_experiment_options(
        argv,
        __doc__,
        DEFAULT_CONFIG,
        ARG_TO_CONFIG,
        add_arguments
    )

    setup_logger()
    logger = logging.getLogger("p4aids")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("Invalid tag")

    if args.full_only and (
        args.missing_kind or args.missing_rate
    ):
        raise ValueError("Conflicting missing filters")

    if args.tag == "main" and (
        args.no_weight
        or args.channels is not None
        or args.use_mask_delta is False
        or args.imputer is not None
        or args.kernels is not None
    ):
        raise ValueError(
            "Use a separate --tag for ablations/quick runs"
        )

    prepared = Path(
        cfg["data"].get("prepared_path")
        or cfg["data"].get("output_path")
        or "outputs/data/windows.npz"
    )

    data = load_prepared_data(str(prepared))

    X, y = data["X"], data["y"]

    tr = data["train_idx"]
    va = data["val_idx"]
    te = data["test_idx"]

    if args.full_only:
        masks = {FULL_DATA_CONFIG: ~np.isnan(X)}
    else:
        masks = resolve_masks(
            X, args, cfg.get("missing", {})
        )

    rc = cfg["rocket"]

    variant = rc.get("variant", "mini")
    seed = int(cfg["experiment"].get("seed", 42))

    kernels = args.kernels

    if kernels is None:
        kernels = rc.get("n_kernels")

    if kernels is None:
        kernels = rc.get("kernel_defaults", {}).get(
            variant, 1008 if variant == "mini" else 336
        )

    if int(kernels) < 84:
        raise ValueError("n_kernels must be at least 84")

    parameters = {
        "variant": variant,
        "seed": seed,
        "n_kernels": int(kernels),

        "use_mask_delta": bool(
            rc.get("use_mask_delta", True)
        ),

        "n_jobs": int(rc.get("n_jobs", 2)),

        "max_fit_samples": int(
            rc.get("max_fit_samples", 2500)
        ),

        "transform_batch_size": int(
            rc.get("transform_batch_size", 512)
        ),

        "max_feature_mib": float(
            rc.get("max_feature_mib", 1024)
        ),

        "alphas": list(
            rc.get("alphas", [0.1, 1.0, 10.0])
        ),

        "class_weight": (
            None if args.no_weight
            else rc.get("class_weight", "balanced")
        ),

        "tune_threshold": bool(
            rc.get("tune_threshold", True)
        ),

        "imputer": (
            args.imputer or rc.get("imputer", "linear")
        ),

        "channels": (
            args.channels or list(
                rc.get("channels", list(CHANNELS))
            )
        )
    }

    out = Path(cfg["experiment"].get(
        "output_dir", "outputs/results/rocket"
    ))

    if args.tag != "main" or seed != 42:
        out = (
            out.parent
            / (out.name + "_extra")
            / args.tag
            / f"seed_{seed}"
        )

    out.mkdir(parents=True, exist_ok=True)

    # Prevent duplicate results from the old runner.
    for path in out.glob("rocket_*.json"):

        if not path.name.startswith(
            ("rocket_mini_", "rocket_multi_")
        ):
            raise RuntimeError(
                "Legacy results found. Back up old results "
                "outside this directory first."
            )

    metadata = {
        "tag": args.tag,
        "data_sha256": hashlib.sha256(
            prepared.read_bytes()
        ).hexdigest(),

        "missing_seed": int(
            cfg.get("missing", {}).get("seed", 123)
        ),

        "python": platform.python_version(),
        "platform": platform.platform(),

        "versions": {
            name: version(name)
            for name in [
                "aeon", "numpy",
                "scikit-learn", "pandas"
            ]
        },

        "parameters": parameters,

        "source_sha256": {
            name: hashlib.sha256(
                (ROOT / name).read_bytes()
            ).hexdigest()
            for name in [
                "models/rocket.py",
                "trainers/rocket_trainer.py",
                "tools/run_rocket.py",
                "datasets/features.py",
                "evaluation/metrics.py"
            ]
        }
    }

    settings_hash = fingerprint(metadata)

    settings_path = out / f"settings_{variant}.json"

    if settings_path.exists():

        old = json.loads(
            settings_path.read_text(encoding="utf-8")
        )

        if old["settings_sha256"] != settings_hash:
            raise RuntimeError(
                "Settings changed: use a new experiment "
                "tag or output directory."
            )

    else:
        write_json(settings_path, {
            **metadata,
            "settings_sha256": settings_hash
        })

    for config, mask in masks.items():

        stem = f"rocket_{variant}_{config.key}"

        result_path = out / f"{stem}.json"

        prediction_path = (
            out / "predictions" / f"{stem}.csv.gz"
        )

        run_hash = fingerprint({
            "settings": settings_hash,
            "config": dataclasses.asdict(config),
            "mask_sha256": hashlib.sha256(
                mask.tobytes()
            ).hexdigest()
        })

        if result_path.exists():

            old = json.loads(
                result_path.read_text(encoding="utf-8")
            )

            if (
                old.get("metadata", {}).get(
                    "run_sha256"
                ) == run_hash
                and prediction_path.exists()
            ):
                logger.info("SKIP %s", stem)
                continue

            raise RuntimeError(
                f"Incomplete or incompatible: {result_path}"
            )

        logger.info(
            "RUN %s | seed=%d | kernels=%d",
            stem, seed, kernels
        )

        result = run_rocket(
            X, y, mask,
            tr, va, te, config,
            end_time=data["end_time"],
            prediction_path=prediction_path,
            **parameters
        )

        record = dataclasses.asdict(result)

        record["metadata"].update({
            **metadata,
            "settings_sha256": settings_hash,
            "run_sha256": run_hash
        })

        write_json(result_path, record)

        logger.info(
            "DONE %s | Macro F1=%.4f | PR-AUC=%.4f",
            stem,
            result.test_metrics["macro_f1"],
            result.test_metrics["pr_auc"]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
