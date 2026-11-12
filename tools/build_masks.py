"""Sinh + cache mặt nạ thiếu dữ liệu cho mọi (kind, rate) cần dùng (Mục 4.3).

    python tools/build_masks.py --config configs/rain_snow_missing.yaml

Mặt nạ được sinh trên TOÀN BỘ cửa sổ (train+val+test gộp) nên mọi nhánh dùng
đúng cùng một file cache theo khoá ``(kind, rate, seed, n_samples)``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.cache import load_prepared_data  # noqa: E402
from datasets.missingness import KINDS, MissingnessConfig, generate_or_load_mask  # noqa: E402
from utils.utils import load_yaml, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_missing.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data", default="outputs/data/windows.npz", help="File .npz từ prepare_data.py.")
    args = parser.parse_args(argv)

    setup_logger()
    logger = logging.getLogger("p4aids")

    cfg = load_yaml(args.config)
    missing_cfg = cfg["missing"]
    kinds = missing_cfg.get("kinds", list(KINDS))
    rates = missing_cfg.get("rates", [0.10, 0.30, 0.50])
    seed = int(missing_cfg.get("seed", 123))
    cache_dir = missing_cfg.get("cache_dir", "outputs/masks")

    data = load_prepared_data(args.data)
    X = data["X"]

    for kind in kinds:
        for rate in rates:
            config = MissingnessConfig(kind=kind, rate=float(rate))
            generate_or_load_mask(X, config, seed=seed, cache_dir=cache_dir)

    logger.info("Hoàn tất: %d mặt nạ (%d kind x %d rate) trong '%s'.",
                len(kinds) * len(rates), len(kinds), len(rates), cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
