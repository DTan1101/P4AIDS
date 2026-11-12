"""Làm sạch dữ liệu Szeged, cắt cửa sổ, chia theo năm, cache ra .npz (Mục 2, 4.2).

    python tools/prepare_data.py --config configs/rain_snow_data.yaml
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

from datasets.cache import save_prepared_data  # noqa: E402
from datasets.splits import year_split  # noqa: E402
from datasets.szeged import DEFAULT_CSV_PATH, clean, load_raw  # noqa: E402
from datasets.windows import build_windows  # noqa: E402
from utils.utils import load_yaml, setup_logger  # noqa: E402

DEFAULT_CONFIG = "configs/rain_snow_data.yaml"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="File YAML config dữ liệu.")
    parser.add_argument("--output", default=None, help="Override data.output_path.")
    args = parser.parse_args(argv)

    setup_logger()
    logger = logging.getLogger("p4aids")

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    df_raw = load_raw(data_cfg.get("csv_path", DEFAULT_CSV_PATH))
    df = clean(df_raw)
    windows = build_windows(df, window=int(data_cfg.get("window", 24)))
    split = year_split(
        windows.end_time,
        train_years=tuple(data_cfg.get("train_years", [2006, 2013])),
        val_years=tuple(data_cfg.get("val_years", [2014, 2014])),
        test_years=tuple(data_cfg.get("test_years", [2015, 2016])),
    )

    output_path = args.output or data_cfg.get("output_path", "outputs/data/windows.npz")
    save_prepared_data(output_path, windows.X, windows.y, windows.end_time,
                        split.train_idx, split.val_idx, split.test_idx)
    logger.info("Đã lưu %d cửa sổ vào '%s' (train=%d, val=%d, test=%d).",
                len(windows.y), output_path, len(split.train_idx), len(split.val_idx), len(split.test_idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
