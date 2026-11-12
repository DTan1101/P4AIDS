"""Tham số dòng lệnh chung cho các script ``tools/run_*.py`` (Chương 4-6 báo cáo).

Theo đúng quy ước cũ của repo (``ConfigDict`` + YAML + override CLI): mỗi script
gọi :func:`build_base_parser` để có sẵn ``--config/--seed/--missing-kind/
--missing-rate/--opts``, tự thêm cờ riêng của mình, rồi gọi :func:`parse_experiment_options`.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from datasets.missingness import KINDS, RATES
from utils.utils import ConfigDict, apply_overrides, load_yaml, override_from_args, set_nested


def build_base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config, help="Đường dẫn file YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Seed, override experiment.seed.")
    parser.add_argument("--missing-kind", type=str, default=None, choices=list(KINDS),
                         help="Chỉ chạy một kiểu mất dữ liệu (mặc định: chạy cả %s)." % list(KINDS))
    parser.add_argument("--missing-rate", type=float, default=None, choices=list(RATES),
                         help="Chỉ chạy một tỷ lệ mất dữ liệu (mặc định: chạy cả %s)." % list(RATES))
    parser.add_argument("--opts", nargs="*", default=None,
                         help="Override dạng key.path=value, ví dụ train.epochs=10.")
    return parser


def parse_experiment_options(argv: Optional[list], description: str, default_config: str,
                              arg_to_config: Mapping[str, str],
                              add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None
                              ) -> Tuple[ConfigDict, argparse.Namespace]:
    """Parse CLI + YAML, trả về ``(config, args)``.

    ``args`` được trả thêm (ngoài ``config``) vì ``--missing-kind``/``--missing-rate``
    là bộ lọc chạy thí nghiệm (không phải một khoá config cố định) -- các script
    ``tools/run_*.py`` tự đọc trực tiếp từ ``args``.
    """
    parser = build_base_parser(description, default_config)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    cfg.setdefault("experiment", ConfigDict())
    if args.seed is not None:
        set_nested(cfg, "experiment.seed", args.seed)
    override_from_args(cfg, args, arg_to_config)
    apply_overrides(cfg, args.opts)
    return cfg, args


def missing_configs_from_args(args: argparse.Namespace):
    """Danh sách :class:`~datasets.missingness.MissingnessConfig` cần chạy theo CLI filter."""
    from datasets.missingness import ALL_CONFIGS, MissingnessConfig

    kinds = [args.missing_kind] if args.missing_kind else list(KINDS)
    rates = [args.missing_rate] if args.missing_rate else list(RATES)
    selected = {MissingnessConfig(k, r) for k in kinds for r in rates}
    return [c for c in ALL_CONFIGS if c in selected]


def resolve_masks(X, args: argparse.Namespace, missing_cfg: Mapping) -> Dict:
    """Mặt nạ cho mọi cấu hình cần chạy theo CLI filter, cộng thêm FULL_DATA_CONFIG
    (0% thiếu, Mục 6.1) khi người dùng không lọc theo một ``(kind, rate)`` cụ thể.
    """
    import numpy as np

    from datasets.missingness import FULL_DATA_CONFIG, load_masks_for_configs

    seed = int(missing_cfg.get("seed", 123))
    cache_dir = missing_cfg.get("cache_dir", "outputs/masks")
    requested = missing_configs_from_args(args)
    masks = load_masks_for_configs(X, requested, seed=seed, cache_dir=cache_dir)

    if not (args.missing_kind or args.missing_rate):
        masks = {FULL_DATA_CONFIG: ~np.isnan(X), **masks}
    return masks
