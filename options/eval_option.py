"""Tham số dòng lệnh cho đánh giá / suy luận, hợp nhất với configs/eval.yaml."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from utils.utils import (
    ConfigDict,
    apply_overrides,
    load_yaml,
    override_from_args,
    to_config,
)

DEFAULT_CONFIG = os.path.join("configs", "eval.yaml")

ARG_TO_CONFIG = {
    "name": "experiment.name",
    "output_dir": "experiment.output_dir",
    "seed": "experiment.seed",
    "device": "experiment.device",
    "test_dir": "data.test_dir",
    "image_size": "data.image_size",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "checkpoints": "eval.checkpoints",
    "checkpoint_dir": "eval.checkpoint_dir",
    "output_csv": "eval.output_csv",
    "ensemble": "eval.ensemble",
    "tta": "eval.tta",
}


def _resolve_run_dir(cfg: ConfigDict, config_file: str) -> str:
    """Chuẩn hoá nhánh ``experiment`` và trả về thư mục của run."""
    cfg.setdefault("experiment", ConfigDict())
    experiment = cfg["experiment"]
    experiment["name"] = experiment.get("name") or "dml_mushroom"
    experiment["output_dir"] = experiment.get("output_dir") or "outputs"
    experiment["config_file"] = config_file
    experiment["run_dir"] = os.path.join(experiment["output_dir"], experiment["name"])
    return experiment["run_dir"]


class EvalOptions:
    """Parse CLI + config cho tools/eval.py."""

    def __init__(self) -> None:
        self.parser = self._build_parser()

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Dự đoán trên tập test và xuất file kết quả CSV.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--config", default=DEFAULT_CONFIG, help="Đường dẫn file YAML.")

        experiment = parser.add_argument_group("thí nghiệm")
        experiment.add_argument("--name", help="Tên run cần đánh giá (khớp với lúc train).")
        experiment.add_argument("--output-dir", help="Thư mục gốc chứa kết quả.")
        experiment.add_argument("--seed", type=int)
        experiment.add_argument("--device", help="cuda / cpu / auto.")

        data = parser.add_argument_group("dữ liệu")
        data.add_argument("--test-dir", help="Thư mục ảnh cần dự đoán.")
        data.add_argument("--image-size", type=int)
        data.add_argument("--batch-size", type=int)
        data.add_argument("--num-workers", type=int)

        evaluation = parser.add_argument_group("suy luận")
        evaluation.add_argument("--checkpoints", nargs="+",
                                help="Danh sách file .pth. Bỏ trống sẽ tự dò trong "
                                     "<run_dir>/checkpoints.")
        evaluation.add_argument("--checkpoint-dir", help="Thư mục chứa checkpoint.")
        evaluation.add_argument("--output-csv", help="Đường dẫn file CSV kết quả.")
        evaluation.add_argument("--ensemble", action=argparse.BooleanOptionalAction,
                                default=None, help="Trung bình xác suất của các model.")
        evaluation.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None,
                                help="Test-time augmentation bằng lật ngang/dọc.")

        parser.add_argument("--opts", nargs="*", metavar="KEY=VALUE", default=[],
                            help="Ghi đè bất kỳ khoá nào, ví dụ: data.batch_size=16")
        return parser

    # ------------------------------------------------------------------ #
    def parse(self, argv: Optional[Sequence[str]] = None) -> ConfigDict:
        args = self.parser.parse_args(argv)
        cfg = to_config(load_yaml(args.config))

        override_from_args(cfg, args, ARG_TO_CONFIG)
        apply_overrides(cfg, args.opts)

        run_dir = _resolve_run_dir(cfg, args.config)

        # Dùng `or` thay cho setdefault vì file YAML khai báo sẵn các khoá này
        # với giá trị null (nghĩa là "tự suy ra"), setdefault sẽ không ghi đè.
        cfg.setdefault("eval", ConfigDict())
        cfg["eval"]["checkpoint_dir"] = (cfg["eval"].get("checkpoint_dir")
                                         or os.path.join(run_dir, "checkpoints"))
        cfg["eval"]["output_csv"] = (cfg["eval"].get("output_csv")
                                     or os.path.join(run_dir, "results.csv"))
        self._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg: ConfigDict) -> None:
        if not cfg.get("data", {}).get("test_dir"):
            raise ValueError("Thiếu 'data.test_dir' (đặt trong config hoặc dùng --test-dir).")


def parse_eval_options(argv: Optional[Sequence[str]] = None) -> ConfigDict:
    """Hàm tiện lợi cho tools/eval.py."""
    return EvalOptions().parse(argv)
