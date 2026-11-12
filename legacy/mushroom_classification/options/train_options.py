"""Tham số dòng lệnh cho huấn luyện, hợp nhất với file YAML trong configs/."""

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

DEFAULT_CONFIG = os.path.join("configs", "train.yaml")

# Ánh xạ tham số CLI -> khoá trong config. CLI luôn thắng file YAML.
ARG_TO_CONFIG = {
    "name": "experiment.name",
    "output_dir": "experiment.output_dir",
    "seed": "experiment.seed",
    "device": "experiment.device",
    "deterministic": "experiment.deterministic",
    "train_dir": "data.train_dir",
    "test_dir": "data.test_dir",
    "image_size": "data.image_size",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "val_split": "data.val_split",
    "multi_augment": "data.use_multi_augment",
    "backbones": "model.backbones",
    "num_classes": "model.num_classes",
    "pretrained": "model.pretrained",
    "epochs": "train.num_epochs",
    "lr": "train.learning_rate",
    "weight_decay": "train.weight_decay",
    "patience": "train.patience",
    "mutual_weight": "train.mutual_weight",
    "amp": "train.amp",
}


class TrainOptions:
    """Gom việc parse CLI và nạp config thành một chỗ duy nhất."""

    def __init__(self) -> None:
        self.parser = self._build_parser()

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Huấn luyện ensemble deep mutual learning cho phân loại ảnh nấm.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--config", default=DEFAULT_CONFIG, help="Đường dẫn file YAML.")

        experiment = parser.add_argument_group("thí nghiệm")
        experiment.add_argument("--name", help="Tên run, dùng làm thư mục con trong outputs/.")
        experiment.add_argument("--output-dir", help="Thư mục gốc chứa kết quả.")
        experiment.add_argument("--seed", type=int, help="Seed cố định kết quả.")
        experiment.add_argument("--device", help="cuda / cpu / auto.")
        experiment.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                                default=None, help="Bật chế độ cudnn deterministic.")

        data = parser.add_argument_group("dữ liệu")
        data.add_argument("--train-dir", help="Thư mục train (mỗi lớp một thư mục con).")
        data.add_argument("--test-dir", help="Thư mục test.")
        data.add_argument("--image-size", type=int, help="Kích thước ảnh đầu vào.")
        data.add_argument("--batch-size", type=int)
        data.add_argument("--num-workers", type=int)
        data.add_argument("--val-split", type=float, help="Tỉ lệ tách validation.")
        data.add_argument("--multi-augment", action=argparse.BooleanOptionalAction,
                          default=None, help="Nhân bản ảnh train qua nhiều transform.")

        model = parser.add_argument_group("model")
        model.add_argument("--backbones", nargs="+",
                           help="Danh sách backbone, ví dụ: efficientnet_b6 resnet101.")
        model.add_argument("--num-classes", type=int,
                           help="Bỏ trống để lấy theo số lớp thật của tập train.")
        model.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)

        training = parser.add_argument_group("huấn luyện")
        training.add_argument("--epochs", type=int)
        training.add_argument("--lr", type=float, help="Learning rate.")
        training.add_argument("--weight-decay", type=float)
        training.add_argument("--patience", type=int, help="0 = tắt early stopping.")
        training.add_argument("--mutual-weight", type=float,
                              help="Trọng số của thành phần KL học lẫn nhau.")
        training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                              help="Mixed precision (chỉ có tác dụng trên CUDA).")

        parser.add_argument("--opts", nargs="*", metavar="KEY=VALUE", default=[],
                            help="Ghi đè bất kỳ khoá nào, ví dụ: train.scheduler.type=cosine")
        return parser

    # ------------------------------------------------------------------ #
    def parse(self, argv: Optional[Sequence[str]] = None) -> ConfigDict:
        args = self.parser.parse_args(argv)
        cfg = to_config(load_yaml(args.config))

        override_from_args(cfg, args, ARG_TO_CONFIG)
        apply_overrides(cfg, args.opts)

        # Dùng `or` thay cho setdefault: file YAML có thể khai báo khoá với
        # giá trị null, khi đó setdefault sẽ không thay bằng giá trị mặc định.
        cfg.setdefault("experiment", ConfigDict())
        experiment = cfg["experiment"]
        experiment["name"] = experiment.get("name") or "dml_mushroom"
        experiment["output_dir"] = experiment.get("output_dir") or "outputs"
        experiment["config_file"] = args.config
        experiment["run_dir"] = os.path.join(experiment["output_dir"], experiment["name"])
        self._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg: ConfigDict) -> None:
        if not cfg.get("data", {}).get("train_dir"):
            raise ValueError("Thiếu 'data.train_dir' (đặt trong config hoặc dùng --train-dir).")
        if not cfg.get("model", {}).get("backbones"):
            raise ValueError("Thiếu 'model.backbones'.")


def parse_train_options(argv: Optional[Sequence[str]] = None) -> ConfigDict:
    """Hàm tiện lợi cho tools/train.py."""
    return TrainOptions().parse(argv)
