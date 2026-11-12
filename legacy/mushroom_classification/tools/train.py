#!/usr/bin/env python3
"""Điểm vào huấn luyện.

    python tools/train.py --config configs/train.yaml
    python tools/train.py --train-dir data/train --epochs 20 --batch-size 32

Thay cho legacy/main.py: phần khai báo cứng nay nằm trong configs/train.yaml.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.dataloader import build_dataloaders  # noqa: E402
from models.model import build_models  # noqa: E402
from options.train_options import parse_train_options  # noqa: E402
from trainers.mutual_trainer import MutualTrainer  # noqa: E402
from utils.utils import (  # noqa: E402
    count_parameters,
    dump_config,
    ensure_dir,
    get_device,
    seed_everything,
    setup_logger,
)


def build_meta(cfg, data) -> dict:
    """Metadata ghi kèm checkpoint để tools/eval.py dựng lại đúng pipeline."""
    return {
        "experiment": cfg["experiment"]["name"],
        "classes": data.classes,
        "class_to_idx": data.class_to_idx,
        "label_mapping": data.label_mapping,
        "mean": list(data.mean),
        "std": list(data.std),
        "image_size": int(cfg["data"].get("image_size", 224)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_train_options(argv)
    run_dir = ensure_dir(cfg["experiment"]["run_dir"])
    logger = setup_logger("p4aids", os.path.join(run_dir, "train.log"))

    seed_everything(int(cfg["experiment"].get("seed", 0)),
                    bool(cfg["experiment"].get("deterministic", False)))
    device = get_device(cfg["experiment"].get("device", "auto"))
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    logger.info("Run: %s | device: %s | kết quả: %s",
                cfg["experiment"]["name"], device, run_dir)

    # --- dữ liệu ---------------------------------------------------------- #
    data = build_dataloaders(cfg, cache_path=os.path.join(run_dir, "dataset_stats.json"),
                             with_test=False)

    configured_classes = cfg["model"].get("num_classes")
    num_classes = int(configured_classes or data.num_classes)
    if configured_classes and int(configured_classes) != data.num_classes:
        logger.warning("model.num_classes=%s khác số lớp thật trong '%s' (%d). "
                       "Đặt num_classes: null để lấy tự động.",
                       configured_classes, cfg["data"]["train_dir"], data.num_classes)
    cfg["model"]["num_classes"] = num_classes

    # --- model ------------------------------------------------------------ #
    models = build_models(cfg["model"], num_classes=num_classes)
    for model in models:
        logger.info("%s: %.1fM tham số huấn luyện được",
                    model.backbone_name, count_parameters(model) / 1e6)

    # --- huấn luyện ------------------------------------------------------- #
    trainer = MutualTrainer(
        models=models,
        train_cfg=cfg["train"],
        device=device,
        output_dir=run_dir,
        meta=build_meta(cfg, data),
        class_names=data.classes,
    )
    result = trainer.fit(data.train_loader, data.val_loader)

    logger.info("Checkpoint đã lưu:")
    for path in result["checkpoints"]:
        logger.info("  %s", path)
    logger.info("Chạy dự đoán: python tools/eval.py --name %s --test-dir %s",
                cfg["experiment"]["name"], cfg["data"].get("test_dir", "data/test"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
