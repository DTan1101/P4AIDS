#!/usr/bin/env python3
"""Điểm vào suy luận / đánh giá.

    python tools/eval.py --name dml_mushroom --test-dir data/test
    python tools/eval.py --checkpoints outputs/run/checkpoints/best_model1_resnet101.pth

Gộp legacy/test.py và libs/predict.py: nạp checkpoint, chạy ensemble, xuất CSV.
Nếu thư mục test được chia theo lớp (có nhãn thật), script sẽ in thêm báo cáo
accuracy / F1 / ma trận nhầm lẫn.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.dataloader import build_test_loader, resolve_label_mapping  # noqa: E402
from datasets.dataset import TEST_LABEL, get_image_paths, get_targets  # noqa: E402
from datasets.preprocess_data import DEFAULT_MEAN, DEFAULT_STD, MushroomTransforms  # noqa: E402
from models.model import build_model_from_meta  # noqa: E402
from options.eval_option import parse_eval_options  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402
from utils.utils import ensure_dir, get_device, load_checkpoint, seed_everything, setup_logger  # noqa: E402

logger = logging.getLogger("p4aids")


def discover_checkpoints(eval_cfg: Mapping) -> List[str]:
    """Lấy danh sách checkpoint từ config, hoặc dò trong thư mục checkpoint."""
    checkpoints = eval_cfg.get("checkpoints")
    if checkpoints:
        paths = [checkpoints] if isinstance(checkpoints, str) else list(checkpoints)
    else:
        checkpoint_dir = eval_cfg.get("checkpoint_dir")
        paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pth"))) if checkpoint_dir else []

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {missing}")
    if not paths:
        raise FileNotFoundError(
            f"Không có checkpoint nào trong '{eval_cfg.get('checkpoint_dir')}'. "
            "Hãy chạy tools/train.py trước, hoặc chỉ định --checkpoints.")
    return paths


def load_models(paths: Sequence[str], device: torch.device,
                fallback_cfg: Optional[Mapping] = None) -> Tuple[List[torch.nn.Module], Dict[str, Any]]:
    """Nạp model từ checkpoint; ưu tiên metadata lưu sẵn trong file."""
    models: List[torch.nn.Module] = []
    merged_meta: Dict[str, Any] = {}

    for index, path in enumerate(paths):
        checkpoint = load_checkpoint(path, map_location=device)
        meta = dict(checkpoint.get("meta") or {})

        if "backbone" not in meta:
            backbones = (fallback_cfg or {}).get("backbones") or []
            if index >= len(backbones):
                raise ValueError(
                    f"Checkpoint '{path}' không có metadata backbone. Hãy chỉ định thủ công, "
                    "ví dụ: --opts model.backbones='[efficientnet_b6, resnet101]'")
            meta.setdefault("backbone", backbones[index])
            meta.setdefault("num_classes", (fallback_cfg or {}).get("num_classes"))
            meta.setdefault("head", (fallback_cfg or {}).get("head"))

        model = build_model_from_meta(meta, pretrained=False)
        model.load_state_dict(checkpoint["model"])
        models.append(model.to(device).eval())

        for key, value in meta.items():
            merged_meta.setdefault(key, value)
        logger.info("Đã nạp %s (%s, epoch %s, score %s)", os.path.basename(path),
                    meta.get("backbone"), meta.get("epoch", "?"),
                    f"{meta['score']:.4f}" if isinstance(meta.get("score"), float) else "?")

    return models, merged_meta


@torch.no_grad()
def predict_probabilities(models: Sequence[torch.nn.Module], loader, device: torch.device,
                          use_tta: bool = False, ensemble: bool = True) -> np.ndarray:
    """Trả về ma trận xác suất (N, num_classes).

    ``ensemble=False`` chỉ dùng model đầu tiên; ``use_tta`` lấy trung bình thêm
    ảnh lật ngang và lật dọc.
    """
    active_models = list(models) if ensemble else [models[0]]
    batches: List[np.ndarray] = []

    for batch_index, (inputs, _) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        views = [inputs]
        if use_tta:
            views += [torch.flip(inputs, dims=[3]), torch.flip(inputs, dims=[2])]

        probabilities = None
        for view in views:
            for model in active_models:
                current = F.softmax(model(view), dim=1)
                probabilities = current if probabilities is None else probabilities + current
        probabilities = probabilities / (len(views) * len(active_models))
        batches.append(probabilities.cpu().numpy())

        if batch_index % 10 == 0:
            logger.info("Đã xử lý %d/%d batch", batch_index, len(loader))

    return np.concatenate(batches, axis=0) if batches else np.empty((0, 0))


def write_submission(image_paths: Sequence[str], labels: Sequence[int],
                     output_csv: str, submission_cfg: Mapping) -> "Any":
    """Ghi file CSV kết quả theo đúng định dạng nộp bài."""
    import pandas as pd

    columns = list(submission_cfg.get("columns", ["image_name", "label"]))
    keep_extension = bool(submission_cfg.get("keep_extension", False))

    names = [os.path.basename(path) for path in image_paths]
    if not keep_extension:
        names = [os.path.splitext(name)[0] for name in names]

    frame = pd.DataFrame({columns[0]: names, columns[1]: list(labels)})
    ensure_dir(os.path.dirname(output_csv))
    frame.to_csv(output_csv, index=False)
    logger.info("Đã ghi %d dòng vào %s", len(frame), output_csv)
    logger.info("Xem trước:\n%s", frame.head(10).to_string(index=False))
    return frame


def report_metrics(predictions: np.ndarray, targets: Sequence[int],
                   class_names: Sequence[str]) -> None:
    """In báo cáo khi tập test có nhãn thật."""
    matrix = ConfusionMatrix(len(class_names), class_names)
    matrix.update(torch.as_tensor(predictions), torch.as_tensor(list(targets)))
    logger.info("Kết quả trên tập có nhãn:\n%s", matrix.report())
    logger.info("Ma trận nhầm lẫn (hàng = nhãn thật):\n%s", matrix.matrix_str())


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_eval_options(argv)
    run_dir = ensure_dir(cfg["experiment"]["run_dir"])
    setup_logger("p4aids", os.path.join(run_dir, "eval.log"))

    seed_everything(int(cfg["experiment"].get("seed", 0)))
    device = get_device(cfg["experiment"].get("device", "auto"))

    # --- model ------------------------------------------------------------ #
    checkpoint_paths = discover_checkpoints(cfg["eval"])
    models, meta = load_models(checkpoint_paths, device, fallback_cfg=cfg.get("model"))

    # --- dữ liệu ----------------------------------------------------------- #
    data_cfg = cfg["data"]
    normalize_cfg = data_cfg.get("normalize") or {}
    mean = normalize_cfg.get("mean") or meta.get("mean") or DEFAULT_MEAN
    std = normalize_cfg.get("std") or meta.get("std") or DEFAULT_STD
    image_size = int(data_cfg.get("image_size") or meta.get("image_size") or 224)
    data_cfg["image_size"] = image_size
    logger.info("Tiền xử lý: image_size=%d, mean=%s", image_size, [round(v, 4) for v in mean])

    eval_transform = MushroomTransforms(mean=mean, std=std, image_size=image_size).eval_transform()
    test_loader = build_test_loader(data_cfg, eval_transform)

    # --- suy luận ---------------------------------------------------------- #
    probabilities = predict_probabilities(
        models, test_loader, device,
        use_tta=bool(cfg["eval"].get("tta", False)),
        ensemble=bool(cfg["eval"].get("ensemble", True)),
    )
    predictions = probabilities.argmax(axis=1)

    # --- ánh xạ nhãn & xuất kết quả ---------------------------------------- #
    class_names = list(meta.get("classes") or [])
    mapping_cfg = data_cfg.get("class_to_label")
    if mapping_cfg:
        label_mapping = resolve_label_mapping(class_names, mapping_cfg)
    else:
        stored = meta.get("label_mapping") or {}
        label_mapping = {int(key): int(value) for key, value in stored.items()}
    mapped = [int(label_mapping.get(int(p), int(p))) for p in predictions]

    output_csv = cfg["eval"]["output_csv"]
    write_submission(get_image_paths(test_loader.dataset), mapped, output_csv,
                     cfg["eval"].get("submission") or {})

    if cfg["eval"].get("save_probabilities"):
        npy_path = os.path.splitext(output_csv)[0] + "_probs.npy"
        np.save(npy_path, probabilities)
        logger.info("Đã lưu xác suất thô: %s", npy_path)

    # --- chấm điểm offline nếu tập test có nhãn ----------------------------- #
    targets = get_targets(test_loader.dataset)
    if class_names and targets and all(t != TEST_LABEL for t in targets):
        report_metrics(predictions, targets, class_names)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
