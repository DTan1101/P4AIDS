"""Ghép backbone + head thành model phân loại và các factory tương ứng."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from models.backbone import build_backbone, normalize_backbone_name
from models.head import build_head

logger = logging.getLogger("p4aids")


class Classifier(nn.Module):
    """Model phân loại = backbone trích đặc trưng + head MLP.

    Thay cho ba class ``efficientnetb6_model`` / ``efficientnetv2_m_model`` /
    ``resnet101_model`` gần như trùng nhau ở code cũ.
    """

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 weights: Optional[str] = None, head: Optional[Mapping] = None,
                 freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone_name = normalize_backbone_name(backbone)
        self.num_classes = int(num_classes)
        self.head_cfg: Dict[str, Any] = dict(head or {})

        self.backbone = build_backbone(self.backbone_name, pretrained=pretrained,
                                       weights=weights, freeze=freeze_backbone)
        self.head = build_head(self.backbone.out_features, self.num_classes, self.head_cfg)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self, frozen: bool = True) -> "Classifier":
        self.backbone.freeze(frozen)
        return self

    @property
    def config(self) -> Dict[str, Any]:
        """Thông tin đủ để dựng lại model khi load checkpoint."""
        return {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "head": dict(self.head_cfg),
        }

    def __repr__(self) -> str:  # pragma: no cover - chỉ để log
        return (f"Classifier(backbone={self.backbone_name}, "
                f"features={self.backbone.out_features}, classes={self.num_classes})")


def _backbone_entries(model_cfg: Mapping) -> List[Dict[str, Any]]:
    """Chuẩn hoá ``model.backbones`` về danh sách dict cấu hình."""
    raw = model_cfg.get("backbones") or model_cfg.get("backbone")
    if raw is None:
        raise KeyError("Config thiếu 'model.backbones'.")
    if isinstance(raw, (str, Mapping)):
        raw = [raw]

    entries: List[Dict[str, Any]] = []
    for item in raw:
        entries.append({"name": item} if isinstance(item, str) else dict(item))
    return entries


def build_model(model_cfg: Mapping, backbone: Optional[str] = None,
                num_classes: Optional[int] = None) -> Classifier:
    """Tạo một model từ ``model_cfg``.

    ``backbone`` cho phép chỉ định thẳng tên backbone; nếu bỏ trống sẽ lấy
    phần tử đầu tiên trong ``model.backbones``.
    """
    entry = {"name": backbone} if backbone else _backbone_entries(model_cfg)[0]
    name = entry["name"]

    overrides = (model_cfg.get("head_overrides") or {}).get(normalize_backbone_name(name), {})
    head_cfg = dict(model_cfg.get("head") or {})
    head_cfg.update(overrides)
    head_cfg.update(entry.get("head") or {})

    return Classifier(
        backbone=name,
        num_classes=int(num_classes if num_classes is not None else model_cfg["num_classes"]),
        pretrained=bool(entry.get("pretrained", model_cfg.get("pretrained", True))),
        weights=entry.get("weights", model_cfg.get("weights")),
        head=head_cfg,
        freeze_backbone=bool(entry.get("freeze_backbone",
                                       model_cfg.get("freeze_backbone", False))),
    )


def build_models(model_cfg: Mapping, num_classes: Optional[int] = None) -> List[Classifier]:
    """Tạo toàn bộ model tham gia deep mutual learning."""
    models = [build_model(model_cfg, backbone=entry["name"], num_classes=num_classes)
              for entry in _backbone_entries(model_cfg)]
    for model in models:
        logger.info("Đã tạo %s", model)
    return models


def build_model_from_meta(meta: Mapping, pretrained: bool = False) -> Classifier:
    """Dựng lại model từ metadata lưu trong checkpoint (dùng khi inference)."""
    return Classifier(
        backbone=meta["backbone"],
        num_classes=int(meta["num_classes"]),
        pretrained=pretrained,
        head=meta.get("head"),
    )
