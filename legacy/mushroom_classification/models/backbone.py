"""Backbone trích đặc trưng, dựng từ torchvision và bỏ lớp phân loại gốc."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models

logger = logging.getLogger("p4aids")

# Tên rút gọn quen dùng ở code cũ -> tên chuẩn của torchvision.
BACKBONE_ALIASES = {
    "efficientnetb6": "efficientnet_b6",
    "efficientnet_b6_model": "efficientnet_b6",
    "efficientnetv2_m": "efficientnet_v2_m",
    "efficientnetv2m": "efficientnet_v2_m",
    "resnet101_model": "resnet101",
}

# Các thuộc tính chứa lớp phân loại cuối, theo thứ tự ưu tiên khi dò tìm.
_CLASSIFIER_ATTRS = ("classifier", "fc", "head", "heads")


def normalize_backbone_name(name: str) -> str:
    key = name.strip().lower()
    return BACKBONE_ALIASES.get(key, key)


def _last_linear_in_features(module: nn.Module) -> Optional[int]:
    """Tìm ``in_features`` của lớp Linear cuối cùng trong module."""
    if isinstance(module, nn.Linear):
        return module.in_features
    last = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last = child
    return last.in_features if last is not None else None


def _strip_classifier(net: nn.Module) -> int:
    """Thay lớp phân loại của backbone bằng Identity, trả về số chiều đặc trưng."""
    for attr in _CLASSIFIER_ATTRS:
        module = getattr(net, attr, None)
        if module is None:
            continue
        in_features = _last_linear_in_features(module)
        if in_features is None:
            continue
        setattr(net, attr, nn.Identity())
        return in_features
    raise ValueError(f"Không xác định được lớp phân loại của {type(net).__name__}.")


def _create_torchvision_model(name: str, weights: Optional[str]) -> nn.Module:
    if hasattr(tv_models, "get_model"):
        return tv_models.get_model(name, weights=weights)
    builder = getattr(tv_models, name, None)  # torchvision cũ
    if builder is None:
        raise ValueError(f"torchvision không có backbone '{name}'.")
    return builder(weights=weights)


class Backbone(nn.Module):
    """Bọc một model torchvision, chỉ giữ phần trích đặc trưng.

    Args:
        name: tên backbone của torchvision, ví dụ ``efficientnet_b6``.
        pretrained: có nạp trọng số ImageNet hay không.
        weights: tên preset trọng số cụ thể (``IMAGENET1K_V2``...). Mặc định
            ``DEFAULT`` - trùng với các preset mà code cũ chọn tay.
        freeze: đóng băng toàn bộ backbone (chỉ train phần head).
    """

    def __init__(self, name: str, pretrained: bool = True,
                 weights: Optional[str] = None, freeze: bool = False) -> None:
        super().__init__()
        self.name = normalize_backbone_name(name)
        weights_name = (weights or "DEFAULT") if pretrained else None

        try:
            self.net = _create_torchvision_model(self.name, weights_name)
        except Exception as exc:
            if weights_name is None:
                raise
            logger.warning("Không tải được trọng số pretrained cho '%s' (%s); "
                           "khởi tạo ngẫu nhiên.", self.name, exc)
            self.net = _create_torchvision_model(self.name, None)

        self.out_features = _strip_classifier(self.net)
        if freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        if features.ndim > 2:  # một số backbone trả về (N, C, 1, 1)
            features = torch.flatten(features, 1)
        return features

    def freeze(self, frozen: bool = True) -> "Backbone":
        for parameter in self.net.parameters():
            parameter.requires_grad = not frozen
        return self

    def unfreeze(self) -> "Backbone":
        return self.freeze(False)


def build_backbone(name: str, pretrained: bool = True, weights: Optional[str] = None,
                   freeze: bool = False) -> Backbone:
    """Factory tạo backbone theo tên."""
    return Backbone(name=name, pretrained=pretrained, weights=weights, freeze=freeze)


def available_backbones() -> list:  # pragma: no cover - tiện tra cứu
    if hasattr(tv_models, "list_models"):
        return sorted(tv_models.list_models(module=tv_models))
    return sorted(n for n in dir(tv_models) if n.islower() and not n.startswith("_"))
