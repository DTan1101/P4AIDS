"""Head phân loại gắn sau backbone."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "gelu": nn.GELU,
    "silu": lambda: nn.SiLU(inplace=True),
}


def _as_dropout_list(dropout, num_layers: int) -> list:
    """Chuẩn hoá tham số dropout thành danh sách đúng ``num_layers`` phần tử."""
    if dropout is None:
        return [0.0] * num_layers
    if isinstance(dropout, (int, float)):
        return [float(dropout)] * num_layers
    values = [float(v) for v in dropout]
    if len(values) < num_layers:
        values += [0.0] * (num_layers - len(values))
    return values[:num_layers]


class ClassifierHead(nn.Module):
    """MLP head: ``[Dropout -> Linear -> Act] * n -> Dropout -> Linear``.

    Mỗi phần tử của ``dropout`` là xác suất dropout **đặt trước** một lớp
    Linear (dropout = 0 thì bỏ qua), nên tái hiện được cả hai kiểu head ở
    code cũ: EfficientNet dùng ``[0.2, 0.3, 0.4]`` còn ResNet dùng
    ``[0.0, 0.5, 0.5]``.
    """

    def __init__(self, in_features: int, num_classes: int,
                 hidden_dims: Sequence[int] = (1024, 512),
                 dropout=(0.2, 0.3, 0.4), activation: str = "relu") -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise KeyError(f"Activation '{activation}' không hỗ trợ. "
                           f"Hợp lệ: {tuple(ACTIVATIONS)}")

        hidden_dims = [int(dim) for dim in (hidden_dims or [])]
        probabilities = _as_dropout_list(dropout, len(hidden_dims) + 1)

        layers: list = []
        current_dim = int(in_features)
        for index, hidden_dim in enumerate(hidden_dims):
            if probabilities[index] > 0:
                layers.append(nn.Dropout(probabilities[index]))
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(ACTIVATIONS[activation]())
            current_dim = hidden_dim

        if probabilities[-1] > 0:
            layers.append(nn.Dropout(probabilities[-1]))
        layers.append(nn.Linear(current_dim, int(num_classes)))

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_head(in_features: int, num_classes: int,
               head_cfg: Optional[Mapping] = None) -> nn.Module:
    """Tạo head từ config.

    ``type: linear`` cho một lớp Linear duy nhất, ``type: mlp`` (mặc định)
    cho MLP nhiều tầng.
    """
    head_cfg = dict(head_cfg or {})
    head_type = str(head_cfg.pop("type", "mlp")).lower()

    if head_type == "linear":
        return nn.Linear(in_features, num_classes)
    if head_type != "mlp":
        raise KeyError(f"Head '{head_type}' không hỗ trợ (dùng 'mlp' hoặc 'linear').")

    return ClassifierHead(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=head_cfg.get("hidden_dims", (1024, 512)),
        dropout=head_cfg.get("dropout", (0.2, 0.3, 0.4)),
        activation=str(head_cfg.get("activation", "relu")).lower(),
    )
