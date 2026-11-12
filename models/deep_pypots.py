"""Wrapper mỏng quanh BRITS/SAITS/GRU-D của thư viện PyPOTS (Mục 4.6.2 báo cáo).

Cài đặt BRITS/SAITS đúng chuẩn từ đầu rủi ro cao và dễ sai; PyPOTS là thư viện
chuẩn, được duy trì tốt, cho đúng bài toán "phân loại trên chuỗi thời gian
thiếu dữ liệu" và đã có sẵn các kiến trúc này -- xem ghi chú lựa chọn thư viện
trong kế hoạch refactor. API đã được xác minh trực tiếp với ``pypots==2.0``
(constructor, ``fit``/``classify``/``predict``) khi viết module này.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pypots.classification import BRITS, SAITS
from pypots.classification import GRUD as PyPOTSGRUD
from pypots.nn.modules.loss import Criterion

PYPOTS_MODELS = ("brits", "saits", "grud_pypots")


class WeightedCrossEntropy(Criterion):
    """Cross-entropy có trọng số lớp (Mục 4.6.3) -- PyPOTS không có sẵn bản này."""

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(lower_better=True)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return torch.nn.functional.cross_entropy(logits, targets, weight=weight)


def build_pypots_model(name: str, n_steps: int, n_features: int, n_classes: int = 2,
                        class_weight: Optional[torch.Tensor] = None,
                        device: str = "cpu", epochs: int = 100, patience: Optional[int] = 10,
                        batch_size: int = 64, saving_path: Optional[str] = None,
                        **kwargs: Any):
    """Factory tạo mô hình PyPOTS chưa fit, theo tên (Mục 4.6.2)."""
    loss_fn = WeightedCrossEntropy(class_weight)
    common: Dict[str, Any] = dict(
        n_steps=n_steps, n_features=n_features, n_classes=n_classes,
        batch_size=batch_size, epochs=epochs, patience=patience,
        training_loss=loss_fn, validation_metric=loss_fn,
        device=device, saving_path=saving_path, model_saving_strategy="best", verbose=True,
    )

    if name == "brits":
        return BRITS(rnn_hidden_size=kwargs.get("rnn_hidden_size", 64), **common)
    if name == "grud_pypots":
        return PyPOTSGRUD(rnn_hidden_size=kwargs.get("rnn_hidden_size", 64), **common)
    if name == "saits":
        return SAITS(
            n_layers=kwargs.get("n_layers", 2), d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4), d_k=kwargs.get("d_k", 16), d_v=kwargs.get("d_v", 16),
            d_ffn=kwargs.get("d_ffn", 128), dropout=kwargs.get("dropout", 0.1),
            attn_dropout=kwargs.get("attn_dropout", 0.1), **common,
        )
    raise ValueError(f"Mô hình PyPOTS không hợp lệ: '{name}'. Hỗ trợ: {PYPOTS_MODELS}")
