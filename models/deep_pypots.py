"""Adapter BRITS/SAITS/GRU-D, cố định API PyPOTS 2.0.

Giữ kiến trúc/loss của thư viện, chọn checkpoint theo Macro F1 toàn validation.
BRITS.fit được sửa tại adapter để dùng đúng val_set thay vì train_set.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import copy
import logging

import numpy as np
import torch
from pypots.classification import BRITS, SAITS
from pypots.classification import GRUD as PyPOTSGRUD
from pypots.nn.modules.loss import Criterion
from pypots.optim import Adam
from pypots.imputation.brits.data import DatasetForBRITS
from torch.utils.data import DataLoader

from evaluation.metrics import macro_f1

PYPOTS_MODELS = ("brits", "saits", "grud_pypots")


class WeightedCrossEntropy(Criterion):
    """Cross-entropy có trọng số lớp (Mục 4.6.3) -- PyPOTS không có sẵn bản này."""

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__(lower_better=True)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return torch.nn.functional.cross_entropy(logits, targets, weight=weight)


class _MacroF1Training:
    """Giữ loss/batch PyPOTS, chọn checkpoint theo prediction toàn validation.

    Override hook của PyPOTS 2.0 vì metric theo batch và trung bình hai chiều
    BRITS không tương đương Macro F1 của prediction cuối cùng trên toàn tập.
    """

    def _train_model(self, train_dataloader, val_dataloader=None):
        if val_dataloader is None:
            raise ValueError("Giao thức DL yêu cầu validation để chọn checkpoint.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.torch_optimizer, mode="max", factor=0.5, patience=3,
        )
        best_score, stale = -1.0, 0
        self.best_model_dict = None
        self.history = {"train_loss": [], "val_macro_f1": []}
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for data in train_dataloader:
                inputs = self._assemble_input_for_training(data)
                self.optimizer.zero_grad()
                loss = self.model(inputs, calc_criterion=True)["loss"].sum()
                if not torch.isfinite(loss):
                    raise ValueError("PyPOTS training loss không hữu hạn.")
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(inputs["y"])
            self.model.eval()
            targets, predictions = [], []
            with torch.no_grad():
                for data in val_dataloader:
                    inputs = self._assemble_input_for_validating(data)
                    proba = self.model(inputs)["classification_proba"]
                    if not torch.isfinite(proba).all():
                        raise ValueError("PyPOTS validation probability không hữu hạn.")
                    targets.append(inputs["y"].cpu().numpy())
                    predictions.append(proba.argmax(dim=1).cpu().numpy())
            score = macro_f1(np.concatenate(targets), np.concatenate(predictions))
            scheduler.step(score)
            self.history["train_loss"].append(total_loss / len(train_dataloader.dataset))
            self.history["val_macro_f1"].append(score)
            logging.getLogger("p4aids").info("[%s] epoch %d: val_macro_f1=%.4f", type(self).__name__, epoch, score)
            if score > best_score:
                best_score, self.best_epoch, stale = score, epoch, 0
                self.best_model_dict = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if self.original_patience is not None and stale >= self.original_patience:
                    break
        if self.best_model_dict is None:
            raise ValueError("Không có checkpoint hợp lệ; epochs phải >= 1.")
        self.best_loss = best_score


class BRITSClassifier(_MacroF1Training, BRITS):
    def fit(self, train_set, val_set=None, file_type="hdf5"):
        # PyPOTS 2.0 BRITS.fit dùng nhầm train_set để tạo val_dataset.
        if val_set is None:
            raise ValueError("Giao thức DL yêu cầu validation để chọn checkpoint.")
        train_dataset = DatasetForBRITS(train_set, return_X_ori=False, return_y=True, file_type=file_type)
        val_dataset = DatasetForBRITS(val_set, return_X_ori=False, return_y=True, file_type=file_type)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")


class SAITSClassifier(_MacroF1Training, SAITS):
    pass


class GRUDClassifier(_MacroF1Training, PyPOTSGRUD):
    pass


def build_pypots_model(name: str, n_steps: int, n_features: int, n_classes: int = 2,
                        class_weight: Optional[torch.Tensor] = None,
                        device: str = "cpu", epochs: int = 100, patience: Optional[int] = 10,
                        batch_size: int = 64, saving_path: Optional[str] = None,
                        hidden_size: int = 64, lr: float = 1e-3,
                        **kwargs: Any):
    """Factory tạo mô hình PyPOTS chưa fit, theo tên (Mục 4.6.2)."""
    loss_fn = WeightedCrossEntropy(class_weight)
    common: Dict[str, Any] = dict(
        n_steps=n_steps, n_features=n_features, n_classes=n_classes,
        batch_size=batch_size, epochs=epochs,
        patience=min(patience, epochs) if patience is not None else None,
        training_loss=loss_fn, optimizer=Adam(lr=lr),
        # _MacroF1Training tính metric toàn tập; Criterion chỉ khai báo hướng tối ưu.
        validation_metric=Criterion(lower_better=False),
        device=device, saving_path=saving_path, model_saving_strategy="best", verbose=True,
    )

    if name == "brits":
        return BRITSClassifier(rnn_hidden_size=kwargs.pop("rnn_hidden_size", hidden_size), **common, **kwargs)
    if name == "grud_pypots":
        return GRUDClassifier(rnn_hidden_size=kwargs.pop("rnn_hidden_size", hidden_size), **common, **kwargs)
    if name == "saits":
        d_model = kwargs.pop("d_model", hidden_size)
        n_heads = kwargs.pop("n_heads", 4)
        if n_heads < 1 or d_model % n_heads:
            raise ValueError("SAITS d_model phải chia hết cho n_heads >= 1.")
        return SAITSClassifier(
            n_layers=kwargs.pop("n_layers", 2), d_model=d_model,
            n_heads=n_heads, d_k=kwargs.pop("d_k", d_model // n_heads),
            d_v=kwargs.pop("d_v", d_model // n_heads),
            d_ffn=kwargs.pop("d_ffn", 128), dropout=kwargs.pop("dropout", 0.1),
            attn_dropout=kwargs.pop("attn_dropout", 0.1), **common, **kwargs,
        )
    raise ValueError(f"Mô hình PyPOTS không hợp lệ: '{name}'. Hỗ trợ: {PYPOTS_MODELS}")
