"""Trainer cho nhánh Deep Learning (Mục 4.6.3 báo cáo).

Mô hình tự cài (GRUFill/GRUMaskDelta/GRUD, :mod:`models.deep`) dùng vòng lặp
PyTorch chuẩn ở đây (loss có trọng số lớp, early stopping, scheduler, nhiều
seed). Mô hình PyPOTS (BRITS/SAITS/GRU-D, :mod:`models.deep_pypots`) dùng
``.fit()``/``.predict()`` qua adapter chọn checkpoint theo cùng Macro F1.
"""

from __future__ import annotations

import copy
import logging
import os
from importlib.metadata import version
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dl_dataset import WeatherWindowDataset, prepare_dl_arrays, prepare_observed_arrays
from datasets.missingness import MissingnessConfig
from evaluation.metrics import compute_all_metrics
from models.deep import build_custom_deep_model
from utils.utils import ensure_dir, get_device, save_checkpoint, seed_everything

logger = logging.getLogger("p4aids")

DEEP_PROTOCOL = "deep-v2-observed-zscore-macro-f1"

CUSTOM_MODEL_NAMES = ("gru_fill", "gru_mask_delta", "gru_d")
PYPOTS_MODEL_NAMES = ("brits", "saits", "grud_pypots")


@dataclass
class DeepRunResult:
    config: MissingnessConfig
    model_name: str
    seed: int
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    best_epoch: int = -1
    history: Dict[str, List[float]] = field(default_factory=dict)
    protocol: str = DEEP_PROTOCOL
    checkpoint_path: Optional[str] = None
    prediction_path: Optional[str] = None


def _class_weight(y_train: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weight = counts.sum() / (n_classes * counts)
    return torch.tensor(weight, dtype=torch.float32)


@torch.no_grad()
def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    all_preds, all_proba, all_targets = [], [], []
    for value, mask, delta, target in loader:
        logits = model(value.to(device), mask.to(device), delta.to(device))
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_proba.append(proba)
        all_targets.append(target.numpy())
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_proba)
    y_true = np.concatenate(all_targets)
    return {"y_true": y_true, "y_pred": y_pred, "y_score": y_proba}


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    return compute_all_metrics(**_predict_loader(model, loader, device))


def _finish_run(result: DeepRunResult, model: nn.Module, meta: Dict[str, Any],
                predictions: Dict[str, Any], artifact_dir: Optional[str]) -> DeepRunResult:
    for split, (_, prediction) in predictions.items():
        if not np.isfinite(prediction["y_score"]).all():
            raise ValueError(f"Probability không hữu hạn trên {split}.")
        setattr(result, f"{split}_metrics", compute_all_metrics(**prediction))
    if artifact_dir is not None:
        folder = ensure_dir(os.path.join(artifact_dir, result.protocol, result.model_name,
                                        result.config.key, f"seed{result.seed}"))
        meta.update(protocol=result.protocol, model_name=result.model_name, seed=result.seed,
                    config=asdict(result.config), best_epoch=result.best_epoch,
                    selection_metric="val_macro_f1", classes=["Rain", "Snow"],
                    torch_version=str(torch.__version__))
        result.checkpoint_path = save_checkpoint(os.path.join(folder, "checkpoint.pt"), model, meta)
        arrays = {f"{split}_{key}": value for split, (idx, prediction) in predictions.items()
                  for key, value in {"idx": idx, **prediction}.items()}
        result.prediction_path = os.path.join(folder, "predictions.npz")
        np.savez_compressed(result.prediction_path, protocol=result.protocol, seed=result.seed, **arrays)
    return result


def run_deep_custom(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                     train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                     config: MissingnessConfig, model_name: str = "gru_fill",
                     hidden_size: int = 64, epochs: int = 50, patience: Optional[int] = 10,
                     batch_size: int = 128, lr: float = 1e-3, seed: int = 42,
                     device: Optional[str] = None, artifact_dir: Optional[str] = None) -> DeepRunResult:
    """Huấn luyện một mô hình tự cài (GRU-Fill/GRU+mask-delta/GRU-D) cho một cấu hình.

    ``patience=None`` tắt early stopping (huấn luyện đủ ``epochs``), đồng nhất với
    quy ước của PyPOTS ở :func:`run_deep_pypots`.
    """
    seed_everything(seed)
    torch_device = get_device(device or "auto")

    if epochs < 1:
        raise ValueError("epochs phải >= 1.")
    value_norm, mask_f, delta, mean, std = prepare_dl_arrays(X, mask, train_idx)
    n_features = X.shape[-1]

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = WeatherWindowDataset(value_norm[idx], mask_f[idx], delta[idx], y[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)
    test_loader = _loader(test_idx, shuffle=False)

    model = build_custom_deep_model(model_name, n_features, hidden_size).to(torch_device)
    criterion = nn.CrossEntropyLoss(weight=_class_weight(y[train_idx]).to(torch_device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score, best_epoch, best_state = -1.0, -1, None
    epochs_without_improve = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_macro_f1": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for value, mask_b, delta_b, target in train_loader:
            value, mask_b, delta_b, target = (value.to(torch_device), mask_b.to(torch_device),
                                               delta_b.to(torch_device), target.to(torch_device))
            optimizer.zero_grad()
            logits = model(value, mask_b, delta_b)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Training loss không hữu hạn.")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * value.size(0)
        epoch_loss /= len(train_loader.dataset)

        val_metrics = _evaluate(model, val_loader, torch_device)
        scheduler.step(val_metrics["macro_f1"])
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        logger.info("[%s|%s] epoch %d: loss=%.4f val_macro_f1=%.4f",
                    config.key, model_name, epoch, epoch_loss, val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_score:
            best_score, best_epoch = val_metrics["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if patience is not None and epochs_without_improve >= patience:
                logger.info("Early stopping tại epoch %d (patience=%d).", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result = DeepRunResult(config=config, model_name=model_name, seed=seed,
                           best_epoch=best_epoch, history=history)
    meta = dict(n_features=n_features, n_steps=X.shape[1], mean=mean.tolist(), std=std.tolist(),
                model_kwargs={"hidden_size": hidden_size}, class_weight=_class_weight(y[train_idx]).tolist(),
                training=dict(epochs=epochs, patience=patience, batch_size=batch_size, lr=lr))
    predictions = {"val": (val_idx, _predict_loader(model, val_loader, torch_device)),
                   "test": (test_idx, _predict_loader(model, test_loader, torch_device))}
    return _finish_run(result, model, meta, predictions, artifact_dir)


def run_deep_pypots(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                     train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                     config: MissingnessConfig, model_name: str = "brits",
                     epochs: int = 100, patience: Optional[int] = 10, batch_size: int = 64,
                     seed: int = 42, device: Optional[str] = None, hidden_size: int = 64,
                     lr: float = 1e-3, artifact_dir: Optional[str] = None,
                     **model_kwargs: Any) -> DeepRunResult:
    """Huấn luyện một mô hình PyPOTS (BRITS/SAITS/GRU-D) cho một cấu hình."""
    from models.deep_pypots import build_pypots_model

    seed_everything(seed)
    torch_device = str(get_device(device or "auto"))

    if epochs < 1:
        raise ValueError("epochs phải >= 1.")
    X_masked, mean, std = prepare_observed_arrays(X, mask, train_idx)
    n_features = X.shape[-1]
    weight = _class_weight(y[train_idx])

    model = build_pypots_model(model_name, n_steps=X.shape[1], n_features=n_features,
                                class_weight=weight, device=torch_device, epochs=epochs,
                                patience=patience, batch_size=batch_size, hidden_size=hidden_size, lr=lr, **model_kwargs)
    model.fit({"X": X_masked[train_idx], "y": y[train_idx]},
              val_set={"X": X_masked[val_idx], "y": y[val_idx]})

    def _predict(idx: np.ndarray):
        prediction = model.predict({"X": X_masked[idx]})
        return {"y_true": y[idx], "y_pred": prediction["classification"],
                "y_score": prediction["classification_proba"][:, 1]}

    result = DeepRunResult(config=config, model_name=model_name, seed=seed,
                           best_epoch=model.best_epoch, history=model.history)
    meta = dict(n_features=n_features, n_steps=X.shape[1], mean=mean.tolist(), std=std.tolist(),
                model_kwargs={"hidden_size": hidden_size, **model_kwargs}, class_weight=weight.tolist(),
                training=dict(epochs=epochs, patience=patience, batch_size=batch_size, lr=lr),
                pypots_version=version("pypots"))
    predictions = {"val": (val_idx, _predict(val_idx)), "test": (test_idx, _predict(test_idx))}
    return _finish_run(result, model.model, meta, predictions, artifact_dir)


def run_deep(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
             train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
             config: MissingnessConfig, model_name: str, seed: int = 42, **kwargs: Any) -> DeepRunResult:
    """Dispatch theo tên mô hình: tự cài hay PyPOTS (Mục 4.6.2)."""
    if model_name in CUSTOM_MODEL_NAMES:
        return run_deep_custom(X, y, mask, train_idx, val_idx, test_idx, config, model_name, seed=seed, **kwargs)
    if model_name in PYPOTS_MODEL_NAMES:
        return run_deep_pypots(X, y, mask, train_idx, val_idx, test_idx, config, model_name, seed=seed, **kwargs)
    raise ValueError(f"Mô hình không hợp lệ: '{model_name}'. "
                      f"Hỗ trợ: {CUSTOM_MODEL_NAMES + PYPOTS_MODEL_NAMES}")


def run_deep_multiseed(seeds: List[int], *args: Any, **kwargs: Any) -> List[DeepRunResult]:
    """Lặp lại :func:`run_deep` với nhiều seed, để báo cáo trung bình/độ lệch chuẩn (Mục 4.6.3)."""
    return [run_deep(*args, seed=seed, **kwargs) for seed in seeds]
