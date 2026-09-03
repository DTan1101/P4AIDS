"""Deep Mutual Learning: huấn luyện đồng thời N model, học lẫn nhau qua KL.

Mỗi model học từ nhãn thật (cross-entropy) và từ phân phối dự đoán của các
model còn lại (KL divergence). Bản refactor này tổng quát cho N model thay vì
cố định 3 như code cũ, và tách phần lặp epoch thành các hàm nhỏ.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter, ConfusionMatrix
from utils.utils import ensure_dir, plot_history, save_checkpoint, save_json

logger = logging.getLogger("p4aids")


def _make_grad_scaler(device: torch.device, enabled: bool, **kwargs):
    """Tạo GradScaler tương thích nhiều phiên bản torch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled, **kwargs)
        except TypeError:  # torch cũ chưa nhận tham số device
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled, **kwargs)


class MutualTrainer:
    """Điều phối vòng huấn luyện mutual learning.

    Args:
        models: danh sách model cùng train (>=1; với 1 model thì thoái hoá về
            huấn luyện thông thường vì không có peer nào để học).
        train_cfg: nhánh ``train`` trong file config.
        device: thiết bị chạy.
        output_dir: thư mục lưu checkpoint, log và biểu đồ.
        meta: metadata ghi kèm checkpoint (danh sách lớp, mean/std...).
    """

    def __init__(self, models: Sequence[nn.Module], train_cfg: Mapping, device: torch.device,
                 output_dir: str, meta: Optional[Mapping[str, Any]] = None,
                 class_names: Optional[Sequence[str]] = None) -> None:
        if not models:
            raise ValueError("Cần ít nhất một model để huấn luyện.")

        self.cfg = dict(train_cfg)
        self.device = device
        self.models = [model.to(device) for model in models]
        self.num_models = len(self.models)
        self.output_dir = ensure_dir(output_dir)
        self.checkpoint_dir = ensure_dir(os.path.join(output_dir, "checkpoints"))
        self.meta = dict(meta or {})
        self.class_names = list(class_names) if class_names else None

        self.num_epochs = int(self.cfg.get("num_epochs", 10))
        self.mutual_weight = float(self.cfg.get("mutual_weight", 0.5))
        self.average_peers = bool(self.cfg.get("average_peers", False))
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.grad_clip = float(self.cfg.get("grad_clip", 0.0) or 0.0)
        self.patience = int(self.cfg.get("patience", 0) or 0)
        self.log_interval = int(self.cfg.get("log_interval", 0) or 0)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(self.cfg.get("label_smoothing", 0.0)))
        self.optimizers = [self._build_optimizer(model) for model in self.models]
        self.schedulers: List[Any] = []

        self.use_amp = bool(self.cfg.get("amp", True)) and device.type == "cuda"
        self.scaler = _make_grad_scaler(device, enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
        }
        self.best_score = -1.0
        self.best_epoch = -1

    # ------------------------------------------------------------------ #
    # Khởi tạo optimizer / scheduler
    # ------------------------------------------------------------------ #
    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        name = str(self.cfg.get("optimizer", "adamw")).lower()
        lr = float(self.cfg.get("learning_rate", 1e-3))
        weight_decay = float(self.cfg.get("weight_decay", 0.01))
        parameters = [p for p in model.parameters() if p.requires_grad]

        if name == "adamw":
            betas = tuple(self.cfg.get("betas", (0.9, 0.999)))
            return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
        if name == "adam":
            betas = tuple(self.cfg.get("betas", (0.9, 0.999)))
            return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
        if name == "sgd":
            return optim.SGD(parameters, lr=lr, weight_decay=weight_decay,
                             momentum=float(self.cfg.get("momentum", 0.9)),
                             nesterov=bool(self.cfg.get("nesterov", True)))
        raise KeyError(f"Optimizer '{name}' không hỗ trợ (adamw / adam / sgd).")

    def _build_schedulers(self, steps_per_epoch: int) -> List[Any]:
        scheduler_cfg = dict(self.cfg.get("scheduler") or {})
        name = str(scheduler_cfg.pop("type", "onecycle")).lower()
        if name in ("none", "off", ""):
            return []

        lr = float(self.cfg.get("learning_rate", 1e-3))
        schedulers = []
        for optimizer in self.optimizers:
            if name == "onecycle":
                schedulers.append(optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_cfg.get("max_lr", lr),
                    epochs=self.num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=float(scheduler_cfg.get("pct_start", 0.1)),
                    div_factor=float(scheduler_cfg.get("div_factor", 25)),
                    final_div_factor=float(scheduler_cfg.get("final_div_factor", 250)),
                    anneal_strategy=str(scheduler_cfg.get("anneal_strategy", "cos")),
                ))
            elif name == "cosine":
                schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.num_epochs * steps_per_epoch,
                    eta_min=float(scheduler_cfg.get("eta_min", 0.0))))
            elif name == "step":
                schedulers.append(optim.lr_scheduler.StepLR(
                    optimizer, step_size=int(scheduler_cfg.get("step_size", 5)) * steps_per_epoch,
                    gamma=float(scheduler_cfg.get("gamma", 0.1))))
            else:
                raise KeyError(f"Scheduler '{name}' không hỗ trợ.")
        return schedulers

    # ------------------------------------------------------------------ #
    # Hàm mất mát
    # ------------------------------------------------------------------ #
    def _mutual_loss(self, student_logits: torch.Tensor,
                     peer_logits: Sequence[torch.Tensor]) -> torch.Tensor:
        """Tổng KL(peer || student) trên các peer, có nhân trọng số.

        Dùng ``F.kl_div`` với log-softmax thay cho công thức thủ công
        ``sum(p * log(p / q + eps))`` ở code cũ - cùng ý nghĩa nhưng ổn định
        số học hơn và không cần epsilon.
        """
        if not peer_logits:
            return student_logits.new_zeros(())

        temperature = max(self.temperature, 1e-6)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)

        total = student_logits.new_zeros(())
        for logits in peer_logits:
            peer_prob = F.softmax(logits.detach() / temperature, dim=1)
            total = total + F.kl_div(student_log_prob, peer_prob, reduction="batchmean")

        total = total * (temperature ** 2)
        if self.average_peers:
            total = total / len(peer_logits)
        return total * self.mutual_weight

    def _compute_losses(self, outputs: Sequence[torch.Tensor],
                        labels: torch.Tensor) -> List[torch.Tensor]:
        losses = []
        for index, logits in enumerate(outputs):
            peers = [out for position, out in enumerate(outputs) if position != index]
            losses.append(self.criterion(logits, labels) + self._mutual_loss(logits, peers))
        return losses

    # ------------------------------------------------------------------ #
    # Vòng huấn luyện
    # ------------------------------------------------------------------ #
    def _optimizer_step(self) -> bool:
        """Unscale -> clip -> step. Trả về ``True`` nếu bước cập nhật hợp lệ."""
        if self.use_amp:
            # Phải unscale trước khi clip, nếu không gradient vẫn đang bị nhân
            # với hệ số scale và ngưỡng clip trở nên vô nghĩa (lỗi ở bản cũ).
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

        if self.grad_clip > 0:
            for model in self.models:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)

        scale_before = self.scaler.get_scale() if self.use_amp else None
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()

        # Khi scale giảm nghĩa là batch có inf/NaN và optimizer đã bị bỏ qua.
        return not self.use_amp or self.scaler.get_scale() >= scale_before

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        for model in self.models:
            model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("acc")
        started_at = time.time()

        for batch_index, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = [model(inputs) for model in self.models]
                losses = self._compute_losses(outputs, labels)

            for loss in losses:
                self.scaler.scale(loss).backward()

            stepped = self._optimizer_step()
            if stepped:
                for scheduler in self.schedulers:
                    scheduler.step()

            batch_size = labels.size(0)
            mean_loss = sum(loss.item() for loss in losses) / self.num_models
            mean_correct = sum((logits.argmax(dim=1) == labels).sum().item()
                               for logits in outputs) / self.num_models
            loss_meter.update(mean_loss, batch_size)
            acc_meter.update(mean_correct / batch_size, batch_size)

            if self.log_interval and batch_index % self.log_interval == 0:
                logger.info("  epoch %d [%d/%d] loss=%.4f acc=%.2f%% lr=%.2e",
                            epoch + 1, batch_index, len(loader), loss_meter.avg,
                            acc_meter.avg * 100, self._current_lr())

        return {"loss": loss_meter.avg, "acc": acc_meter.avg,
                "time": time.time() - started_at, "lr": self._current_lr()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, Any]:
        for model in self.models:
            model.eval()

        loss_meter = AverageMeter("val_loss")
        correct = [0] * self.num_models
        ensemble_correct = 0
        total = 0
        num_classes = getattr(self.models[0], "num_classes", None)
        matrix = (ConfusionMatrix(num_classes, self.class_names)
                  if num_classes and self.class_names else None)

        for inputs, labels in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = [model(inputs) for model in self.models]
            loss = sum(self.criterion(logits, labels) for logits in outputs) / self.num_models
            loss_meter.update(loss.item(), labels.size(0))

            for index, logits in enumerate(outputs):
                correct[index] += int((logits.argmax(dim=1) == labels).sum().item())

            ensemble_prob = sum(F.softmax(logits, dim=1) for logits in outputs) / self.num_models
            ensemble_preds = ensemble_prob.argmax(dim=1)
            ensemble_correct += int((ensemble_preds == labels).sum().item())
            if matrix is not None:
                matrix.update(ensemble_preds, labels)
            total += labels.size(0)

        per_model = [count / total for count in correct] if total else [0.0] * self.num_models
        return {
            "loss": loss_meter.avg,
            "acc": sum(per_model) / self.num_models,
            "per_model_acc": per_model,
            "ensemble_acc": ensemble_correct / total if total else 0.0,
            "confusion": matrix,
        }

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Chạy toàn bộ quá trình huấn luyện và trả về lịch sử + kết quả tốt nhất."""
        self.schedulers = self._build_schedulers(len(train_loader))
        logger.info("Bắt đầu huấn luyện %d model trên %s trong %d epoch.",
                    self.num_models, self.device, self.num_epochs)

        epochs_without_improvement = 0
        for epoch in range(self.num_epochs):
            train_stats = self._train_one_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_stats["loss"])
            self.history["train_acc"].append(train_stats["acc"])
            self.history["lr"].append(train_stats["lr"])

            message = (f"Epoch {epoch + 1}/{self.num_epochs} "
                       f"- Train Loss: {train_stats['loss']:.4f}, "
                       f"Train Acc: {train_stats['acc'] * 100:.2f}%")

            score = train_stats["acc"]
            confusion = None
            if val_loader is not None:
                val_stats = self._validate(val_loader)
                confusion = val_stats["confusion"]
                self.history["val_loss"].append(val_stats["loss"])
                self.history["val_acc"].append(val_stats["acc"])
                score = val_stats["acc"]
                message += (f" - Val Loss: {val_stats['loss']:.4f}, "
                            f"Val Acc: {val_stats['acc'] * 100:.2f}% "
                            f"(ensemble {val_stats['ensemble_acc'] * 100:.2f}%, "
                            f"từng model: "
                            f"{', '.join(f'{a * 100:.2f}%' for a in val_stats['per_model_acc'])})")
            message += f" - {train_stats['time']:.1f}s"
            logger.info(message)

            if score > self.best_score:
                self.best_score, self.best_epoch = score, epoch
                epochs_without_improvement = 0
                self._save_checkpoints(epoch, score)
                logger.info("Đã lưu checkpoint tốt nhất (score=%.4f).", score)
                if confusion is not None:
                    logger.info("Chi tiết theo lớp (ensemble):\n%s", confusion.report())
            else:
                epochs_without_improvement += 1
                if self.patience and epochs_without_improvement >= self.patience:
                    logger.info("Early stopping sau %d epoch không cải thiện.", self.patience)
                    break

        self._save_artifacts()
        logger.info("Hoàn tất. Score tốt nhất: %.4f tại epoch %d.",
                    self.best_score, self.best_epoch + 1)
        return {"history": self.history, "best_score": self.best_score,
                "best_epoch": self.best_epoch, "checkpoints": self.checkpoint_paths()}

    # ------------------------------------------------------------------ #
    def _current_lr(self) -> float:
        return float(self.optimizers[0].param_groups[0]["lr"])

    def checkpoint_paths(self) -> List[str]:
        return [os.path.join(self.checkpoint_dir, f"best_model{index + 1}_{name}.pth")
                for index, name in enumerate(self._model_names())]

    def _model_names(self) -> List[str]:
        return [getattr(model, "backbone_name", f"model{index + 1}")
                for index, model in enumerate(self.models)]

    def _save_checkpoints(self, epoch: int, score: float) -> None:
        for model, path in zip(self.models, self.checkpoint_paths()):
            meta = dict(self.meta)
            meta.update(getattr(model, "config", {}))
            meta.update({"epoch": epoch + 1, "score": score})
            save_checkpoint(path, model, meta=meta, epoch=epoch + 1, score=score)

    def _save_artifacts(self) -> None:
        save_json({"history": self.history, "best_score": self.best_score,
                   "best_epoch": self.best_epoch + 1, "meta": self.meta},
                  os.path.join(self.output_dir, "history.json"))
        plot_history(self.history, os.path.join(self.output_dir, "training_history.png"))
