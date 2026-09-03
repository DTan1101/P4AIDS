"""Các độ đo dùng cho bài toán phân loại ảnh nấm.

Chỉ phụ thuộc numpy/torch nên không cần cài thêm scikit-learn.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


class AverageMeter:
    """Theo dõi giá trị trung bình có trọng số (thường là batch size)."""

    def __init__(self, name: str = "meter", fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, n: float = 1) -> None:
        self.value = float(value)
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:  # pragma: no cover - chỉ dùng để log
        return f"{self.name} {self.avg:{self.fmt.lstrip(':')}}"


@torch.no_grad()
def correct_count(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Số mẫu dự đoán đúng trong batch."""
    preds = logits.argmax(dim=1)
    return int((preds == targets).sum().item())


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy của một batch, giá trị trong [0, 1]."""
    if targets.numel() == 0:
        return 0.0
    return correct_count(logits, targets) / targets.numel()


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                  topk: Sequence[int] = (1,)) -> List[float]:
    """Accuracy top-k, hữu ích khi số lớp tăng lên."""
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    hits = pred.eq(targets.view(-1, 1).expand_as(pred))
    batch_size = targets.size(0)
    return [float(hits[:, :min(k, maxk)].any(dim=1).sum().item()) / batch_size for k in topk]


class ConfusionMatrix:
    """Ma trận nhầm lẫn tích luỹ qua nhiều batch."""

    def __init__(self, num_classes: int, class_names: Optional[Sequence[str]] = None) -> None:
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names else [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """``preds``/``targets`` là nhãn dạng chỉ số lớp (không phải logits)."""
        preds_np = preds.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        indices = targets_np[valid] * self.num_classes + preds_np[valid]
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def update_from_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.update(logits.argmax(dim=1), targets)

    @property
    def total(self) -> int:
        return int(self.matrix.sum())

    def accuracy(self) -> float:
        return float(np.trace(self.matrix)) / self.total if self.total else 0.0

    def per_class(self) -> Dict[str, np.ndarray]:
        """Precision / recall / f1 / support cho từng lớp."""
        true_positive = np.diag(self.matrix).astype(np.float64)
        predicted = self.matrix.sum(axis=0).astype(np.float64)
        actual = self.matrix.sum(axis=1).astype(np.float64)

        precision = np.divide(true_positive, predicted,
                              out=np.zeros_like(true_positive), where=predicted > 0)
        recall = np.divide(true_positive, actual,
                           out=np.zeros_like(true_positive), where=actual > 0)
        denominator = precision + recall
        f1 = np.divide(2 * precision * recall, denominator,
                       out=np.zeros_like(true_positive), where=denominator > 0)
        return {"precision": precision, "recall": recall, "f1": f1, "support": actual}


    def macro_f1(self) -> float:
        return float(self.per_class()["f1"].mean())

    def summary(self) -> Dict[str, float]:
        scores = self.per_class()
        return {
            "accuracy": self.accuracy(),
            "macro_precision": float(scores["precision"].mean()),
            "macro_recall": float(scores["recall"].mean()),
            "macro_f1": float(scores["f1"].mean()),
        }

    def report(self, digits: int = 4) -> str:
        """Bảng kết quả dạng text, tương tự classification_report."""
        scores = self.per_class()
        width = max([len(name) for name in self.class_names] + [9])
        lines = [f"{'class':<{width}} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}"]
        for index, name in enumerate(self.class_names):
            lines.append(
                f"{name:<{width}} {scores['precision'][index]:>10.{digits}f}"
                f" {scores['recall'][index]:>10.{digits}f} {scores['f1'][index]:>10.{digits}f}"
                f" {int(scores['support'][index]):>9d}"
            )
        summary = self.summary()
        lines.append("-" * len(lines[0]))
        lines.append(
            f"{'macro avg':<{width}} {summary['macro_precision']:>10.{digits}f}"
            f" {summary['macro_recall']:>10.{digits}f} {summary['macro_f1']:>10.{digits}f}"
            f" {self.total:>9d}"
        )
        lines.append(f"{'accuracy':<{width}} {summary['accuracy']:>10.{digits}f}")
        return "\n".join(lines)

    def matrix_str(self) -> str:
        """In ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán)."""
        width = max([len(name) for name in self.class_names] + [5])
        header = " " * (width + 1) + " ".join(f"{name[:width]:>{width}}" for name in self.class_names)
        rows = [header]
        for index, name in enumerate(self.class_names):
            cells = " ".join(f"{value:>{width}d}" for value in self.matrix[index])
            rows.append(f"{name[:width]:<{width}} {cells}")
        return "\n".join(rows)
