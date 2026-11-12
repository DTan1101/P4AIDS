"""Các độ đo dùng cho bài toán Rain/Snow (Mục 5.3 báo cáo).

Nhãn nhị phân: 0 = Rain, 1 = Snow (:data:`datasets.szeged.LABEL_MAP`). Khác với
``utils/metrics.py`` (đa lớp, dùng cho pipeline ảnh nấm cũ), module này dùng
sklearn trực tiếp vì cần PR-AUC và các độ đo nhạy với mất cân bằng lớp.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """PR-AUC cho lớp dương (Snow=1). ``y_score`` là xác suất/điểm số của lớp 1."""
    return float(average_precision_score(y_true, y_score))


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def precision_recall_per_class(y_true: Sequence[int], y_pred: Sequence[int],
                                class_names: Sequence[str] = ("Rain", "Snow")) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0,
    )
    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }


def compute_all_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                          y_score: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """Gộp toàn bộ chỉ số cho một lần chạy, dùng để ghi JSON kết quả thống nhất."""
    result: Dict[str, object] = {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "per_class": precision_recall_per_class(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }
    if y_score is not None:
        result["pr_auc"] = pr_auc(y_true, y_score)
    return result
