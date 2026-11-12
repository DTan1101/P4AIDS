"""Baseline cho Chương 5.2 của báo cáo: lớp đa số, persistence, ngưỡng nhiệt độ,
logistic regression, k-NN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import macro_f1


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    """Luôn dự đoán lớp xuất hiện nhiều hơn trong ``y_train``."""
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = int(values[np.argmax(counts)])
    return np.full(n, majority_class, dtype=np.int64)


def persistence_predict(df: pd.DataFrame, end_time: pd.DatetimeIndex) -> np.ndarray:
    """Dự đoán nhãn tại t+1 bằng nhãn thật tại giờ cuối cửa sổ (t = end_time - 1h).

    Dùng trực tiếp nhãn trong dữ liệu đã làm sạch (không áp mặt nạ giả lập):
    baseline này mô phỏng "giữ nguyên trạng thái gần nhất", không phụ thuộc
    vào các biến quan trắc nên không bị ảnh hưởng bởi thiếu dữ liệu giả lập.
    """
    last_hour = end_time - pd.Timedelta(hours=1)
    labels = df["label"].reindex(last_hour).to_numpy()
    fallback = int(df["label"].mode(dropna=True).iloc[0])
    labels = np.where(np.isnan(labels), fallback, labels).astype(np.int64)
    return labels


def fit_temperature_threshold(mean_temperature: np.ndarray, y_train: np.ndarray,
                                candidates: Optional[np.ndarray] = None) -> float:
    """Hiệu chỉnh ngưỡng nhiệt độ trên tập huấn luyện bằng grid-search Macro F1."""
    if candidates is None:
        candidates = np.linspace(-10, 10, 81)
    best_threshold, best_score = float(candidates[0]), -1.0
    for threshold in candidates:
        preds = (mean_temperature < threshold).astype(np.int64)
        score = macro_f1(y_train, preds)
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


def temperature_threshold_predict(mean_temperature: np.ndarray, threshold: float) -> np.ndarray:
    """Dự đoán Snow (1) nếu nhiệt độ trung bình cửa sổ dưới ``threshold``."""
    return (mean_temperature < threshold).astype(np.int64)


def build_logreg_baseline() -> Pipeline:
    """Logistic Regression trên đặc trưng LiteStatTemp, impute median + chuẩn hoá."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def build_knn_baseline(n_neighbors: int = 5) -> Pipeline:
    """k-NN trên đặc trưng LiteStatTemp, impute median + chuẩn hoá."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])


def mean_channel(X: np.ndarray, channel: int) -> np.ndarray:
    """Trung bình (bỏ NaN) của một kênh/biến qua toàn bộ cửa sổ -- dùng cho baseline ngưỡng nhiệt độ."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(X[:, :, channel], axis=1)
