"""Nhánh ROCKET: nội suy + MiniRocket/MultiRocket + RidgeClassifierCV (Mục 4.5)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.linear_model import RidgeClassifierCV

ROCKET_VARIANTS = ("mini", "multi")


def interpolate_missing(X: np.ndarray) -> np.ndarray:
    """Nội suy tuyến tính theo thời gian cho từng biến/cửa sổ; đoạn đầu/cuối thiếu
    được điền bằng giá trị hợp lệ gần nhất (``np.interp`` tự ngoại suy hằng số ở
    biên), theo đúng Mục 4.5. Cửa sổ mất trọn một biến được điền bằng 0.
    """
    n_samples, window, n_features = X.shape
    out = X.copy()
    t = np.arange(window)
    for i in range(n_samples):
        for c in range(n_features):
            series = out[i, :, c]
            valid = ~np.isnan(series)
            n_valid = int(valid.sum())
            if n_valid == window:
                continue
            if n_valid == 0:
                out[i, :, c] = 0.0
                continue
            out[i, :, c] = np.interp(t, t[valid], series[valid])
    return out


def to_aeon_format(X: np.ndarray) -> np.ndarray:
    """``(N, window, n_feat)`` -> ``(N, n_feat, window)`` theo định dạng aeon."""
    return np.ascontiguousarray(X.transpose(0, 2, 1))


def build_rocket_transformer(variant: str = "mini", random_state: int = 42, n_jobs: int = 2,
                              n_kernels: Optional[int] = None):
    """``n_jobs`` mặc định nhỏ (không dùng ``-1``/tất cả lõi): MiniRocket/MultiRocket
    cấp phát bộ nhớ đệm song song theo từng luồng, với ~70 nghìn cửa sổ train có thể
    vượt RAM trên máy không có swap nếu chạy hết số lõi. Tăng giá trị này nếu máy
    nhiều RAM.

    ``n_kernels`` mặc định ``None`` = giữ nguyên mặc định gốc của aeon (MiniRocket
    10000, MultiRocket 6250) -- đúng chuẩn phương pháp theo Mục 4.5. Với ma trận đặc
    trưng ~70 nghìn x ~10 nghìn, RidgeClassifierCV có thể cần ép kiểu float64 nội bộ
    và vượt RAM trên máy không có swap/ít RAM (đã xác minh thực nghiệm); hạ
    ``n_kernels`` (ví dụ 1000-2000) đánh đổi một phần độ chính xác để chạy được trên
    máy yếu hơn.
    """
    kwargs = dict(random_state=random_state, n_jobs=n_jobs)
    if n_kernels is not None:
        kwargs["n_kernels"] = n_kernels
    if variant == "mini":
        return MiniRocket(**kwargs)
    if variant == "multi":
        return MultiRocket(**kwargs)
    raise ValueError(f"variant phải thuộc {ROCKET_VARIANTS}, nhận '{variant}'.")


def build_ridge_classifier(cv: int = 5) -> RidgeClassifierCV:
    """``cv`` cố định (k-fold tường minh), KHÔNG dùng ``cv=None``: mặc định của
    sklearn khi ``cv=None`` dùng một "trick" generalized-CV dựa trên SVD của toàn
    bộ ma trận thiết kế, cấp phát một ma trận ``U`` cùng kích thước với đặc trưng
    đầu vào -- với ~70 nghìn cửa sổ train x ~10 nghìn đặc trưng ROCKET, đã xác minh
    thực nghiệm việc này gây OOM trên máy không có swap. ``cv`` tường minh chậm hơn
    (phải refit theo từng fold) nhưng bộ nhớ ổn định, không phụ thuộc cỡ dữ liệu.
    """
    return RidgeClassifierCV(alphas=np.logspace(-3, 3, 5), cv=cv)


class RocketClassifier:
    """Nội suy -> MiniRocket/MultiRocket -> (nối mask/Δt tuỳ chọn) -> RidgeClassifierCV.

    ``predict_proba`` không có sẵn ở ``RidgeClassifierCV`` (chỉ là linear SVM
    kiểu hồi quy), nên dùng ``decision_function`` đã chuẩn hoá qua sigmoid làm
    điểm số xác suất xấp xỉ cho PR-AUC (Mục 5.3).
    """

    def __init__(self, variant: str = "mini", random_state: int = 42, n_jobs: int = 2,
                 max_fit_samples: int = 2500, transform_batch_size: int = 2000,
                 ridge_cv: int = 5, n_kernels: Optional[int] = None) -> None:
        """Hai chốt chặn bộ nhớ, cả hai đều cần thiết (đã xác minh thực nghiệm mỗi bước
        có thể tự gây OOM riêng trên máy không có swap với ~70 nghìn cửa sổ train):

        - ``max_fit_samples``: ``fit()`` (ước lượng dilation/bias bằng quantile) cấp
          phát bộ nhớ tỉ lệ thuận số mẫu đưa vào. Theo đúng khuyến nghị gốc của
          ROCKET/MiniRocket, việc này chỉ cần một mẫu con nhỏ là đủ ổn định, nên
          ``fit()`` chỉ gọi trên một mẫu con ngẫu nhiên tối đa ``max_fit_samples`` dòng.
        - ``transform_batch_size``: ``transform()`` CŨNG cấp phát bộ nhớ tỉ lệ thuận số
          mẫu đưa vào trong một lần gọi (giới hạn riêng ``max_fit_samples`` không đủ --
          gọi ``transform()`` trên toàn bộ ~70 nghìn mẫu cùng lúc vẫn OOM), nên luôn gọi
          theo từng lô nhỏ dù là tập train/val/test nào.
        """
        self.variant = variant
        self.random_state = random_state
        self.max_fit_samples = max_fit_samples
        self.transform_batch_size = transform_batch_size
        self.transformer = build_rocket_transformer(variant, random_state, n_jobs, n_kernels)
        self.classifier = build_ridge_classifier(cv=ridge_cv)
        self._fitted = False

    def _transform_in_batches(self, X_aeon: np.ndarray) -> np.ndarray:
        batches = [
            self.transformer.transform(X_aeon[start:start + self.transform_batch_size])
            for start in range(0, X_aeon.shape[0], self.transform_batch_size)
        ]
        return np.concatenate(batches, axis=0)

    def _features(self, X: np.ndarray, mask_delta: Optional[np.ndarray], fit: bool) -> np.ndarray:
        X_filled = interpolate_missing(X)
        X_aeon = to_aeon_format(X_filled)

        if fit:
            n_samples = X_aeon.shape[0]
            if self.max_fit_samples and n_samples > self.max_fit_samples:
                rng = np.random.default_rng(self.random_state)
                fit_idx = rng.choice(n_samples, size=self.max_fit_samples, replace=False)
                self.transformer.fit(X_aeon[fit_idx])
            else:
                self.transformer.fit(X_aeon)

        rocket_feats = self._transform_in_batches(X_aeon)
        if mask_delta is not None:
            return np.concatenate([rocket_feats, mask_delta], axis=1)
        return rocket_feats

    def fit(self, X: np.ndarray, y: np.ndarray, mask_delta: Optional[np.ndarray] = None) -> "RocketClassifier":
        features = self._features(X, mask_delta, fit=True)
        self.classifier.fit(features, y)
        self._fitted = True
        return self

    def decision_scores(self, X: np.ndarray, mask_delta: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước.")
        features = self._features(X, mask_delta, fit=False)
        return self.classifier.decision_function(features)

    def predict(self, X: np.ndarray, mask_delta: Optional[np.ndarray] = None) -> np.ndarray:
        scores = self.decision_scores(X, mask_delta)
        return (scores >= 0.0).astype(np.int64)

    def predict_proba_positive(self, X: np.ndarray, mask_delta: Optional[np.ndarray] = None) -> np.ndarray:
        """Xấp xỉ P(Snow) bằng sigmoid(decision_function) -- dùng cho PR-AUC."""
        scores = self.decision_scores(X, mask_delta)
        return 1.0 / (1.0 + np.exp(-scores))
