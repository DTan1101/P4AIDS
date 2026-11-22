"""Nhánh Classic ML: XGBoost + các chiến lược xử lý dữ liệu thiếu (Mục 4.4 báo cáo)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (bắt buộc để bật IterativeImputer)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBClassifier

#: Các chiến lược xử lý dữ liệu thiếu cho đặc trưng (Mục 4.4.2).
IMPUTE_STRATEGIES = ("xgboost_native", "median_mask", "missforest")


def build_xgboost(params: Optional[Mapping[str, Any]] = None,
                   scale_pos_weight: Optional[float] = None) -> XGBClassifier:
    """XGBoost nhận NaN trực tiếp (``missing=np.nan`` là mặc định của XGBClassifier)."""
    default_params: dict = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", missing=np.nan, random_state=42,
    )
    default_params.update(params or {})
    if scale_pos_weight is not None:
        default_params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**default_params)


def build_median_mask_imputer() -> SimpleImputer:
    """Median kèm mask: impute trung vị + tự thêm cột chỉ báo thiếu (Mục 4.4.2)."""
    return SimpleImputer(strategy="median", add_indicator=True)


def build_missforest_imputer(n_estimators: int = 50, max_iter: int = 10,
                              random_state: int = 42) -> IterativeImputer:
    """MissForest (Stekhoven & Bühlmann, 2012): impute lặp bằng random forest."""
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
        max_iter=max_iter, random_state=random_state,
    )


def build_imputer(strategy: str, **kwargs: Any):
    """Factory theo tên chiến lược; trả về ``None`` cho ``xgboost_native`` (không impute)."""
    if strategy == "xgboost_native":
        return None
    if strategy == "median_mask":
        return build_median_mask_imputer()
    if strategy == "missforest":
        return build_missforest_imputer(**kwargs)
    raise ValueError(f"Chiến lược xử lý thiếu không hợp lệ: '{strategy}'. Hỗ trợ: {IMPUTE_STRATEGIES}")


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """``scale_pos_weight`` = (số mẫu âm) / (số mẫu dương), cho XGBoost (Mục 4.4.3)."""
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    return float(n_neg) / float(max(n_pos, 1))


def build_random_forest(params: Optional[Mapping[str, Any]] = None,
                        class_weight: Optional[str] = "balanced") -> RandomForestClassifier:
    default_params: dict = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        class_weight=class_weight,
        random_state=42,
    )
    default_params.update(params or {})
    return RandomForestClassifier(**default_params)


def build_lightgbm(params: Optional[Mapping[str, Any]] = None,
                   scale_pos_weight: Optional[float] = None):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM chưa được cài: pip install lightgbm") from exc
    default_params: dict = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    default_params.update(params or {})
    if scale_pos_weight is not None:
        default_params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**default_params)


def build_catboost(params: Optional[Mapping[str, Any]] = None,
                   scale_pos_weight: Optional[float] = None):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("CatBoost chưa được cài: pip install catboost") from exc
    default_params: dict = dict(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    default_params.update(params or {})
    if scale_pos_weight is not None:
        default_params["class_weights"] = [1.0, scale_pos_weight]
    return CatBoostClassifier(**default_params)
