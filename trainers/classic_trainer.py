"""Trainer cho nhánh Classic ML: feature engineering + impute + XGBoost (Mục 4.4)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import optuna

from datasets.features import lite_stat_temp, mask_delta_features
from datasets.missingness import MissingnessConfig, apply_mask
from evaluation.metrics import compute_all_metrics
from models.classic import build_imputer, build_xgboost, compute_scale_pos_weight

logger = logging.getLogger("p4aids")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ClassicRunResult:
    config: MissingnessConfig
    impute_strategy: str
    imbalance_strategy: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)


def build_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LiteStatTemp + đặc trưng mask/Δt, ghép lại thành một ma trận đặc trưng (Mục 4.4.1)."""
    Xm = apply_mask(X, mask)
    stat_feats, _ = lite_stat_temp(Xm)
    mask_feats, _ = mask_delta_features(mask)
    return np.concatenate([stat_feats, mask_feats], axis=1)


def _apply_imbalance(features: np.ndarray, y: np.ndarray, strategy: str,
                      imputed: bool, seed: int = 42):
    """Trả về ``(features, y)`` đã xử lý mất cân bằng, và ``scale_pos_weight`` (None nếu không dùng)."""
    if strategy == "none":
        return features, y, None
    if strategy == "scale_pos_weight":
        return features, y, compute_scale_pos_weight(y)
    if strategy == "smote":
        if not imputed:
            raise ValueError(
                "SMOTE cần dữ liệu không NaN -- không dùng được với impute_strategy='xgboost_native'."
            )
        from imblearn.over_sampling import SMOTE
        features_res, y_res = SMOTE(random_state=seed).fit_resample(features, y)
        return features_res, y_res, None
    raise ValueError(f"Chiến lược mất cân bằng không hợp lệ: '{strategy}'.")


def _fit_predict(features_train: np.ndarray, y_train: np.ndarray,
                  features_eval: np.ndarray, impute_strategy: str,
                  imbalance_strategy: str, xgb_params: Optional[Dict[str, Any]] = None,
                  seed: int = 42):
    imputer = build_imputer(impute_strategy)
    if imputer is not None:
        features_train = imputer.fit_transform(features_train)
        features_eval = imputer.transform(features_eval)

    features_train, y_train, scale_pos_weight = _apply_imbalance(
        features_train, y_train, imbalance_strategy, imputed=imputer is not None, seed=seed,
    )
    model = build_xgboost(xgb_params, scale_pos_weight=scale_pos_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(features_train, y_train)

    proba = model.predict_proba(features_eval)[:, 1]
    preds = (proba >= 0.5).astype(np.int64)
    return model, preds, proba


def tune_xgboost(features_train: np.ndarray, y_train: np.ndarray,
                  features_val: np.ndarray, y_val: np.ndarray,
                  impute_strategy: str, imbalance_strategy: str,
                  n_trials: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Optuna search cho siêu tham số XGBoost, mục tiêu = Macro F1 trên validation (Mục 4.4.4)."""

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        _, preds, _ = _fit_predict(features_train, y_train, features_val, impute_strategy,
                                    imbalance_strategy, xgb_params=params, seed=seed)
        return compute_all_metrics(y_val, preds)["macro_f1"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna: best macro_f1=%.4f, params=%s", study.best_value, study.best_params)
    return dict(study.best_params)


def run_classic(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                 config: MissingnessConfig, impute_strategy: str = "xgboost_native",
                 imbalance_strategy: str = "scale_pos_weight",
                 tune: bool = False, n_trials: int = 30, seed: int = 42) -> ClassicRunResult:
    """Chạy một cấu hình đầy đủ: feature engineering -> (tuning) -> fit -> eval val/test."""
    features = build_features(X, mask)
    f_train, f_val, f_test = features[train_idx], features[val_idx], features[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    best_params: Dict[str, Any] = {}
    if tune:
        best_params = tune_xgboost(f_train, y_train, f_val, y_val,
                                    impute_strategy, imbalance_strategy, n_trials, seed)

    _, val_preds, val_proba = _fit_predict(f_train, y_train, f_val, impute_strategy,
                                            imbalance_strategy, xgb_params=best_params, seed=seed)
    _, test_preds, test_proba = _fit_predict(f_train, y_train, f_test, impute_strategy,
                                              imbalance_strategy, xgb_params=best_params, seed=seed)

    return ClassicRunResult(
        config=config, impute_strategy=impute_strategy, imbalance_strategy=imbalance_strategy,
        best_params=best_params,
        val_metrics=compute_all_metrics(y_val, val_preds, val_proba),
        test_metrics=compute_all_metrics(y_test, test_preds, test_proba),
    )


def run_classic_sweep(X: np.ndarray, y: np.ndarray, masks: Dict[MissingnessConfig, np.ndarray],
                       train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                       impute_strategy: str = "xgboost_native",
                       imbalance_strategy: str = "scale_pos_weight",
                       tune: bool = False, n_trials: int = 30,
                       seed: int = 42) -> List[ClassicRunResult]:
    """Chạy ``run_classic`` cho nhiều cấu hình mất dữ liệu, dùng cho Mục 6.2."""
    results = []
    for config, mask in masks.items():
        logger.info("Classic ML | %s | impute=%s | imbalance=%s",
                    config.key, impute_strategy, imbalance_strategy)
        results.append(run_classic(X, y, mask, train_idx, val_idx, test_idx, config,
                                    impute_strategy, imbalance_strategy, tune, n_trials, seed))
    return results
