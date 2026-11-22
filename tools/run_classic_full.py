"""Runner Classic ML đầy đủ cho benchmark Rain/Snow.

Chạy các model tree-based, feature-set/ablation và mọi cấu hình missingness,
ghi JSON/CSV tổng hợp để tái lập và phân tích thống kê.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.cache import load_prepared_data
from datasets.features import lite_stat_temp, mask_delta_features
from datasets.missingness import ALL_CONFIGS, FULL_DATA_CONFIG, apply_mask
from evaluation.metrics import compute_all_metrics
from models.classic import (
    build_catboost,
    build_lightgbm,
    build_missforest_imputer,
    build_random_forest,
    build_xgboost,
    compute_scale_pos_weight,
)
from utils.utils import ensure_dir, setup_logger


@dataclass
class ClassicFullRun:
    model: str
    feature_set: str
    imputation: str
    imbalance: str
    missing_kind: str
    missing_rate: float
    seed: int
    train_seconds: float
    infer_seconds: float
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    test_predictions: List[int] = field(default_factory=list)
    test_probability: List[float] = field(default_factory=list)
    test_true: List[int] = field(default_factory=list)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def lite_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xm = apply_mask(X, mask)
    stats, _ = lite_stat_temp(Xm)
    missing, _ = mask_delta_features(mask)
    return np.concatenate([stats, missing], axis=1)


def raw_lag_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xm = apply_mask(X, mask)
    missing, _ = mask_delta_features(mask)
    return np.concatenate([Xm.reshape(Xm.shape[0], -1), missing], axis=1)


def catch22_feature_matrix(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from datasets.features import catch22_features

    Xm = apply_mask(X, mask)
    features, _ = catch22_features(Xm)
    missing, _ = mask_delta_features(mask)
    return np.nan_to_num(np.concatenate([features, missing], axis=1), nan=0.0)


def tsfresh_feature_matrix(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from datasets.features import tsfresh_features

    Xm = apply_mask(X, mask)
    # tsfresh không cho phép NaN trong dataframe wide; nội suy từng chuỗi
    # (biên = quan trắc gần nhất) trước khi trích xuất feature.
    filled = Xm.copy()
    t = np.arange(Xm.shape[1])
    for i in range(Xm.shape[0]):
        for c in range(Xm.shape[2]):
            series = filled[i, :, c]
            valid = ~np.isnan(series)
            if valid.sum() == Xm.shape[1]:
                continue
            if valid.sum() == 0:
                filled[i, :, c] = 0.0
            else:
                filled[i, :, c] = np.interp(t, t[valid], series[valid])
    features, _ = tsfresh_features(filled)
    missing, _ = mask_delta_features(mask)
    return np.nan_to_num(np.concatenate([features, missing], axis=1), nan=0.0)


def build_features(name: str, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if name == "lite_mask_delta":
        return lite_features(X, mask)
    if name == "raw_lag_mask_delta":
        return raw_lag_features(X, mask)
    if name == "catch22_mask_delta":
        return catch22_feature_matrix(X, mask)
    if name == "tsfresh_mask_delta":
        return tsfresh_feature_matrix(X, mask)
    raise ValueError(f"Feature set không hợp lệ: {name}")


def make_model(model_name: str, seed: int, imbalance: str, y_train: np.ndarray,
               params: Optional[Dict[str, Any]] = None):
    scale = compute_scale_pos_weight(y_train) if imbalance == "scale_pos_weight" else None
    if model_name == "xgboost":
        return build_xgboost(params, scale_pos_weight=scale)
    if model_name == "random_forest":
        rf_params = dict(params or {})
        rf_params.setdefault("random_state", seed)
        rf_params["class_weight"] = "balanced" if imbalance == "scale_pos_weight" else None
        return build_random_forest(rf_params)
    if model_name == "lightgbm":
        return build_lightgbm(params, scale_pos_weight=scale)
    if model_name == "catboost":
        return build_catboost(params, scale_pos_weight=scale)
    raise ValueError(f"Model không hợp lệ: {model_name}")


def prepare_train_eval(features: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                       test_idx: np.ndarray, imputation: str, seed: int):
    f_train, f_val, f_test = features[train_idx], features[val_idx], features[test_idx]
    if imputation == "native":
        return f_train, f_val, f_test
    if imputation == "median_mask":
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        return imputer.fit_transform(f_train), imputer.transform(f_val), imputer.transform(f_test)
    if imputation == "missforest":
        imputer = build_missforest_imputer(n_estimators=30, max_iter=5, random_state=seed)
        return imputer.fit_transform(f_train), imputer.transform(f_val), imputer.transform(f_test)
    raise ValueError(f"Imputation không hợp lệ: {imputation}")


def smote_resample(features: np.ndarray, y: np.ndarray, seed: int):
    from imblearn.over_sampling import SMOTE

    return SMOTE(random_state=seed).fit_resample(features, y)


def fit_predict(model_name: str, f_train: np.ndarray, y_train: np.ndarray,
                f_eval: np.ndarray, imbalance: str, seed: int,
                imputation: str, params: Optional[Dict[str, Any]] = None,
                resample_eval: bool = False):
    if imbalance == "smote":
        f_train, y_train = smote_resample(f_train, y_train, seed)
    model = make_model(model_name, seed, imbalance, y_train, params)
    model.fit(f_train, y_train)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(f_eval)[:, 1]
    else:
        scores = model.decision_function(f_eval)
        proba = 1.0 / (1.0 + np.exp(-scores))
    preds = (proba >= 0.5).astype(np.int64)
    return model, preds, proba


def tune_xgboost(f_train: np.ndarray, y_train: np.ndarray, f_val: np.ndarray,
                 y_val: np.ndarray, imputation: str, imbalance: str, seed: int,
                 n_trials: int = 12) -> Dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }
        _, preds, _ = fit_predict(
            "xgboost", f_train, y_train, f_val, imbalance, seed, imputation, params
        )
        return float(compute_all_metrics(y_val, preds)["macro_f1"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params)


def run_one(model_name: str, feature_set: str, imputation: str, imbalance: str,
            X: np.ndarray, y: np.ndarray, mask: np.ndarray,
            train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
            kind: str, rate: float, seed: int,
            tune: bool = False) -> ClassicFullRun:
    features = build_features(feature_set, X, mask)
    f_train, f_val, f_test = prepare_train_eval(
        features, train_idx, val_idx, test_idx, imputation, seed
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    best_params: Dict[str, Any] = {}
    if tune and model_name == "xgboost":
        best_params = tune_xgboost(
            f_train, y_train, f_val, y_val, imputation, imbalance, seed, n_trials=12
        )

    started = time.perf_counter()
    _, val_preds, val_proba = fit_predict(
        model_name, f_train, y_train, f_val, imbalance, seed, imputation, best_params
    )
    _, test_preds, test_proba = fit_predict(
        model_name, f_train, y_train, f_test, imbalance, seed, imputation, best_params
    )
    train_seconds = time.perf_counter() - started

    return ClassicFullRun(
        model=model_name,
        feature_set=feature_set,
        imputation=imputation,
        imbalance=imbalance,
        missing_kind=kind,
        missing_rate=rate,
        seed=seed,
        train_seconds=train_seconds,
        infer_seconds=0.0,
        val_metrics=compute_all_metrics(y_val, val_preds, val_proba),
        test_metrics=compute_all_metrics(y_test, test_preds, test_proba),
        best_params=best_params,
        test_predictions=[int(v) for v in test_preds],
        test_probability=[float(v) for v in test_proba],
        test_true=[int(v) for v in y_test],
    )


def run_key(run: ClassicFullRun) -> str:
    tuned = "_tuned" if run.best_params else ""
    return (
        f"{run.model}_{run.feature_set}_{run.imputation}_{run.imbalance}"
        f"{tuned}_{run.missing_kind}_{round(run.missing_rate * 100):02d}_seed{run.seed}"
    )


def run_exists(output_dir: str, run: ClassicFullRun) -> bool:
    return os.path.isfile(os.path.join(output_dir, f"{run_key(run)}.json"))


def write_run(output_dir: str, run: ClassicFullRun) -> None:
    with open(os.path.join(output_dir, f"{run_key(run)}.json"), "w", encoding="utf-8") as fp:
        json.dump(dataclasses.asdict(run), fp, ensure_ascii=False, indent=2)


def write_summary(output_dir: str, runs: List[ClassicFullRun]) -> None:
    rows = []
    for run in runs:
        row = {
            "model": run.model,
            "feature_set": run.feature_set,
            "imputation": run.imputation,
            "imbalance": run.imbalance,
            "missing_kind": run.missing_kind,
            "missing_rate": run.missing_rate,
            "seed": run.seed,
            "train_seconds": run.train_seconds,
            "val_macro_f1": run.val_metrics.get("macro_f1"),
            "val_pr_auc": run.val_metrics.get("pr_auc"),
            "val_balanced_accuracy": run.val_metrics.get("balanced_accuracy"),
            "test_macro_f1": run.test_metrics.get("macro_f1"),
            "test_pr_auc": run.test_metrics.get("pr_auc"),
            "test_balanced_accuracy": run.test_metrics.get("balanced_accuracy"),
        }
        per_class = run.test_metrics.get("per_class", {})
        snow = per_class.get("Snow", {})
        rain = per_class.get("Rain", {})
        row.update({
            "test_snow_precision": snow.get("precision"),
            "test_snow_recall": snow.get("recall"),
            "test_snow_f1": snow.get("f1"),
            "test_rain_precision": rain.get("precision"),
            "test_rain_recall": rain.get("recall"),
            "test_rain_f1": rain.get("f1"),
        })
        rows.append(row)
    if not rows:
        return
    columns = list(rows[0])
    with open(os.path.join(output_dir, "classic_full_summary.csv"), "w",
              newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def package_versions() -> Dict[str, str]:
    names = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost",
             "optuna", "imbalanced-learn", "pycatch22", "tsfresh")
    versions: Dict[str, str] = {}
    import importlib.metadata
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> int:
    setup_logger()
    logger = logging.getLogger("p4aids")
    data = load_prepared_data("outputs/data/windows.npz")
    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    masks: Dict[Any, np.ndarray] = {FULL_DATA_CONFIG: ~np.isnan(X)}
    for config in ALL_CONFIGS:
        path = os.path.join(
            "outputs", "masks",
            f"mask_{config.key}_seed123_n{X.shape[0]}.npz",
        )
        masks[config] = np.load(path)["mask"]

    output_dir = "outputs/results/classic_full"
    ensure_dir(output_dir)
    runs: List[ClassicFullRun] = []

    main_matrix = (
        ("xgboost", "lite_mask_delta", "native", "scale_pos_weight"),
        ("random_forest", "lite_mask_delta", "median_mask", "scale_pos_weight"),
        ("lightgbm", "lite_mask_delta", "native", "scale_pos_weight"),
        ("catboost", "lite_mask_delta", "native", "scale_pos_weight"),
        ("xgboost", "lite_mask_delta", "median_mask", "scale_pos_weight"),
    )

    for config, mask in masks.items():
        for model_name, feature_set, imputation, imbalance in main_matrix:
            logger.info(
                "Classic FULL | %s | %s | %s | %s",
                model_name, feature_set, imputation, config.key,
            )
            run = run_one(
                model_name, feature_set, imputation, imbalance,
                X, y, mask, train_idx, val_idx, test_idx,
                config.kind, config.rate, seed=42,
            )
            runs.append(run)
            write_run(output_dir, run)
            logger.info(
                "test_macro_f1=%.4f test_pr_auc=%.4f",
                run.test_metrics["macro_f1"], run.test_metrics.get("pr_auc", 0.0),
            )

    # Ablation feature set chỉ chạy trên 0%, point50 và block50 để kiểm soát chi phí.
    for config in (FULL_DATA_CONFIG,
                   ALL_CONFIGS[ALL_CONFIGS.index(max(
                       ALL_CONFIGS, key=lambda c: c.rate if c.kind == "point" else -1.0
                   ))],
                   next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5)):
        mask = masks[config]
        for feature_set in ("raw_lag_mask_delta", "catch22_mask_delta", "tsfresh_mask_delta"):
            logger.info("Classic ablation | %s | %s", feature_set, config.key)
            run = run_one(
                "xgboost", feature_set, "native", "scale_pos_weight",
                X, y, mask, train_idx, val_idx, test_idx,
                config.kind, config.rate, seed=42,
            )
            runs.append(run)
            write_run(output_dir, run)

    # So sánh imputation bổ sung; MissForest chỉ cần cài trên ba cấu hình cao nhất.
    for config in (FULL_DATA_CONFIG,
                   next(c for c in ALL_CONFIGS if c.kind == "point" and c.rate == 0.3),
                   next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5)):
        run = run_one(
            "xgboost", "lite_mask_delta", "missforest", "scale_pos_weight",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42,
        )
        runs.append(run)
        write_run(output_dir, run)

    # SMOTE và tuning XGBoost ở một cấu hình đại diện 30% point và 50% block.
    for config in (
        next(c for c in ALL_CONFIGS if c.kind == "point" and c.rate == 0.3),
        next(c for c in ALL_CONFIGS if c.kind == "block" and c.rate == 0.5),
    ):
        smote_run = run_one(
            "xgboost", "lite_mask_delta", "median_mask", "smote",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42,
        )
        runs.append(smote_run)
        write_run(output_dir, smote_run)

        tuned_run = run_one(
            "xgboost", "lite_mask_delta", "native", "scale_pos_weight",
            X, y, masks[config], train_idx, val_idx, test_idx,
            config.kind, config.rate, seed=42, tune=True,
        )
        runs.append(tuned_run)
        write_run(output_dir, tuned_run)

    write_summary(output_dir, runs)
    environment = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_versions": package_versions(),
        "data": {
            "n_windows": int(X.shape[0]),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "seed": 42,
        "missing_mask_seed": 123,
    }
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as fp:
        json.dump(environment, fp, ensure_ascii=False, indent=2)
    logger.info("Hoàn tất %d run Classic ML; summary=%s", len(runs),
                os.path.join(output_dir, "classic_full_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
