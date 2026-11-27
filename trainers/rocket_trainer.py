"""ROCKET trainer using the shared team protocol."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
import hashlib

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from datasets.features import mask_delta_features
from datasets.missingness import MissingnessConfig, apply_mask
from datasets.szeged import FEATURE_COLUMNS
from evaluation.metrics import compute_all_metrics
from models.rocket import RocketClassifier


CHANNELS = {
    "temperature": 0,
    "humidity": 1,
    "wind": 2,
    "pressure": 3,
    "visibility": 4
}


@dataclass
class RocketRunResult:
    config: MissingnessConfig
    variant: str
    use_mask_delta: bool

    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)

    seed: int = 42
    channels: list = field(default_factory=list)

    metadata: dict = field(default_factory=dict)
    subgroup_metrics: dict = field(default_factory=dict)
    validation_trials: list = field(default_factory=list)


def subgroup_result(y, pred, score, selected):

    n = int(selected.sum())
    snow = int(y[selected].sum())

    if n == 0 or np.unique(y[selected]).size < 2:
        return {
            "n_samples": n,
            "snow_support": snow,
            "macro_f1": None,
            "pr_auc": None
        }

    return compute_all_metrics(
        y[selected], pred[selected], score[selected]
    )


def run_rocket(
    X, y, mask,
    train_idx, val_idx, test_idx, config,
    variant="mini",
    use_mask_delta=True,
    seed=42,
    n_jobs=2,
    max_fit_samples=2500,
    transform_batch_size=512,
    n_kernels=None,
    alphas=(0.1, 1.0, 10.0),
    class_weight="balanced",
    tune_threshold=True,
    imputer="linear",
    max_feature_mib=1024,
    channels=None,
    end_time=None,
    prediction_path=None
):
    channels = (
        list(CHANNELS)
        if channels is None else list(channels)
    )

    if (
        not channels
        or len(set(channels)) != len(channels)
        or not set(channels) <= set(CHANNELS)
    ):
        raise ValueError("Invalid channels")

    if X.ndim != 3 or X.shape[1:] != (24, 5):
        raise ValueError("Expected (N,24,5)")

    if mask.shape != X.shape or mask.dtype != np.bool_:
        raise ValueError("Invalid mask")

    if np.any(mask & np.isnan(X)):
        raise ValueError("Mask marks natural NaN as observed")

    indices = [CHANNELS[c] for c in channels]
    feature_names = [FEATURE_COLUMNS[i] for i in indices]

    start = perf_counter()

    # True = observed, False = missing.
    selected_mask = mask[:, :, indices]

    Xm = apply_mask(
        X[:, :, indices], selected_mask
    )

    md = None

    if use_mask_delta:
        md, _ = mask_delta_features(
            selected_mask, feature_names
        )

        md = md.astype(np.float32)

    prep_seconds = perf_counter() - start

    def subset(idx):
        return (
            Xm[idx],
            None if md is None else md[idx]
        )

    xtr, mtr = subset(train_idx)
    xv, mv = subset(val_idx)
    xt, mt = subset(test_idx)

    ytr = y[train_idx]
    yv = y[val_idx]
    yt = y[test_idx]

    clf = RocketClassifier(
        variant=variant,
        random_state=seed,
        n_jobs=n_jobs,
        max_fit_samples=max_fit_samples,
        transform_batch_size=transform_batch_size,
        n_kernels=n_kernels,
        alphas=alphas,
        class_weight=class_weight,
        tune_threshold=tune_threshold,
        imputer=imputer,
        max_feature_mib=max_feature_mib
    )

    start = perf_counter()

    clf.fit(
        xtr, ytr, mtr,
        X_val=xv,
        y_val=yv,
        mask_delta_val=mv
    )

    training_seconds = perf_counter() - start

    # Reuse validation scores.
    sv = clf.val_scores_

    start = perf_counter()

    # Transform test only once.
    st = clf.decision_scores(xt, mt)

    inference_seconds = perf_counter() - start

    pv = (sv >= clf.threshold).astype(np.int64)
    pt = (st >= clf.threshold).astype(np.int64)

    vm = compute_all_metrics(yv, pv, sv)
    tm = compute_all_metrics(yt, pt, st)

    tm["confusion_matrix"] = confusion_matrix(
        yt, pt, labels=[0, 1]
    ).tolist()

    tm["macro_f1_default_threshold"] = compute_all_metrics(
        yt, (st >= 0).astype(np.int64), st
    )["macro_f1"]

    # Original last historical temperature.
    # Used for diagnostics only, not model inputs.
    temperature = X[test_idx, -1, 0]

    near = (
        np.isfinite(temperature)
        & (temperature >= 0)
        & (temperature <= 4)
    )

    groups = {
        "temperature_0_4": subgroup_result(
            yt, pt, st, near
        )
    }

    times = None

    if end_time is not None:

        times = pd.DatetimeIndex(end_time)[test_idx]

        seasons = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11]
        }

        for name, months in seasons.items():

            groups[name] = subgroup_result(
                yt, pt, st,
                np.isin(times.month, months)
            )

    if prediction_path is not None:

        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "sample_idx": test_idx,
            "target_time": (
                times.astype(str)
                if times is not None else ""
            ),
            "y_true": yt,
            "y_pred": pt,
            "decision_score": st,
            "original_last_temperature": temperature,
            "missing_rate": (
                1 - mask[test_idx].mean(axis=(1, 2))
            )
        }).to_csv(path, index=False)

    metadata = {
        "n_kernels": n_kernels,
        "n_features": clf.n_features,
        "best_alpha": clf.best_alpha,
        "threshold": clf.threshold,
        "class_weight": class_weight,
        "imputer": imputer,
        "tune_threshold": bool(tune_threshold),

        "prep_seconds": prep_seconds,
        "training_tuning_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,

        "inference_ms_per_sample": (
            1000 * inference_seconds / len(yt)
        ),

        "linear_parameters": int(
            clf.classifier.coef_.size
            + clf.classifier.intercept_.size
        ),

        "train_class_counts": np.bincount(
            ytr, minlength=2
        ).tolist(),

        "actual_missing_train": float(
            1 - mask[train_idx].mean()
        ),

        "actual_missing_val": float(
            1 - mask[val_idx].mean()
        ),

        "actual_missing_test": float(
            1 - mask[test_idx].mean()
        ),

        "selected_channel_missing_test": float(
            np.isnan(xt).mean()
        ),

        "mask_sha256": hashlib.sha256(
            mask.tobytes()
        ).hexdigest()
    }

    return RocketRunResult(
        config=config,
        variant=variant,
        use_mask_delta=use_mask_delta,
        seed=seed,
        channels=channels,
        val_metrics=vm,
        test_metrics=tm,
        metadata=metadata,
        subgroup_metrics=groups,
        validation_trials=clf.validation_trials
    )


def run_rocket_sweep(
    X, y, masks,
    train_idx, val_idx, test_idx,
    **kwargs
):
    return [
        run_rocket(
            X, y, mask,
            train_idx, val_idx, test_idx,
            config, **kwargs
        )
        for config, mask in masks.items()
    ]
