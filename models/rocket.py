"""MiniRocket/MultiRocket with validation-tuned Ridge."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROCKET_VARIANTS = ("mini", "multi")


def interpolate_missing(X, fallback=None, method="linear"):
    out = np.asarray(X, dtype=np.float32).copy()

    if out.ndim != 3:
        raise ValueError("Expected (N, time, channels)")

    if method not in ("linear", "ffill"):
        raise ValueError("Invalid imputation method")

    out[~np.isfinite(out)] = np.nan

    if fallback is None:
        fallback = np.zeros(out.shape[2])

    t = np.arange(out.shape[1])

    for i, c in np.argwhere(np.isnan(out).any(axis=1)):
        row = out[i, :, c]
        valid = np.isfinite(row)

        if not valid.any():
            row[:] = fallback[c]

        elif method == "linear":
            row[~valid] = np.interp(
                t[~valid], t[valid], row[valid]
            )

        else:
            last = fallback[c]

            for j in range(len(row)):
                if valid[j]:
                    last = row[j]
                else:
                    row[j] = last

    if not np.isfinite(out).all():
        raise ValueError("Imputation failed")

    return out


def to_aeon_format(X):
    return np.ascontiguousarray(
        X.transpose(0, 2, 1),
        dtype=np.float32
    )


def build_rocket_transformer(
    variant="mini",
    random_state=42,
    n_jobs=2,
    n_kernels=None
):
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket
    )

    kwargs = dict(
        random_state=random_state,
        n_jobs=n_jobs
    )

    if n_kernels is not None:
        kwargs["n_kernels"] = int(n_kernels)

    if variant == "mini":
        return MiniRocket(**kwargs)

    if variant == "multi":
        return MultiRocket(**kwargs)

    raise ValueError(f"Invalid variant: {variant}")


def build_ridge_classifier(alpha=1.0, class_weight="balanced"):
    return RidgeClassifier(
        alpha=float(alpha),
        class_weight=class_weight,
        solver="lsqr",
        tol=1e-3,
        max_iter=1000
    )


def macro_f1(y, pred):
    return float(f1_score(
        y,
        pred,
        labels=[0, 1],
        average="macro",
        zero_division=0
    ))


def choose_threshold(y, scores):
    scores = np.asarray(scores, dtype=np.float64)

    if not scores.size or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores")

    thresholds = np.unique(np.r_[
        0.0,
        np.quantile(scores, np.linspace(0, 1, 101)),
        np.nextafter(scores.max(), np.inf)
    ])

    return float(max(
        thresholds,
        key=lambda t: (
            macro_f1(y, scores >= t),
            -abs(t)
        )
    ))


class RocketClassifier:

    def __init__(
        self,
        variant="mini",
        random_state=42,
        n_jobs=2,
        max_fit_samples=2500,
        transform_batch_size=512,
        n_kernels=None,
        alphas=(0.1, 1.0, 10.0),
        class_weight="balanced",
        tune_threshold=True,
        imputer="linear",
        max_feature_mib=1024
    ):
        self.variant = variant
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

        self.max_fit_samples = int(max_fit_samples)
        self.batch_size = int(transform_batch_size)

        self.alphas = tuple(float(a) for a in alphas)
        self.class_weight = class_weight
        self.tune_threshold = tune_threshold
        self.imputer = imputer
        self.max_feature_mib = float(max_feature_mib)

        if (
            min(self.n_jobs, self.batch_size,
                self.max_fit_samples) <= 0
            or not self.alphas
            or min(self.alphas) <= 0
            or self.max_feature_mib <= 0
        ):
            raise ValueError("Invalid configuration")

        self.transformer = build_rocket_transformer(
            variant, random_state, n_jobs, n_kernels
        )

        self.scaler = StandardScaler(copy=False)
        self.classifier = None

        self.best_alpha = None
        self.threshold = 0.0
        self.validation_trials = []
        self.val_scores_ = None
        self._fitted = False

    def _batch_features(self, X, mask_delta=None):

        filled = interpolate_missing(
            X, self.fallback, self.imputer
        )

        z = np.asarray(
            self.transformer.transform(
                to_aeon_format(filled)
            ),
            dtype=np.float32
        )

        if mask_delta is not None:
            z = np.concatenate([
                z,
                np.asarray(mask_delta, dtype=np.float32)
            ], axis=1)

        if not np.isfinite(z).all():
            raise ValueError("Non-finite features")

        return z

    def _batches(self, X, mask_delta=None):

        for start in range(0, len(X), self.batch_size):

            end = min(start + self.batch_size, len(X))

            md = (
                None if mask_delta is None
                else mask_delta[start:end]
            )

            yield start, end, self._batch_features(
                X[start:end], md
            )

    def _features(self, X, mask_delta=None):

        if not len(X):
            raise ValueError("Empty split")

        result = None

        for start, end, z in self._batches(X, mask_delta):

            if result is None:
                width = z.shape[1]

                mib = len(X) * width * 4 / 1024**2

                if mib > self.max_feature_mib:
                    raise MemoryError(
                        f"Feature matrix: {mib:.1f} MiB. "
                        "Reduce n_kernels."
                    )

                result = np.empty(
                    (len(X), width),
                    dtype=np.float32
                )

            result[start:end] = z

        return result

    def fit(
        self, X, y, mask_delta=None,
        X_val=None, y_val=None,
        mask_delta_val=None
    ):
        y = np.asarray(y)

        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Train must contain both classes")

        if X_val is not None and y_val is None:
            raise ValueError("Validation labels missing")

        self._fitted = False
        self.val_scores_ = None
        self.scaler = StandardScaler(copy=False)

        # Train-only imputation statistics.
        self.fallback = np.nanmedian(
            X, axis=(0, 1)
        ).astype(np.float32)

        if not np.isfinite(self.fallback).all():
            raise ValueError("Entire training channel missing")

        rng = np.random.default_rng(self.random_state)

        ids = rng.choice(
            len(X),
            size=min(len(X), self.max_fit_samples),
            replace=False
        )

        fit_data = interpolate_missing(
            X[ids], self.fallback, self.imputer
        )

        self.transformer.fit(
            to_aeon_format(fit_data)
        )

        ztr = self._features(X, mask_delta)
        self.n_features = int(ztr.shape[1])

        # Train-only scaling, using batches.
        for start in range(0, len(ztr), self.batch_size):
            self.scaler.partial_fit(
                ztr[start:start + self.batch_size]
            )

        for start in range(0, len(ztr), self.batch_size):
            part = ztr[start:start + self.batch_size]
            part[:] = self.scaler.transform(part)

        zv = None

        if X_val is not None:
            zv = self.scaler.transform(
                self._features(X_val, mask_delta_val)
            )

        candidates = (
            self.alphas
            if zv is not None
            else self.alphas[:1]
        )

        best = None
        self.validation_trials = []

        for alpha in candidates:

            clf = build_ridge_classifier(
                alpha, self.class_weight
            )

            with threadpool_limits(limits=self.n_jobs):
                clf.fit(ztr, y)

            threshold = 0.0
            score = None
            scores = None

            if zv is not None:

                scores = clf.decision_function(zv)

                if self.tune_threshold:
                    threshold = choose_threshold(
                        y_val, scores
                    )

                score = macro_f1(
                    y_val, scores >= threshold
                )

            self.validation_trials.append({
                "alpha": alpha,
                "threshold": threshold,
                "val_macro_f1": score
            })

            rank = (
                score if score is not None else 0,
                -abs(threshold),
                -alpha
            )

            if best is None or rank > best:

                best = rank
                self.classifier = clf
                self.best_alpha = alpha
                self.threshold = threshold

                self.val_scores_ = (
                    scores.copy()
                    if scores is not None else None
                )

        self._fitted = True
        return self

    def decision_scores(self, X, mask_delta=None):

        if not self._fitted:
            raise RuntimeError("Call fit() first")

        parts = []

        for _, _, z in self._batches(X, mask_delta):

            z = self.scaler.transform(z)

            with threadpool_limits(limits=self.n_jobs):
                parts.append(
                    self.classifier.decision_function(z)
                )

        if not parts:
            return np.empty(0)

        return np.concatenate(parts)

    def predict(self, X, mask_delta=None):
        return (
            self.decision_scores(X, mask_delta)
            >= self.threshold
        ).astype(np.int64)

    def predict_proba_positive(self, X, mask_delta=None):
        # Compatibility only, not calibrated probabilities.
        return expit(
            self.decision_scores(X, mask_delta)
        )
