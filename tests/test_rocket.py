import numpy as np
import pandas as pd
import pytest

from datasets.missingness import FULL_DATA_CONFIG, apply_mask
from datasets.szeged import FEATURE_COLUMNS

from models.rocket import (
    interpolate_missing,
    to_aeon_format,
    choose_threshold,
    RocketClassifier
)

from trainers.rocket_trainer import CHANNELS, run_rocket


def test_channel_order():

    assert CHANNELS["pressure"] == 3
    assert CHANNELS["visibility"] == 4

    assert FEATURE_COLUMNS[3] == "Pressure (millibars)"
    assert FEATURE_COLUMNS[4] == "Visibility (km)"


def test_interpolation():

    x = np.array([[[2.0], [np.nan], [6.0]]])

    np.testing.assert_allclose(
        interpolate_missing(x).ravel(),
        [2, 4, 6]
    )


def test_missing_edges():

    x = np.array([[[np.nan], [4.0], [np.nan]]])

    np.testing.assert_allclose(
        interpolate_missing(x).ravel(),
        [4, 4, 4]
    )


def test_full_missing():

    x = np.full((2, 24, 1), np.nan)

    assert np.all(
        interpolate_missing(x, fallback=[7]) == 7
    )


def test_ffill():

    x = np.array([[[np.nan], [4.0], [np.nan]]])

    np.testing.assert_allclose(
        interpolate_missing(
            x, fallback=[2], method="ffill"
        ).ravel(),
        [2, 4, 4]
    )


def test_mask():

    x = np.ones((1, 24, 5))
    mask = np.ones_like(x, dtype=bool)

    mask[0, 0, 3] = False

    result = apply_mask(x, mask)

    assert np.isnan(result[0, 0, 3])
    assert result[0, 0, 4] == 1


def test_shape():

    assert to_aeon_format(
        np.zeros((8, 24, 5))
    ).shape == (8, 5, 24)


def test_threshold():

    y = np.array([0, 0, 1, 1])
    scores = np.array([-2, -1, 1, 2])

    threshold = choose_threshold(y, scores)

    np.testing.assert_equal(
        scores >= threshold, y
    )


def synthetic():

    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(80, 24, 5)
    ).astype(np.float32)

    y = np.tile([0, 1], 40)

    x[:, :, 0] += 4 * y[:, None]
    x[0, 2, 0] = np.nan

    return x, y


@pytest.mark.parametrize("variant", ["mini", "multi"])
def test_actual_aeon(variant):

    x, y = synthetic()

    clf = RocketClassifier(
        variant=variant,
        n_kernels=168,
        n_jobs=1,
        max_fit_samples=32,
        transform_batch_size=16,
        alphas=[0.1, 1.0]
    )

    clf.fit(
        x[:48], y[:48],
        X_val=x[48:64],
        y_val=y[48:64]
    )

    scores = clf.decision_scores(x[64:])

    assert scores.shape == (16,)
    assert np.isfinite(scores).all()

    assert clf.best_alpha in (0.1, 1.0)
    assert len(clf.validation_trials) == 2
    assert clf.val_scores_.shape == (16,)


def test_pipeline(tmp_path):

    x, y = synthetic()
    mask = ~np.isnan(x)

    path = tmp_path / "predictions.csv.gz"

    result = run_rocket(
        x, y, mask,
        np.arange(0, 48),
        np.arange(48, 64),
        np.arange(64, 80),
        FULL_DATA_CONFIG,

        variant="mini",
        n_kernels=168,
        n_jobs=1,
        max_fit_samples=32,
        transform_batch_size=16,
        alphas=[1.0],

        end_time=pd.date_range(
            "2015-01-01", periods=80, freq="h"
        ),

        prediction_path=path
    )

    pred = pd.read_csv(path)

    assert len(pred) == 16

    np.testing.assert_equal(
        pred["y_true"].to_numpy(), y[64:]
    )

    assert np.isfinite(
        result.test_metrics["macro_f1"]
    )

    assert np.isfinite(
        result.test_metrics["pr_auc"]
    )

    assert sum(map(
        sum, result.test_metrics["confusion_matrix"]
    )) == 16


def test_channel_ablation(tmp_path):

    x, y = synthetic()

    result = run_rocket(
        x, y, ~np.isnan(x),
        np.arange(48),
        np.arange(48, 64),
        np.arange(64, 80),
        FULL_DATA_CONFIG,

        channels=["temperature"],
        variant="mini",
        n_kernels=168,
        n_jobs=1,
        max_fit_samples=32,
        transform_batch_size=16,
        alphas=[1.0],

        prediction_path=tmp_path / "temp.csv.gz"
    )

    assert result.channels == ["temperature"]
    assert np.isfinite(result.test_metrics["macro_f1"])
