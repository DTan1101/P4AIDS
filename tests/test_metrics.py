"""Kiểm tra evaluation/metrics.py, so khớp trực tiếp với sklearn (Mục 5.3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
)

from evaluation.metrics import (
    balanced_accuracy, compute_all_metrics, macro_f1, pr_auc, precision_recall_per_class,
)


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_pred = y_true.copy()
    flip = rng.random(500) < 0.2
    y_pred[flip] = 1 - y_pred[flip]
    y_score = rng.random(500)
    return y_true, y_pred, y_score


def test_macro_f1_matches_sklearn(sample):
    y_true, y_pred, _ = sample
    assert macro_f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred, average="macro"))


def test_pr_auc_matches_sklearn(sample):
    y_true, _, y_score = sample
    assert pr_auc(y_true, y_score) == pytest.approx(average_precision_score(y_true, y_score))


def test_balanced_accuracy_matches_sklearn(sample):
    y_true, y_pred, _ = sample
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_macro_f1_perfect_predictions_is_one():
    y = [0, 0, 1, 1, 1]
    assert macro_f1(y, y) == 1.0


def test_precision_recall_per_class_known_confusion():
    # 4 Rain đúng, 1 Rain bị đoán thành Snow; 2 Snow đúng, 1 Snow bị đoán thành Rain.
    y_true = [0, 0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 0]
    result = precision_recall_per_class(y_true, y_pred, class_names=("Rain", "Snow"))

    assert result["Rain"]["support"] == 5
    assert result["Rain"]["recall"] == pytest.approx(4 / 5)
    assert result["Snow"]["support"] == 3
    assert result["Snow"]["recall"] == pytest.approx(2 / 3)


def test_compute_all_metrics_includes_pr_auc_only_when_score_given(sample):
    y_true, y_pred, y_score = sample
    without_score = compute_all_metrics(y_true, y_pred)
    with_score = compute_all_metrics(y_true, y_pred, y_score)

    assert "pr_auc" not in without_score
    assert "pr_auc" in with_score
    assert with_score["n_samples"] == len(y_true)
