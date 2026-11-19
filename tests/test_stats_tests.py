"""Kiểm tra evaluation/stats_tests.py: McNemar, Friedman, critical difference (Mục 5.4/6.5)."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.stats_tests import (
    NEMENYI_Q_ALPHA_005, compute_average_ranks, critical_difference,
    friedman_test, mcnemar_test,
)


def test_mcnemar_identical_predictors_gives_near_one_p_value():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    pred = y_true.copy()
    flip = rng.random(300) < 0.3
    pred[flip] = 1 - pred[flip]

    # Hai mô hình giống hệt nhau -> không có bất đồng -> không có bằng chứng khác biệt.
    _, p_value = mcnemar_test(y_true, pred, pred)
    assert p_value > 0.99


def test_mcnemar_detects_clear_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    pred_good = y_true.copy()
    flip_good = rng.random(1000) < 0.05
    pred_good[flip_good] = 1 - pred_good[flip_good]

    pred_bad = y_true.copy()
    flip_bad = rng.random(1000) < 0.45
    pred_bad[flip_bad] = 1 - pred_bad[flip_bad]

    _, p_value = mcnemar_test(y_true, pred_good, pred_bad)
    assert p_value < 0.01


def test_friedman_detects_dominant_model():
    rng = np.random.default_rng(0)
    # Model A áp đảo rõ ràng model B, C trên mọi fold.
    n_folds = 10
    score_a = 0.9 + rng.normal(0, 0.01, n_folds)
    score_b = 0.6 + rng.normal(0, 0.01, n_folds)
    score_c = 0.5 + rng.normal(0, 0.01, n_folds)
    score_matrix = np.stack([score_a, score_b, score_c], axis=1)

    _, p_value = friedman_test(score_matrix)
    assert p_value < 0.01


def test_friedman_rejects_single_model_matrix():
    with pytest.raises(ValueError):
        friedman_test(np.ones((5, 1)))


def test_compute_average_ranks_best_model_gets_rank_one():
    # 3 fold, model 0 luôn tốt nhất, model 2 luôn tệ nhất.
    score_matrix = np.array([
        [0.9, 0.5, 0.3],
        [0.8, 0.6, 0.2],
        [0.95, 0.55, 0.25],
    ])
    ranks = compute_average_ranks(score_matrix)
    assert ranks[0] == pytest.approx(1.0)
    assert ranks[2] == pytest.approx(3.0)


def test_critical_difference_matches_known_nemenyi_constant():
    # Công thức: CD = q_alpha * sqrt(k(k+1) / (6N)); kiểm tra trực tiếp với k=3, N=10.
    k, n = 3, 10
    expected = NEMENYI_Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    assert critical_difference(k, n) == pytest.approx(expected)


def test_critical_difference_rejects_unsupported_model_count():
    with pytest.raises(ValueError):
        critical_difference(n_models=20, n_folds=10)


def test_critical_difference_diagram_runs_and_saves_file(tmp_path):
    from evaluation.stats_tests import critical_difference_diagram

    rng = np.random.default_rng(0)
    score_matrix = rng.normal(loc=[0.9, 0.7, 0.5], scale=0.02, size=(6, 3))
    save_path = tmp_path / "cd.png"
    result_path = critical_difference_diagram(["A", "B", "C"], score_matrix, str(save_path))
    assert save_path.is_file()
    assert result_path == str(save_path)
