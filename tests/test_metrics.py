"""Tests for validation metrics."""

import math

import numpy as np
import pytest

from seed_preview_cv.training.metrics import (
    compute_all_metrics,
    compute_quality,
    online_filtering_metrics,
    regression_metrics,
    score_prediction_metrics,
)


def test_compute_quality_shape_and_zero_scores():
    scores = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    quality = compute_quality(scores, eps=1e-6)
    assert quality.shape == (3,)
    assert np.all(np.isfinite(quality))
    assert quality[2] > quality[0] > quality[1]


def test_regression_metrics_keys():
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.1, 0.4, 0.9])
    metrics = regression_metrics(y_true, y_pred, "forest")
    assert metrics["forest/mae"] == pytest.approx(0.1)
    assert metrics["forest/rmse"] > 0
    assert -1.0 <= metrics["forest/pearson"] <= 1.0
    assert -1.0 <= metrics["forest/spearman"] <= 1.0


def test_regression_metrics_constant_returns_nan_correlation():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.2, 0.3])
    metrics = regression_metrics(y_true, y_pred, "ocean")
    assert math.isnan(metrics["ocean/pearson"])
    assert math.isnan(metrics["ocean/spearman"])


def test_score_prediction_metrics_without_total():
    y_true = np.array([[1.0, 0.8, 0.6], [0.5, 0.4, 0.3]])
    y_pred = np.array([[0.9, 0.7, 0.5], [0.4, 0.3, 0.2]])
    metrics = score_prediction_metrics(y_true, y_pred)
    assert "forest/mae" in metrics
    assert "quality/spearman" in metrics
    assert metrics["quality/target_source"] == "computed_from_scores"


def test_score_prediction_metrics_with_total():
    y_true = np.array([[1.0, 0.8, 0.6], [0.5, 0.4, 0.3]])
    y_pred = np.array([[0.9, 0.7, 0.5], [0.4, 0.3, 0.2]])
    y_total = np.array([0.9, 0.2])
    metrics = score_prediction_metrics(y_true, y_pred, y_true_total=y_total)
    assert metrics["quality/target_source"] == "s_total"


def test_online_filtering_metrics_keys():
    true_q = np.linspace(0.0, 1.0, 100)
    pred_q = true_q + np.random.default_rng(0).normal(0, 0.05, size=100)
    metrics = online_filtering_metrics(true_q, pred_q, accept_rates=[0.10], true_good_rate=0.10)
    assert "filter/accept_0.10/count" in metrics
    assert "filter/accept_0.10/precision" in metrics
    assert "filter/accept_0.10/recall" in metrics
    assert "filter/accept_0.10/enrichment" in metrics


def test_compute_all_metrics_runs():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 1, size=(32, 3))
    y_pred = rng.uniform(0, 1, size=(32, 3))
    metrics = compute_all_metrics(y_true, y_pred)
    assert "forest/mae" in metrics
    assert "quality/mae" in metrics
    assert "filter/accept_0.05/count" in metrics


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        score_prediction_metrics(np.zeros((2, 3)), np.zeros((3, 3)))
