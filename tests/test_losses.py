"""Tests for ScoreSmoothL1Loss."""

import pytest
import torch

from seed_preview_cv.training.losses import (
    ScoreSmoothL1Loss,
    build_loss_from_config,
    sanity_check_loss,
)


def _example_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.25],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.5, 0.5, 0.75],
        ],
        dtype=torch.float32,
    )
    return pred, target


def test_score_smooth_l1_loss_instantiates():
    criterion = ScoreSmoothL1Loss()
    assert criterion.beta == 0.05
    assert criterion.weights["forest"] == 1.0


def test_score_smooth_l1_loss_forward_returns_dict():
    criterion = ScoreSmoothL1Loss()
    pred, target = _example_batch()
    loss_dict = criterion(pred, target)
    assert set(loss_dict.keys()) == {
        "loss",
        "loss_forest",
        "loss_ocean",
        "loss_beach",
    }
    for key, value in loss_dict.items():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value)
        assert float(value) >= 0.0


def test_custom_weights_affect_total_loss():
    pred, target = _example_batch()
    default_loss = ScoreSmoothL1Loss()(pred, target)["loss"]
    weighted_loss = ScoreSmoothL1Loss(weights={"forest": 2.0, "ocean": 1.0, "beach": 1.0})(
        pred, target
    )["loss"]
    assert float(weighted_loss) != float(default_loss)


def test_invalid_shape_raises():
    criterion = ScoreSmoothL1Loss()
    pred = torch.randn(2, 2)
    target = torch.randn(2, 2)
    with pytest.raises(ValueError, match="pred.shape"):
        criterion(pred, target)


def test_build_loss_from_config_full_yaml():
    config = {
        "loss": {
            "name": "smooth_l1",
            "beta": 0.1,
            "weights": {"forest": 1.0, "ocean": 2.0, "beach": 1.0},
        },
        "model": {"name": "scratch_resnet_cnn"},
    }
    criterion = build_loss_from_config(config)
    assert isinstance(criterion, ScoreSmoothL1Loss)
    assert criterion.beta == 0.1
    assert criterion.weights["ocean"] == 2.0


def test_build_loss_from_config_loss_subconfig():
    loss_cfg = {
        "name": "smooth_l1",
        "beta": 0.05,
        "weights": {"beach": 0.5},
    }
    criterion = build_loss_from_config(loss_cfg)
    assert criterion.weights["beach"] == 0.5
    assert criterion.weights["forest"] == 1.0


def test_build_loss_from_config_invalid_name():
    with pytest.raises(ValueError, match="smooth_l1"):
        build_loss_from_config({"name": "mse"})


def test_sanity_check_loss_runs():
    sanity_check_loss()
