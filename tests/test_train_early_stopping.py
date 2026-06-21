"""Tests for val_loss early stopping in training."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from seed_preview_cv.training.train import (
    LOSS_KEYS,
    checkpoint_val_loss_improved,
    early_stop_val_loss_improved,
    early_stopping_config_from_training,
    restore_early_stopping_state,
    train,
)


def _make_tiny_csv(tmp_path: Path) -> tuple[Path, Path]:
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for i in range(4):
        image_path = image_dir / f"sample_{i}.png"
        Image.new("RGB", (512, 320), color=(i * 20, 50, 80)).save(image_path)
        paths.append(str(image_path))

    import pandas as pd

    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "image_path": paths,
            "seed": [-1, -2, -3, -4],
            "s_forest": [0.8, 0.6, 0.4, 0.2],
            "s_ocean": [0.5, 0.4, 0.3, 0.2],
            "s_beach": [0.3, 0.2, 0.1, 0.05],
            "split": ["train", "train", "train", "train"],
            "source": ["balanced", "balanced", "balanced", "balanced"],
            "s_total": [0.55, 0.45, 0.35, 0.25],
        }
    ).to_csv(csv_path, index=False)
    return csv_path, tmp_path


def _base_config(
    tmp_path: Path,
    csv_path: Path,
    image_root: Path,
    run_dir: Path,
    *,
    epochs: int = 5,
    early_stopping: dict | None = None,
    evaluate_test_after_training: bool = True,
) -> dict:
    training: dict = {
        "epochs": epochs,
        "batch_size": 2,
        "num_workers": 0,
        "optimizer": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "mixed_precision": False,
        "seed": 42,
        "save_best": True,
        "save_last": True,
    }
    if early_stopping is not None:
        training["early_stopping"] = early_stopping

    return {
        "data": {
            "train_csv": str(csv_path),
            "val_csv": str(csv_path),
            "test_balanced_csv": str(csv_path),
            "test_natural_csv": str(csv_path),
            "image_root": str(image_root),
            "target_columns": {"forest": "s_forest", "ocean": "s_ocean", "beach": "s_beach"},
        },
        "image": {"width": 512, "height": 320, "normalize": "imagenet"},
        "model": {
            "name": "scratch_resnet_cnn",
            "input_channels": 3,
            "output_dim": 3,
            "dropout": 0.1,
            "activation": "silu",
        },
        "loss": {
            "name": "smooth_l1",
            "beta": 0.05,
            "weights": {"forest": 1, "ocean": 1, "beach": 1},
        },
        "training": training,
        "output": {"run_dir": str(run_dir)},
        "evaluation": {
            "eps": 1e-6,
            "total_loss_weights": {"forest": 0.4, "ocean": 0.4, "beach": 0.2},
            "accept_rates": [0.10],
            "true_good_rate": 0.10,
            "use_s_total_if_available": True,
            "evaluate_test_after_training": evaluate_test_after_training,
        },
    }


def _loss_metrics(loss: float) -> dict[str, float]:
    return {key: loss for key in LOSS_KEYS}


def test_early_stopping_config_defaults_disabled():
    cfg = early_stopping_config_from_training({"training": {}})
    assert cfg["enabled"] is False
    assert cfg["patience"] == 15
    assert cfg["min_delta"] == 0.0001
    assert cfg["metric"] == "val_loss"


def test_early_stopping_config_unsupported_metric_raises():
    with pytest.raises(ValueError, match="Unsupported early_stopping.metric"):
        early_stopping_config_from_training(
            {"training": {"early_stopping": {"enabled": True, "metric": "quality/spearman"}}}
        )


def test_checkpoint_val_loss_improved():
    assert checkpoint_val_loss_improved(0.310868, 0.310942)
    assert not checkpoint_val_loss_improved(0.310942, 0.310942)
    assert checkpoint_val_loss_improved(0.99, 1.0)


def test_early_stop_val_loss_improved_with_min_delta():
    assert early_stop_val_loss_improved(0.5, 1.0, 0.1)
    assert not early_stop_val_loss_improved(0.95, 1.0, 0.1)
    assert not early_stop_val_loss_improved(0.91, 1.0, 0.1)
    # bs128 scenario: real improvement but below min_delta
    assert not early_stop_val_loss_improved(0.310868, 0.310942, 0.0001)


def test_restore_early_stopping_state_defaults_when_missing():
    state = restore_early_stopping_state({}, 0.42, enabled=True)
    assert state == {"bad_epochs": 0, "best_metric": 0.42}


def test_restore_early_stopping_state_ignored_when_disabled():
    state = restore_early_stopping_state(
        {"early_stopping_state": {"bad_epochs": 5, "best_metric": 0.1}},
        0.42,
        enabled=False,
    )
    assert state == {"bad_epochs": 0, "best_metric": 0.42}


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_early_stopping_disabled_runs_all_epochs(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    mock_val_epoch.return_value = {
        "loss": _loss_metrics(1.0),
        "metrics": {"quality/spearman": 0.1, "quality/target_source": "s_total"},
    }

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=3,
        early_stopping={"enabled": False},
        evaluate_test_after_training=False,
    )
    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    assert result["early_stopped"] is False
    assert len(result["history"]) == 3
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_early_stopping_stops_after_patience(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    val_losses = [1.0, 0.5, 0.49, 0.49]
    mock_val_epoch.side_effect = [
        {"loss": _loss_metrics(loss), "metrics": {"quality/spearman": 0.1}}
        for loss in val_losses
    ]

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=10,
        early_stopping={
            "enabled": True,
            "patience": 2,
            "min_delta": 0.01,
            "metric": "val_loss",
        },
        evaluate_test_after_training=False,
    )
    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    assert result["early_stopped"] is True
    assert result["stop_epoch"] == 4
    assert len(result["history"]) == 4
    assert result["history"][-1]["early_stopped"] is True
    assert (run_dir / "train_history.json").is_file()
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_early_stopping_runs_test_evaluation_after_stop(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    mock_val_epoch.return_value = {
        "loss": _loss_metrics(1.0),
        "metrics": {"quality/spearman": 0.1, "quality/target_source": "s_total"},
    }
    mock_run_test.return_value = {
        "test_balanced": {"quality/spearman": 0.2},
        "test_natural": {"quality/spearman": 0.3},
    }

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=5,
        early_stopping={
            "enabled": True,
            "patience": 1,
            "min_delta": 999.0,
            "metric": "val_loss",
        },
        evaluate_test_after_training=True,
    )
    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    assert result["early_stopped"] is True
    assert len(result["history"]) == 2
    mock_run_test.assert_called_once()
    assert "test" in result


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_checkpoint_saved_when_improvement_below_min_delta(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    """Real val_loss improvement below min_delta still updates best.pt."""
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    val_losses = [0.310942, 0.310868]
    mock_val_epoch.side_effect = [
        {"loss": _loss_metrics(loss), "metrics": {"quality/spearman": 0.1}}
        for loss in val_losses
    ]

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=2,
        early_stopping={
            "enabled": True,
            "patience": 5,
            "min_delta": 0.0001,
            "metric": "val_loss",
        },
        evaluate_test_after_training=False,
    )
    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    best = torch.load(run_dir / "best.pt", weights_only=False)
    assert best["epoch"] == 2
    assert best["best_val_loss"] == 0.310868

    history = result["history"]
    assert history[0]["checkpoint_improved"] is True
    assert history[0]["early_stop_improved"] is True
    assert history[0]["early_stopping_bad_epochs"] == 0
    assert history[1]["checkpoint_improved"] is True
    assert history[1]["early_stop_improved"] is False
    assert history[1]["early_stopping_bad_epochs"] == 1
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_early_stop_resets_when_improvement_meets_min_delta(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    mock_train_epoch.return_value = _loss_metrics(1.0)
    val_losses = [1.0, 0.5]
    mock_val_epoch.side_effect = [
        {"loss": _loss_metrics(loss), "metrics": {"quality/spearman": 0.1}}
        for loss in val_losses
    ]

    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=2,
        early_stopping={
            "enabled": True,
            "patience": 5,
            "min_delta": 0.01,
            "metric": "val_loss",
        },
        evaluate_test_after_training=False,
    )
    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    assert result["history"][1]["checkpoint_improved"] is True
    assert result["history"][1]["early_stop_improved"] is True
    assert result["history"][1]["early_stopping_bad_epochs"] == 0
    best = torch.load(run_dir / "best.pt", weights_only=False)
    assert best["best_val_loss"] == 0.5
    last = torch.load(run_dir / "last.pt", weights_only=False)
    assert last["early_stopping_state"]["best_metric"] == 0.5
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_resume_restores_early_stopping_best_metric(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    mock_val_epoch.return_value = {
        "loss": _loss_metrics(1.0),
        "metrics": {"quality/spearman": 0.1},
    }

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=1,
        early_stopping={
            "enabled": True,
            "patience": 5,
            "min_delta": 0.01,
            "metric": "val_loss",
        },
        evaluate_test_after_training=False,
    )
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    payload = torch.load(run_dir / "last.pt", weights_only=False)
    payload["best_val_loss"] = 0.42
    payload["early_stopping_state"] = {"bad_epochs": 2, "best_metric": 0.55}
    torch.save(payload, run_dir / "last.pt")

    config["training"]["epochs"] = 2
    train(
        config,
        resume_path=run_dir / "last.pt",
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    resumed = torch.load(run_dir / "last.pt", weights_only=False)
    assert resumed["best_val_loss"] == 0.42
    assert resumed["early_stopping_state"]["bad_epochs"] == 3
    assert resumed["early_stopping_state"]["best_metric"] == 0.55
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_resume_restores_early_stopping_bad_epochs(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    mock_val_epoch.return_value = {
        "loss": _loss_metrics(1.0),
        "metrics": {"quality/spearman": 0.1},
    }

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=1,
        early_stopping={
            "enabled": True,
            "patience": 5,
            "min_delta": 0.01,
            "metric": "val_loss",
        },
        evaluate_test_after_training=False,
    )
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    payload = torch.load(run_dir / "last.pt", weights_only=False)
    payload["early_stopping_state"] = {"bad_epochs": 3, "best_metric": payload["best_val_loss"]}
    torch.save(payload, run_dir / "last.pt")

    config["training"]["epochs"] = 2
    train(
        config,
        resume_path=run_dir / "last.pt",
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    resumed = torch.load(run_dir / "last.pt", weights_only=False)
    assert resumed["early_stopping_state"]["bad_epochs"] == 4
    mock_run_test.assert_not_called()
