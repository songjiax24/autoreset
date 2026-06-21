"""Tests for ReduceLROnPlateau scheduler in training."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from seed_preview_cv.training.train import (
    LOSS_KEYS,
    build_scheduler_from_config,
    current_learning_rates,
    scheduler_config_from_training,
    step_scheduler,
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
    scheduler: dict | None = None,
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
    if scheduler is not None:
        training["scheduler"] = scheduler
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


def test_scheduler_config_defaults_disabled():
    cfg = scheduler_config_from_training({"training": {}})
    assert cfg["enabled"] is False


def test_scheduler_config_unsupported_name_raises():
    with pytest.raises(ValueError, match="Unsupported scheduler.name"):
        scheduler_config_from_training(
            {
                "training": {
                    "scheduler": {"enabled": True, "name": "cosine", "monitor": "val_loss"},
                }
            }
        )


def test_scheduler_config_unsupported_monitor_raises():
    with pytest.raises(ValueError, match="Unsupported scheduler.monitor"):
        scheduler_config_from_training(
            {
                "training": {
                    "scheduler": {
                        "enabled": True,
                        "name": "reduce_on_plateau",
                        "monitor": "train_loss",
                    },
                }
            }
        )


def test_build_scheduler_from_config_creates_reduce_on_plateau():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    config = {
        "training": {
            "scheduler": {
                "enabled": True,
                "name": "reduce_on_plateau",
                "monitor": "val_loss",
                "mode": "min",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1e-5,
                "threshold": 0.0001,
                "threshold_mode": "abs",
            }
        }
    }
    scheduler = build_scheduler_from_config(optimizer, config)
    assert scheduler is not None
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_step_scheduler_reduces_lr():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    config = {
        "training": {
            "scheduler": {
                "enabled": True,
                "name": "reduce_on_plateau",
                "monitor": "val_loss",
                "patience": 1,
                "factor": 0.5,
                "min_lr": 1e-5,
                "threshold": 999.0,
                "threshold_mode": "abs",
            }
        }
    }
    scheduler = build_scheduler_from_config(optimizer, config)
    assert scheduler is not None
    _, _, changed = step_scheduler(scheduler, optimizer, 1.0)
    assert not changed
    _, _, changed = step_scheduler(scheduler, optimizer, 1.0)
    assert not changed
    _, _, changed = step_scheduler(scheduler, optimizer, 1.0)
    assert changed
    assert current_learning_rates(optimizer)[0] == 5e-4


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_scheduler_disabled_compatible(
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
        epochs=2,
        scheduler={"enabled": False},
        early_stopping={"enabled": False},
        evaluate_test_after_training=False,
    )
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)
    payload = torch.load(run_dir / "last.pt", weights_only=False)
    assert "scheduler_state_dict" not in payload
    history = json.loads((run_dir / "train_history.json").read_text())
    assert "lr" in history[0]


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_scheduler_history_records_lr_and_checkpoint_state(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    mock_train_epoch.return_value = _loss_metrics(1.0)
    val_losses = [1.0, 1.0, 1.0]
    mock_val_epoch.side_effect = [
        {"loss": _loss_metrics(loss), "metrics": {"quality/spearman": 0.1}}
        for loss in val_losses
    ]

    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=3,
        scheduler={
            "enabled": True,
            "name": "reduce_on_plateau",
            "monitor": "val_loss",
            "patience": 1,
            "factor": 0.5,
            "min_lr": 1e-5,
            "threshold": 999.0,
            "threshold_mode": "abs",
        },
        early_stopping={"enabled": False},
        evaluate_test_after_training=False,
    )
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    history = json.loads((run_dir / "train_history.json").read_text())
    assert history[0]["train_lr"] == 0.001
    assert history[0]["next_lr"] == 0.001
    assert history[0]["lr"] == 0.001
    assert history[1]["train_lr"] == 0.001
    assert history[1]["next_lr"] == 0.001
    assert history[2]["train_lr"] == 0.001
    assert history[2]["next_lr"] == 0.0005

    payload = torch.load(run_dir / "last.pt", weights_only=False)
    assert "scheduler_state_dict" in payload


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_scheduler_resume_restores_state(
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

    scheduler_cfg = {
        "enabled": True,
        "name": "reduce_on_plateau",
        "monitor": "val_loss",
        "patience": 1,
        "factor": 0.5,
        "min_lr": 1e-5,
        "threshold": 999.0,
        "threshold_mode": "abs",
    }
    config = _base_config(
        tmp_path,
        csv_path,
        image_root,
        run_dir,
        epochs=1,
        scheduler=scheduler_cfg,
        early_stopping={"enabled": False},
        evaluate_test_after_training=False,
    )
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    config["training"]["epochs"] = 2
    train(
        config,
        resume_path=run_dir / "last.pt",
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    payload = torch.load(run_dir / "last.pt", weights_only=False)
    assert payload["scheduler_state_dict"] is not None
    mock_run_test.assert_not_called()


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_early_stopping_with_scheduler_runs_test_eval(
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
        scheduler={
            "enabled": True,
            "name": "reduce_on_plateau",
            "monitor": "val_loss",
            "patience": 1,
            "factor": 0.5,
            "min_lr": 1e-5,
            "threshold": 999.0,
            "threshold_mode": "abs",
        },
        early_stopping={
            "enabled": True,
            "patience": 2,
            "min_delta": 999.0,
            "metric": "val_loss",
        },
        evaluate_test_after_training=True,
    )
    result = train(config, max_train_batches=1, max_val_batches=1, show_progress=False)
    assert result["early_stopped"] is True
    mock_run_test.assert_called_once()
