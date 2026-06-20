"""Tests for training resume support."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from seed_preview_cv.training.train import (
    load_resume_checkpoint,
    load_train_history,
    train,
    validate_resume_config,
)


def _make_tiny_csv(tmp_path: Path) -> tuple[Path, Path]:
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
    epochs: int = 1,
    evaluate_test_after_training: bool = False,
) -> dict:
    return {
        "data": {
            "train_csv": str(csv_path),
            "val_csv": str(csv_path),
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
        "loss": {"name": "smooth_l1", "beta": 0.05, "weights": {"forest": 1, "ocean": 1, "beach": 1}},
        "training": {
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
        },
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


@patch("seed_preview_cv.training.train.run_test_evaluations")
def test_resume_continues_from_next_epoch(mock_run_test, tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    config = _base_config(tmp_path, csv_path, image_root, run_dir, epochs=1)
    train(
        config,
        max_train_batches=2,
        max_val_batches=2,
        show_progress=False,
    )

    ckpt1 = torch.load(run_dir / "last.pt", weights_only=False)
    assert ckpt1["epoch"] == 1
    assert "optimizer_state_dict" in ckpt1
    assert "scaler_state_dict" in ckpt1

    config["training"]["epochs"] = 2
    train(
        config,
        resume_path=run_dir / "last.pt",
        max_train_batches=2,
        max_val_batches=2,
        show_progress=False,
    )

    history = json.loads((run_dir / "train_history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == [1, 2]

    ckpt2 = torch.load(run_dir / "last.pt", weights_only=False)
    assert ckpt2["epoch"] == 2
    assert ckpt2["best_val_loss"] == ckpt1["best_val_loss"] or ckpt2["best_val_loss"] <= ckpt1["best_val_loss"]
    mock_run_test.assert_not_called()


def test_load_resume_checkpoint_restores_optimizer_and_scaler(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _base_config(tmp_path, csv_path, image_root, run_dir, epochs=1)
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    payload = torch.load(run_dir / "last.pt", weights_only=False)
    device = torch.device("cpu")
    model, optimizer, scaler, resume_epoch, best_val_loss = load_resume_checkpoint(
        run_dir / "last.pt",
        config,
        device,
        use_amp=False,
    )
    assert resume_epoch == 1
    assert best_val_loss == payload["best_val_loss"]
    assert len(optimizer.state_dict()["state"]) > 0
    assert scaler.state_dict() is not None


def test_load_train_history_appends_existing(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history_path = run_dir / "train_history.json"
    history_path.write_text(json.dumps([{"epoch": 1, "best_val_loss": 0.5}]), encoding="utf-8")
    loaded = load_train_history(run_dir, 1)
    assert loaded == [{"epoch": 1, "best_val_loss": 0.5}]


def test_validate_resume_config_raises_on_model_structure_mismatch():
    ckpt_cfg = {
        "model": {
            "name": "scratch_resnet_cnn",
            "input_channels": 3,
            "output_dim": 3,
            "dropout": 0.1,
        },
        "image": {"width": 512, "height": 320},
    }
    current_cfg = {
        "model": {
            "name": "scratch_resnet_cnn",
            "input_channels": 3,
            "output_dim": 3,
            "dropout": 0.5,
        },
        "image": {"width": 512, "height": 320},
    }
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_config(ckpt_cfg, current_cfg)


@patch("seed_preview_cv.training.train.run_test_evaluations")
def test_resume_skips_training_when_epochs_complete(mock_run_test, tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _base_config(tmp_path, csv_path, image_root, run_dir, epochs=1)
    train(config, max_train_batches=1, max_val_batches=1, show_progress=False)

    result = train(
        config,
        resume_path=run_dir / "last.pt",
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )
    assert result["training_skipped"] is True
    history = json.loads((run_dir / "train_history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    mock_run_test.assert_not_called()
