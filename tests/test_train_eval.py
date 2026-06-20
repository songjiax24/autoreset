"""Lightweight tests for post-training evaluation helpers."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from seed_preview_cv.training.dataset import SeedPreviewDataset
from seed_preview_cv.training.metrics import (
    PREDICTION_CSV_COLUMNS,
    compute_all_metrics,
    evaluation_config_from_training,
    write_predictions_csv,
)
from seed_preview_cv.training.train import (
    LOSS_KEYS,
    predict_dataset,
    train,
    validate_training_config,
)


class _ConstantModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.full((batch, 3), 0.5, device=x.device, dtype=x.dtype)


def _make_tiny_csv(tmp_path: Path, with_total: bool = True) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for i in range(4):
        image_path = image_dir / f"sample_{i}.png"
        Image.new("RGB", (512, 320), color=(i * 20, 50, 80)).save(image_path)
        paths.append(str(image_path))

    rows: dict[str, list] = {
        "image_path": paths,
        "seed": [-1, -2, -3, -4],
        "s_forest": [0.8, 0.6, 0.4, 0.2],
        "s_ocean": [0.5, 0.4, 0.3, 0.2],
        "s_beach": [0.3, 0.2, 0.1, 0.05],
        "split": ["test_balanced", "test_balanced", "test_balanced", "test_balanced"],
        "source": ["balanced", "balanced", "balanced", "balanced"],
    }
    if with_total:
        rows["s_total"] = [0.55, 0.45, 0.35, 0.25]

    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, tmp_path


def test_predict_dataset_shapes_and_metadata(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    dataset = SeedPreviewDataset(csv_path=csv_path, image_root=image_root)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    device = torch.device("cpu")
    model = _ConstantModel().to(device)

    result = predict_dataset(model, loader, device, use_amp=False, show_progress=False)

    assert result["targets"].shape == (4, 3)
    assert result["predictions"].shape == (4, 3)
    assert result["target_total"] is not None
    assert result["target_total"].shape == (4,)
    assert len(result["image_paths"]) == 4
    assert len(result["seeds"]) == 4
    assert result["splits"] == ["test_balanced"] * 4
    assert result["sources"] == ["balanced"] * 4


def test_predict_dataset_respects_use_s_total_if_available(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    dataset = SeedPreviewDataset(csv_path=csv_path, image_root=image_root)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = _ConstantModel()

    with_total = predict_dataset(
        model,
        loader,
        torch.device("cpu"),
        use_s_total_if_available=True,
        show_progress=False,
    )
    without_total = predict_dataset(
        model,
        loader,
        torch.device("cpu"),
        use_s_total_if_available=False,
        show_progress=False,
    )

    assert with_total["target_total"] is not None
    assert without_total["target_total"] is None


def test_predict_dataset_target_source_matches_validation_logic(tmp_path: Path):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    dataset = SeedPreviewDataset(csv_path=csv_path, image_root=image_root)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = _ConstantModel()

    enabled = predict_dataset(
        model,
        loader,
        torch.device("cpu"),
        use_s_total_if_available=True,
        show_progress=False,
    )
    metrics_enabled = compute_all_metrics(
        enabled["targets"],
        enabled["predictions"],
        y_true_total=enabled["target_total"],
    )
    assert metrics_enabled["quality/target_source"] == "s_total"

    disabled = predict_dataset(
        model,
        loader,
        torch.device("cpu"),
        use_s_total_if_available=False,
        show_progress=False,
    )
    metrics_disabled = compute_all_metrics(
        disabled["targets"],
        disabled["predictions"],
        y_true_total=disabled["target_total"],
    )
    assert metrics_disabled["quality/target_source"] == "computed_from_scores"


def test_validate_training_config_rejects_save_best_false_with_test_eval():
    config = {
        "training": {"save_best": False},
        "evaluation": {"evaluate_test_after_training": True},
    }
    with pytest.raises(ValueError, match="save_best=true"):
        validate_training_config(config)


def test_write_predictions_csv_columns_and_target_total(tmp_path: Path):
    out_path = tmp_path / "preds.csv"
    targets = [[0.8, 0.5, 0.3], [0.6, 0.4, 0.2]]
    predictions = [[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]
    target_total = [0.55, 0.45]
    true_quality = [0.55, 0.45]
    pred_quality = [0.5, 0.4]

    write_predictions_csv(
        out_path,
        image_paths=["a.png", "b.png"],
        seeds=[1, 2],
        splits=["test_balanced", "test_balanced"],
        sources=["balanced", "balanced"],
        targets=np.array(targets, dtype=float),
        predictions=np.array(predictions, dtype=float),
        target_total=np.array(target_total, dtype=float),
        true_quality=np.array(true_quality, dtype=float),
        pred_quality=np.array(pred_quality, dtype=float),
    )

    df = pd.read_csv(out_path)
    assert list(df.columns) == list(PREDICTION_CSV_COLUMNS)
    assert df["target_total"].tolist() == target_total
    assert df["true_quality"].tolist() == true_quality
    assert df["pred_quality"].tolist() == pred_quality


def test_evaluation_config_evaluate_flag():
    cfg = evaluation_config_from_training(
        {"evaluation": {"evaluate_test_after_training": False}}
    )
    assert cfg["evaluate_test_after_training"] is False


@patch("seed_preview_cv.training.train.run_test_evaluations")
@patch("seed_preview_cv.training.train.validate_one_epoch")
@patch("seed_preview_cv.training.train.train_one_epoch")
def test_train_skips_test_evaluation_when_disabled(
    mock_train_epoch,
    mock_val_epoch,
    mock_run_test,
    tmp_path: Path,
):
    csv_path, image_root = _make_tiny_csv(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    loss_metrics = {key: 1.0 for key in LOSS_KEYS}
    mock_train_epoch.return_value = loss_metrics
    mock_val_epoch.return_value = {
        "loss": loss_metrics,
        "metrics": {"quality/spearman": 0.1, "quality/target_source": "s_total"},
    }

    config = {
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
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "mixed_precision": False,
            "seed": 42,
            "save_best": True,
            "save_last": False,
        },
        "output": {"run_dir": str(run_dir)},
        "evaluation": {
            "eps": 1e-6,
            "total_loss_weights": {"forest": 0.4, "ocean": 0.4, "beach": 0.2},
            "accept_rates": [0.10],
            "true_good_rate": 0.10,
            "use_s_total_if_available": True,
            "evaluate_test_after_training": False,
        },
    }

    result = train(
        config,
        max_train_batches=1,
        max_val_batches=1,
        show_progress=False,
    )

    mock_run_test.assert_not_called()
    assert "test" not in result
