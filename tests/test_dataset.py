"""Tests for SeedPreviewDataset."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from seed_preview_cv.training.dataset import (
    DEFAULT_TARGET_COLUMNS,
    SeedPreviewDataset,
    resolve_dataframe_column,
    sanity_check_dataset,
)


def test_resolve_dataframe_column_case_variants():
    df = pd.DataFrame({"S_forest": [1.0], "s_ocean": [0.5], "S_beach": [0.0]})
    assert resolve_dataframe_column(df, "s_forest") == "S_forest"
    assert resolve_dataframe_column(df, "s_ocean") == "s_ocean"
    assert resolve_dataframe_column(df, "s_beach") == "S_beach"


def test_seed_preview_dataset_shapes_and_targets(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    Image.new("RGB", (512, 320), color=(100, 150, 200)).save(image_path)

    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "image_path": [str(image_path)],
            "seed": [-123],
            "s_forest": [0.8],
            "s_ocean": [0.4],
            "s_beach": [0.2],
        }
    ).to_csv(csv_path, index=False)

    dataset = SeedPreviewDataset(
        csv_path=csv_path,
        image_root=tmp_path,
        target_columns=DEFAULT_TARGET_COLUMNS,
    )
    sample = dataset[0]

    assert sample["image"].shape == torch.Size([3, 320, 512])
    assert sample["target"].shape == torch.Size([3])
    assert torch.allclose(sample["target"], torch.tensor([0.8, 0.4, 0.2]))
    assert sample["seed"] == -123
    assert "target_total" not in sample


def test_seed_preview_dataset_returns_target_total_when_present(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    Image.new("RGB", (512, 320), color=(100, 150, 200)).save(image_path)

    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "image_path": [str(image_path)],
            "seed": [-123],
            "s_forest": [0.8],
            "s_ocean": [0.4],
            "s_beach": [0.2],
            "s_total": [0.55],
        }
    ).to_csv(csv_path, index=False)

    dataset = SeedPreviewDataset(csv_path=csv_path, image_root=tmp_path)
    sample = dataset[0]
    assert "target_total" in sample
    assert sample["target_total"].dtype == torch.float32
    assert float(sample["target_total"]) == pytest.approx(0.55)
    assert sample["target"].shape == torch.Size([3])


def test_seed_preview_dataset_missing_image_raises(tmp_path: Path):
    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "image_path": ["images/missing.png"],
            "seed": [1],
            "s_forest": [1.0],
            "s_ocean": [0.0],
            "s_beach": [0.0],
        }
    ).to_csv(csv_path, index=False)

    dataset = SeedPreviewDataset(csv_path=csv_path, image_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="missing.png"):
        dataset[0]


def test_sanity_check_dataset(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(3):
        path = image_dir / f"img_{i}.png"
        Image.new("RGB", (400, 250), color=(i * 10, 50, 80)).save(path)
        rows.append(
            {
                "image_path": str(path),
                "seed": i,
                "s_forest": 0.5 + i * 0.1,
                "s_ocean": 0.3,
                "s_beach": 0.1,
            }
        )
    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    sanity_check_dataset(csv_path=csv_path, image_root=tmp_path, num_samples=2)
