"""Tests for split CSV generation."""

from pathlib import Path

import pandas as pd
import pytest

from seed_preview_cv.training.make_splits import make_splits, merge_index_and_labels


def _write_balanced_fixtures(
    tmp_path: Path,
    n: int = 100,
    *,
    with_total: bool = False,
) -> tuple[Path, Path]:
    index_path = tmp_path / "balanced_index.csv"
    labels_path = tmp_path / "balanced_labels.csv"
    seeds = list(range(n))
    pd.DataFrame(
        {
            "seed": seeds,
            "image_path": [f"images/{s}.png" for s in seeds],
        }
    ).to_csv(index_path, index=False)
    label_rows: dict[str, list] = {
        "seed": seeds,
        "s_forest": [0.5] * n,
        "s_ocean": [0.4] * n,
        "s_beach": [0.3] * n,
    }
    if with_total:
        label_rows["s_total"] = [0.2] * n
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    return index_path, labels_path


def test_merge_index_and_labels_inner_join(tmp_path: Path):
    index_path, labels_path = _write_balanced_fixtures(tmp_path, n=5)
    merged = merge_index_and_labels(index_path, labels_path, "balanced")
    assert len(merged) == 5
    assert set(merged.columns) >= {"image_path", "seed", "s_forest", "source"}


def test_merge_index_and_labels_includes_s_total(tmp_path: Path):
    index_path, labels_path = _write_balanced_fixtures(tmp_path, n=5, with_total=True)
    merged = merge_index_and_labels(index_path, labels_path, "balanced")
    assert "s_total" in merged.columns


def test_make_splits_preserves_s_total_column(tmp_path: Path):
    from PIL import Image

    n_balanced = 20
    index_path, labels_path = _write_balanced_fixtures(tmp_path, n=n_balanced, with_total=True)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for s in range(n_balanced):
        Image.new("RGB", (512, 320), color=(10, 20, 30)).save(image_dir / f"{s}.png")

    natural_index = tmp_path / "natural_index.csv"
    natural_labels = tmp_path / "natural_labels.csv"
    natural_seeds = list(range(100, 110))
    for s in natural_seeds:
        Image.new("RGB", (512, 320), color=(40, 50, 60)).save(image_dir / f"{s}.png")
    pd.DataFrame(
        {
            "seed": natural_seeds,
            "image_path": [f"images/{s}.png" for s in natural_seeds],
        }
    ).to_csv(natural_index, index=False)
    pd.DataFrame(
        {
            "seed": natural_seeds,
            "s_forest": [0.6] * len(natural_seeds),
            "s_ocean": [0.5] * len(natural_seeds),
            "s_beach": [0.4] * len(natural_seeds),
            "s_total": [0.3] * len(natural_seeds),
        }
    ).to_csv(natural_labels, index=False)

    output_dir = tmp_path / "splits"
    make_splits(
        balanced_index=index_path,
        balanced_labels=labels_path,
        natural_index=natural_index,
        natural_labels=natural_labels,
        output_dir=output_dir,
        train_size=12,
        val_size=4,
        test_size=4,
        natural_test_size=10,
        seed=42,
        label_mode="directional",
        image_root=tmp_path,
        run_sanity=False,
    )
    train_df = pd.read_csv(output_dir / "train_balanced.csv")
    assert "s_total" in train_df.columns


def test_make_splits_small_fixture(tmp_path: Path):
    from PIL import Image

    n_balanced = 20
    index_path, labels_path = _write_balanced_fixtures(tmp_path, n=n_balanced)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for s in range(n_balanced):
        Image.new("RGB", (512, 320), color=(10, 20, 30)).save(image_dir / f"{s}.png")

    natural_index = tmp_path / "natural_index.csv"
    natural_labels = tmp_path / "natural_labels.csv"
    natural_seeds = list(range(100, 110))
    for s in natural_seeds:
        Image.new("RGB", (512, 320), color=(40, 50, 60)).save(image_dir / f"{s}.png")
    pd.DataFrame(
        {
            "seed": natural_seeds,
            "image_path": [f"images/{s}.png" for s in natural_seeds],
        }
    ).to_csv(natural_index, index=False)
    pd.DataFrame(
        {
            "seed": natural_seeds,
            "s_forest": [0.6] * len(natural_seeds),
            "s_ocean": [0.5] * len(natural_seeds),
            "s_beach": [0.4] * len(natural_seeds),
        }
    ).to_csv(natural_labels, index=False)

    output_dir = tmp_path / "splits"
    manifest = make_splits(
        balanced_index=index_path,
        balanced_labels=labels_path,
        natural_index=natural_index,
        natural_labels=natural_labels,
        output_dir=output_dir,
        train_size=12,
        val_size=4,
        test_size=4,
        natural_test_size=10,
        seed=42,
        label_mode="directional",
        image_root=tmp_path,
        run_sanity=True,
        sanity_num_samples=2,
    )

    assert manifest["splits"][0]["row_count"] == 12
    assert (output_dir / "train_balanced.csv").is_file()
    assert (output_dir / "test_natural.csv").is_file()
    train_df = pd.read_csv(output_dir / "train_balanced.csv")
    val_df = pd.read_csv(output_dir / "val_balanced.csv")
    test_df = pd.read_csv(output_dir / "test_balanced.csv")
    natural_df = pd.read_csv(output_dir / "test_natural.csv")

    train_seeds = set(train_df["seed"])
    val_seeds = set(val_df["seed"])
    test_seeds = set(test_df["seed"])
    assert not (train_seeds & val_seeds)
    assert not (train_seeds & test_seeds)
    assert not (train_seeds & set(natural_df["seed"]))
