"""PyTorch Dataset for mask+resize World Preview images and proxy scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.image_preprocess import build_image_transform
from seed_preview_cv.common.paths import PROJECT_ROOT

TARGET_ORDER = ("forest", "ocean", "beach")

DEFAULT_TARGET_COLUMNS = {
    "forest": "s_forest",
    "ocean": "s_ocean",
    "beach": "s_beach",
}

DEFAULT_TOTAL_COLUMN = "s_total"


def _column_variants(name: str) -> list[str]:
    variants = [name]
    if len(name) > 1 and name[0] in ("s", "S"):
        swapped = name[0].swapcase() + name[1:]
        if swapped not in variants:
            variants.append(swapped)
    return variants


def resolve_dataframe_column(df: pd.DataFrame, configured: str) -> str:
    """Return the actual column name, trying s_/S_ case variants."""
    for candidate in _column_variants(configured):
        if candidate in df.columns:
            return candidate
    tried = ", ".join(_column_variants(configured))
    raise KeyError(f"Column '{configured}' not found in CSV (tried: {tried})")


def resolve_image_path(image_path: str | Path, image_root: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (image_root / path).resolve()


class SeedPreviewDataset(Dataset):
    """Load World Preview images and forest/ocean/beach proxy scores from a CSV."""

    def __init__(
        self,
        csv_path: Path | str,
        image_root: Path | str = ".",
        image_path_column: str = "image_path",
        seed_column: str = "seed",
        target_columns: dict[str, str] | None = None,
        image_width: int = 512,
        image_height: int = 320,
        normalize: str = "imagenet",
    ) -> None:
        csv_path = Path(csv_path)
        self.csv_path = csv_path.resolve()
        self.image_root = Path(image_root).resolve()
        self.image_path_column = image_path_column
        self.seed_column = seed_column

        self.df = pd.read_csv(self.csv_path)
        if self.image_path_column not in self.df.columns:
            raise KeyError(
                f"Column '{self.image_path_column}' not found in {self.csv_path}"
            )
        if self.seed_column not in self.df.columns:
            raise KeyError(f"Column '{self.seed_column}' not found in {self.csv_path}")

        configured = target_columns or DEFAULT_TARGET_COLUMNS
        self.target_columns: dict[str, str] = {}
        for key in TARGET_ORDER:
            if key not in configured:
                raise KeyError(f"target_columns missing key '{key}'")
            self.target_columns[key] = resolve_dataframe_column(self.df, configured[key])

        self.total_column: str | None = None
        try:
            self.total_column = resolve_dataframe_column(self.df, DEFAULT_TOTAL_COLUMN)
        except KeyError:
            pass

        self.image_width = image_width
        self.image_height = image_height
        self.transform = build_image_transform(
            width=image_width,
            height=image_height,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = str(row[self.image_path_column])
        full_path = resolve_image_path(image_path, self.image_root)
        if not full_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {full_path} (csv image_path={image_path})"
            )

        with Image.open(full_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)

        target_tensor = torch.tensor(
            [
                float(row[self.target_columns["forest"]]),
                float(row[self.target_columns["ocean"]]),
                float(row[self.target_columns["beach"]]),
            ],
            dtype=torch.float32,
        )

        sample: dict[str, Any] = {
            "image": image_tensor,
            "target": target_tensor,
            "image_path": image_path,
            "seed": int(row[self.seed_column]),
        }
        if self.total_column is not None:
            sample["target_total"] = torch.tensor(
                float(row[self.total_column]),
                dtype=torch.float32,
            )
        if "split" in self.df.columns:
            sample["split"] = str(row["split"])
        if "source" in self.df.columns:
            sample["source"] = str(row["source"])
        return sample


def dataset_from_training_config(
    config: dict[str, Any],
    split: str,
) -> SeedPreviewDataset:
    """Build a dataset from configs/training.yaml-style nested dict."""
    data_cfg = config.get("data", {})
    image_cfg = config.get("image", {})

    split_key = {
        "train": "train_csv",
        "val": "val_csv",
        "test_balanced": "test_balanced_csv",
        "test_natural": "test_natural_csv",
    }.get(split, split)

    csv_key = split_key if split_key.endswith("_csv") else f"{split_key}_csv"
    csv_path = data_cfg.get(csv_key)
    if csv_path is None:
        raise KeyError(f"Config data.{csv_key} is required for split '{split}'")

    image_root = resolve_path(data_cfg.get("image_root", "."))
    target_columns = data_cfg.get("target_columns", DEFAULT_TARGET_COLUMNS)
    augmentation_enabled = image_cfg.get("augmentation", {}).get("enabled", False)
    if augmentation_enabled:
        raise ValueError("Data augmentation is disabled for Stage 1; set augmentation.enabled=false")

    return SeedPreviewDataset(
        csv_path=resolve_path(csv_path),
        image_root=image_root,
        image_path_column=data_cfg.get("image_path_column", "image_path"),
        seed_column=data_cfg.get("seed_column", "seed"),
        target_columns=target_columns,
        image_width=int(image_cfg.get("width", 512)),
        image_height=int(image_cfg.get("height", 320)),
        normalize=image_cfg.get("normalize", "imagenet"),
    )


def sanity_check_dataset(
    csv_path: str | Path,
    image_root: str | Path = ".",
    num_samples: int = 3,
    **dataset_kwargs: Any,
) -> None:
    """Load samples and verify tensor shapes and target ranges."""
    dataset = SeedPreviewDataset(
        csv_path=csv_path,
        image_root=image_root,
        **dataset_kwargs,
    )

    print(f"dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {csv_path}")

    samples_to_check = min(num_samples, len(dataset))
    for i in range(samples_to_check):
        sample = dataset[i]
        image = sample["image"]
        target = sample["target"]

        print(f"--- sample {i} ---")
        print(f"image_path: {sample['image_path']}")
        print(f"seed: {sample['seed']}")
        print(f"image tensor shape: {list(image.shape)}")
        print(f"target tensor shape: {list(target.shape)}")
        print(f"target tensor values: {target.tolist()}")

        expected_image_shape = [3, dataset.image_height, dataset.image_width]
        if list(image.shape) != expected_image_shape:
            raise ValueError(
                f"image shape {list(image.shape)} != expected {expected_image_shape}"
            )
        if list(target.shape) != [3]:
            raise ValueError(f"target shape {list(target.shape)} != expected [3]")

        target_min = float(target.min())
        target_max = float(target.max())
        print(f"target min: {target_min}")
        print(f"target max: {target_max}")

        if target_min < 0.0 or target_max > 1.0:
            raise ValueError(
                f"target values out of [0, 1]: min={target_min}, max={target_max}"
            )

    print("sanity check passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity check SeedPreviewDataset")
    parser.add_argument("--csv", type=Path, default=None, help="Path to split CSV")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root for relative image_path values (default: project root)",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional configs/training.yaml; --csv overrides data.train_csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_kwargs: dict[str, Any] = {}
    image_root = resolve_path(args.image_root)

    if args.config is not None:
        config = load_yaml(resolve_path(args.config))
        data_cfg = config.get("data", {})
        image_cfg = config.get("image", {})
        dataset_kwargs = {
            "image_path_column": data_cfg.get("image_path_column", "image_path"),
            "seed_column": data_cfg.get("seed_column", "seed"),
            "target_columns": data_cfg.get("target_columns", DEFAULT_TARGET_COLUMNS),
            "image_width": int(image_cfg.get("width", 512)),
            "image_height": int(image_cfg.get("height", 320)),
            "normalize": image_cfg.get("normalize", "imagenet"),
        }
        image_root = resolve_path(data_cfg.get("image_root", "."))

    if args.csv is not None:
        csv_path = resolve_path(args.csv)
    elif args.config is not None:
        train_csv = load_yaml(resolve_path(args.config)).get("data", {}).get("train_csv")
        if not train_csv:
            raise ValueError("Provide --csv or set data.train_csv in --config")
        csv_path = resolve_path(train_csv)
    else:
        raise ValueError("Provide --csv or --config")

    sanity_check_dataset(
        csv_path=csv_path,
        image_root=image_root,
        num_samples=args.num_samples,
        **dataset_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
