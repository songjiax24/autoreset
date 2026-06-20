"""Batch inference for ScratchResNetCNN on World Preview images."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.image_preprocess import (
    preprocess_prepared_image,
    preprocess_raw_world_preview,
)
from seed_preview_cv.training.dataset import (
    DEFAULT_TARGET_COLUMNS,
    DEFAULT_TOTAL_COLUMN,
    resolve_dataframe_column,
    resolve_image_path,
)
from seed_preview_cv.training.metrics import compute_quality, evaluation_config_from_training
from seed_preview_cv.training.train import build_model_from_config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

INFERENCE_OUTPUT_COLUMNS = (
    "image_path",
    "seed",
    "split",
    "source",
    "preprocess_mode",
    "target_forest",
    "target_ocean",
    "target_beach",
    "target_total",
    "pred_forest",
    "pred_ocean",
    "pred_beach",
    "pred_quality",
    "accept_decision",
)


def validate_input_args(input_csv: Path | None, image_dir: Path | None) -> None:
    if input_csv is not None and image_dir is not None:
        raise ValueError("Provide only one of --input-csv or --image-dir, not both")
    if input_csv is None and image_dir is None:
        raise ValueError("One of --input-csv or --image-dir is required")


def load_checkpoint(
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def resolve_inference_config(
    checkpoint: dict[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config")
    if config_path is not None:
        if checkpoint_config is not None:
            warnings.warn(
                "Overriding checkpoint config with --config; model architecture must match checkpoint",
                stacklevel=2,
            )
        return load_yaml(resolve_path(config_path))

    if checkpoint_config is None:
        raise ValueError(
            "Checkpoint has no embedded config; provide --config configs/training.yaml"
        )

    return checkpoint_config


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing model_state_dict")

    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def resolve_image_root(image_root: Path | None, config: dict[str, Any]) -> Path:
    if image_root is not None:
        return resolve_path(image_root)
    return resolve_path(config.get("data", {}).get("image_root", "."))


def _optional_column(df: pd.DataFrame, name: str) -> str | None:
    try:
        return resolve_dataframe_column(df, name)
    except KeyError:
        return None


def records_from_csv(csv_path: Path, image_root: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise KeyError(f"input CSV must contain image_path: {csv_path}")

    forest_col = _optional_column(df, DEFAULT_TARGET_COLUMNS["forest"])
    ocean_col = _optional_column(df, DEFAULT_TARGET_COLUMNS["ocean"])
    beach_col = _optional_column(df, DEFAULT_TARGET_COLUMNS["beach"])
    total_col = _optional_column(df, DEFAULT_TOTAL_COLUMN)
    has_targets = all(col is not None for col in (forest_col, ocean_col, beach_col))

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        record: dict[str, Any] = {
            "image_path": str(row["image_path"]),
            "seed": int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else None,
            "split": str(row["split"]) if "split" in row and pd.notna(row["split"]) else None,
            "source": str(row["source"]) if "source" in row and pd.notna(row["source"]) else None,
        }
        if has_targets:
            assert forest_col is not None and ocean_col is not None and beach_col is not None
            record["target_forest"] = float(row[forest_col])
            record["target_ocean"] = float(row[ocean_col])
            record["target_beach"] = float(row[beach_col])
        if total_col is not None and total_col in row and pd.notna(row[total_col]):
            record["target_total"] = float(row[total_col])
        records.append(record)

    for record in records:
        full_path = resolve_image_path(record["image_path"], image_root)
        if not full_path.is_file():
            raise FileNotFoundError(f"Image not found: {full_path}")
    return records


def records_from_image_dir(image_dir: Path) -> list[dict[str, Any]]:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image-dir is not a directory: {image_dir}")

    records: list[dict[str, Any]] = []
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir} (expected {sorted(IMAGE_EXTENSIONS)})")

    for path in paths:
        records.append(
            {
                "image_path": str(path.resolve()),
                "seed": None,
                "split": None,
                "source": None,
            }
        )
    return records


class InferenceImageDataset(Dataset):
    """Load images for inference with prepared or raw preprocessing."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        image_root: Path,
        preprocess_mode: str,
        image_width: int = 512,
        image_height: int = 320,
        normalize: str = "imagenet",
    ) -> None:
        if preprocess_mode not in {"prepared", "raw"}:
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode!r}")
        self.records = records
        self.image_root = image_root
        self.preprocess_mode = preprocess_mode
        self.image_width = image_width
        self.image_height = image_height
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        full_path = resolve_image_path(record["image_path"], self.image_root)
        with Image.open(full_path) as img:
            if self.preprocess_mode == "prepared":
                image_tensor = preprocess_prepared_image(
                    img,
                    width=self.image_width,
                    height=self.image_height,
                    normalize=self.normalize,
                    warn_on_resize=True,
                    image_path=record["image_path"],
                )
            else:
                image_tensor = preprocess_raw_world_preview(
                    img,
                    width=self.image_width,
                    height=self.image_height,
                    normalize=self.normalize,
                )

        sample: dict[str, Any] = {
            "image": image_tensor,
            "image_path": record["image_path"],
            "seed": record.get("seed"),
            "split": record.get("split"),
            "source": record.get("source"),
            "preprocess_mode": self.preprocess_mode,
        }
        for key in ("target_forest", "target_ocean", "target_beach", "target_total"):
            if key in record:
                sample[key] = record[key]
        return sample


def collate_inference_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "seed": [item.get("seed") for item in batch],
        "split": [item.get("split") for item in batch],
        "source": [item.get("source") for item in batch],
        "preprocess_mode": batch[0]["preprocess_mode"],
    }
    for key in ("target_forest", "target_ocean", "target_beach", "target_total"):
        if all(key in item for item in batch):
            collated[key] = [item[key] for item in batch]
    return collated


@torch.no_grad()
def run_batch_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    evaluation_cfg: dict[str, Any],
    *,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    batch_iter = dataloader
    if show_progress:
        batch_iter = tqdm(dataloader, desc="inference", unit="batch")

    for batch in batch_iter:
        images = batch["image"].to(device, non_blocking=True)
        pred = model(images).detach().cpu().numpy()

        pred_quality = compute_quality(
            pred,
            eps=evaluation_cfg["eps"],
            weights=evaluation_cfg.get("quality_weights"),
        )

        batch_size = pred.shape[0]
        for i in range(batch_size):
            row: dict[str, Any] = {
                "image_path": batch["image_path"][i],
                "seed": batch["seed"][i],
                "split": batch["split"][i],
                "source": batch["source"][i],
                "preprocess_mode": batch["preprocess_mode"],
                "pred_forest": float(pred[i, 0]),
                "pred_ocean": float(pred[i, 1]),
                "pred_beach": float(pred[i, 2]),
                "pred_quality": float(pred_quality[i]),
            }
            if "target_forest" in batch:
                row["target_forest"] = batch["target_forest"][i]
                row["target_ocean"] = batch["target_ocean"][i]
                row["target_beach"] = batch["target_beach"][i]
            if "target_total" in batch:
                row["target_total"] = batch["target_total"][i]
            rows.append(row)
    return rows


def write_inference_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    threshold: float | None = None,
) -> Path:
    df = pd.DataFrame(rows)
    for col in INFERENCE_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if threshold is not None:
        df["accept_decision"] = df["pred_quality"] >= threshold
    else:
        df["accept_decision"] = None

    df = df[list(INFERENCE_OUTPUT_COLUMNS)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def predict(
    *,
    checkpoint: Path,
    output: Path,
    input_csv: Path | None = None,
    image_dir: Path | None = None,
    config_path: Path | None = None,
    image_root: Path | None = None,
    preprocess_mode: str = "prepared",
    batch_size: int = 32,
    num_workers: int = 4,
    threshold: float | None = None,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> Path:
    validate_input_args(input_csv, image_dir)

    checkpoint_path = resolve_path(checkpoint)
    output = resolve_path(output)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = resolve_inference_config(checkpoint_payload, config_path)

    image_cfg = config.get("image", {})
    image_width = int(image_cfg.get("width", 512))
    image_height = int(image_cfg.get("height", 320))
    normalize = str(image_cfg.get("normalize", "imagenet"))
    evaluation_cfg = evaluation_config_from_training(config)

    root = resolve_image_root(image_root, config)
    print(f"image_root: {root}")

    if input_csv is not None:
        records = records_from_csv(resolve_path(input_csv), root)
    else:
        assert image_dir is not None
        records = records_from_image_dir(resolve_path(image_dir))

    dataset = InferenceImageDataset(
        records,
        image_root=root,
        preprocess_mode=preprocess_mode,
        image_width=image_width,
        image_height=image_height,
        normalize=normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_inference_batch,
    )

    model = load_model_from_checkpoint(checkpoint_payload, config, device)
    rows = run_batch_inference(
        model,
        loader,
        device,
        evaluation_cfg,
        show_progress=show_progress,
    )
    return write_inference_csv(rows, output, threshold=threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ScratchResNetCNN inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root for relative image_path values; defaults to checkpoint/config data.image_root, else '.'",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("prepared", "raw"),
        default="prepared",
        help=(
            "prepared: input images are already masked and resized to training format (512x320). "
            "raw: original World Preview screenshots; applies dataset mask + resize pipeline. "
            "Use raw for deployment on full-resolution captures."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device is not None else None
    output_path = predict(
        checkpoint=args.checkpoint,
        output=args.output,
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        config_path=args.config,
        image_root=args.image_root,
        preprocess_mode=args.preprocess_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        device=device,
        show_progress=not args.no_progress,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
