"""Deterministic image preprocessing shared by training and inference."""

from __future__ import annotations

import warnings
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from seed_preview_cv.collection.mask_screenshots import (
    DATASET_HEIGHT,
    DATASET_WIDTH,
    prepare_dataset_image,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(
    width: int = 512,
    height: int = 320,
    normalize: str = "imagenet",
) -> transforms.Compose:
    ops: list[Any] = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ]
    if normalize == "imagenet":
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif normalize and normalize != "none":
        raise ValueError(f"Unsupported normalize mode: {normalize}")
    return transforms.Compose(ops)


def _normalize_tensor(tensor: torch.Tensor, normalize: str) -> torch.Tensor:
    if normalize == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    if normalize and normalize != "none":
        raise ValueError(f"Unsupported normalize mode: {normalize}")
    return tensor


def preprocess_prepared_image(
    image: Image.Image,
    width: int = 512,
    height: int = 320,
    normalize: str = "imagenet",
    warn_on_resize: bool = False,
    image_path: str | None = None,
) -> torch.Tensor:
    """Prepare an already mask+resize dataset image for model input."""
    rgb = image.convert("RGB")
    if warn_on_resize and rgb.size != (width, height):
        size_label = f"{rgb.size[0]}x{rgb.size[1]}"
        path_label = image_path if image_path else size_label
        warnings.warn(
            f"Prepared mode received a non-512x320 image: {path_label}. "
            "Prepared mode is intended for images already masked and resized to the training format. "
            "For original World Preview screenshots, use --preprocess-mode raw.",
            stacklevel=2,
        )
    transform = build_image_transform(width=width, height=height, normalize=normalize)
    return transform(rgb)


def apply_dataset_mask_and_resize(
    image: Image.Image,
    width: int = DATASET_WIDTH,
    height: int = DATASET_HEIGHT,
) -> Image.Image:
    """Apply dataset HUD mask then resize (mask -> resize, same as dataset construction)."""
    rgb = image.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    prepared_bgr = prepare_dataset_image(bgr)
    if prepared_bgr.shape[1] != width or prepared_bgr.shape[0] != height:
        raise RuntimeError(
            f"Unexpected prepared size {prepared_bgr.shape[1]}x{prepared_bgr.shape[0]}, "
            f"expected {width}x{height}"
        )
    prepared_rgb = cv2.cvtColor(prepared_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(prepared_rgb)


def preprocess_raw_world_preview(
    image: Image.Image,
    width: int = 512,
    height: int = 320,
    normalize: str = "imagenet",
    mask_config: dict | None = None,
) -> torch.Tensor:
    """Prepare a raw World Preview screenshot using dataset mask + resize pipeline."""
    del mask_config  # reserved for future mask overrides; uses collection/mask_screenshots.py
    prepared = apply_dataset_mask_and_resize(image, width=width, height=height)
    tensor = transforms.ToTensor()(prepared)
    return _normalize_tensor(tensor, normalize)
