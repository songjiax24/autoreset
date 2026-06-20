"""Mask World Preview UI overlays from screenshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Reference resolution used when defining mask regions in capture.py
REF_WIDTH = 2560
REF_HEIGHT = 1600

# Model input resolution (5x downscale from reference)
DATASET_WIDTH = 512
DATASET_HEIGHT = 320


@dataclass(frozen=True)
class MaskRegions:
    """Pixel regions to black out (y, x slices in reference resolution)."""

    minimap: tuple[slice, slice]
    percent_text: tuple[slice, slice]
    seed_text: tuple[slice, slice]
    hand_polygon: np.ndarray


def _ref_mask_regions() -> MaskRegions:
    # Coordinates from capture.py (2560x1600 World Preview HUD)
    hand_pts = np.array(
        [
            [1870, 1600],
            [1860, 1245],
            [1997, 1158],
            [2247, 1299],
            [2340, 1600],
        ],
        dtype=np.int32,
    )
    return MaskRegions(
        minimap=(slice(1150, 1600), slice(0, 450)),
        percent_text=(slice(1050, 1100), slice(180, 270)),
        seed_text=(slice(950, 1000), slice(0, 530)),
        hand_polygon=hand_pts.reshape((-1, 1, 2)),
    )


def _scale_slice(s: slice, scale: float, limit: int) -> slice:
    start = int(round(s.start * scale))
    end = int(round(s.stop * scale))
    start = max(0, min(start, limit))
    end = max(0, min(end, limit))
    if end < start:
        end = start
    return slice(start, end)


def mask_regions_for_size(width: int, height: int) -> MaskRegions:
    """Scale reference mask regions to the target image size."""
    if width == REF_WIDTH and height == REF_HEIGHT:
        return _ref_mask_regions()

    sx = width / REF_WIDTH
    sy = height / REF_HEIGHT
    ref = _ref_mask_regions()
    hand_scaled = np.round(ref.hand_polygon.astype(np.float64) * [sx, sy]).astype(np.int32)

    return MaskRegions(
        minimap=(
            _scale_slice(ref.minimap[0], sy, height),
            _scale_slice(ref.minimap[1], sx, width),
        ),
        percent_text=(
            _scale_slice(ref.percent_text[0], sy, height),
            _scale_slice(ref.percent_text[1], sx, width),
        ),
        seed_text=(
            _scale_slice(ref.seed_text[0], sy, height),
            _scale_slice(ref.seed_text[1], sx, width),
        ),
        hand_polygon=hand_scaled,
    )


def apply_ui_masks(frame: np.ndarray) -> np.ndarray:
    """
    Black out HUD / UI regions on a BGR screenshot.

    Regions match capture.py masks for 2560x1600 World Preview captures.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {frame.shape}")

    height, width = frame.shape[:2]
    masked = frame.copy()
    regions = mask_regions_for_size(width, height)

    masked[regions.minimap] = (0, 0, 0)
    masked[regions.percent_text] = (0, 0, 0)
    masked[regions.seed_text] = (0, 0, 0)
    cv2.fillPoly(masked, [regions.hand_polygon], (0, 0, 0))
    return masked


def resize_for_dataset(frame: np.ndarray) -> np.ndarray:
    """Downscale a BGR image to dataset resolution (512x320)."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {frame.shape}")
    return cv2.resize(
        frame,
        (DATASET_WIDTH, DATASET_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def prepare_dataset_image(frame: np.ndarray) -> np.ndarray:
    """Mask HUD regions then resize to dataset resolution."""
    return resize_for_dataset(apply_ui_masks(frame))


def prepare_dataset_image_file(input_path: Path, output_path: Path) -> Path:
    """Read, mask, resize, and write a dataset-ready screenshot."""
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read image: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset_image(frame)
    if prepared.shape[1] != DATASET_WIDTH or prepared.shape[0] != DATASET_HEIGHT:
        raise RuntimeError(
            f"Unexpected output size {prepared.shape[1]}x{prepared.shape[0]}, "
            f"expected {DATASET_WIDTH}x{DATASET_HEIGHT}"
        )
    if not cv2.imwrite(str(output_path), prepared):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def mask_image_file(input_path: Path, output_path: Path) -> Path:
    """Read, mask, and write a screenshot."""
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read image: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked = apply_ui_masks(frame)
    if not cv2.imwrite(str(output_path), masked):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def mask_screenshot_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.png",
) -> list[Path]:
    """Mask all matching images in a directory."""
    written: list[Path] = []
    for input_path in sorted(input_dir.glob(pattern)):
        output_path = output_dir / input_path.name
        written.append(mask_image_file(input_path, output_path))
    return written
