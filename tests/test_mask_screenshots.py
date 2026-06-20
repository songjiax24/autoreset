"""Tests for screenshot UI masking."""

from pathlib import Path

import cv2
import numpy as np

from seed_preview_cv.collection.mask_screenshots import apply_ui_masks, mask_regions_for_size


def test_apply_ui_masks_blackens_regions():
    frame = np.full((1600, 2560, 3), 255, dtype=np.uint8)
    masked = apply_ui_masks(frame)
    regions = mask_regions_for_size(2560, 1600)

    assert masked[regions.minimap].max() == 0
    assert masked[regions.percent_text].max() == 0
    assert masked[regions.seed_text].max() == 0

    # Center pixel outside masked HUD should stay white.
    assert masked[400, 1280].max() == 255


def test_mask_image_file(tmp_path: Path):
    from seed_preview_cv.collection.mask_screenshots import mask_image_file

    src = tmp_path / "test.png"
    out = tmp_path / "masked.png"
    cv2.imwrite(str(src), np.full((1600, 2560, 3), 128, dtype=np.uint8))
    saved = mask_image_file(src, out)
    assert saved.is_file()
    assert cv2.imread(str(out)) is not None


def test_prepare_dataset_image_size():
    from seed_preview_cv.collection.mask_screenshots import (
        DATASET_HEIGHT,
        DATASET_WIDTH,
        prepare_dataset_image,
    )

    frame = np.full((1600, 2560, 3), 200, dtype=np.uint8)
    out = prepare_dataset_image(frame)
    assert out.shape == (DATASET_HEIGHT, DATASET_WIDTH, 3)
