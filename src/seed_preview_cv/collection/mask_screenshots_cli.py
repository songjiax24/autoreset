"""CLI: mask UI overlays from collected screenshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import (
    COLLECTION_INDEX_CSV,
    DATA_INTERIM_DIR,
    DATA_SCREENSHOTS_DIR,
    DATA_SCREENSHOTS_MASKED_DIR,
)
from seed_preview_cv.collection.mask_screenshots import mask_image_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mask HUD regions on World Preview screenshots")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_SCREENSHOTS_DIR,
        help="Directory of raw screenshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_SCREENSHOTS_MASKED_DIR,
        help="Directory for masked screenshots",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Update collection_index.csv screenshot paths to masked images",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return 1

    iterator = png_files
    if not args.no_progress:
        iterator = tqdm(png_files, desc="mask screenshots")

    for input_path in iterator:
        output_path = output_dir / input_path.name
        mask_image_file(input_path, output_path)

    if args.update_index:
        index_path = resolve_path(COLLECTION_INDEX_CSV)
        if index_path.is_file():
            index = pd.read_csv(index_path)
            index["screenshot_path"] = index["seed"].map(
                lambda s: str(output_dir.relative_to(resolve_path(".")) / f"{s}.png")
            )
            index.to_csv(index_path, index=False)
            print(f"Updated {index_path}")

    print(f"Masked {len(png_files)} images -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
