"""Prepare dataset images: mask HUD + resize to 512x320."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import (
    DATA_DATASET_IMAGES_DIR,
    DATA_INTERIM_DIR,
    DATA_SCREENSHOTS_DIR,
)
from seed_preview_cv.collection.mask_screenshots import (
    DATASET_HEIGHT,
    DATASET_WIDTH,
    REF_HEIGHT,
    REF_WIDTH,
    prepare_dataset_image_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask World Preview HUD and resize screenshots for model input",
    )
    parser.add_argument(
        "--seeds-csv",
        type=Path,
        default=DATA_INTERIM_DIR / "pilot_spawns.csv",
        help="CSV with seed column (default: pilot 200)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_SCREENSHOTS_DIR,
        help="Directory of raw 2560x1600 screenshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DATASET_IMAGES_DIR / "pilot200",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seeds_path = resolve_path(args.seeds_csv)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    seeds_df = pd.read_csv(seeds_path)
    if "seed" not in seeds_df.columns:
        raise ValueError(f"{seeds_path} must contain a seed column")

    iterator = seeds_df.itertuples(index=False)
    if not args.no_progress:
        iterator = tqdm(list(seeds_df.itertuples(index=False)), desc="prepare dataset images")

    written: list[dict] = []
    for row in iterator:
        seed = int(row.seed)
        input_path = input_dir / f"{seed}.png"
        if not input_path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {input_path}")

        output_path = output_dir / f"{seed}.png"
        prepare_dataset_image_file(input_path, output_path)
        written.append(
            {
                "seed": seed,
                "image_path": str(output_path.relative_to(resolve_path("."))),
                "source_screenshot": str(input_path.relative_to(resolve_path("."))),
            }
        )

    index_path = output_dir / "dataset_index.csv"
    index_df = pd.DataFrame(written)
    index_df.to_csv(index_path, index=False)

    manifest = {
        "seeds_csv": str(seeds_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "count": len(written),
        "source_size": f"{REF_WIDTH}x{REF_HEIGHT}",
        "output_size": f"{DATASET_WIDTH}x{DATASET_HEIGHT}",
        "pipeline": "mask_ui_masks + resize INTER_AREA",
        "dataset_index": str(index_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(written)} images -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
