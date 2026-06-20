"""CLI: plot buried treasure distance histogram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR
from seed_preview_cv.seed_selection.plot_distances import plot_distance_histogram
from seed_preview_cv.seed_selection.summarize_distances import load_distance_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot treasure distance histogram")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "seed_selection.yaml")
    parser.add_argument("--input", type=Path, help="Override input distances table")
    parser.add_argument("--output", type=Path, help="Override histogram image path")
    parser.add_argument("--bin-width", type=int, default=16, help="Histogram bin width in blocks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    treasure = config.get("treasure", {})
    output_cfg = config.get("output", {})

    in_path = resolve_path(
        args.input or output_cfg.get("distances", "outputs/seed_selection/distances.csv")
    )
    out_path = resolve_path(
        args.output or output_cfg.get("distance_histogram", "outputs/seed_selection/distance_histogram.png")
    )
    search_radius = int(treasure.get("search_radius_blocks", 512))

    meta_path = in_path.with_suffix(in_path.suffix + ".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        search_radius = int(meta.get("search_radius_blocks", search_radius))

    df = load_distance_table(in_path)
    saved = plot_distance_histogram(
        df,
        out_path,
        search_radius_blocks=search_radius,
        bin_width=args.bin_width,
    )
    print(f"Wrote histogram to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
