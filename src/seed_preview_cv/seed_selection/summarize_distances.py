"""CLI: summarize buried treasure distance distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR
from seed_preview_cv.seed_selection.distances import summarize_distances


def load_distance_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize treasure distance scan results")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "seed_selection.yaml",
    )
    parser.add_argument("--input", type=Path, help="Override input distances table")
    parser.add_argument("--output", type=Path, help="Override summary JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    output_cfg = config.get("output", {})

    in_path = resolve_path(
        args.input or output_cfg.get("distances", "outputs/seed_selection/distances.csv")
    )
    out_path = resolve_path(
        args.output or output_cfg.get("distance_summary", "outputs/seed_selection/distance_summary.json")
    )

    df = load_distance_table(in_path)
    summary = summarize_distances(df)
    summary["input"] = str(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
