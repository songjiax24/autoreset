"""CLI: scan random seeds for buried treasure distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR
from seed_preview_cv.seed_selection.distances import ScanConfig, scan_seed_distances, write_distance_table
from seed_preview_cv.seed_selection.seeds import generate_seed_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan seeds for nearest buried treasure distance")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "seed_selection.yaml",
        help="YAML config path",
    )
    parser.add_argument("--num-seeds", type=int, help="Override sampling.num_seeds")
    parser.add_argument("--seed", type=int, help="Override sampling.seed (RNG seed)")
    parser.add_argument("--start-index", type=int, default=0, help="RNG offset for chunked scans")
    parser.add_argument(
        "--search-radius",
        type=int,
        help="Override treasure.search_radius_blocks",
    )
    parser.add_argument("--output", type=Path, help="Override output.distances path")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    sampling = config.get("sampling", {})
    treasure = config.get("treasure", {})
    output = config.get("output", {})

    num_seeds = args.num_seeds if args.num_seeds is not None else int(sampling.get("num_seeds", 1000))
    rng_seed = args.seed if args.seed is not None else int(sampling.get("seed", 42))
    search_radius = (
        args.search_radius
        if args.search_radius is not None
        else int(treasure.get("search_radius_blocks", 512))
    )

    out_path = resolve_path(args.output or output.get("distances", "outputs/seed_selection/distances.csv"))

    seeds = generate_seed_batch(num_seeds, rng_seed, start_index=args.start_index)
    df = scan_seed_distances(
        [int(s) for s in seeds],
        ScanConfig(search_radius_blocks=search_radius, show_progress=not args.no_progress),
    )
    write_distance_table(df, out_path)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta = {
        "num_seeds": num_seeds,
        "rng_seed": rng_seed,
        "start_index": args.start_index,
        "search_radius_blocks": search_radius,
        "output": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
