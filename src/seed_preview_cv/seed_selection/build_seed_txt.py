"""Build a fixed-size seed list file (one seed per line)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR, PROJECT_ROOT
from seed_preview_cv.seed_selection.acceptance import AcceptanceConfig, SEED_LIST_COLUMNS, apply_acceptance
from seed_preview_cv.seed_selection.distances import ScanConfig, scan_seed_distances
from seed_preview_cv.seed_selection.seeds import generate_seed_batch


def write_seed_txt(path: Path, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(s) for s in seeds) + "\n", encoding="utf-8")


def collect_accepted_seeds(
    target_count: int,
    *,
    sampling_seed: int,
    acceptance_config: AcceptanceConfig,
    search_radius_blocks: int,
    batch_size: int = 5000,
    max_scanned: int | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], int]:
    """Scan and sample until at least target_count seeds are accepted."""
    if max_scanned is None:
        # ~10% acceptance at current params; scan up to 20x target with headroom.
        max_scanned = max(target_count * 20, 50000)

    accepted_rows: list[dict] = []
    start_index = 0
    scanned = 0

    while len(accepted_rows) < target_count and scanned < max_scanned:
        batch = min(batch_size, max_scanned - scanned)
        seeds = generate_seed_batch(batch, sampling_seed, start_index=start_index)
        df = scan_seed_distances(
            [int(s) for s in seeds],
            ScanConfig(search_radius_blocks=search_radius_blocks, show_progress=False),
        )
        batch_accept_cfg = AcceptanceConfig(
            p_min=acceptance_config.p_min,
            p_max=acceptance_config.p_max,
            d0=acceptance_config.d0,
            scale=acceptance_config.scale,
            rng_seed=acceptance_config.rng_seed + start_index,
        )
        sampled = apply_acceptance(df, batch_accept_cfg)
        batch_accepted = sampled[sampled["accepted"]]
        accepted_rows.extend(batch_accepted.to_dict("records"))

        start_index += batch
        scanned += batch
        if show_progress:
            rate = len(accepted_rows) / scanned if scanned else 0.0
            print(
                f"scanned {scanned}/{max_scanned}, "
                f"accepted {len(accepted_rows)}/{target_count} "
                f"(running rate {rate:.1%})"
            )

    if len(accepted_rows) < target_count:
        raise RuntimeError(
            f"Only collected {len(accepted_rows)} accepted seeds after scanning {scanned}"
        )

    return accepted_rows[:target_count], scanned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build seeds.txt with accepted seeds")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "seed_selection.yaml")
    parser.add_argument("--count", type=int, default=200, help="Number of seeds to collect")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "seeds.txt",
        help="Output seeds.txt path",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--max-scanned",
        type=int,
        help="Max random seeds to scan (default: count * 20)",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    sampling = config.get("sampling", {})
    treasure = config.get("treasure", {})
    output_cfg = config.get("output", {})
    acceptance_cfg = AcceptanceConfig.from_mapping(config.get("acceptance", {}))

    sampling_seed = int(sampling.get("seed", 42))
    search_radius = int(treasure.get("search_radius_blocks", 512))
    max_scanned = args.max_scanned if args.max_scanned is not None else args.count * 20

    accepted_rows, scanned = collect_accepted_seeds(
        args.count,
        sampling_seed=sampling_seed,
        acceptance_config=acceptance_cfg,
        search_radius_blocks=search_radius,
        batch_size=args.batch_size,
        max_scanned=max_scanned,
        show_progress=not args.no_progress,
    )

    seeds = [int(row["seed"]) for row in accepted_rows]
    out_path = resolve_path(args.output)
    write_seed_txt(out_path, seeds)

    seed_list_path = resolve_path(output_cfg.get("seed_list", "outputs/seed_selection/seed_list.csv"))
    pd.DataFrame(accepted_rows)[list(SEED_LIST_COLUMNS)].to_csv(seed_list_path, index=False)

    meta = {
        "count": args.count,
        "scanned": scanned,
        "actual_acceptance_rate": len(seeds) / scanned,
        "sampling_seed": sampling_seed,
        "max_scanned": max_scanned,
        "acceptance": {
            "p_min": acceptance_cfg.p_min,
            "p_max": acceptance_cfg.p_max,
            "d0": acceptance_cfg.d0,
            "scale": acceptance_cfg.scale,
            "rng_seed": acceptance_cfg.rng_seed,
        },
        "search_radius_blocks": search_radius,
        "output": str(out_path),
        "seed_list": str(seed_list_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"Wrote {len(seeds)} seeds to {out_path}")
    print(f"Wrote seed metadata to {seed_list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
