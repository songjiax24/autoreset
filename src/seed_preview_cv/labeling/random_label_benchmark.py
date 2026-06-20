"""Label benchmark on uniformly random int64 seeds with cubiomes getSpawn()."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import DATA_INTERIM_DIR, DATA_LABELS_DIR
from seed_preview_cv.cubiomes_bindings.ffi import get_world_spawn
from seed_preview_cv.labeling.compute_labels import compute_label_rows
from seed_preview_cv.labeling.summarize_labels import plot_label_distributions, summarize_labels
from seed_preview_cv.labeling.weights import WeightMode


def sample_uniform_int64_seeds(count: int, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 2**64, size=count, dtype=np.uint64)


def resolve_spawns_getspawn(seeds: np.ndarray, progress: bool = True) -> pd.DataFrame:
    iterator = seeds
    if progress:
        iterator = tqdm(seeds, desc="getSpawn")

    rows: list[dict[str, int]] = []
    for seed_u64 in iterator:
        seed = int(seed_u64)
        spawn_x, spawn_z = get_world_spawn(seed)
        rows.append({"seed": seed, "x": spawn_x, "z": spawn_z})

    return pd.DataFrame(rows)


def compare_mode_stats(
    isotropic: dict[str, Any],
    directional: dict[str, Any],
) -> dict[str, Any]:
    keys = ("s_forest", "s_ocean", "s_beach", "l_forest", "l_ocean", "l_beach", "l_total")
    comparison: dict[str, Any] = {"total": isotropic.get("total")}

    def _median(stats: dict[str, Any]) -> float | None:
        if "median" in stats:
            return float(stats["median"])
        finite = stats.get("finite")
        if isinstance(finite, dict) and "median" in finite:
            return float(finite["median"])
        return None

    for key in keys:
        iso = isotropic.get(key, {})
        dire = directional.get(key, {})
        iso_med = _median(iso)
        dir_med = _median(dire)
        if iso_med is not None and dir_med is not None:
            comparison[f"{key}_median_iso"] = iso_med
            comparison[f"{key}_median_dir"] = dir_med
        if "infinite_rate" in iso and "infinite_rate" in dire:
            comparison[f"{key}_infinite_rate_iso"] = iso["infinite_rate"]
            comparison[f"{key}_infinite_rate_dir"] = dire["infinite_rate"]

    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark isotropic/directional labels on random int64 seeds (getSpawn)",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--spawns-output",
        type=Path,
        default=DATA_INTERIM_DIR / "random200_getspawn_spawns.csv",
    )
    parser.add_argument(
        "--isotropic-output",
        type=Path,
        default=DATA_LABELS_DIR / "random200_labels_isotropic.csv",
    )
    parser.add_argument(
        "--directional-output",
        type=Path,
        default=DATA_LABELS_DIR / "random200_labels_directional.csv",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--skip-spawns",
        action="store_true",
        help="Reuse existing spawns CSV (skip getSpawn)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    progress = not args.no_progress

    spawns_path = resolve_path(args.spawns_output)
    iso_path = resolve_path(args.isotropic_output)
    dir_path = resolve_path(args.directional_output)
    iso_stats_path = iso_path.with_name(iso_path.stem + "_stats.json")
    dir_stats_path = dir_path.with_name(dir_path.stem + "_stats.json")
    compare_stats_path = iso_path.with_name("random200_labels_compare_stats.json")
    iso_plot_path = iso_path.with_name(iso_path.stem + "_distributions.png")
    dir_plot_path = dir_path.with_name(dir_path.stem + "_distributions.png")

    if args.skip_spawns and spawns_path.is_file():
        spawns = pd.read_csv(spawns_path)
    else:
        seeds = sample_uniform_int64_seeds(args.count, args.rng_seed)
        spawns = resolve_spawns_getspawn(seeds, progress=progress)
        spawns_path.parent.mkdir(parents=True, exist_ok=True)
        spawns.to_csv(spawns_path, index=False)
        summary = {
            "count": int(len(spawns)),
            "rng_seed": args.rng_seed,
            "spawn_method": "getSpawn",
            "minecraft_version": "1.16.1",
            "seed_sampling": "uniform_uint64 [0, 2^64)",
            "output": str(spawns_path),
            "seeds": [int(s) for s in spawns["seed"]],
        }
        spawns_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    for weight_mode, out_path in (
        ("isotropic", iso_path),
        ("directional", dir_path),
    ):
        rows = compute_label_rows(
            spawns,
            weight_mode=weight_mode,
            progress=progress,
        )
        out_df = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        stats = summarize_labels(out_df)
        stats["input"] = str(out_path)
        stats["spawns"] = str(spawns_path)
        stats["weight_mode"] = weight_mode
        stats_path = out_path.with_name(out_path.stem + "_stats.json")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        if not args.no_plot:
            plot_path = out_path.with_name(out_path.stem + "_distributions.png")
            plot_label_distributions(out_df, plot_path)
            stats["distribution_plot"] = str(plot_path)

        print(json.dumps({k: v for k, v in stats.items() if k != "veto_breakdown"}, indent=2))
        print(f"Wrote {out_path}")
        print(f"Wrote {stats_path}")

    iso_stats = json.loads(iso_stats_path.read_text(encoding="utf-8"))
    dir_stats = json.loads(dir_stats_path.read_text(encoding="utf-8"))
    compare = {
        "spawns": str(spawns_path),
        "isotropic_labels": str(iso_path),
        "directional_labels": str(dir_path),
        "comparison": compare_mode_stats(iso_stats, dir_stats),
    }
    compare_stats_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"Wrote {compare_stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
