"""CLI: compute biome-based labels for collected seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import COLLECTION_SPAWNS_CSV, DATA_LABELS_DIR
from seed_preview_cv.cubiomes_bindings.ffi import generate_spawn_chunk_biomes
from seed_preview_cv.labeling.scoring import compute_scores_from_grid, scores_to_row
from seed_preview_cv.labeling.weights import WeightMode


def default_output_for_mode(mode: WeightMode) -> Path:
    if mode == "directional":
        return DATA_LABELS_DIR / "pilot_labels_directional.csv"
    return DATA_LABELS_DIR / "pilot_labels.csv"


def compute_label_rows(
    spawns: pd.DataFrame,
    weight_mode: WeightMode,
    progress: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    iterator = spawns.itertuples(index=False)
    if progress:
        iterator = tqdm(
            list(spawns.itertuples(index=False)),
            desc=f"compute labels ({weight_mode})",
        )

    for row in iterator:
        seed = int(row.seed)
        spawn_x = int(row.x)
        spawn_z = int(row.z)
        grid, grid_x0, grid_z0 = generate_spawn_chunk_biomes(seed, spawn_x, spawn_z)
        scores = compute_scores_from_grid(
            grid,
            seed=seed,
            spawn_x=spawn_x,
            spawn_z=spawn_z,
            grid_x0=grid_x0,
            grid_z0=grid_z0,
            weight_mode=weight_mode,
        )
        rows.append(scores_to_row(scores))

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute seed biome labels")
    parser.add_argument(
        "--input",
        type=Path,
        default=COLLECTION_SPAWNS_CSV,
        help="CSV with seed,x,z spawn coordinates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: pilot_labels.csv or pilot_labels_directional.csv)",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("isotropic", "directional"),
        default="isotropic",
        help="isotropic: W_dist(d); directional: W_eff(d,theta) facing +z",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    weight_mode: WeightMode = args.weight_mode
    in_path = resolve_path(args.input)
    out_path = resolve_path(args.output or default_output_for_mode(weight_mode))

    spawns = pd.read_csv(in_path)
    rows = compute_label_rows(spawns, weight_mode=weight_mode, progress=not args.no_progress)
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    finite = out_df["l_total"].apply(lambda v: v != "inf")
    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "weight_mode": weight_mode,
        "count": int(len(out_df)),
        "finite_loss_count": int(finite.sum()),
        "infinite_loss_count": int((~finite).sum()),
        "l_total_min": float(out_df.loc[finite, "l_total"].astype(float).min()) if finite.any() else None,
        "l_total_median": float(out_df.loc[finite, "l_total"].astype(float).median()) if finite.any() else None,
        "l_total_max": float(out_df.loc[finite, "l_total"].astype(float).max()) if finite.any() else None,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote labels to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
