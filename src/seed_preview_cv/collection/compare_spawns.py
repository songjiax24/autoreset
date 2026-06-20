"""Compare cubiomes estimated spawn with mod-collected spawn coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import COLLECTION_SPAWNS_CSV, DATA_INTERIM_DIR, OUTPUTS_DIR, PROJECT_ROOT
from seed_preview_cv.cubiomes_bindings import nearest_buried_treasure_dist


def estimate_spawn(seed: int) -> tuple[int, int]:
    """Return cubiomes estimateSpawn for a seed (search radius 0, treasure ignored)."""
    result = nearest_buried_treasure_dist(seed, search_radius_blocks=0)
    return result.spawn_x, result.spawn_z


def compare_spawns(spawns_df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    iterator = spawns_df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(list(spawns_df.itertuples(index=False)), desc="estimate spawn")

    for row in iterator:
        seed = int(row.seed)
        true_x, true_z = int(row.x), int(row.z)
        est_x, est_z = estimate_spawn(seed)
        dx = est_x - true_x
        dz = est_z - true_z
        err = float(np.hypot(dx, dz))
        rows.append(
            {
                "seed": seed,
                "spawn_x": true_x,
                "spawn_z": true_z,
                "estimated_spawn_x": est_x,
                "estimated_spawn_z": est_z,
                "delta_x": dx,
                "delta_z": dz,
                "error_blocks": err,
                "exact_match": bool(dx == 0 and dz == 0),
            }
        )

    return pd.DataFrame(rows)


def summarize_spawn_comparison(df: pd.DataFrame) -> dict:
    err = df["error_blocks"].to_numpy()
    return {
        "total": int(len(df)),
        "exact_match_count": int(df["exact_match"].sum()),
        "exact_match_rate": float(df["exact_match"].mean()),
        "error_min": float(err.min()),
        "error_max": float(err.max()),
        "error_mean": float(err.mean()),
        "error_median": float(np.median(err)),
        "error_p90": float(np.percentile(err, 90)),
        "error_p95": float(np.percentile(err, 95)),
        "error_p99": float(np.percentile(err, 99)),
        "delta_x_mean": float(df["delta_x"].mean()),
        "delta_z_mean": float(df["delta_z"].mean()),
        "within_threshold": {
            str(t): int((err <= t).sum()) for t in (0, 4, 8, 16, 32, 64, 128)
        },
    }


def plot_spawn_comparison(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ax_x, ax_z, ax_err = axes
    ax_x.scatter(df["spawn_x"], df["estimated_spawn_x"], alpha=0.65, s=18, c="#3b82f6")
    lim_x = (
        min(df["spawn_x"].min(), df["estimated_spawn_x"].min()),
        max(df["spawn_x"].max(), df["estimated_spawn_x"].max()),
    )
    ax_x.plot(lim_x, lim_x, "--", color="#ef4444", linewidth=1.2, label="perfect")
    ax_x.set_xlabel("True spawn x")
    ax_x.set_ylabel("Estimated spawn x")
    ax_x.set_title("Spawn X")
    ax_x.legend()
    ax_x.grid(alpha=0.25)

    ax_z.scatter(df["spawn_z"], df["estimated_spawn_z"], alpha=0.65, s=18, c="#3b82f6")
    lim_z = (
        min(df["spawn_z"].min(), df["estimated_spawn_z"].min()),
        max(df["spawn_z"].max(), df["estimated_spawn_z"].max()),
    )
    ax_z.plot(lim_z, lim_z, "--", color="#ef4444", linewidth=1.2, label="perfect")
    ax_z.set_xlabel("True spawn z")
    ax_z.set_ylabel("Estimated spawn z")
    ax_z.set_title("Spawn Z")
    ax_z.legend()
    ax_z.grid(alpha=0.25)

    ax_err.hist(df["error_blocks"], bins=32, color="#f97316", edgecolor="white", alpha=0.9)
    median = float(df["error_blocks"].median())
    ax_err.axvline(median, color="#ef4444", linestyle="--", label=f"median = {median:.1f}")
    ax_err.set_xlabel("Euclidean error (blocks)")
    ax_err.set_ylabel("Seed count")
    ax_err.set_title("Position error")
    ax_err.legend()
    ax_err.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Spawn estimate vs mod ground truth (n={len(df)}, "
        f"exact={int(df['exact_match'].sum())})",
        fontsize=13,
        y=1.02,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare estimated and true spawn positions")
    parser.add_argument(
        "--input",
        type=Path,
        default=COLLECTION_SPAWNS_CSV,
        help="Mod-collected spawns CSV (seed,x,z)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_INTERIM_DIR / "spawn_comparison.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=OUTPUTS_DIR / "collection" / "spawn_comparison_summary.json",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=OUTPUTS_DIR / "collection" / "spawn_comparison.png",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    in_path = resolve_path(args.input)
    out_csv = resolve_path(args.output_csv)
    out_summary = resolve_path(args.output_summary)
    out_plot = resolve_path(args.output_plot)

    spawns = pd.read_csv(in_path)
    comparison = compare_spawns(spawns, show_progress=not args.no_progress)
    summary = summarize_spawn_comparison(comparison)
    summary["input"] = str(in_path)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_csv, index=False)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_spawn_comparison(comparison, out_plot)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
