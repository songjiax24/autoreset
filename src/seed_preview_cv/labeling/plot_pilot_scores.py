"""Histograms for isotropic vs directional score distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import DATA_INTERIM_DIR, DATA_LABELS_DIR
from seed_preview_cv.labeling.compute_labels import compute_label_rows

SCORE_COLUMNS = ("s_forest", "s_ocean", "s_beach", "s_total")
SCORE_TITLES = {
    "s_forest": "Forest score",
    "s_ocean": "Ocean score",
    "s_beach": "Beach score",
    "s_total": "Total score (exp(-l_total))",
}


def plot_score_histograms(
    isotropic_df: pd.DataFrame,
    directional_df: pd.DataFrame,
    output_path: Path,
    *,
    separate: bool = False,
    title: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(isotropic_df)
    suptitle = title or f"Score distributions (n={n})"

    if separate:
        fig, axes = plt.subplots(4, 2, figsize=(11, 13), constrained_layout=True)
        for row, col in enumerate(SCORE_COLUMNS):
            iso_vals = isotropic_df[col].astype(float)
            dir_vals = directional_df[col].astype(float)
            ax_iso, ax_dir = axes[row, 0], axes[row, 1]
            ax_iso.hist(
                iso_vals,
                bins=30,
                color="#3b82f6",
                edgecolor="white",
                alpha=0.9,
            )
            ax_dir.hist(
                dir_vals,
                bins=30,
                color="#f59e0b",
                edgecolor="white",
                alpha=0.9,
            )
            ax_iso.set_title(f"isotropic — {SCORE_TITLES[col]}")
            ax_dir.set_title(f"directional — {SCORE_TITLES[col]}")
            ax_iso.set_xlabel(col)
            ax_dir.set_xlabel(col)
            ax_iso.set_ylabel("count")
            ax_dir.set_ylabel("count")
            ax_iso.set_xlim(0.0, 1.0)
            ax_dir.set_xlim(0.0, 1.0)
            ax_iso.grid(axis="y", alpha=0.25)
            ax_dir.grid(axis="y", alpha=0.25)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for ax, col in zip(axes.ravel(), SCORE_COLUMNS):
            iso = isotropic_df[col].astype(float)
            dire = directional_df[col].astype(float)
            ax.hist(
                iso,
                bins=30,
                alpha=0.55,
                color="#3b82f6",
                edgecolor="white",
                label="isotropic",
            )
            ax.hist(
                dire,
                bins=30,
                alpha=0.55,
                color="#f59e0b",
                edgecolor="white",
                label="directional",
            )
            ax.set_title(SCORE_TITLES[col])
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            ax.set_xlim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_labels_for_spawns(spawns: pd.DataFrame, weight_mode: str) -> pd.DataFrame:
    rows = compute_label_rows(spawns, weight_mode=weight_mode, progress=True)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot isotropic vs directional score histograms",
    )
    parser.add_argument(
        "--seeds-csv",
        type=Path,
        default=DATA_INTERIM_DIR / "pilot_spawns.csv",
    )
    parser.add_argument(
        "--isotropic-labels",
        type=Path,
        default=None,
        help="If omitted, compute labels from --seeds-csv",
    )
    parser.add_argument(
        "--directional-labels",
        type=Path,
        default=None,
        help="If omitted, compute labels from --seeds-csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_scores_compare_histogram.png",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Use 8 separate histograms (4 metrics x 2 modes), no overlay",
    )
    parser.add_argument("--title", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spawns_path = resolve_path(args.seeds_csv)
    spawns = pd.read_csv(spawns_path)
    seeds = spawns["seed"].astype(int)

    if args.isotropic_labels:
        iso = pd.read_csv(resolve_path(args.isotropic_labels))
    else:
        iso = compute_labels_for_spawns(spawns, "isotropic")

    if args.directional_labels:
        dire = pd.read_csv(resolve_path(args.directional_labels))
    else:
        dire = compute_labels_for_spawns(spawns, "directional")

    iso = iso[iso["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)
    dire = dire[dire["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)
    if len(iso) != len(seeds) or len(dire) != len(seeds):
        raise ValueError("Label rows do not cover all seeds in --seeds-csv")

    out = plot_score_histograms(
        iso,
        dire,
        resolve_path(args.output),
        separate=args.separate,
        title=args.title,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
