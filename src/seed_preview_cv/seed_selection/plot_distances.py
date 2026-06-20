"""Plot buried treasure distance distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distance_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    search_radius_blocks: int = 512,
    bin_width: int = 16,
    title: str | None = None,
) -> Path:
    """Save a two-panel figure: distance histogram + found/not-found overview."""
    dist = df["nearest_treasure_dist"].to_numpy()
    found_mask = dist >= 0
    found_dist = dist[found_mask]
    not_found_count = int((~found_mask).sum())
    total = len(dist)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_hist, ax_overview = axes

    if found_dist.size:
        upper = min(search_radius_blocks, int(found_dist.max()))
        bins = np.arange(0, upper + bin_width, bin_width)
        ax_hist.hist(
            found_dist,
            bins=bins,
            color="#3b82f6",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        median = float(np.median(found_dist))
        ax_hist.axvline(median, color="#ef4444", linestyle="--", linewidth=1.5, label=f"median = {median:.0f}")
        ax_hist.legend(loc="upper right")
    else:
        ax_hist.text(0.5, 0.5, "No treasure found in scan", ha="center", va="center", transform=ax_hist.transAxes)

    ax_hist.set_xlabel("Nearest buried treasure distance (blocks)")
    ax_hist.set_ylabel("Seed count")
    ax_hist.set_title("Distance distribution (treasure found)")
    ax_hist.grid(axis="y", alpha=0.25)

    labels = ["Found", "Not found"]
    counts = [int(found_mask.sum()), not_found_count]
    colors = ["#3b82f6", "#94a3b8"]
    bars = ax_overview.bar(labels, counts, color=colors, edgecolor="white", width=0.55)
    ax_overview.set_ylabel("Seed count")
    ax_overview.set_title("Overview")
    ax_overview.grid(axis="y", alpha=0.25)

    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total if total else 0.0
        ax_overview.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        fig.suptitle(
            f"Buried treasure distance scan (n={total}, radius={search_radius_blocks} blocks)",
            fontsize=13,
            y=1.02,
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_acceptance_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    output_path: Path,
    *,
    search_radius_blocks: int = 512,
    bin_width: int = 16,
    title: str | None = None,
) -> Path:
    """Overlay distance histograms before and after acceptance sampling."""
    before_dist = before_df["nearest_treasure_dist"].to_numpy()
    after_dist = after_df["nearest_treasure_dist"].to_numpy()

    before_found = before_dist >= 0
    after_found = after_dist >= 0
    before_found_dist = before_dist[before_found]
    after_found_dist = after_dist[after_found]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_hist, ax_overview = axes

    upper = search_radius_blocks
    if before_found_dist.size:
        upper = min(search_radius_blocks, int(before_found_dist.max()))
    bins = np.arange(0, upper + bin_width, bin_width)

    if before_found_dist.size:
        ax_hist.hist(
            before_found_dist,
            bins=bins,
            color="#94a3b8",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.55,
            label=f"Before ({before_found_dist.size})",
        )
    if after_found_dist.size:
        ax_hist.hist(
            after_found_dist,
            bins=bins,
            color="#f97316",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.75,
            label=f"After ({after_found_dist.size})",
        )

    ax_hist.set_xlabel("Nearest buried treasure distance (blocks)")
    ax_hist.set_ylabel("Seed count")
    ax_hist.set_title("Distance distribution (treasure found)")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(axis="y", alpha=0.25)

    categories = ["Found", "Not found"]
    before_counts = [int(before_found.sum()), int((~before_found).sum())]
    after_counts = [int(after_found.sum()), int((~after_found).sum())]
    x = np.arange(len(categories))
    width = 0.35
    ax_overview.bar(x - width / 2, before_counts, width, label="Before", color="#94a3b8", edgecolor="white")
    ax_overview.bar(x + width / 2, after_counts, width, label="After", color="#f97316", edgecolor="white")
    ax_overview.set_xticks(x)
    ax_overview.set_xticklabels(categories)
    ax_overview.set_ylabel("Seed count")
    ax_overview.set_title("Overview")
    ax_overview.legend()
    ax_overview.grid(axis="y", alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        fig.suptitle(
            f"Acceptance comparison (before n={len(before_df)}, after n={len(after_df)})",
            fontsize=13,
            y=1.02,
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
