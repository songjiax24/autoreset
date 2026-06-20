"""Summarize label distributions for parameter tuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import DATA_LABELS_DIR


LOSS_COLUMNS = ("l_forest", "l_ocean", "l_beach", "l_total")
SCORE_COLUMNS = ("s_forest", "s_ocean", "s_beach", "s_total")
DISTANCE_COLUMNS = ("d_min_forest_high", "d_min_forest_low", "d_min_ocean")
RATIO_COLUMNS = ("ocean_tier_worst_ratio", "ocean_tier_mid_ratio", "ocean_tier_best_ratio")
COUNT_COLUMNS = ("n_ocean", "n_beach", "s_beach_base")


def _parse_loss_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: math.inf if v == "inf" else float(v))


def _distribution_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _loss_column_stats(series: pd.Series) -> dict[str, Any]:
    parsed = _parse_loss_series(series)
    finite = parsed[parsed < math.inf]
    infinite_count = int((parsed >= math.inf).sum())
    stats: dict[str, Any] = {
        "finite_count": int(finite.size),
        "infinite_count": infinite_count,
        "infinite_rate": float(infinite_count / len(parsed)) if len(parsed) else 0.0,
    }
    if finite.size:
        stats["finite"] = _distribution_stats(finite.to_numpy())
    return stats


def summarize_labels(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": int(len(df))}

    for col in SCORE_COLUMNS:
        summary[col] = _distribution_stats(df[col].astype(float).to_numpy())

    for col in LOSS_COLUMNS:
        summary[col] = _loss_column_stats(df[col])

    for col in DISTANCE_COLUMNS:
        finite = df[col].dropna().astype(float).to_numpy()
        stats = _distribution_stats(finite)
        stats["missing_count"] = int(df[col].isna().sum())
        summary[col] = stats

    for col in RATIO_COLUMNS + COUNT_COLUMNS:
        summary[col] = _distribution_stats(df[col].astype(float).to_numpy())

    # Veto breakdown: which loss component is infinite?
    l_f = _parse_loss_series(df["l_forest"])
    l_o = _parse_loss_series(df["l_ocean"])
    l_b = _parse_loss_series(df["l_beach"])
    l_t = _parse_loss_series(df["l_total"])

    summary["veto_breakdown"] = {
        "l_forest_infinite": int((l_f >= math.inf).sum()),
        "l_ocean_infinite": int((l_o >= math.inf).sum()),
        "l_beach_infinite": int((l_b >= math.inf).sum()),
        "l_total_infinite": int((l_t >= math.inf).sum()),
        "only_forest_infinite": int(((l_f >= math.inf) & (l_o < math.inf) & (l_b < math.inf)).sum()),
        "only_ocean_infinite": int(((l_o >= math.inf) & (l_f < math.inf) & (l_b < math.inf)).sum()),
        "only_beach_infinite": int(((l_b >= math.inf) & (l_f < math.inf) & (l_o < math.inf)).sum()),
    }

    finite_total = l_t[l_t < math.inf]
    if finite_total.size:
        summary["l_total_buckets"] = {
            "lt_0.2": int((finite_total < 0.2).sum()),
            "0.2_0.3": int(((finite_total >= 0.2) & (finite_total < 0.3)).sum()),
            "0.3_0.7": int(((finite_total >= 0.3) & (finite_total < 0.7)).sum()),
            "0.7_1.0": int(((finite_total >= 0.7) & (finite_total <= 1.0)).sum()),
            "gt_1.0": int((finite_total > 1.0).sum()),
        }

    return summary


def plot_label_distributions(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)
    score_panels = [
        ("s_forest", df["s_forest"], "#22c55e", "Forest score"),
        ("s_ocean", df["s_ocean"], "#3b82f6", "Ocean score"),
        ("s_beach", df["s_beach"], "#f59e0b", "Beach score"),
    ]
    for ax, (name, series, color, title) in zip(axes[0], score_panels):
        ax.hist(series.astype(float), bins=30, color=color, edgecolor="white", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)

    axes[1, 0].hist(
        df["s_total"].astype(float),
        bins=30,
        color="#a855f7",
        edgecolor="white",
        alpha=0.9,
    )
    axes[1, 0].set_title("Total score")
    axes[1, 0].set_xlabel("s_total")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(axis="y", alpha=0.25)

    loss_panels = [
        ("l_forest", df["l_forest"], "#22c55e"),
        ("l_ocean", df["l_ocean"], "#3b82f6"),
        ("l_beach", df["l_beach"], "#f59e0b"),
    ]
    for ax, (name, series, color) in zip(axes[1, 1:], loss_panels[:2]):
        vals = _parse_loss_series(series)
        finite = vals[vals < math.inf]
        ax.hist(finite, bins=30, color=color, edgecolor="white", alpha=0.9)
        inf_n = int((vals >= math.inf).sum())
        ax.set_title(f"{name} (finite only, inf={inf_n})")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)

    l_b_vals = _parse_loss_series(df["l_beach"])
    l_b_finite = l_b_vals[l_b_vals < math.inf]
    axes[2, 0].hist(l_b_finite, bins=30, color="#f59e0b", edgecolor="white", alpha=0.9)
    inf_n = int((l_b_vals >= math.inf).sum())
    axes[2, 0].set_title(f"l_beach (finite only, inf={inf_n})")
    axes[2, 0].set_xlabel("l_beach")
    axes[2, 0].set_ylabel("count")
    axes[2, 0].grid(axis="y", alpha=0.25)

    l_t_vals = _parse_loss_series(df["l_total"])
    l_t_finite = l_t_vals[l_t_vals < math.inf]
    axes[2, 1].hist(l_t_finite, bins=30, color="#a855f7", edgecolor="white", alpha=0.9)
    inf_n = int((l_t_vals >= math.inf).sum())
    axes[2, 1].set_title(f"l_total (finite only, inf={inf_n})")
    axes[2, 1].set_xlabel("l_total")
    axes[2, 1].set_ylabel("count")
    axes[2, 1].grid(axis="y", alpha=0.25)

    dist_panels = [
        ("d_min_forest_high", df["d_min_forest_high"].dropna(), "#22c55e"),
        ("d_min_forest_low", df["d_min_forest_low"].dropna(), "#84cc16"),
        ("d_min_ocean", df["d_min_ocean"].dropna(), "#3b82f6"),
    ]
    for ax, (name, series, color) in zip(axes[2, 2:], dist_panels[:1]):
        ax.hist(series.astype(float), bins=30, color=color, edgecolor="white", alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("blocks")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)

    for ax, (name, series, color) in zip(axes[3], dist_panels[1:]):
        ax.hist(series.astype(float), bins=30, color=color, edgecolor="white", alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("blocks")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)

    axes[3, 2].axis("off")

    fig.suptitle(f"Label distributions (n={len(df)})", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize label CSV distributions")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels_stats.json",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels_distributions.png",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    in_path = resolve_path(args.input)
    out_json = resolve_path(args.output_json)
    out_plot = resolve_path(args.output_plot)

    df = pd.read_csv(in_path)
    summary = summarize_labels(df)
    summary["input"] = str(in_path)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_label_distributions(df, out_plot)
        summary["distribution_plot"] = str(out_plot)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    if not args.no_plot:
        print(f"Wrote {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
