"""Scan seeds for nearest buried treasure distance from estimated spawn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from seed_preview_cv.cubiomes_bindings import (
    NO_TREASURE_DISTANCE,
    nearest_buried_treasure_dist,
)

DISTANCE_COLUMNS = (
    "seed",
    "estimated_spawn_x",
    "estimated_spawn_z",
    "nearest_treasure_dist",
)


@dataclass(frozen=True)
class ScanConfig:
    search_radius_blocks: int = 512
    show_progress: bool = True


def scan_seed_distances(
    seeds: Iterable[int],
    config: ScanConfig = ScanConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    iterator = seeds
    if config.show_progress:
        iterator = tqdm(list(seeds), desc="scan treasure distances")

    for seed in iterator:
        result = nearest_buried_treasure_dist(seed, config.search_radius_blocks)
        rows.append(
            {
                "seed": int(result.seed),
                "estimated_spawn_x": result.spawn_x,
                "estimated_spawn_z": result.spawn_z,
                "nearest_treasure_dist": result.treasure_dist,
            }
        )

    return pd.DataFrame(rows, columns=list(DISTANCE_COLUMNS))


def write_distance_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def summarize_distances(df: pd.DataFrame) -> dict:
    dist = df["nearest_treasure_dist"].to_numpy()
    found = dist >= 0
    found_dist = dist[found]

    summary: dict = {
        "total": int(len(dist)),
        "found_count": int(found.sum()),
        "not_found_count": int((~found).sum()),
        "not_found_rate": float((~found).mean()) if len(dist) else 0.0,
    }

    if found_dist.size:
        summary.update(
            {
                "min": int(found_dist.min()),
                "max": int(found_dist.max()),
                "mean": float(found_dist.mean()),
                "median": float(np.median(found_dist)),
                "p90": float(np.percentile(found_dist, 90)),
                "p95": float(np.percentile(found_dist, 95)),
                "p99": float(np.percentile(found_dist, 99)),
            }
        )

    thresholds = (32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512)
    summary["within_threshold"] = {
        str(t): int((found_dist <= t).sum()) if found_dist.size else 0 for t in thresholds
    }

    return summary
