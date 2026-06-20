"""Acceptance probability for dataset seed selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AcceptanceConfig:
    p_min: float = 0.02
    p_max: float = 1.0
    d0: float = 128.0
    scale: float = 32.0
    rng_seed: int = 42

    @classmethod
    def from_mapping(cls, data: dict) -> AcceptanceConfig:
        return cls(
            p_min=float(data.get("p_min", 0.02)),
            p_max=float(data.get("p_max", 1.0)),
            d0=float(data.get("d0", 128.0)),
            scale=float(data.get("scale", 32.0)),
            rng_seed=int(data.get("rng_seed", 42)),
        )


def treasure_accept_prob(distance: int, config: AcceptanceConfig = AcceptanceConfig()) -> float:
    """Logistic acceptance probability from nearest treasure distance (blocks)."""
    if distance < 0:
        return config.p_min

    logistic = 1.0 / (1.0 + math.exp((distance - config.d0) / config.scale))
    return config.p_min + (config.p_max - config.p_min) * logistic


def acceptance_probs(distances: np.ndarray, config: AcceptanceConfig = AcceptanceConfig()) -> np.ndarray:
    """Vectorized acceptance probabilities for a distance array."""
    distances = np.asarray(distances)
    probs = np.full(distances.shape, config.p_min, dtype=np.float64)
    found = distances >= 0
    if found.any():
        d = distances[found].astype(np.float64)
        probs[found] = config.p_min + (config.p_max - config.p_min) / (
            1.0 + np.exp((d - config.d0) / config.scale)
        )
    return probs


def simulate_acceptance(df: pd.DataFrame, config: AcceptanceConfig = AcceptanceConfig()) -> dict:
    """Estimate mean acceptance rate without sampling."""
    dist = df["nearest_treasure_dist"].to_numpy()
    probs = acceptance_probs(dist, config)
    found = dist >= 0

    summary = {
        "total": int(len(dist)),
        "expected_accepted": float(probs.sum()),
        "expected_acceptance_rate": float(probs.mean()) if len(probs) else 0.0,
        "found_count": int(found.sum()),
        "not_found_count": int((~found).sum()),
        "params": {
            "p_min": config.p_min,
            "p_max": config.p_max,
            "d0": config.d0,
            "scale": config.scale,
        },
    }

    if found.any():
        found_probs = probs[found]
        found_dist = dist[found]
        summary["found_expected_accepted"] = float(found_probs.sum())
        summary["found_expected_rate"] = float(found_probs.mean())
        summary["not_found_expected_accepted"] = float(probs[~found].sum())
        thresholds = (32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512)
        summary["expected_within_threshold"] = {
            str(t): float(found_probs[found_dist <= t].sum()) for t in thresholds
        }

    return summary


def apply_acceptance(
    df: pd.DataFrame,
    config: AcceptanceConfig = AcceptanceConfig(),
) -> pd.DataFrame:
    """Bernoulli sample seeds using per-seed acceptance probabilities."""
    dist = df["nearest_treasure_dist"].to_numpy()
    probs = acceptance_probs(dist, config)
    rng = np.random.default_rng(config.rng_seed)
    accepted_mask = rng.random(len(probs)) < probs

    out = df.copy()
    out["acceptance_prob"] = probs
    out["accepted"] = accepted_mask
    return out


def select_seeds(df: pd.DataFrame, config: AcceptanceConfig = AcceptanceConfig()) -> pd.DataFrame:
    """Return accepted rows with selection metadata."""
    sampled = apply_acceptance(df, config)
    return sampled[sampled["accepted"]].reset_index(drop=True)


SEED_LIST_COLUMNS = (
    "seed",
    "estimated_spawn_x",
    "estimated_spawn_z",
    "nearest_treasure_dist",
    "acceptance_prob",
)
