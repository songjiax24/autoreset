"""Compute biome-based seed scores from a spawn-centered chunk grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from seed_preview_cv.labeling.biome_groups import (
    BEACH,
    BEACH_SATURATION_K,
    BEACH_SATURATION_K_DIRECTIONAL,
    F_HIGH,
    F_LOW,
    OCEAN_ALL,
    OCEAN_TIER_BEST,
    OCEAN_TIER_MID,
    OCEAN_TIER_WEIGHTS,
    OCEAN_TIER_WORST,
)
from seed_preview_cv.labeling.weights import (
    WeightMode,
    compute_ocean_weight_grid,
    compute_weight_grids,
    loss_from_score,
    score_from_loss,
    total_loss,
)


@dataclass(frozen=True)
class SeedScores:
    seed: int
    spawn_x: int
    spawn_z: int
    grid_x0: int
    grid_z0: int
    weight_mode: str
    s_forest: float
    s_ocean: float
    s_beach: float
    s_total: float
    l_forest: float
    l_ocean: float
    l_beach: float
    l_total: float
    d_min_forest_high: float
    d_min_forest_low: float
    d_min_ocean: float
    ocean_tier_worst_ratio: float
    ocean_tier_mid_ratio: float
    ocean_tier_best_ratio: float
    s_beach_base: float
    n_ocean: int
    n_beach: int


def _min_distance(mask: np.ndarray, dist: np.ndarray) -> float:
    if not mask.any():
        return math.inf
    return float(dist[mask].min())


def _weight_at_min_distance(mask: np.ndarray, dist: np.ndarray, eff: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    masked_dist = dist[mask]
    flat_idx = int(np.argmin(masked_dist))
    return float(eff[mask].flat[flat_idx])


def _max_weighted_score(mask: np.ndarray, eff: np.ndarray, coefficient: float) -> float:
    if not mask.any():
        return 0.0
    return float(coefficient * eff[mask].max())


def _beach_saturation_k(weight_mode: WeightMode) -> float:
    if weight_mode == "directional":
        return BEACH_SATURATION_K_DIRECTIONAL
    return BEACH_SATURATION_K


def _forest_scores(
    mask_high: np.ndarray,
    mask_low: np.ndarray,
    dist: np.ndarray,
    eff: np.ndarray,
    weight_mode: WeightMode,
) -> tuple[float, float, float]:
    if weight_mode == "directional":
        s_high = _max_weighted_score(mask_high, eff, 1.0)
        s_low = _max_weighted_score(mask_low, eff, 0.3)
    else:
        s_high = 1.0 * _weight_at_min_distance(mask_high, dist, eff)
        s_low = 0.3 * _weight_at_min_distance(mask_low, dist, eff)
    return s_high, s_low, max(s_high, s_low)


def _ocean_scores(
    ids: np.ndarray,
    mask_ocean: np.ndarray,
    dist: np.ndarray,
    eff: np.ndarray,
    ocean_eff: np.ndarray,
    weight_mode: WeightMode,
) -> tuple[float, float, float, float, float]:
    n_ocean = int(mask_ocean.sum())
    if n_ocean == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mask_worst = np.isin(ids, list(OCEAN_TIER_WORST))
    mask_mid = np.isin(ids, list(OCEAN_TIER_MID))
    mask_best = np.isin(ids, list(OCEAN_TIER_BEST))

    if weight_mode == "directional":
        w_worst = float(ocean_eff[mask_worst].sum())
        w_mid = float(ocean_eff[mask_mid].sum())
        w_best = float(ocean_eff[mask_best].sum())
        w_total = w_worst + w_mid + w_best
        if w_total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        ocean_tier_worst_ratio = w_worst / w_total
        ocean_tier_mid_ratio = w_mid / w_total
        ocean_tier_best_ratio = w_best / w_total
        w_ocean = float(ocean_eff[mask_ocean].max())
    else:
        n_worst = int(mask_worst.sum())
        n_mid = int(mask_mid.sum())
        n_best = int(mask_best.sum())
        ocean_tier_worst_ratio = n_worst / n_ocean
        ocean_tier_mid_ratio = n_mid / n_ocean
        ocean_tier_best_ratio = n_best / n_ocean
        w_ocean = _weight_at_min_distance(mask_ocean, dist, eff)

    c_ocean = (
        ocean_tier_worst_ratio * OCEAN_TIER_WEIGHTS["worst"]
        + ocean_tier_mid_ratio * OCEAN_TIER_WEIGHTS["mid"]
        + ocean_tier_best_ratio * OCEAN_TIER_WEIGHTS["best"]
    )
    s_ocean = c_ocean * w_ocean
    return (
        s_ocean,
        ocean_tier_worst_ratio,
        ocean_tier_mid_ratio,
        ocean_tier_best_ratio,
        c_ocean,
    )


def compute_scores_from_grid(
    biome_grid: np.ndarray,
    *,
    seed: int,
    spawn_x: int,
    spawn_z: int,
    grid_x0: int,
    grid_z0: int,
    weight_mode: WeightMode = "isotropic",
) -> SeedScores:
    """Score a seed from a 176x176 biome ID grid."""
    height, width = biome_grid.shape
    xs = grid_x0 + np.arange(width, dtype=np.float64)
    zs = grid_z0 + np.arange(height, dtype=np.float64)
    dx = xs - spawn_x
    dz = zs[:, None] - spawn_z
    dist, eff = compute_weight_grids(dx, dz, weight_mode)
    ocean_eff = compute_ocean_weight_grid(dx, dz, weight_mode)

    ids = biome_grid
    mask_high = np.isin(ids, list(F_HIGH))
    mask_low = np.isin(ids, list(F_LOW))
    mask_ocean = np.isin(ids, list(OCEAN_ALL))
    mask_beach = np.isin(ids, list(BEACH))

    d_min_high = _min_distance(mask_high, dist)
    d_min_low = _min_distance(mask_low, dist)
    d_min_ocean = _min_distance(mask_ocean, dist)

    _, _, s_forest = _forest_scores(mask_high, mask_low, dist, eff, weight_mode)
    s_ocean, ocean_tier_worst_ratio, ocean_tier_mid_ratio, ocean_tier_best_ratio, _ = (
        _ocean_scores(ids, mask_ocean, dist, eff, ocean_eff, weight_mode)
    )

    s_beach_base = float(eff[mask_beach].sum()) if mask_beach.any() else 0.0
    beach_k = _beach_saturation_k(weight_mode)
    s_beach = 1.0 - math.exp(-s_beach_base / beach_k)

    l_forest = loss_from_score(s_forest)
    l_ocean = loss_from_score(s_ocean)
    l_beach = loss_from_score(s_beach)
    l_total = total_loss(l_forest, l_ocean, l_beach)
    s_total = score_from_loss(l_total)

    return SeedScores(
        seed=seed,
        spawn_x=spawn_x,
        spawn_z=spawn_z,
        grid_x0=grid_x0,
        grid_z0=grid_z0,
        weight_mode=weight_mode,
        s_forest=s_forest,
        s_ocean=s_ocean,
        s_beach=s_beach,
        s_total=s_total,
        l_forest=l_forest,
        l_ocean=l_ocean,
        l_beach=l_beach,
        l_total=l_total,
        d_min_forest_high=d_min_high,
        d_min_forest_low=d_min_low,
        d_min_ocean=d_min_ocean,
        ocean_tier_worst_ratio=ocean_tier_worst_ratio,
        ocean_tier_mid_ratio=ocean_tier_mid_ratio,
        ocean_tier_best_ratio=ocean_tier_best_ratio,
        s_beach_base=s_beach_base,
        n_ocean=int(mask_ocean.sum()),
        n_beach=int(mask_beach.sum()),
    )


def scores_to_row(scores: SeedScores) -> dict:
    def _fmt_dist(v: float) -> float | None:
        return None if math.isinf(v) else v

    def _fmt_loss(v: float) -> float | str:
        return "inf" if math.isinf(v) else v

    return {
        "seed": scores.seed,
        "spawn_x": scores.spawn_x,
        "spawn_z": scores.spawn_z,
        "grid_x0": scores.grid_x0,
        "grid_z0": scores.grid_z0,
        "weight_mode": scores.weight_mode,
        "s_forest": scores.s_forest,
        "s_ocean": scores.s_ocean,
        "s_beach": scores.s_beach,
        "s_total": scores.s_total,
        "l_forest": _fmt_loss(scores.l_forest),
        "l_ocean": _fmt_loss(scores.l_ocean),
        "l_beach": _fmt_loss(scores.l_beach),
        "l_total": _fmt_loss(scores.l_total),
        "d_min_forest_high": _fmt_dist(scores.d_min_forest_high),
        "d_min_forest_low": _fmt_dist(scores.d_min_forest_low),
        "d_min_ocean": _fmt_dist(scores.d_min_ocean),
        "ocean_tier_worst_ratio": scores.ocean_tier_worst_ratio,
        "ocean_tier_mid_ratio": scores.ocean_tier_mid_ratio,
        "ocean_tier_best_ratio": scores.ocean_tier_best_ratio,
        "s_beach_base": scores.s_beach_base,
        "n_ocean": scores.n_ocean,
        "n_beach": scores.n_beach,
    }
