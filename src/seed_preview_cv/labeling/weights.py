"""Distance and directional weight helpers for seed labeling."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from seed_preview_cv.labeling.biome_groups import LOSS_WEIGHTS

WeightMode = Literal["isotropic", "directional"]

THETA_0 = math.radians(55.0)
THETA_1 = math.pi  # 180 degrees
THETA_RANGE = THETA_1 - THETA_0  # 125 degrees

OCEAN_D_NEAR = 24.0
OCEAN_D_FRONT = 80.0
OCEAN_D_BACK = 24.0
OCEAN_D_RANGE = OCEAN_D_FRONT - OCEAN_D_NEAR  # 56
OCEAN_D_MAX_EPS = 1e-9


def distance_weight(d: float) -> float:
    """Isotropic distance weight W_dist(d) in blocks."""
    if d <= 16:
        return 1.0
    if d < 80:
        return 1.0 - (d - 16) / 64.0
    return 0.0


def angular_decay_q(theta: float) -> float:
    """Angular decay q(theta), theta in radians in [0, pi]."""
    if theta <= THETA_0:
        return 1.0
    if theta < THETA_1:
        return (1.0 + math.cos(math.pi * (theta - THETA_0) / THETA_RANGE)) / 2.0
    return 0.0


def angular_confidence(theta: float) -> float:
    """Angular confidence A(theta) = 0.3 + 0.7 * q(theta)."""
    return 0.3 + 0.7 * angular_decay_q(theta)


def d_max(theta: float) -> float:
    """Angle-dependent maximum distance d_max(theta) = 16 + 64*q(theta)."""
    return 16.0 + 64.0 * angular_decay_q(theta)


def directional_distance_weight(d: float, theta: float) -> float:
    """Directional distance weight W_dist(d; theta)."""
    d_max_val = d_max(theta)
    if d <= 16.0:
        return 1.0
    if d_max_val <= 16.0:
        return 0.0
    if d < d_max_val:
        return 1.0 - (d - 16.0) / (d_max_val - 16.0)
    return 0.0


def effective_weight(d: float, theta: float) -> float:
    """Effective weight W_eff(d, theta) = A(theta) * W_dist(d; theta)."""
    return angular_confidence(theta) * directional_distance_weight(d, theta)


def d_max_ocean(theta: float) -> float:
    """Ocean angle-dependent max distance: 24 + 56*q(theta)."""
    return OCEAN_D_NEAR + OCEAN_D_RANGE * angular_decay_q(theta)


def ocean_directional_distance_weight(d: float, theta: float) -> float:
    """Ocean-only directional distance weight W_dist_ocean(d; theta)."""
    d_max_val = d_max_ocean(theta)
    if d <= OCEAN_D_NEAR:
        return 1.0
    if d_max_val <= OCEAN_D_NEAR + OCEAN_D_MAX_EPS:
        return 0.0
    if d < d_max_val:
        return 1.0 - (d - OCEAN_D_NEAR) / (d_max_val - OCEAN_D_NEAR)
    return 0.0


def ocean_effective_weight(d: float, theta: float) -> float:
    """Ocean effective weight W_eff_ocean = A(theta) * W_dist_ocean."""
    return angular_confidence(theta) * ocean_directional_distance_weight(d, theta)


def compute_weight_grids(
    dx: np.ndarray,
    dz: np.ndarray,
    mode: WeightMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dist, effective_weights) for a spawn-centered offset grid."""
    dist = np.hypot(dx, dz)
    if mode == "isotropic":
        eff = np.vectorize(distance_weight)(dist)
        return dist, eff

    theta = np.abs(np.arctan2(dx, dz))
    eff = np.vectorize(effective_weight)(dist, theta)
    return dist, eff


def compute_ocean_weight_grid(
    dx: np.ndarray,
    dz: np.ndarray,
    mode: WeightMode,
) -> np.ndarray:
    """Return per-cell weights for ocean scoring (directional uses W_eff_ocean)."""
    dist = np.hypot(dx, dz)
    if mode == "isotropic":
        return np.vectorize(distance_weight)(dist)

    theta = np.abs(np.arctan2(dx, dz))
    return np.vectorize(ocean_effective_weight)(dist, theta)


def loss_from_score(score: float) -> float:
    if score <= 0.0:
        return math.inf
    return -math.log(score)


def score_from_loss(loss: float) -> float:
    if math.isinf(loss):
        return 0.0
    return math.exp(-loss)


def total_loss(l_forest: float, l_ocean: float, l_beach: float) -> float:
    if any(math.isinf(x) for x in (l_forest, l_ocean, l_beach)):
        return math.inf
    return (
        LOSS_WEIGHTS["forest"] * l_forest
        + LOSS_WEIGHTS["ocean"] * l_ocean
        + LOSS_WEIGHTS["beach"] * l_beach
    )
