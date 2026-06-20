"""Tests for labeling weights and scoring."""

import math

import numpy as np

from seed_preview_cv.labeling.biome_groups import BEACH, F_HIGH, F_LOW, OCEAN_TIER_BEST, OCEAN_TIER_WORST
from seed_preview_cv.labeling.scoring import compute_scores_from_grid
from seed_preview_cv.labeling.weights import (
    angular_confidence,
    angular_decay_q,
    d_max,
    d_max_ocean,
    directional_distance_weight,
    distance_weight,
    effective_weight,
    loss_from_score,
    ocean_directional_distance_weight,
    ocean_effective_weight,
    total_loss,
)


def test_distance_weight():
    assert distance_weight(0) == 1.0
    assert distance_weight(16) == 1.0
    assert distance_weight(48) == 0.5
    assert distance_weight(80) == 0.0
    assert distance_weight(200) == 0.0


def test_directional_weight_key_angles():
    theta_0 = 0.0
    theta_55 = math.radians(55.0)
    theta_180 = math.pi

    assert angular_decay_q(theta_0) == 1.0
    assert angular_decay_q(theta_55) == 1.0
    assert angular_decay_q(theta_180) == 0.0

    assert angular_confidence(theta_0) == 1.0
    assert angular_confidence(theta_55) == 1.0
    assert angular_confidence(theta_180) == 0.3

    assert d_max(theta_0) == 80.0
    assert d_max(theta_55) == 80.0
    assert d_max(theta_180) == 16.0

    assert directional_distance_weight(16.0, theta_0) == 1.0
    assert directional_distance_weight(17.0, theta_180) == 0.0
    assert effective_weight(48.0, theta_0) == 0.5
    assert effective_weight(16.0, theta_180) == 0.3


def test_ocean_directional_distance_weight_extended_near_field():
    theta_180 = math.pi
    assert d_max_ocean(theta_180) == 24.0
    assert ocean_directional_distance_weight(20.0, theta_180) == 1.0
    assert ocean_directional_distance_weight(25.0, theta_180) == 0.0
    assert ocean_effective_weight(20.0, theta_180) == 0.3
    assert directional_distance_weight(20.0, theta_180) == 0.0


def test_directional_ocean_extended_near_increases_score_behind_camera():
    grid = np.full((5, 5), 1, dtype=np.int32)
    grid[0, 2] = next(iter(OCEAN_TIER_BEST))  # behind, d=2 blocks

    isotropic = compute_scores_from_grid(
        grid,
        seed=7,
        spawn_x=2,
        spawn_z=2,
        grid_x0=0,
        grid_z0=0,
        weight_mode="isotropic",
    )
    directional = compute_scores_from_grid(
        grid,
        seed=7,
        spawn_x=2,
        spawn_z=2,
        grid_x0=0,
        grid_z0=0,
        weight_mode="directional",
    )
    assert isotropic.s_ocean == 1.0
    assert directional.s_ocean == 0.3


def test_directional_forest_uses_max_weight_not_min_distance():
    grid = np.full((5, 5), 1, dtype=np.int32)
    grid[0, 2] = next(iter(F_HIGH))  # behind camera, within 16 blocks

    isotropic = compute_scores_from_grid(
        grid,
        seed=4,
        spawn_x=2,
        spawn_z=2,
        grid_x0=0,
        grid_z0=0,
        weight_mode="isotropic",
    )
    directional = compute_scores_from_grid(
        grid,
        seed=4,
        spawn_x=2,
        spawn_z=2,
        grid_x0=0,
        grid_z0=0,
        weight_mode="directional",
    )
    assert isotropic.s_forest == 1.0
    assert directional.s_forest == 0.3


def test_directional_ocean_tier_ratios_use_weights():
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[0, 0] = next(iter(OCEAN_TIER_BEST))  # behind, low W_eff
    grid[2, 2] = next(iter(OCEAN_TIER_WORST))  # ahead, higher W_eff

    scores = compute_scores_from_grid(
        grid,
        seed=6,
        spawn_x=1,
        spawn_z=1,
        grid_x0=0,
        grid_z0=0,
        weight_mode="directional",
    )
    assert scores.ocean_tier_worst_ratio > scores.ocean_tier_best_ratio


def test_directional_beach_uses_lower_saturation_k():
    from seed_preview_cv.labeling.biome_groups import BEACH_SATURATION_K, BEACH_SATURATION_K_DIRECTIONAL

    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = next(iter(BEACH))

    isotropic = compute_scores_from_grid(
        grid,
        seed=5,
        spawn_x=1,
        spawn_z=1,
        grid_x0=0,
        grid_z0=0,
        weight_mode="isotropic",
    )
    directional = compute_scores_from_grid(
        grid,
        seed=5,
        spawn_x=1,
        spawn_z=1,
        grid_x0=0,
        grid_z0=0,
        weight_mode="directional",
    )
    assert isotropic.s_beach_base == directional.s_beach_base
    expected_iso = 1.0 - math.exp(-isotropic.s_beach_base / BEACH_SATURATION_K)
    expected_dir = 1.0 - math.exp(-directional.s_beach_base / BEACH_SATURATION_K_DIRECTIONAL)
    assert isotropic.s_beach == expected_iso
    assert directional.s_beach == expected_dir
    assert directional.s_beach > isotropic.s_beach


def test_loss_from_score():
    assert loss_from_score(0) == math.inf
    assert loss_from_score(1.0) == 0.0


def test_compute_scores_synthetic():
    grid = np.zeros((3, 3), dtype=np.int32)
    grid[1, 1] = next(iter(F_HIGH))  # forest at spawn
    grid[0, 0] = next(iter(OCEAN_TIER_BEST))  # deep_ocean
    grid[2, 2] = next(iter(BEACH))

    scores = compute_scores_from_grid(
        grid,
        seed=1,
        spawn_x=1,
        spawn_z=1,
        grid_x0=0,
        grid_z0=0,
    )
    assert scores.s_forest > 0
    assert scores.s_ocean > 0
    assert scores.s_beach > 0
    assert scores.l_total < math.inf
    assert scores.s_total == math.exp(-scores.l_total)


def test_score_from_loss():
    from seed_preview_cv.labeling.weights import score_from_loss

    assert score_from_loss(0.0) == 1.0
    assert score_from_loss(math.inf) == 0.0
    assert score_from_loss(1.0) == math.exp(-1.0)


def test_compute_scores_no_ocean_is_infinite():
    grid = np.full((3, 3), next(iter(F_LOW)), dtype=np.int32)
    grid[1, 1] = next(iter(BEACH))
    scores = compute_scores_from_grid(
        grid,
        seed=2,
        spawn_x=1,
        spawn_z=1,
        grid_x0=0,
        grid_z0=0,
    )
    assert scores.s_ocean == 0.0
    assert math.isinf(scores.l_ocean)
    assert math.isinf(scores.l_total)
    assert scores.s_total == 0.0


def test_total_loss_with_infinity():
    assert math.isinf(total_loss(math.inf, 0.5, 0.3))


def test_format_label_value():
    from seed_preview_cv.labeling.annotate_labels import format_label_value

    assert format_label_value("s_forest", 0.5) == "0.500"
    assert format_label_value("s_total", 0.0) == "0.000"
