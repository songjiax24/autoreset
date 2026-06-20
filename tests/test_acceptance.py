"""Tests for acceptance probability and sampling."""

import math

import numpy as np
import pandas as pd

from seed_preview_cv.seed_selection.acceptance import (
    AcceptanceConfig,
    acceptance_probs,
    apply_acceptance,
    simulate_acceptance,
    treasure_accept_prob,
)


def test_treasure_accept_prob_not_found():
    cfg = AcceptanceConfig()
    assert treasure_accept_prob(-1, cfg) == cfg.p_min


def test_treasure_accept_prob_at_d0():
    cfg = AcceptanceConfig(p_min=0.02, p_max=1.0, d0=128.0, scale=32.0)
    p = treasure_accept_prob(128, cfg)
    assert abs(p - 0.51) < 0.001


def test_treasure_accept_prob_near_max():
    cfg = AcceptanceConfig()
    assert treasure_accept_prob(0, cfg) > 0.95


def test_simulate_acceptance_on_sample_data():
    df = pd.DataFrame(
        {
            "nearest_treasure_dist": [0, 128, -1, 256],
        }
    )
    summary = simulate_acceptance(df)
    assert summary["total"] == 4
    assert summary["expected_acceptance_rate"] > 0
    assert summary["expected_accepted"] == sum(
        treasure_accept_prob(d) for d in df["nearest_treasure_dist"]
    )


def test_apply_acceptance_reproducible():
    df = pd.DataFrame(
        {
            "seed": [1, 2, 3, 4],
            "estimated_spawn_x": [0, 0, 0, 0],
            "estimated_spawn_z": [0, 0, 0, 0],
            "nearest_treasure_dist": [0, 64, 200, -1],
        }
    )
    cfg = AcceptanceConfig(rng_seed=99)
    a = apply_acceptance(df, cfg)
    b = apply_acceptance(df, cfg)
    assert a["accepted"].tolist() == b["accepted"].tolist()
    assert np.allclose(a["acceptance_prob"], b["acceptance_prob"])
