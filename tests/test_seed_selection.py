"""Tests for seed selection helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from seed_preview_cv.cubiomes_bindings import CubiomesNotBuiltError, wrapper_library_path
from seed_preview_cv.cubiomes_bindings.ffi import get_world_spawn, nearest_buried_treasure_dist
from seed_preview_cv.seed_selection.distances import summarize_distances
from seed_preview_cv.seed_selection.seeds import MIN_MC_SEED, MAX_MC_SEED, generate_seed_batch


def test_generate_seed_batch_reproducible():
    a = generate_seed_batch(10, 42)
    b = generate_seed_batch(10, 42)
    assert np.array_equal(a, b)


def test_generate_seed_batch_chunked_matches_full_stream():
    full = generate_seed_batch(20, 7)
    second_chunk = generate_seed_batch(10, 7, start_index=10)
    assert np.array_equal(full[10:], second_chunk)


def test_generate_seed_batch_range():
    seeds = generate_seed_batch(5000, 123)
    assert seeds.min() >= MIN_MC_SEED
    assert seeds.max() <= MAX_MC_SEED
    assert (seeds < 0).any()


def test_summarize_distances():
    df = pd.DataFrame(
        {
            "seed": [1, 2, 3, 4],
            "estimated_spawn_x": [0, 0, 0, 0],
            "estimated_spawn_z": [0, 0, 0, 0],
            "nearest_treasure_dist": [10, 80, -1, 40],
        }
    )
    summary = summarize_distances(df)
    assert summary["total"] == 4
    assert summary["found_count"] == 3
    assert summary["not_found_count"] == 1
    assert summary["within_threshold"]["80"] == 3


@pytest.mark.skipif(not wrapper_library_path().is_file(), reason="wrapper not built")
def test_get_world_spawn_smoke():
    spawn_x, spawn_z = get_world_spawn(3055141959546)
    assert isinstance(spawn_x, int)
    assert isinstance(spawn_z, int)


@pytest.mark.skipif(not wrapper_library_path().is_file(), reason="wrapper not built")
def test_nearest_buried_treasure_dist_smoke():
    result = nearest_buried_treasure_dist(3055141959546, search_radius_blocks=512)
    assert isinstance(result.spawn_x, int)
    assert isinstance(result.spawn_z, int)
    assert result.treasure_dist >= -1


def test_wrapper_missing_raises():
    if wrapper_library_path().is_file():
        pytest.skip("wrapper already built")
    with pytest.raises(CubiomesNotBuiltError):
        nearest_buried_treasure_dist(0, 64)
