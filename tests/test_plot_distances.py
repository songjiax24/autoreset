"""Tests for distance histogram plotting."""

from pathlib import Path

import pandas as pd

from seed_preview_cv.seed_selection.plot_distances import plot_distance_histogram


def test_plot_distance_histogram(tmp_path: Path):
    df = pd.DataFrame(
        {
            "seed": [1, 2, 3, 4, 5],
            "estimated_spawn_x": [0, 0, 0, 0, 0],
            "estimated_spawn_z": [0, 0, 0, 0, 0],
            "nearest_treasure_dist": [16, 48, 80, -1, 120],
        }
    )
    out = tmp_path / "hist.png"
    saved = plot_distance_histogram(df, out, search_radius_blocks=512, bin_width=16)
    assert saved.is_file()
    assert saved.stat().st_size > 0
