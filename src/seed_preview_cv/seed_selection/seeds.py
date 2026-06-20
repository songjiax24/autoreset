"""Generate reproducible seed streams for dataset construction."""

from __future__ import annotations

import numpy as np

# Minecraft Java Edition world seeds are signed 64-bit integers.
MIN_MC_SEED = np.iinfo(np.int64).min  # -2^63
MAX_MC_SEED = np.iinfo(np.int64).max  # 2^63 - 1


def generate_seed_batch(num_seeds: int, rng_seed: int, start_index: int = 0) -> np.ndarray:
    """Return int64 seeds uniformly sampled from [-2^63, 2^63 - 1]."""
    if num_seeds < 0:
        raise ValueError("num_seeds must be non-negative")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    rng = np.random.default_rng(rng_seed)
    if start_index > 0:
        rng.bit_generator.advance(start_index)

    # numpy integers use an exclusive upper bound.
    return rng.integers(MIN_MC_SEED, MAX_MC_SEED + 1, size=num_seeds, dtype=np.int64)
