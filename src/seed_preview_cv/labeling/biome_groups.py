"""Biome ID sets for seed labeling (Minecraft 1.16.1, biomes.h)."""

from __future__ import annotations

from typing import FrozenSet

# Enum order in third_party/cubiomes/biomes.h (MC 1.16.1).
# Oceans: 0 ocean, 10 frozen_ocean, 24 deep_ocean,
# 44 warm_ocean, 45 lukewarm_ocean, 46 cold_ocean,
# 47 deep_warm_ocean, 48 deep_lukewarm_ocean,
# 49 deep_cold_ocean, 50 deep_frozen_ocean.

# (1) High-quality forest F_high
F_HIGH: FrozenSet[int] = frozenset(
    {
        4,  # forest
        132,  # flower_forest
        18,  # wooded_hills
        29,  # dark_forest
        157,  # dark_forest_hills
        27,  # birch_forest
        28,  # birch_forest_hills
        155,  # tall_birch_forest
        156,  # tall_birch_hills
        5,  # taiga
        19,  # taiga_hills
        133,  # taiga_mountains
        30,  # snowy_taiga
        31,  # snowy_taiga_hills
        158,  # snowy_taiga_mountains
        32,  # giant_tree_taiga
        33,  # giant_tree_taiga_hills
        160,  # giant_spruce_taiga
        161,  # giant_spruce_taiga_hills
        21,  # jungle
        22,  # jungle_hills
        23,  # jungle_edge (sparse_jungle)
        149,  # modified_jungle
        151,  # modified_jungle_edge
    }
)

# (2) Low-quality wood F_low
F_LOW: FrozenSet[int] = frozenset(
    {
        1,  # plains
        129,  # sunflower_plains
        35,  # savanna
        36,  # savanna_plateau
        163,  # shattered_savanna
        164,  # shattered_savanna_plateau
        6,  # swamp
        134,  # swamp_hills
    }
)

# (3) Ocean worst tier (weight 0.5)
OCEAN_TIER_WORST: FrozenSet[int] = frozenset(
    {
        44,  # warm_ocean
        45,  # lukewarm_ocean
    }
)

# (4) Ocean middle tier (weight 0.8)
OCEAN_TIER_MID: FrozenSet[int] = frozenset(
    {
        0,  # ocean
        10,  # frozen_ocean
        46,  # cold_ocean
        47,  # deep_warm_ocean
        48,  # deep_lukewarm_ocean
    }
)

# (5) Ocean best tier (weight 1.0)
OCEAN_TIER_BEST: FrozenSet[int] = frozenset(
    {
        24,  # deep_ocean
        49,  # deep_cold_ocean
        50,  # deep_frozen_ocean
    }
)

OCEAN_ALL: FrozenSet[int] = OCEAN_TIER_WORST | OCEAN_TIER_MID | OCEAN_TIER_BEST

# (6) Beach
BEACH: FrozenSet[int] = frozenset({16})  # beach

OCEAN_TIER_WEIGHTS = {
    "worst": 0.5,
    "mid": 0.8,
    "best": 1.0,
}

LOSS_WEIGHTS = {
    "forest": 0.4,
    "ocean": 0.4,
    "beach": 0.2,
}

BEACH_SATURATION_K = 1000.0
BEACH_SATURATION_K_DIRECTIONAL = 800.0

BIOME_GRID_DIM = 176
BIOME_CHUNK_RADIUS = 5
