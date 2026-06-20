"""Validate biome ID sets against cubiomes biomes.h (MC 1.16.1)."""

from seed_preview_cv.labeling.biome_groups import (
    BEACH,
    F_HIGH,
    F_LOW,
    OCEAN_ALL,
    OCEAN_TIER_BEST,
    OCEAN_TIER_MID,
    OCEAN_TIER_WORST,
)

# Numeric IDs from third_party/cubiomes/biomes.h enum BiomeID.
MC_1_16_OCEAN_IDS = frozenset(
    {
        0,  # ocean
        10,  # frozen_ocean
        24,  # deep_ocean
        44,  # warm_ocean
        45,  # lukewarm_ocean
        46,  # cold_ocean
        47,  # deep_warm_ocean
        48,  # deep_lukewarm_ocean
        49,  # deep_cold_ocean
        50,  # deep_frozen_ocean
    }
)

MC_1_16_END_IDS = frozenset(
    {
        40,  # small_end_islands
        41,  # end_midlands
        42,  # end_highlands
        43,  # end_barrens
    }
)


def test_ocean_tiers_partition_mc_1_16_oceans():
    assert OCEAN_TIER_WORST == frozenset({44, 45})
    assert OCEAN_TIER_MID == frozenset({0, 10, 46, 47, 48})
    assert OCEAN_TIER_BEST == frozenset({24, 49, 50})
    assert OCEAN_ALL == MC_1_16_OCEAN_IDS
    assert not (OCEAN_TIER_WORST & OCEAN_TIER_MID)
    assert not (OCEAN_TIER_WORST & OCEAN_TIER_BEST)
    assert not (OCEAN_TIER_MID & OCEAN_TIER_BEST)
    assert not (OCEAN_ALL & MC_1_16_END_IDS)


F_HIGH_JUNGLE_IDS = frozenset(
    {
        21,  # jungle
        22,  # jungle_hills
        23,  # jungle_edge
        149,  # modified_jungle
        151,  # modified_jungle_edge
    }
)

MC_1_16_JUNGLE_IDS = F_HIGH_JUNGLE_IDS | frozenset(
    {
        168,  # bamboo_jungle
        169,  # bamboo_jungle_hills
    }
)


def test_forest_ids_include_jungle_biomes_in_f_high():
    assert F_HIGH_JUNGLE_IDS <= F_HIGH
    assert 168 not in F_HIGH
    assert 169 not in F_HIGH


def test_forest_ids_exclude_end_biomes():
    labeled = F_HIGH | F_LOW
    assert not (labeled & MC_1_16_END_IDS)
    assert 160 in F_HIGH  # giant_spruce_taiga
    assert 161 in F_HIGH  # giant_spruce_taiga_hills
    assert 136 not in F_HIGH
    assert 137 not in F_HIGH


def test_beach_id():
    assert BEACH == frozenset({16})
