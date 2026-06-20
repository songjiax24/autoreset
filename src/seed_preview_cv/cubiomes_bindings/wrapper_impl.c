#include "seed_preview_cubiomes_wrapper.h"

#include "finders.h"
#include "generator.h"
#include "rng.h"

#include <math.h>
#include <stdlib.h>

static double block_distance(int x0, int z0, int x1, int z1)
{
    const double dx = (double)(x1 - x0);
    const double dz = (double)(z1 - z0);
    return sqrt(dx * dx + dz * dz);
}

int nearest_buried_treasure_dist(
    uint64_t seed,
    int search_radius_blocks,
    int *out_spawn_x,
    int *out_spawn_z,
    int *out_dist)
{
    if (!out_spawn_x || !out_spawn_z || !out_dist || search_radius_blocks < 0) {
        return 1;
    }

    Generator g;
    setupGenerator(&g, MC_1_16_1, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    uint64_t rng;
    const Pos spawn = estimateSpawn(&g, &rng);
    *out_spawn_x = spawn.x;
    *out_spawn_z = spawn.z;

    StructureConfig sconf;
    if (!getStructureConfig(Treasure, MC_1_16_1, &sconf)) {
        *out_dist = SEED_PREVIEW_NO_TREASURE;
        return 2;
    }

    const int x0 = spawn.x - search_radius_blocks;
    const int z0 = spawn.z - search_radius_blocks;
    const int x1 = spawn.x + search_radius_blocks;
    const int z1 = spawn.z + search_radius_blocks;

    const double blocks_per_region = (double)sconf.regionSize * 16.0;
    const int rx0 = (int)floor(x0 / blocks_per_region);
    const int rz0 = (int)floor(z0 / blocks_per_region);
    const int rx1 = (int)ceil(x1 / blocks_per_region);
    const int rz1 = (int)ceil(z1 / blocks_per_region);

    int best_dist = SEED_PREVIEW_NO_TREASURE;

    for (int rz = rz0; rz <= rz1; rz++) {
        for (int rx = rx0; rx <= rx1; rx++) {
            Pos pos;
            if (!getStructurePos(Treasure, MC_1_16_1, seed, rx, rz, &pos)) {
                continue;
            }
            if (pos.x < x0 || pos.x > x1 || pos.z < z0 || pos.z > z1) {
                continue;
            }
            if (!isViableStructurePos(Treasure, &g, pos.x, pos.z, 0)) {
                continue;
            }

            const int dist = (int)block_distance(spawn.x, spawn.z, pos.x, pos.z);
            if (best_dist < 0 || dist < best_dist) {
                best_dist = dist;
            }
        }
    }

    *out_dist = best_dist;
    return 0;
}

int get_world_spawn(uint64_t seed, int *out_spawn_x, int *out_spawn_z)
{
    if (!out_spawn_x || !out_spawn_z) {
        return 1;
    }

    Generator g;
    setupGenerator(&g, MC_1_16_1, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    const Pos spawn = getSpawn(&g);
    *out_spawn_x = spawn.x;
    *out_spawn_z = spawn.z;
    return 0;
}

int generate_spawn_chunk_biomes(
    uint64_t seed,
    int spawn_x,
    int spawn_z,
    int *out_biomes,
    int *out_x0,
    int *out_z0)
{
    if (!out_biomes || !out_x0 || !out_z0) {
        return 1;
    }

    Generator g;
    setupGenerator(&g, MC_1_16_1, 0);
    applySeed(&g, DIM_OVERWORLD, seed);

    const int cx = floordiv(spawn_x, 16);
    const int cz = floordiv(spawn_z, 16);
    const int x0 = (cx - SEED_PREVIEW_BIOME_CHUNK_RADIUS) * 16;
    const int z0 = (cz - SEED_PREVIEW_BIOME_CHUNK_RADIUS) * 16;

    Range r;
    r.scale = 1;
    r.x = x0;
    r.z = z0;
    r.sx = SEED_PREVIEW_BIOME_GRID_DIM;
    r.sz = SEED_PREVIEW_BIOME_GRID_DIM;
    r.y = 63;
    r.sy = 1;

    int *cache = allocCache(&g, r);
    if (!cache) {
        return 2;
    }
    if (genBiomes(&g, cache, r) != 0) {
        free(cache);
        return 3;
    }

    for (int iz = 0; iz < r.sz; iz++) {
        for (int ix = 0; ix < r.sx; ix++) {
            out_biomes[iz * r.sx + ix] = cache[iz * r.sx + ix];
        }
    }

    free(cache);
    *out_x0 = x0;
    *out_z0 = z0;
    return 0;
}
