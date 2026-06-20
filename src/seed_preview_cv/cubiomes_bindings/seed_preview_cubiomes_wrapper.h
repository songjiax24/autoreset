#ifndef SEED_PREVIEW_CUBIOMES_WRAPPER_H
#define SEED_PREVIEW_CUBIOMES_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* No buried treasure found within search radius (blocks). */
#define SEED_PREVIEW_NO_TREASURE (-1)

/*
 * Estimate spawn and find nearest viable buried treasure within a square search
 * radius (Chebyshev bounds on block coordinates).
 *
 * Returns 0 on success, non-zero on error.
 * out_dist is Euclidean distance in blocks, or SEED_PREVIEW_NO_TREASURE.
 */
int nearest_buried_treasure_dist(
    uint64_t seed,
    int search_radius_blocks,
    int *out_spawn_x,
    int *out_spawn_z,
    int *out_dist
);

/*
 * World spawn via cubiomes getSpawn() (slow; searches for grass/podzol near estimateSpawn).
 */
int get_world_spawn(uint64_t seed, int *out_spawn_x, int *out_spawn_z);

#define SEED_PREVIEW_BIOME_GRID_DIM 176
#define SEED_PREVIEW_BIOME_CHUNK_RADIUS 5

/*
 * Generate a 176x176 biome grid centered on the spawn chunk (11x11 chunks).
 * out_biomes must hold BIOME_GRID_DIM * BIOME_GRID_DIM integers.
 * out_x0/out_z0 receive the world block coordinate of the north-west corner.
 */
int generate_spawn_chunk_biomes(
    uint64_t seed,
    int spawn_x,
    int spawn_z,
    int *out_biomes,
    int *out_x0,
    int *out_z0
);

#ifdef __cplusplus
}
#endif

#endif
