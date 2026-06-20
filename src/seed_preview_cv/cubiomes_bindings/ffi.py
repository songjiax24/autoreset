"""ctypes bindings for libcubiomes_wrapper."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

BINDINGS_DIR = Path(__file__).resolve().parent
NO_TREASURE_DISTANCE = -1

_LIB: ctypes.CDLL | None = None


class CubiomesNotBuiltError(RuntimeError):
    """Raised when the wrapper shared library has not been built."""


def wrapper_library_path() -> Path:
    return BINDINGS_DIR / "libcubiomes_wrapper.so"


def load_library(path: Path | None = None) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = path or wrapper_library_path()
    if not lib_path.is_file():
        raise CubiomesNotBuiltError(
            f"Missing {lib_path}. Run: uv run python -m seed_preview_cv.cubiomes_bindings.build"
        )

    lib = ctypes.CDLL(str(lib_path))
    lib.nearest_buried_treasure_dist.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.nearest_buried_treasure_dist.restype = ctypes.c_int
    lib.generate_spawn_chunk_biomes.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.generate_spawn_chunk_biomes.restype = ctypes.c_int
    lib.get_world_spawn.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.get_world_spawn.restype = ctypes.c_int
    _LIB = lib
    return lib


@dataclass(frozen=True)
class TreasureDistanceResult:
    seed: int
    spawn_x: int
    spawn_z: int
    treasure_dist: int  # blocks, or NO_TREASURE_DISTANCE


def nearest_buried_treasure_dist(
    seed: int,
    search_radius_blocks: int,
    *,
    lib_path: Path | None = None,
) -> TreasureDistanceResult:
    """Return estimated spawn and nearest viable buried treasure distance."""
    lib = load_library(lib_path)
    seed_u64 = ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF)

    spawn_x = ctypes.c_int()
    spawn_z = ctypes.c_int()
    dist = ctypes.c_int()

    rc = lib.nearest_buried_treasure_dist(
        seed_u64,
        ctypes.c_int(search_radius_blocks),
        ctypes.byref(spawn_x),
        ctypes.byref(spawn_z),
        ctypes.byref(dist),
    )
    if rc != 0:
        raise RuntimeError(f"nearest_buried_treasure_dist failed with code {rc}")

    return TreasureDistanceResult(
        seed=seed,
        spawn_x=spawn_x.value,
        spawn_z=spawn_z.value,
        treasure_dist=dist.value,
    )


def get_world_spawn(
    seed: int,
    *,
    lib_path: Path | None = None,
) -> tuple[int, int]:
    """Return (spawn_x, spawn_z) using cubiomes getSpawn() for MC 1.16.1."""
    lib = load_library(lib_path)
    seed_u64 = ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF)
    spawn_x = ctypes.c_int()
    spawn_z = ctypes.c_int()
    rc = lib.get_world_spawn(
        seed_u64,
        ctypes.byref(spawn_x),
        ctypes.byref(spawn_z),
    )
    if rc != 0:
        raise RuntimeError(f"get_world_spawn failed with code {rc}")
    return spawn_x.value, spawn_z.value


def generate_spawn_chunk_biomes(
    seed: int,
    spawn_x: int,
    spawn_z: int,
    *,
    grid_dim: int = 176,
    lib_path: Path | None = None,
) -> tuple["np.ndarray", int, int]:
    """Return (biome_grid[H,W], grid_x0, grid_z0) for the spawn chunk neighborhood."""
    import numpy as np

    lib = load_library(lib_path)
    seed_u64 = ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF)
    bufsize = grid_dim * grid_dim
    out_biomes = (ctypes.c_int * bufsize)()
    out_x0 = ctypes.c_int()
    out_z0 = ctypes.c_int()

    rc = lib.generate_spawn_chunk_biomes(
        seed_u64,
        ctypes.c_int(spawn_x),
        ctypes.c_int(spawn_z),
        out_biomes,
        ctypes.byref(out_x0),
        ctypes.byref(out_z0),
    )
    if rc != 0:
        raise RuntimeError(f"generate_spawn_chunk_biomes failed with code {rc}")

    grid = np.frombuffer(out_biomes, dtype=np.int32, count=bufsize).reshape(grid_dim, grid_dim)
    return grid.copy(), out_x0.value, out_z0.value
