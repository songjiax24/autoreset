"""Build cubiomes and the seed-preview wrapper shared library."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from seed_preview_cv.common.paths import CUBIOMES_DIR

BINDINGS_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = BINDINGS_DIR
LIB_NAME = "libcubiomes_wrapper.so"
WRAPPER_SOURCES = ("wrapper_impl.c",)


def _run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build(force: bool = False) -> Path:
    if not (CUBIOMES_DIR / "finders.h").is_file():
        raise FileNotFoundError(
            f"cubiomes not found at {CUBIOMES_DIR}. Run scripts/setup_cubiomes.sh"
        )

    out_path = BINDINGS_DIR / LIB_NAME
    if out_path.exists() and not force:
        return out_path

    _run(["make", "release"], CUBIOMES_DIR)

    flags = os.environ.get("CFLAGS", "")
    compile_cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O3",
        "-fwrapv",
        f"-I{CUBIOMES_DIR}",
        f"-I{INCLUDE_DIR}",
        "-o",
        str(out_path),
        *[str(BINDINGS_DIR / src) for src in WRAPPER_SOURCES],
        str(CUBIOMES_DIR / "libcubiomes.a"),
        "-lm",
    ]
    if flags:
        compile_cmd[1:1] = flags.split()

    _run(compile_cmd, BINDINGS_DIR)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build cubiomes wrapper library")
    parser.add_argument("--force", action="store_true", help="Rebuild even if .so exists")
    args = parser.parse_args(argv)
    path = build(force=args.force)
    print(f"Built {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
