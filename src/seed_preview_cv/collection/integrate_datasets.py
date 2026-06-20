"""Integrate mod-collected dataset folders into the project data layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import (
    COLLECTION_INDEX_CSV,
    COLLECTION_MANIFEST_JSON,
    COLLECTION_SOURCES_CSV,
    COLLECTION_SPAWNS_CSV,
    DATA_SCREENSHOTS_DIR,
    PROJECT_ROOT,
    SEEDS_TXT,
)


def default_dataset_dirs(project_root: Path) -> list[Path]:
    dirs = [project_root / f"dataset_{i}" for i in range(1, 8)]
    return [d for d in dirs if d.is_dir()]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def integrate_collection(
    dataset_dirs: list[Path],
    screenshots_dir: Path,
    spawns_csv: Path,
    index_csv: Path,
    sources_csv: Path,
    manifest_json: Path,
    seeds_txt: Path | None = None,
    progress: bool = True,
) -> dict:
    """Merge dataset_* folders into data/screenshots and interim CSVs."""
    rows: list[dict] = []
    per_dataset: list[dict] = []
    missing_png_by_dataset: dict[str, list[int]] = {}

    for dataset_dir in dataset_dirs:
        spawns_path = dataset_dir / "spawns.csv"
        if not spawns_path.is_file():
            raise FileNotFoundError(f"Missing spawns.csv in {dataset_dir}")

        df = pd.read_csv(spawns_path)
        required = {"seed", "x", "z"}
        if not required.issubset(df.columns):
            raise ValueError(f"{spawns_path} must contain columns: seed,x,z")

        pngs = {p.stem: p for p in dataset_dir.glob("*.png")}
        missing_png: list[int] = []
        kept = 0
        for record in df.itertuples(index=False):
            seed = int(record.seed)
            seed_str = str(seed)
            png_path = pngs.get(seed_str)
            if png_path is None:
                missing_png.append(seed)
                continue
            rows.append(
                {
                    "seed": seed,
                    "x": int(record.x),
                    "z": int(record.z),
                    "source_dataset": dataset_dir.name,
                    "screenshot_src": str(png_path.resolve()),
                }
            )
            kept += 1

        missing_png_by_dataset[dataset_dir.name] = missing_png
        per_dataset.append(
            {
                "dataset": dataset_dir.name,
                "spawns_rows": int(len(df)),
                "screenshots": int(len(pngs)),
                "integrated": kept,
                "missing_screenshot": len(missing_png),
            }
        )

    if not rows:
        raise ValueError("No records with both spawn coordinates and screenshots")

    merged = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if merged["seed"].duplicated().any():
        dupes = merged[merged["seed"].duplicated(keep=False)]
        raise ValueError(f"Duplicate seeds across datasets: {dupes['seed'].tolist()[:10]}")

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    iterator = merged.itertuples(index=False)
    if progress:
        iterator = tqdm(list(merged.itertuples(index=False)), desc="link screenshots")

    for row in iterator:
        dst = screenshots_dir / f"{int(row.seed)}.png"
        _link_or_copy(Path(row.screenshot_src), dst)

    spawns_out = merged[["seed", "x", "z"]]
    spawns_csv.parent.mkdir(parents=True, exist_ok=True)
    spawns_out.to_csv(spawns_csv, index=False)

    rel_prefix = Path("data/screenshots")
    index_rows = [
        {
            "seed": int(row.seed),
            "spawn_x": int(row.x),
            "spawn_z": int(row.z),
            "screenshot_path": str(rel_prefix / f"{int(row.seed)}.png"),
        }
        for row in merged.itertuples(index=False)
    ]
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)

    sources_out = merged[["seed", "x", "z", "source_dataset"]]
    sources_out.to_csv(sources_csv, index=False)

    collected_seeds = set(merged["seed"].astype(int))
    seeds_txt_missing: list[int] = []
    seeds_txt_count: int | None = None
    if seeds_txt is not None and seeds_txt.is_file():
        txt_seeds = [int(line.strip()) for line in seeds_txt.read_text().splitlines() if line.strip()]
        seeds_txt_count = len(txt_seeds)
        seeds_txt_missing = sorted(set(txt_seeds) - collected_seeds)

    manifest = {
        "dataset_dirs": [str(d.resolve()) for d in dataset_dirs],
        "screenshots_dir": str(screenshots_dir.resolve()),
        "spawns_csv": str(spawns_csv.resolve()),
        "collection_index_csv": str(index_csv.resolve()),
        "integrated_count": int(len(merged)),
        "per_dataset": per_dataset,
        "missing_screenshot_by_dataset": {
            k: v for k, v in missing_png_by_dataset.items() if v
        },
        "seeds_txt_count": seeds_txt_count,
        "seeds_txt_missing_count": len(seeds_txt_missing),
        "seeds_txt_missing": seeds_txt_missing,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate dataset_1..dataset_7 into data/screenshots and interim CSVs",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        default=None,
        help="Dataset directory (default: dataset_1 .. dataset_6 under project root)",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=DATA_SCREENSHOTS_DIR,
    )
    parser.add_argument(
        "--spawns-csv",
        type=Path,
        default=COLLECTION_SPAWNS_CSV,
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=COLLECTION_INDEX_CSV,
    )
    parser.add_argument(
        "--sources-csv",
        type=Path,
        default=COLLECTION_SOURCES_CSV,
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=COLLECTION_MANIFEST_JSON,
    )
    parser.add_argument(
        "--seeds-txt",
        type=Path,
        default=SEEDS_TXT,
        help="Reference seed list for gap reporting (optional)",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dataset_dir:
        dataset_dirs = [resolve_path(d) for d in args.dataset_dir]
    else:
        dataset_dirs = default_dataset_dirs(PROJECT_ROOT)

    manifest = integrate_collection(
        dataset_dirs=dataset_dirs,
        screenshots_dir=resolve_path(args.screenshots_dir),
        spawns_csv=resolve_path(args.spawns_csv),
        index_csv=resolve_path(args.index_csv),
        sources_csv=resolve_path(args.sources_csv),
        manifest_json=resolve_path(args.manifest_json),
        seeds_txt=resolve_path(args.seeds_txt) if args.seeds_txt else None,
        progress=not args.no_progress,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Integrated {manifest['integrated_count']} seeds")
    print(f"Manifest: {resolve_path(args.manifest_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
