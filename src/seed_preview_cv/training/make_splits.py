"""Generate train/val/test split CSVs for balanced and natural datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import (
    DATA_DATASET_IMAGES_DIR,
    DATA_LABELS_DIR,
    PROJECT_ROOT,
)
from seed_preview_cv.training.dataset import sanity_check_dataset

SCORE_COLUMNS = ("s_forest", "s_ocean", "s_beach")
TOTAL_COLUMN = "s_total"
BASE_OUTPUT_COLUMNS = ("image_path", "seed", *SCORE_COLUMNS)
META_OUTPUT_COLUMNS = ("split", "source")

LABEL_MODE_PATHS = {
    "directional": {
        "balanced_labels": DATA_LABELS_DIR / "full_labels_directional.csv",
        "natural_labels": DATA_LABELS_DIR / "test_random_labels_directional.csv",
    },
    "isotropic": {
        "balanced_labels": DATA_LABELS_DIR / "full_labels_isotropic.csv",
        "natural_labels": DATA_LABELS_DIR / "test_random_labels_isotropic.csv",
    },
}

DEFAULT_BALANCED_INDEX = DATA_DATASET_IMAGES_DIR / "full" / "dataset_index.csv"
DEFAULT_NATURAL_INDEX = DATA_DATASET_IMAGES_DIR / "test_random" / "dataset_index.csv"

BALANCED_OUTPUT_FILES = {
    "train": "train_balanced.csv",
    "val": "val_balanced.csv",
    "test": "test_balanced.csv",
}
NATURAL_OUTPUT_FILE = "test_natural.csv"


def default_paths_for_label_mode(label_mode: str) -> dict[str, Path]:
    if label_mode not in LABEL_MODE_PATHS:
        raise ValueError(f"label_mode must be one of {list(LABEL_MODE_PATHS)}")
    paths = LABEL_MODE_PATHS[label_mode]
    return {
        "balanced_index": DEFAULT_BALANCED_INDEX,
        "balanced_labels": paths["balanced_labels"],
        "natural_index": DEFAULT_NATURAL_INDEX,
        "natural_labels": paths["natural_labels"],
    }


def _resolve_total_column(labels_df: pd.DataFrame) -> str | None:
    for name in (TOTAL_COLUMN, "S_total"):
        if name in labels_df.columns:
            return name
    return None


def _output_columns_for_df(df: pd.DataFrame) -> list[str]:
    cols = list(BASE_OUTPUT_COLUMNS)
    if TOTAL_COLUMN in df.columns:
        cols.append(TOTAL_COLUMN)
    cols.extend(META_OUTPUT_COLUMNS)
    return cols


def merge_index_and_labels(index_path: Path, labels_path: Path, source: str) -> pd.DataFrame:
    index_df = pd.read_csv(index_path)
    labels_df = pd.read_csv(labels_path)

    required_index = {"seed", "image_path"}
    missing_index = required_index - set(index_df.columns)
    if missing_index:
        raise ValueError(f"{index_path} missing columns: {sorted(missing_index)}")

    required_labels = {"seed", *SCORE_COLUMNS}
    missing_labels = required_labels - set(labels_df.columns)
    if missing_labels:
        raise ValueError(f"{labels_path} missing columns: {sorted(missing_labels)}")

    label_cols = ["seed", *SCORE_COLUMNS]
    total_col = _resolve_total_column(labels_df)
    if total_col is not None:
        label_cols.append(total_col)

    merged = index_df.merge(
        labels_df[label_cols],
        on="seed",
        how="inner",
    )
    if total_col is not None and total_col != TOTAL_COLUMN:
        merged = merged.rename(columns={total_col: TOTAL_COLUMN})
    if len(merged) != len(index_df):
        raise ValueError(
            f"Inner join {index_path} ({len(index_df)} rows) + {labels_path} "
            f"({len(labels_df)} rows) produced {len(merged)} rows; "
            "expected same count as index (missing labels for some seeds?)"
        )
    if len(merged) != len(labels_df):
        raise ValueError(
            f"Inner join produced {len(merged)} rows but labels has {len(labels_df)}; "
            "duplicate or unmatched seeds in labels"
        )

    dup_seeds = merged["seed"].duplicated()
    if dup_seeds.any():
        n_dup = int(dup_seeds.sum())
        raise ValueError(f"Duplicate seeds after merge for source={source}: {n_dup}")

    merged["source"] = source
    return merged


def compute_quality(df: pd.DataFrame) -> np.ndarray:
    forest = df["s_forest"].astype(float).to_numpy()
    ocean = df["s_ocean"].astype(float).to_numpy()
    beach = df["s_beach"].astype(float).to_numpy()
    quality = np.power(forest, 0.4) * np.power(ocean, 0.4) * np.power(beach, 0.2)
    zero_mask = (forest == 0.0) | (ocean == 0.0) | (beach == 0.0)
    quality[zero_mask] = 0.0
    return quality


def score_stats(series: pd.Series | np.ndarray) -> dict[str, float]:
    arr = np.asarray(series, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "zero_ratio": float(np.mean(arr == 0.0)),
    }


def summarize_split(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    if TOTAL_COLUMN in df.columns:
        quality = df[TOTAL_COLUMN].astype(float).to_numpy()
        quality_source = "s_total"
    else:
        quality = compute_quality(df)
        quality_source = "computed_from_scores"
    summary: dict[str, Any] = {
        "split": split_name,
        "row_count": int(len(df)),
        "unique_seed_count": int(df["seed"].nunique()),
        "s_forest": score_stats(df["s_forest"]),
        "s_ocean": score_stats(df["s_ocean"]),
        "s_beach": score_stats(df["s_beach"]),
        "quality": score_stats(quality),
        "quality_source": quality_source,
    }
    if TOTAL_COLUMN in df.columns:
        summary["s_total"] = score_stats(df[TOTAL_COLUMN])
    return summary


def split_balanced(
    balanced_df: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    expected_total = train_size + val_size + test_size
    if len(balanced_df) != expected_total:
        raise ValueError(
            f"Balanced merged count {len(balanced_df)} != "
            f"train+val+test ({expected_total})"
        )

    indices = rng.permutation(len(balanced_df))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    splits = {
        "train": balanced_df.iloc[train_idx].copy(),
        "val": balanced_df.iloc[val_idx].copy(),
        "test": balanced_df.iloc[test_idx].copy(),
    }
    split_labels = {"train": "train", "val": "val", "test": "test"}
    for key, label in split_labels.items():
        splits[key]["split"] = label

    return splits


def prepare_natural_test(
    natural_df: pd.DataFrame,
    natural_test_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(natural_df) < natural_test_size:
        raise ValueError(
            f"Natural merged count {len(natural_df)} < required {natural_test_size}"
        )
    if len(natural_df) == natural_test_size:
        out = natural_df.copy()
    else:
        indices = rng.choice(len(natural_df), size=natural_test_size, replace=False)
        out = natural_df.iloc[indices].copy()

    out["split"] = "test_natural"
    return out


def check_seed_uniqueness_within(df: pd.DataFrame, split_name: str) -> None:
    if df["seed"].duplicated().any():
        raise ValueError(f"Duplicate seeds within split '{split_name}'")


def check_no_overlap(seed_sets: dict[str, set[int]], pairs: list[tuple[str, str]]) -> None:
    for a, b in pairs:
        overlap = seed_sets[a] & seed_sets[b]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"Seed overlap between {a} and {b}: {len(overlap)} seeds "
                f"(examples: {sample})"
            )


def write_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _output_columns_for_df(df)
    out = df[columns].copy()
    out.to_csv(path, index=False)


def run_sanity_checks(output_dir: Path, image_root: Path, num_samples: int = 3) -> None:
    for filename in [
        BALANCED_OUTPUT_FILES["train"],
        BALANCED_OUTPUT_FILES["val"],
        BALANCED_OUTPUT_FILES["test"],
        NATURAL_OUTPUT_FILE,
    ]:
        csv_path = output_dir / filename
        print(f"\n=== sanity check: {csv_path} ===")
        sanity_check_dataset(
            csv_path=csv_path,
            image_root=image_root,
            num_samples=num_samples,
        )


def make_splits(
    balanced_index: Path,
    balanced_labels: Path,
    natural_index: Path,
    natural_labels: Path,
    output_dir: Path,
    train_size: int = 20000,
    val_size: int = 4000,
    test_size: int = 4000,
    natural_test_size: int = 5000,
    seed: int = 42,
    label_mode: str = "directional",
    image_root: Path = PROJECT_ROOT,
    run_sanity: bool = True,
    sanity_num_samples: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    balanced_merged = merge_index_and_labels(balanced_index, balanced_labels, "balanced")
    natural_merged = merge_index_and_labels(natural_index, natural_labels, "natural")

    balanced_splits = split_balanced(
        balanced_merged,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        rng=rng,
    )
    natural_test = prepare_natural_test(natural_merged, natural_test_size, rng)

    all_splits: dict[str, pd.DataFrame] = {
        **balanced_splits,
        "test_natural": natural_test,
    }

    seed_sets = {name: set(df["seed"].astype(int)) for name, df in all_splits.items()}
    for name, df in all_splits.items():
        check_seed_uniqueness_within(df, name)

    check_no_overlap(
        seed_sets,
        [
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
            ("train", "test_natural"),
            ("val", "test_natural"),
            ("test", "test_natural"),
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(balanced_splits["train"], output_dir / BALANCED_OUTPUT_FILES["train"])
    write_split_csv(balanced_splits["val"], output_dir / BALANCED_OUTPUT_FILES["val"])
    write_split_csv(balanced_splits["test"], output_dir / BALANCED_OUTPUT_FILES["test"])
    write_split_csv(natural_test, output_dir / NATURAL_OUTPUT_FILE)

    summaries = []
    for key, df in all_splits.items():
        file_name = BALANCED_OUTPUT_FILES.get(key, NATURAL_OUTPUT_FILE)
        split_label = df["split"].iloc[0]
        summary = summarize_split(df, split_label)
        summary["file"] = file_name
        summaries.append(summary)
        print(f"\n--- {file_name} ({split_label}) ---")
        print(f"row count: {summary['row_count']}")
        print(f"unique seed count: {summary['unique_seed_count']}")
        for col in SCORE_COLUMNS:
            stats = summary[col]
            print(
                f"{col}: mean={stats['mean']:.4f} median={stats['median']:.4f} "
                f"p10={stats['p10']:.4f} p90={stats['p90']:.4f} "
                f"zero_ratio={stats['zero_ratio']:.4f}"
            )
        q = summary["quality"]
        print(
            f"quality: mean={q['mean']:.4f} median={q['median']:.4f} "
            f"p10={q['p10']:.4f} p90={q['p90']:.4f} zero_ratio={q['zero_ratio']:.4f}"
        )

    manifest: dict[str, Any] = {
        "label_mode": label_mode,
        "balanced_index": str(balanced_index),
        "balanced_labels": str(balanced_labels),
        "natural_index": str(natural_index),
        "natural_labels": str(natural_labels),
        "output_dir": str(output_dir),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "natural_test_size": natural_test_size,
        "seed": seed,
        "splits": summaries,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")

    if run_sanity:
        run_sanity_checks(output_dir, image_root, num_samples=sanity_num_samples)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate training split CSVs")
    parser.add_argument(
        "--label-mode",
        choices=tuple(LABEL_MODE_PATHS),
        default="directional",
        help="Label scoring mode (default: directional)",
    )
    parser.add_argument("--balanced-index", type=Path, default=None)
    parser.add_argument("--balanced-labels", type=Path, default=None)
    parser.add_argument("--natural-index", type=Path, default=None)
    parser.add_argument("--natural-labels", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_LABELS_DIR / "splits",
    )
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--natural-test-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root for sanity check image paths",
    )
    parser.add_argument("--no-sanity", action="store_true")
    parser.add_argument("--sanity-num-samples", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    defaults = default_paths_for_label_mode(args.label_mode)
    balanced_index = resolve_path(args.balanced_index or defaults["balanced_index"])
    balanced_labels = resolve_path(args.balanced_labels or defaults["balanced_labels"])
    natural_index = resolve_path(args.natural_index or defaults["natural_index"])
    natural_labels = resolve_path(args.natural_labels or defaults["natural_labels"])
    output_dir = resolve_path(args.output_dir)
    image_root = resolve_path(args.image_root)

    make_splits(
        balanced_index=balanced_index,
        balanced_labels=balanced_labels,
        natural_index=natural_index,
        natural_labels=natural_labels,
        output_dir=output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        natural_test_size=args.natural_test_size,
        seed=args.seed,
        label_mode=args.label_mode,
        image_root=image_root,
        run_sanity=not args.no_sanity,
        sanity_num_samples=args.sanity_num_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
