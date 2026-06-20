"""Draw loss labels on collected screenshots for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import (
    COLLECTION_INDEX_CSV,
    DATA_LABELS_DIR,
    DATA_SCREENSHOTS_ANNOTATED_DIR,
    DATA_SCREENSHOTS_ANNOTATED_DIRECTIONAL_DIR,
    PROJECT_ROOT,
)

ANNOTATE_COLUMNS = ("s_forest", "s_ocean", "s_beach", "s_total")


def format_label_value(name: str, value: object) -> str:
    return f"{float(value):.3f}"


def _font_scale_for_width(width: int) -> float:
    return max(0.5, width / 2560.0 * 1.1)


def draw_label_overlay(image: np.ndarray, labels: dict[str, object]) -> np.ndarray:
    """Return a copy of image with score/loss text in the upper-left HUD-safe area."""
    out = image.copy()
    height, width = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = _font_scale_for_width(width)
    thickness = max(1, int(round(font_scale * 2)))
    line_gap = int(round(34 * font_scale))
    margin_x = int(round(24 * width / 2560.0))
    margin_y = int(round(36 * height / 1600.0))

    lines = [f"{name}: {format_label_value(name, labels[name])}" for name in ANNOTATE_COLUMNS]

    y = margin_y
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        pad = int(round(6 * font_scale))
        box_top = y - text_h - pad
        box_bottom = y + baseline + pad
        box_right = margin_x + text_w + 2 * pad
        cv2.rectangle(
            out,
            (margin_x - pad, box_top),
            (box_right, box_bottom),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            out,
            line,
            (margin_x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap

    return out


def resolve_screenshot_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def annotate_labels(
    labels_df: pd.DataFrame,
    index_df: pd.DataFrame,
    output_dir: Path,
    progress: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = labels_df.merge(index_df[["seed", "screenshot_path"]], on="seed", how="inner")
    if merged.empty:
        raise ValueError("No rows after merging labels with collection index on seed")

    iterator = merged.itertuples(index=False)
    if progress:
        iterator = tqdm(list(merged.itertuples(index=False)), desc="annotate labels")

    written: list[Path] = []
    for row in iterator:
        input_path = resolve_screenshot_path(row.screenshot_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {input_path}")

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        labels = {name: getattr(row, name) for name in ANNOTATE_COLUMNS}
        annotated = draw_label_overlay(image, labels)

        output_path = output_dir / f"{int(row.seed)}.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to write {output_path}")
        written.append(output_path)

    return written


def default_output_for_labels(labels_path: Path) -> Path:
    name = labels_path.name.lower()
    if "directional" in name:
        return DATA_SCREENSHOTS_ANNOTATED_DIRECTIONAL_DIR
    return DATA_SCREENSHOTS_ANNOTATED_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate screenshots with score and loss labels")
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels.csv",
        help="CSV with s_forest, s_ocean, s_beach, l_total",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=COLLECTION_INDEX_CSV,
        help="CSV mapping seed to screenshot_path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated PNGs (auto from --labels if omitted)",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labels_path = resolve_path(args.labels)
    index_path = resolve_path(args.index)
    output_dir = resolve_path(args.output_dir or default_output_for_labels(labels_path))

    labels_df = pd.read_csv(labels_path)
    index_df = pd.read_csv(index_path)

    written = annotate_labels(
        labels_df,
        index_df,
        output_dir,
        progress=not args.no_progress,
    )
    print(f"Annotated {len(written)} screenshots -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
