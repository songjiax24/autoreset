"""Draw isotropic vs directional labels on the same screenshot."""

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
    DATA_SCREENSHOTS_ANNOTATED_COMPARE_DIR,
)
from seed_preview_cv.labeling.annotate_labels import (
    ANNOTATE_COLUMNS,
    format_label_value,
    resolve_screenshot_path,
)


def _font_scale_for_width(width: int) -> float:
    return max(0.5, width / 2560.0 * 1.1)


def _draw_text_column(
    image,
    x: int,
    y_start: int,
    title: str,
    labels: dict[str, object],
    font_scale: float,
    thickness: int,
    line_gap: int,
    color: tuple[int, int, int],
) -> int:
    """Draw a titled label column; return bottom y of the column."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = int(round(6 * font_scale))
    y = y_start

    header = f"[{title}]"
    (header_w, header_h), header_base = cv2.getTextSize(header, font, font_scale, thickness)
    cv2.rectangle(
        image,
        (x - pad, y - header_h - pad),
        (x + header_w + pad, y + header_base + pad),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(image, header, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    y += line_gap

    for name in ANNOTATE_COLUMNS:
        line = f"{name}: {format_label_value(name, labels[name])}"
        (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(
            image,
            (x - pad, y - text_h - pad),
            (x + text_w + pad, y + baseline + pad),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(image, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_gap

    return y


def draw_compare_overlay(
    image,
    isotropic: dict[str, object],
    directional: dict[str, object],
) -> np.ndarray:
    """Draw isotropic (left) and directional (right) scores on one image."""
    out = image.copy()
    height, width = out.shape[:2]
    font_scale = _font_scale_for_width(width)
    thickness = max(1, int(round(font_scale * 2)))
    line_gap = int(round(34 * font_scale))
    margin_x = int(round(24 * width / 2560.0))
    margin_y = int(round(36 * height / 1600.0))
    column_gap = int(round(320 * width / 2560.0))

    iso_color = (255, 255, 255)
    dir_color = (120, 220, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = int(round(6 * font_scale))
    iso_lines = [f"{name}: {format_label_value(name, isotropic[name])}" for name in ANNOTATE_COLUMNS]
    dir_lines = [f"{name}: {format_label_value(name, directional[name])}" for name in ANNOTATE_COLUMNS]
    header_iso = "[isotropic]"
    header_dir = "[directional]"
    max_iso_w = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in [header_iso] + iso_lines
    )
    x_iso = margin_x
    x_dir = margin_x + max_iso_w + column_gap

    bottom_iso = _draw_text_column(
        out, x_iso, margin_y, "isotropic", isotropic, font_scale, thickness, line_gap, iso_color
    )
    bottom_dir = _draw_text_column(
        out, x_dir, margin_y, "directional", directional, font_scale, thickness, line_gap, dir_color
    )

    divider_x = x_dir - column_gap // 2
    divider_top = margin_y - pad
    divider_bottom = max(bottom_iso, bottom_dir)
    cv2.line(out, (divider_x, divider_top), (divider_x, divider_bottom), (80, 80, 80), 1, cv2.LINE_AA)

    return out


def annotate_compare(
    isotropic_df: pd.DataFrame,
    directional_df: pd.DataFrame,
    index_df: pd.DataFrame,
    output_dir: Path,
    progress: bool = True,
) -> list[Path]:
    iso_cols = ["seed"] + list(ANNOTATE_COLUMNS)
    dir_cols = ["seed"] + list(ANNOTATE_COLUMNS)
    merged = isotropic_df[iso_cols].merge(
        directional_df[dir_cols],
        on="seed",
        suffixes=("_iso", "_dir"),
    )
    merged = merged.merge(index_df[["seed", "screenshot_path"]], on="seed", how="inner")
    if merged.empty:
        raise ValueError("No rows after merging label CSVs with collection index")

    iterator = merged.itertuples(index=False)
    if progress:
        iterator = tqdm(list(merged.itertuples(index=False)), desc="annotate compare")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in iterator:
        input_path = resolve_screenshot_path(row.screenshot_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {input_path}")

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        iso_labels = {name: getattr(row, f"{name}_iso") for name in ANNOTATE_COLUMNS}
        dir_labels = {name: getattr(row, f"{name}_dir") for name in ANNOTATE_COLUMNS}
        annotated = draw_compare_overlay(image, iso_labels, dir_labels)

        output_path = output_dir / f"{int(row.seed)}.png"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to write {output_path}")
        written.append(output_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate screenshots with isotropic and directional labels side by side",
    )
    parser.add_argument(
        "--isotropic-labels",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels.csv",
    )
    parser.add_argument(
        "--directional-labels",
        type=Path,
        default=DATA_LABELS_DIR / "pilot_labels_directional.csv",
    )
    parser.add_argument("--index", type=Path, default=COLLECTION_INDEX_CSV)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_SCREENSHOTS_ANNOTATED_COMPARE_DIR,
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    iso_df = pd.read_csv(resolve_path(args.isotropic_labels))
    dir_df = pd.read_csv(resolve_path(args.directional_labels))
    index_df = pd.read_csv(resolve_path(args.index))
    output_dir = resolve_path(args.output_dir)

    written = annotate_compare(
        iso_df,
        dir_df,
        index_df,
        output_dir,
        progress=not args.no_progress,
    )
    print(f"Annotated {len(written)} comparison screenshots -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
