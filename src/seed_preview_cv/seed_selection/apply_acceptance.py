"""CLI: apply acceptance sampling to distance scan results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR
from seed_preview_cv.seed_selection.acceptance import (
    AcceptanceConfig,
    SEED_LIST_COLUMNS,
    apply_acceptance,
    simulate_acceptance,
)
from seed_preview_cv.seed_selection.plot_distances import plot_acceptance_comparison
from seed_preview_cv.seed_selection.summarize_distances import load_distance_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply treasure-distance acceptance sampling")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "seed_selection.yaml")
    parser.add_argument("--input", type=Path, help="Override distances table path")
    parser.add_argument("--output", type=Path, help="Override seed list output path")
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Print expected acceptance rate without sampling",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip comparison histogram")
    parser.add_argument("--rng-seed", type=int, help="Override acceptance.rng_seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    acceptance_cfg = AcceptanceConfig.from_mapping(config.get("acceptance", {}))
    if args.rng_seed is not None:
        acceptance_cfg = AcceptanceConfig(
            p_min=acceptance_cfg.p_min,
            p_max=acceptance_cfg.p_max,
            d0=acceptance_cfg.d0,
            scale=acceptance_cfg.scale,
            rng_seed=args.rng_seed,
        )

    output_cfg = config.get("output", {})
    treasure_cfg = config.get("treasure", {})

    in_path = resolve_path(
        args.input or output_cfg.get("distances", "outputs/seed_selection/distances.csv")
    )
    out_path = resolve_path(
        args.output or output_cfg.get("seed_list", "outputs/seed_selection/seed_list.csv")
    )
    summary_path = resolve_path(
        output_cfg.get("acceptance_summary", "outputs/seed_selection/acceptance_summary.json")
    )
    hist_path = resolve_path(
        output_cfg.get("acceptance_histogram", "outputs/seed_selection/acceptance_histogram.png")
    )
    search_radius = int(treasure_cfg.get("search_radius_blocks", 512))

    df = load_distance_table(in_path)
    expected = simulate_acceptance(df, acceptance_cfg)
    expected["input"] = str(in_path)

    if args.simulate_only:
        print(json.dumps(expected, indent=2))
        return 0

    sampled = apply_acceptance(df, acceptance_cfg)
    accepted = sampled[sampled["accepted"]].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted[list(SEED_LIST_COLUMNS)].to_csv(out_path, index=False)

    result = {
        **expected,
        "actual_accepted": int(len(accepted)),
        "actual_acceptance_rate": float(len(accepted) / len(df)) if len(df) else 0.0,
        "seed_list": str(out_path),
        "rng_seed": acceptance_cfg.rng_seed,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_acceptance_comparison(
            df,
            accepted,
            hist_path,
            search_radius_blocks=search_radius,
            title=(
                f"Before vs after acceptance "
                f"(accepted {len(accepted)}/{len(df)}, "
                f"rate {result['actual_acceptance_rate']:.1%})"
            ),
        )
        result["acceptance_histogram"] = str(hist_path)

    print(json.dumps(result, indent=2))
    print(f"Wrote {len(accepted)} seeds to {out_path}")
    if not args.no_plot:
        print(f"Wrote comparison histogram to {hist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
