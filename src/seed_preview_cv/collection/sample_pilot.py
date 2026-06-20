"""Sample a pilot subset from the integrated collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seed_preview_cv.common.config import resolve_path
from seed_preview_cv.common.paths import COLLECTION_SPAWNS_CSV, DATA_INTERIM_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample pilot spawns from integrated collection")
    parser.add_argument(
        "--input",
        type=Path,
        default=COLLECTION_SPAWNS_CSV,
        help="Full spawns CSV (seed,x,z)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_INTERIM_DIR / "pilot_spawns.csv",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    in_path = resolve_path(args.input)
    out_path = resolve_path(args.output)

    spawns = pd.read_csv(in_path)
    if args.count > len(spawns):
        raise ValueError(f"Requested {args.count} samples but only {len(spawns)} rows in {in_path}")

    sampled = spawns.sample(n=args.count, random_state=args.rng_seed).sort_values("seed").reset_index(
        drop=True
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, index=False)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "count": int(len(sampled)),
        "rng_seed": args.rng_seed,
        "seeds": [int(s) for s in sampled["seed"]],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "seeds"}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
