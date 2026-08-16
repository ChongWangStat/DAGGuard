#!/usr/bin/env python3
"""Run one scale setting of the d=40 primary simulation.

Splitting the d=40 panel by scale preserves the original seeds and simulation
procedure while keeping each GitHub Actions job below the six-hour job limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from reproduce_primary_simulations import (
    BASE_SEED,
    LAMBDA1,
    N_SAMPLES,
    THRESHOLD,
    run_one,
)

VALID_S = {
    "uniform": [1, 4, 7, 10],
    "modnormal": [1, 2, 3, 4],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-kind", choices=sorted(VALID_S), required=True)
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.s not in VALID_S[args.weight_kind]:
        parser.error(
            f"s={args.s} is invalid for {args.weight_kind}; "
            f"choose from {VALID_S[args.weight_kind]}"
        )
    if args.M < 1:
        parser.error("--M must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(40, args.s, args.weight_kind, rep)
        for rep in range(args.M)
    )
    rows = [row for group in nested for row in group]
    results = pd.DataFrame(rows).sort_values(["s", "rep", "method"])

    expected_rows = args.M * 5
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows but obtained {len(results)}"
        )

    results.to_csv(args.out / "replicate_metrics.csv", index=False)
    summary = (
        results.groupby(["d", "weight_kind", "s", "method"], as_index=False)
        .agg(
            fdr_mean=("fdr", "mean"),
            tpr_mean=("tpr", "mean"),
            shd_mean=("shd", "mean"),
            fdr_sd=("fdr", "std"),
            tpr_sd=("tpr", "std"),
            shd_sd=("shd", "std"),
            estimated_edges_mean=("estimated_edges", "mean"),
        )
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    config = {
        "d": 40,
        "weight_kind": args.weight_kind,
        "s": args.s,
        "M": args.M,
        "n_samples": N_SAMPLES,
        "lambda1": LAMBDA1,
        "threshold": THRESHOLD,
        "base_seed": BASE_SEED,
        "seed_formula": "12123 + 1000 * index(s) + replicate",
    }
    (args.out / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
