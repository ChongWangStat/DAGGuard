#!/usr/bin/env python3
"""Run a small replicate block for one unfinished d=40 simulation setting.

This keeps the original seed formula and analysis code unchanged while splitting
long settings into smaller GitHub Actions jobs that fit within the six-hour
runner limit.
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
METHOD_COUNT = 5
TOTAL_REPLICATES = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-kind", choices=sorted(VALID_S), required=True)
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--rep-start", type=int, required=True)
    parser.add_argument("--rep-count", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.s not in VALID_S[args.weight_kind]:
        parser.error(
            f"s={args.s} is invalid for {args.weight_kind}; "
            f"choose from {VALID_S[args.weight_kind]}"
        )
    if args.rep_start < 0 or args.rep_count < 1:
        parser.error("replicate start must be nonnegative and count must be positive")
    rep_stop = args.rep_start + args.rep_count
    if rep_stop > TOTAL_REPLICATES:
        parser.error(
            f"requested replicates [{args.rep_start}, {rep_stop}) exceed "
            f"the study range 0..{TOTAL_REPLICATES - 1}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    reps = list(range(args.rep_start, rep_stop))
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(40, args.s, args.weight_kind, rep) for rep in reps
    )
    rows = [row for group in nested for row in group]
    results = pd.DataFrame(rows).sort_values(["s", "rep", "method"])

    expected_rows = args.rep_count * METHOD_COUNT
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows but obtained {len(results)}"
        )
    observed_reps = sorted(results["rep"].unique().tolist())
    if observed_reps != reps:
        raise RuntimeError(
            f"Replicate mismatch: expected {reps}, observed {observed_reps}"
        )

    suffix = f"rep{args.rep_start:02d}_{rep_stop - 1:02d}"
    results.to_csv(args.out / f"replicate_metrics_{suffix}.csv", index=False)
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
    summary.to_csv(args.out / f"summary_{suffix}.csv", index=False)

    config = {
        "d": 40,
        "weight_kind": args.weight_kind,
        "s": args.s,
        "rep_start": args.rep_start,
        "rep_stop_exclusive": rep_stop,
        "n_samples": N_SAMPLES,
        "lambda1": LAMBDA1,
        "threshold": THRESHOLD,
        "base_seed": BASE_SEED,
        "seed_formula": "12123 + 1000 * index(s) + replicate",
    }
    (args.out / f"run_config_{suffix}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
