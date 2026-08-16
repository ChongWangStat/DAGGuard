#!/usr/bin/env python3
"""Validate, merge, summarize, and plot chunked d=40 simulation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from reproduce_primary_simulations import (
    BASE_SEED,
    LAMBDA1,
    METHODS,
    N_SAMPLES,
    THRESHOLD,
    draw_combined,
    draw_metric,
)

VALID_S = {
    "uniform": [1, 4, 7, 10],
    "modnormal": [1, 2, 3, 4],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-kind", choices=sorted(VALID_S), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--M", type=int, default=20)
    args = parser.parse_args()

    files = sorted(args.input.rglob("replicate_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No replicate_metrics.csv files found under {args.input}")

    frames = [pd.read_csv(path) for path in files]
    results = pd.concat(frames, ignore_index=True)
    results = results[
        (results["d"] == 40)
        & (results["weight_kind"] == args.weight_kind)
    ].copy()

    s_values = VALID_S[args.weight_kind]
    expected = {
        (s, rep, method)
        for s in s_values
        for rep in range(args.M)
        for method in METHODS
    }
    observed = {
        (int(row.s), int(row.rep), str(row.method))
        for row in results.itertuples(index=False)
    }

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicate_count = int(
        results.duplicated(subset=["d", "weight_kind", "s", "rep", "method"]).sum()
    )
    if missing or unexpected or duplicate_count:
        details = {
            "missing_count": len(missing),
            "missing_first_20": missing[:20],
            "unexpected_count": len(unexpected),
            "unexpected_first_20": unexpected[:20],
            "duplicate_count": duplicate_count,
            "source_files": [str(path) for path in files],
        }
        raise RuntimeError("Chunk validation failed:\n" + json.dumps(details, indent=2))

    expected_rows = len(s_values) * args.M * len(METHODS)
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows but found {len(results)} after filtering"
        )

    results["s"] = pd.Categorical(results["s"], categories=s_values, ordered=True)
    results["method"] = pd.Categorical(
        results["method"], categories=METHODS, ordered=True
    )
    results = results.sort_values(["s", "rep", "method"]).reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "replicate_metrics.csv", index=False)

    summary = (
        results.groupby(
            ["d", "weight_kind", "s", "method"],
            as_index=False,
            observed=True,
        )
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

    draw_metric(
        results,
        "fdr",
        "False Discovery Rate",
        "FDR",
        s_values,
        args.out / "fdr",
    )
    draw_metric(
        results,
        "tpr",
        "True Positive Rate",
        "TPR",
        s_values,
        args.out / "tpr",
    )
    draw_metric(
        results,
        "shd",
        "SHD",
        "SHD",
        s_values,
        args.out / "shd",
    )
    draw_combined(results, s_values, args.out / "combined")

    config = {
        "d": 40,
        "weight_kind": args.weight_kind,
        "M": args.M,
        "n_samples": N_SAMPLES,
        "lambda1": LAMBDA1,
        "threshold": THRESHOLD,
        "base_seed": BASE_SEED,
        "s_values": s_values,
        "source_files": [str(path) for path in files],
        "validation": {
            "expected_rows": expected_rows,
            "observed_rows": len(results),
            "missing_combinations": 0,
            "unexpected_combinations": 0,
            "duplicate_combinations": 0,
        },
    }
    (args.out / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
