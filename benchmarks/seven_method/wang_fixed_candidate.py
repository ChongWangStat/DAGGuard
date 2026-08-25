#!/usr/bin/env python3
"""Audit the Wang et al. (2026) local-BIC pruning precedent on fixed candidates.

This is distinct from the end-to-end Wang hybrid adaptation in ``wang_full.py``.
It applies Algorithm 2's batched backward-parent deletion logic to the same
oriented continuous-data candidate DAGs used in the controlled DAGGuard study.
The published Wang score is higher-is-better; here the identical deletion logic
uses the Gaussian local BIC used for these continuous simulations, where lower
is better.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_contamination_simulations import SETTINGS, run_one
from local_bic_refinement import edge_jaccard, gaussian_local_bic, total_gaussian_bic


EXPECTED = {
    "clean_sparse": (1.00, 1.00, 1.0000, 0.000, 0.000, 0.000),
    "combined_contamination": (0.68, 0.96, 0.9890, 2.956, 65.598, 0.145),
    "dense_high_indegree": (0.87, 0.97, 0.9959, 0.316, 8.380, 0.065),
    "dense_moderate": (0.96, 1.00, 0.9988, 0.037, 2.023, 0.000),
    "fp_025": (1.00, 1.00, 1.0000, 0.000, 0.000, 0.000),
    "fp_050": (0.99, 1.00, 0.9995, 0.000, 0.009, 0.000),
    "fp_100": (0.98, 1.00, 0.9991, 0.010, 0.626, 0.000),
    "lowvar_heterogeneous": (1.00, 1.00, 1.0000, 0.000, 0.000, 0.000),
    "lowvar_weak_fp": (0.93, 0.99, 0.9960, 0.296, 10.227, 0.022),
    "missing_010": (1.00, 1.00, 1.0000, 0.000, 0.000, 0.000),
    "reversal_010": (1.00, 1.00, 1.0000, 0.000, 0.000, 0.000),
    "weak_fp": (0.96, 1.00, 0.9980, 0.406, 30.943, 0.000),
}


def wang_style_prune(X: np.ndarray, candidate: np.ndarray,
                     score_tolerance: float = 1e-10) -> np.ndarray:
    """Algorithm-2-style batched backward deletion with Gaussian local BIC."""
    selected = (np.asarray(candidate) != 0).astype(np.int8).copy()
    for child in range(selected.shape[0]):
        current = [int(p) for p in np.flatnonzero(selected[:, child])]
        while current:
            baseline = gaussian_local_bic(X, child, current)
            to_remove = []
            for parent in current:
                trial = [p for p in current if p != parent]
                if gaussian_local_bic(X, child, trial) < baseline - score_tolerance:
                    to_remove.append(parent)
            if not to_remove:
                break
            remove = set(to_remove)
            current = [p for p in current if p not in remove]
        selected[:, child] = 0
        if current:
            selected[np.asarray(current, dtype=int), child] = 1
    return selected


def regenerate(replicates: int = 100, d: int = 20, n: int = 500) -> pd.DataFrame:
    rows = []
    for setting_index, setting in enumerate(SETTINGS):
        for rep in range(replicates):
            base = run_one(setting, setting_index, rep, d, n)
            # run_one is deterministic; rebuild its candidate/data through the archived helper
            # by importing the simulation primitives only when needed below.
            from candidate_contamination_simulations import (
                BASE_SEED, assign_weights, contaminate_candidate, generate_dag,
                population_standardize, simulate_sem,
            )
            seed = BASE_SEED + 100_000 * setting_index + rep
            rng = np.random.default_rng(seed)
            truth, _ = generate_dag(d, setting.true_edges, setting.true_max_indegree, rng)
            W = assign_weights(truth, setting.weak_fraction, rng)
            if setting.error_regime == "lowvar_heterogeneous":
                W, error_scales = population_standardize(W)
            else:
                error_scales = np.ones(d)
            X = simulate_sem(W, n, error_scales, rng)
            candidate, _ = contaminate_candidate(truth, setting, rng)
            from local_bic_refinement import exact_refine_dag, greedy_refine_dag
            exact = exact_refine_dag(X, candidate)
            greedy = greedy_refine_dag(X, candidate)
            wang = wang_style_prune(X, candidate)
            wang_gap = total_gaussian_bic(X, wang) - exact.total_bic
            rows.append({
                "setting": setting.name,
                "rep": rep,
                "wang_equals_exact": np.array_equal(wang, exact.adjacency),
                "greedy_equals_exact": np.array_equal(greedy.adjacency, exact.adjacency),
                "wang_exact_jaccard": edge_jaccard(wang, exact.adjacency),
                "wang_gap": max(0.0, float(wang_gap)),
                "greedy_gap": max(0.0, float(greedy.total_bic - exact.total_bic)),
            })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby("setting", as_index=False).agg(
        wang_exact=("wang_equals_exact", "mean"),
        greedy_exact=("greedy_equals_exact", "mean"),
        jaccard=("wang_exact_jaccard", "mean"),
        wang_gap=("wang_gap", "mean"),
        wang_max_gap=("wang_gap", "max"),
        greedy_gap=("greedy_gap", "mean"),
    )


def check_expected(summary: pd.DataFrame) -> None:
    failures = []
    for row in summary.itertuples(index=False):
        observed = (round(row.wang_exact, 2), round(row.greedy_exact, 2),
                    round(row.jaccard, 4), round(row.wang_gap, 3),
                    round(row.wang_max_gap, 3), round(row.greedy_gap, 3))
        if observed != EXPECTED[row.setting]:
            failures.append((row.setting, observed, EXPECTED[row.setting]))
    if failures:
        raise AssertionError("Supplement S7 mismatch: " + repr(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-manuscript", action="store_true")
    args = parser.parse_args()
    raw = regenerate(args.replicates)
    summary = summarize(raw)
    print(summary.to_string(index=False))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.out / "wang_fixed_candidate_replicates.csv", index=False)
        summary.to_csv(args.out / "wang_fixed_candidate_summary.csv", index=False)
    if args.check_manuscript:
        if args.replicates != 100:
            raise ValueError("Manuscript check requires 100 replicates per setting")
        check_expected(summary)


if __name__ == "__main__":
    main()
