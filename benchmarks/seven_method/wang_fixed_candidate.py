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

from candidate_contamination_simulations import (
    BASE_SEED,
    SETTINGS,
    assign_weights,
    contaminate_candidate,
    generate_dag,
    population_standardize,
    simulate_sem,
)
from local_bic_refinement import (
    edge_jaccard,
    exact_refine_dag,
    gaussian_local_bic,
    greedy_refine_dag,
    total_gaussian_bic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_SUMMARY = REPO_ROOT / "results" / "seven_method_benchmark" / "wang_fixed_candidate_summary.csv"


def wang_style_prune(X: np.ndarray, candidate: np.ndarray,
                     score_tolerance: float = 1e-10) -> np.ndarray:
    """Algorithm-2-style batched backward deletion with Gaussian local BIC.

    Wang et al. Algorithm 2 evaluates every current parent relative to the same
    current-parent score, collects all individually improving deletions in a
    ``ToRemove`` set, removes that set, and repeats. This differs from
    DAGGuard-Greedy, which removes only the single best improving parent per
    iteration.
    """
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

            exact = exact_refine_dag(X, candidate)
            greedy = greedy_refine_dag(X, candidate)
            wang = wang_style_prune(X, candidate)
            wang_gap = total_gaussian_bic(X, wang) - exact.total_bic
            greedy_gap = greedy.total_bic - exact.total_bic
            if wang_gap < -1e-7 or greedy_gap < -1e-7:
                raise AssertionError("A comparator scored below the exact optimum")
            rows.append({
                "setting": setting.name,
                "rep": rep,
                "seed": seed,
                "wang_equals_exact": np.array_equal(wang, exact.adjacency),
                "greedy_equals_exact": np.array_equal(greedy.adjacency, exact.adjacency),
                "wang_exact_jaccard": edge_jaccard(wang, exact.adjacency),
                "wang_gap": max(0.0, float(wang_gap)),
                "greedy_gap": max(0.0, float(greedy_gap)),
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


def check_archived_summary(summary: pd.DataFrame) -> None:
    expected = pd.read_csv(ARCHIVED_SUMMARY).sort_values("setting").reset_index(drop=True)
    observed = summary.sort_values("setting").reset_index(drop=True)
    if list(observed["setting"]) != list(expected["setting"]):
        raise AssertionError("Archived Wang fixed-candidate setting names do not match")
    digits = {
        "wang_exact": 2,
        "greedy_exact": 2,
        "jaccard": 4,
        "wang_gap": 3,
        "wang_max_gap": 3,
        "greedy_gap": 3,
    }
    failures = []
    for column, ndigits in digits.items():
        got = observed[column].round(ndigits).to_numpy()
        want = expected[column].round(ndigits).to_numpy()
        if not np.array_equal(got, want):
            failures.append((column, got.tolist(), want.tolist()))
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
        check_archived_summary(summary)


if __name__ == "__main__":
    main()
