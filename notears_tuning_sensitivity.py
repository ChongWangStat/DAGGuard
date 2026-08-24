#!/usr/bin/env python3
"""NOTEARS penalty/threshold sensitivity for the DAGGuard JDS submission.

The experiment varies the upstream NOTEARS L1 penalty and coefficient threshold,
then applies certified DAGGuard-Exact to each resulting candidate. It uses the
same data-generating mechanism and seed schedule as the Gaussian/non-Gaussian
end-to-end experiment.

Primary design
--------------
* lambda1 in {0.05, 0.10, 0.20}
* coefficient threshold in {0.20, 0.30, 0.40}
* d=10, s in {2,5}, three error distributions, 20 replicates/setting
* targeted d=20, s=5, three error distributions, 5 replicates/setting

Each NOTEARS fit is reused across the three thresholds. The reported refinement
is the certified exact deletion-subgraph optimum.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from additional_noise_sensitivity import (
    simulate_dag_seeded,
    simulate_lsem_noise,
    simulate_weights_seeded,
)
from local_bic_refinement import exact_refine_dag, graph_metrics
from reproduce_simulations import BASE_SEED, notears_linear, thresholded_candidate_dag

LAMBDAS = (0.05, 0.10, 0.20)
THRESHOLDS = (0.20, 0.30, 0.40)
NOISES = ("normal", "exponential", "gumbel")


def _one_fit(args):
    d, s, noise, rep, lambda1, n = args
    noise_index = NOISES.index(noise)
    seed = BASE_SEED + 100000 * d + 1000 * int(10 * s) + 10 * noise_index + rep
    truth = simulate_dag_seeded(d, 2 * d, seed)
    weights = simulate_weights_seeded(truth, s, seed + 1)
    X = simulate_lsem_noise(weights, n, noise, seed + 2)

    start = time.perf_counter()
    W = notears_linear(X, lambda1=lambda1)
    notears_runtime = time.perf_counter() - start
    rows = []
    for threshold in THRESHOLDS:
        candidate, cycle_edges_omitted = thresholded_candidate_dag(W, threshold)
        start = time.perf_counter()
        exact = exact_refine_dag(X, candidate)
        exact_runtime = time.perf_counter() - start
        if not exact.globally_optimal:
            raise RuntimeError("DAGGuard-Exact search was not certified")
        cand = graph_metrics(truth, candidate)
        ref = graph_metrics(truth, exact.adjacency)
        ce = int(candidate.sum())
        ee = int(exact.adjacency.sum())
        rows.append({
            "d": d, "s": s, "noise": noise, "rep": rep, "seed": seed,
            "lambda1": lambda1, "threshold": threshold,
            "notears_runtime_seconds": notears_runtime,
            "exact_runtime_seconds": exact_runtime,
            "cycle_edges_omitted": int(cycle_edges_omitted),
            "candidate_edges": ce, "exact_edges": ee,
            "deletion_fraction": (ce - ee) / ce if ce else 0.0,
            "candidate_fdr": cand["fdr"], "exact_fdr": ref["fdr"],
            "candidate_tpr": cand["tpr"], "exact_tpr": ref["tpr"],
            "candidate_shd": cand["shd"], "exact_shd": ref["shd"],
            "delta_fdr": ref["fdr"] - cand["fdr"],
            "delta_tpr": ref["tpr"] - cand["tpr"],
            "delta_shd": ref["shd"] - cand["shd"],
        })
    return rows


def _settings(n):
    jobs = []
    for s in (2, 5):
        for noise in NOISES:
            for rep in range(20):
                for lambda1 in LAMBDAS:
                    jobs.append((10, s, noise, rep, lambda1, n))
    for noise in NOISES:
        for rep in range(5):
            for lambda1 in LAMBDAS:
                jobs.append((20, 5, noise, rep, lambda1, n))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/notears_tuning_sensitivity"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = _settings(args.n)
    rows = []
    if args.workers == 1:
        for k, job in enumerate(jobs, 1):
            rows.extend(_one_fit(job))
            if k % 20 == 0:
                print(f"completed {k} of {len(jobs)} NOTEARS fits", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_one_fit, job): job for job in jobs}
            for k, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if k % 20 == 0:
                    print(f"completed {k} of {len(jobs)} NOTEARS fits", flush=True)

    df = pd.DataFrame(rows).sort_values(
        ["d", "s", "noise", "rep", "lambda1", "threshold"]
    ).reset_index(drop=True)
    df.to_csv(args.out / "notears_penalty_threshold_sensitivity_replicates.csv", index=False)

    setting = df.groupby(
        ["d", "s", "noise", "lambda1", "threshold"], as_index=False
    ).agg(
        n_replicates=("rep", "size"),
        candidate_edges=("candidate_edges", "mean"),
        exact_edges=("exact_edges", "mean"),
        candidate_fdr=("candidate_fdr", "mean"),
        exact_fdr=("exact_fdr", "mean"),
        candidate_tpr=("candidate_tpr", "mean"),
        exact_tpr=("exact_tpr", "mean"),
        candidate_shd=("candidate_shd", "mean"),
        exact_shd=("exact_shd", "mean"),
        deletion_fraction=("deletion_fraction", "mean"),
        delta_fdr=("delta_fdr", "mean"),
        delta_tpr=("delta_tpr", "mean"),
        delta_shd=("delta_shd", "mean"),
    )
    setting.to_csv(args.out / "notears_penalty_threshold_sensitivity_summary.csv", index=False)

    grid = df.groupby(["d", "lambda1", "threshold"], as_index=False).agg(
        n=("rep", "size"),
        candidate_edges=("candidate_edges", "mean"),
        exact_edges=("exact_edges", "mean"),
        candidate_fdr=("candidate_fdr", "mean"),
        exact_fdr=("exact_fdr", "mean"),
        candidate_tpr=("candidate_tpr", "mean"),
        exact_tpr=("exact_tpr", "mean"),
        candidate_shd=("candidate_shd", "mean"),
        exact_shd=("exact_shd", "mean"),
        deletion_fraction=("deletion_fraction", "mean"),
        delta_fdr=("delta_fdr", "mean"),
        delta_tpr=("delta_tpr", "mean"),
        delta_shd=("delta_shd", "mean"),
    )
    grid.to_csv(args.out / "notears_penalty_threshold_grid_by_dimension.csv", index=False)
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
