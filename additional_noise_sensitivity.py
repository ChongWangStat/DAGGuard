#!/usr/bin/env python3
"""Corrected Gaussian and non-Gaussian NOTEARS sensitivity experiment.

The authorized swine application has a separate, single source of truth in
``realdata_postselection_diagnostics.py``. This script contains no proprietary-
data path and writes only the secondary end-to-end noise sensitivity outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from local_bic_refinement import (
    candidate_indegree_summary,
    edge_jaccard,
    exact_refine_dag,
    graph_metrics,
    greedy_refine_dag,
)
from reproduce_simulations import (
    BASE_SEED, LAMBDA1, THRESHOLD, notears_linear, thresholded_candidate_dag,
)

def adjacency(W):
    return thresholded_candidate_dag(W, THRESHOLD)[0]


def simulate_dag_seeded(d, num_edges, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=int)
    pairs = [(order[i], order[j]) for i in range(d) for j in range(i + 1, d)]
    chosen = rng.choice(len(pairs), size=min(num_edges, len(pairs)), replace=False)
    for idx in chosen:
        u, v = pairs[idx]
        A[u, v] = 1
    assert int(A.sum()) == num_edges
    return A


def simulate_weights_seeded(A, s, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros_like(A, dtype=float)
    mask = A != 0
    signs = rng.choice([-1, 1], size=A.shape)
    mags = rng.uniform(0.5, s, size=A.shape)
    W[mask] = signs[mask] * mags[mask]
    return W


def simulate_lsem_noise(W, n, noise, seed):
    rng = np.random.default_rng(seed)
    import networkx as nx

    G = nx.DiGraph(W)
    X = np.zeros((n, W.shape[0]))
    for j in nx.topological_sort(G):
        parents = np.where(W[:, j] != 0)[0]
        if noise == "normal":
            eps = rng.normal(0, 1, n)
        elif noise == "exponential":
            eps = rng.exponential(1, n) - 1
        elif noise == "gumbel":
            # A Gumbel(0, beta) variable has variance pi^2 beta^2 / 6.
            # beta=sqrt(6)/pi therefore gives unit variance. Center by beta*gamma.
            beta = np.sqrt(6.0) / np.pi
            eps = rng.gumbel(0, beta, n) - beta * np.euler_gamma
        else:
            raise ValueError(noise)
        X[:, j] = X[:, parents] @ W[parents, j] + eps if len(parents) else eps
    return X


def run_additional_simulations(out, M=20, n=500):
    rows = []
    pruning = []
    for d in [10, 20]:
        for s in [2, 5]:
            noises = ["normal", "exponential", "gumbel"]
            for noise in noises:
                for rep in range(M):
                    seed = BASE_SEED + 100000 * d + 1000 * int(10 * s) + 10 * noises.index(noise) + rep
                    B = simulate_dag_seeded(d, 2 * d, seed)
                    W = simulate_weights_seeded(B, s, seed + 1)
                    X = simulate_lsem_noise(W, n, noise, seed + 2)
                    Wn = notears_linear(X, lambda1=LAMBDA1)
                    An = adjacency(Wn)
                    exact = exact_refine_dag(X, An)
                    greedy = greedy_refine_dag(X, An)
                    if not exact.globally_optimal:
                        raise RuntimeError("Exact local-BIC search was not certified")
                    Aexact = exact.adjacency
                    Agreedy = greedy.adjacency
                    Xs = StandardScaler().fit_transform(X)
                    As = adjacency(notears_linear(Xs, lambda1=LAMBDA1))
                    std_exact = exact_refine_dag(Xs, As)
                    if not std_exact.globally_optimal:
                        raise RuntimeError(
                            "Standardized exact local-BIC search was not certified"
                        )
                    gap = greedy.total_bic - exact.total_bic
                    for method, A in [
                        ("NOTEARS", An),
                        ("LOCAL_BIC_EXACT", Aexact),
                        ("LOCAL_BIC_GREEDY", Agreedy),
                        ("STD_NOTEARS", As),
                        ("STD_LOCAL_BIC_EXACT", std_exact.adjacency),
                    ]:
                        metrics = graph_metrics(B, A)
                        rows.append({"d": d, "s": s, "noise": noise, "rep": rep,
                                     "seed": seed, "method": method,
                                     "fdr": metrics["fdr"], "tpr": metrics["tpr"],
                                     "shd": metrics["shd"],
                                     "tp": metrics["true_positives"],
                                     "fp": metrics["false_positives"],
                                     "fn": metrics["false_negatives"],
                                     "estimated_edges": int(A.sum()),
                                     "exact_certified": bool(exact.globally_optimal),
                                     "standardized_exact_certified": bool(
                                         std_exact.globally_optimal
                                     ),
                                     "greedy_bic_gap": gap,
                                     "greedy_suboptimal": bool(gap > 1e-8),
                                     "exact_greedy_jaccard": edge_jaccard(Aexact, Agreedy)})
                    removed = (An == 1) & (Aexact == 0)
                    pruning.append({
                        "d": d, "s": s, "noise": noise, "rep": rep,
                        "seed": seed, "notears_edges": int(An.sum()),
                        "exact_edges": int(Aexact.sum()),
                        "greedy_edges": int(Agreedy.sum()),
                        "false_positive_removals": int(np.sum(removed & (B == 0))),
                        "true_positive_removals": int(np.sum(removed & (B == 1))),
                        "candidate_max_indegree": candidate_indegree_summary(An)["maximum_indegree"],
                        "greedy_bic_gap": gap,
                        "greedy_suboptimal": bool(gap > 1e-8),
                        "exact_greedy_jaccard": edge_jaccard(Aexact, Agreedy),
                    })
                print(f"completed d={d}, s={s}, noise={noise}, replicates={M}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "additional_simulation_metrics.csv", index=False)
    summary = df.groupby(["method"], as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("estimated_edges", "mean"), tp=("tp", "mean"),
        fp=("fp", "mean"), fn=("fn", "mean"),
        greedy_suboptimal_frequency=("greedy_suboptimal", "mean"),
        greedy_bic_gap_mean=("greedy_bic_gap", "mean"))
    summary.to_csv(out / "additional_simulation_summary.csv", index=False)
    pd.DataFrame(pruning).to_csv(out / "pruning_diagnostics.csv", index=False)
    (out / "additional_simulation_run_config.json").write_text(
        json.dumps({
            "base_seed": BASE_SEED,
            "replicates_per_setting": M,
            "n": n,
            "d": [10, 20],
            "s": [2, 5],
            "noise": ["normal", "exponential", "gumbel"],
            "gumbel_scale": float(np.sqrt(6.0) / np.pi),
            "gumbel_center": "beta * Euler gamma",
            "gumbel_variance": 1.0,
            "shd_convention": "one operation per reversal",
        }, indent=2) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--simulation-replicates", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_additional_simulations(args.out, M=args.simulation_replicates)


if __name__ == "__main__":
    main()
