#!/usr/bin/env python3
"""Synthetic application twin for the DAGGuard workflow.

This script mirrors the public workflow without proprietary swine observations:
it simulates a 37-variable linear Gaussian DAG, creates an intentionally
overconnected but acyclic candidate, and runs both DAGGuard algorithms.  It is
not intended to mimic the confidential data distribution; its purpose is to
provide an executable end-to-end example with the same dimensionality and
public API as the application.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from dagguard import dagguard_exact, dagguard_greedy, edge_jaccard, graph_metrics


def make_problem(n: int = 1200, d: int = 37, seed: int = 20260824):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    truth = np.zeros((d, d), dtype=int)
    weights = np.zeros((d, d), dtype=float)
    # Sparse forward DAG with at most two true parents per child.
    for pos in range(1, d):
        child = int(order[pos])
        possible = order[:pos]
        q = min(2, len(possible))
        if q:
            parents = rng.choice(possible, size=q, replace=False)
            for parent in parents:
                truth[int(parent), child] = 1
                weights[int(parent), child] = rng.choice([-1.0, 1.0]) * rng.uniform(0.55, 1.15)

    X = np.zeros((n, d), dtype=float)
    graph = nx.DiGraph(truth)
    for child in nx.topological_sort(graph):
        parents = np.flatnonzero(truth[:, child])
        signal = X[:, parents] @ weights[parents, child] if len(parents) else 0.0
        X[:, child] = signal + rng.normal(size=n)

    candidate = truth.copy()
    # Add false forward edges while capping candidate indegree at six.
    pairs = []
    pos_of = {int(node): i for i, node in enumerate(order)}
    for u in range(d):
        for v in range(d):
            if u != v and pos_of[u] < pos_of[v] and candidate[u, v] == 0:
                pairs.append((u, v))
    rng.shuffle(pairs)
    added = 0
    for u, v in pairs:
        if int(candidate[:, v].sum()) >= 6:
            continue
        candidate[u, v] = 1
        added += 1
        if added >= 40:
            break
    return X, truth, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/synthetic_application_twin")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, truth, candidate = make_problem(seed=args.seed)
    exact = dagguard_exact(X, candidate)
    greedy = dagguard_greedy(X, candidate)
    rows = []
    for name, A, runtime in [
        ("Candidate", candidate, 0.0),
        ("DAGGuard-Exact", exact.adjacency, exact.runtime_seconds),
        ("DAGGuard-Greedy", greedy.adjacency, greedy.runtime_seconds),
    ]:
        row = {"method": name, "runtime_seconds": runtime, **graph_metrics(truth, A)}
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["exact_jaccard"] = [
        edge_jaccard(candidate, exact.adjacency),
        1.0,
        edge_jaccard(greedy.adjacency, exact.adjacency),
    ]
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(candidate).to_csv(out / "candidate_adjacency.csv", index=False)
    pd.DataFrame(exact.adjacency).to_csv(out / "exact_adjacency.csv", index=False)
    pd.DataFrame(greedy.adjacency).to_csv(out / "greedy_adjacency.csv", index=False)
    print(summary.to_string(index=False))
    print(f"exact_certified={exact.globally_optimal}")


if __name__ == "__main__":
    main()
