#!/usr/bin/env python3
"""Additional NOTEARS-BP validation and proprietary-data workflow.

This script complements reproduce_simulations.py. It reproduces the compact
Gaussian/non-Gaussian validation experiment and, when an authorized
train2023cw_simple.csv is supplied locally, the original-scale, standardized,
continuous-only, runtime, and bootstrap-stability analyses reported in the
manuscript. No proprietary observations are included in this repository.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from reproduce_simulations import (
    BASE_SEED, LAMBDA1, THRESHOLD, directed_metrics, greedy_edge_removal,
    notears_linear, simulate_dag, simulate_parameter_uniform,
)

BINARY_VARS = [
    "PRRS_binary", "MYCO_binary", "LateralPRRS_binary", "Q2", "Q3", "Q4"
]
REAL_VARS = [
    "PRRS_binary", "MYCO_binary", "LateralPRRS_binary", "Q2", "Q3", "Q4",
    "Avg_parity_farrow", "Litters_female_year", "mated_inventory_20wks",
    "PWMFyear", "nonproductive_days", "number_services", "wean_to_service",
    "abortions_rate", "Total_born_avg", "Stillborn_avg", "Mummies_avg",
    "prenatal_losses_avg", "Born_alive_avg", "Gestation_days",
    "Interval_farrows", "Pre_weaning_mortality", "PWSow",
    "productive_days_rate", "services_per_inventory_N_rate", "repeats__rate",
    "gilts_bred_rate", "Last_week_wean_bred_rate", "pregnant_105days_rate",
    "Cull_rate_annual", "Sow_Death_rate", "avg_parity_at_farrow",
    "Lactation_days", "final_inventory", "Farrowing__rate", "HeadIn",
    "mortality_60days",
]


def adjacency(W):
    A = (np.abs(W) > THRESHOLD).astype(int)
    np.fill_diagonal(A, 0)
    return A


def prune(X, A):
    return nx.to_numpy_array(greedy_edge_removal(X, nx.DiGraph(A)), dtype=int)


def simulate_lsem_noise(W, n, noise, rng):
    G = nx.DiGraph(W)
    X = np.zeros((n, W.shape[0]))
    for j in nx.topological_sort(G):
        parents = list(G.predecessors(j))
        if noise == "normal":
            eps = rng.normal(0, 1, n)
        elif noise == "exponential":
            eps = rng.exponential(1, n) - 1
        elif noise == "gumbel":
            eps = rng.gumbel(0, 1, n) - 0.5772156649015329
        else:
            raise ValueError(noise)
        X[:, j] = X[:, parents] @ W[parents, j] + eps if parents else eps
    return X


def local_bic(X, child, parents):
    n = X.shape[0]
    y = X[:, child]
    D = np.column_stack([np.ones(n), X[:, parents]]) if parents else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    rss = max(float(np.sum((y - D @ coef) ** 2)), np.finfo(float).eps)
    return n * np.log(rss / n) + D.shape[1] * np.log(n)


def hc_bic(X, max_iter=200):
    """Simple add/delete Gaussian-BIC hill climbing from the empty DAG."""
    X = X - X.mean(axis=0, keepdims=True)
    d = X.shape[1]
    A = np.zeros((d, d), dtype=int)
    for _ in range(max_iter):
        best = (0.0, None, None)
        for u in range(d):
            for v in range(d):
                if u == v:
                    continue
                old_parents = list(np.where(A[:, v] == 1)[0])
                if A[u, v] == 0:
                    A2 = A.copy(); A2[u, v] = 1
                    if not nx.is_directed_acyclic_graph(nx.DiGraph(A2)):
                        continue
                    new_parents = old_parents + [u]
                    move = "add"
                else:
                    new_parents = [p for p in old_parents if p != u]
                    move = "delete"
                delta = local_bic(X, v, new_parents) - local_bic(X, v, old_parents)
                if delta < best[0]:
                    best = (delta, move, (u, v))
        if best[1] is None:
            break
        u, v = best[2]
        A[u, v] = 1 if best[1] == "add" else 0
    return A


def run_additional_simulations(out, M=20, n=500):
    rows, pruning = [], []
    noises = ["normal", "exponential", "gumbel"]
    for d in [10, 20]:
        for s in [2, 5]:
            for noise in noises:
                for rep in range(M):
                    seed = BASE_SEED + 100000 * d + 1000 * s + 100 * noises.index(noise) + rep
                    np.random.seed(seed)
                    B = simulate_dag(d, 2 * d)
                    W = simulate_parameter_uniform(B, s)
                    X = simulate_lsem_noise(W, n, noise, np.random.default_rng(seed + 17))
                    Wn = notears_linear(X, lambda1=LAMBDA1)
                    An = adjacency(Wn)
                    Abp = prune(X, An)
                    Xs = StandardScaler().fit_transform(X)
                    As = adjacency(notears_linear(Xs, lambda1=LAMBDA1))
                    Ahc = hc_bic(X)
                    for method, A in [("NOTEARS", An), ("NOTEARS_BP", Abp),
                                      ("STD_NOTEARS", As), ("HC_BIC", Ahc)]:
                        fdr, tpr, shd = directed_metrics(B, A)
                        rows.append({"d": d, "s": s, "noise": noise, "rep": rep,
                                     "method": method, "fdr": fdr, "tpr": tpr,
                                     "shd": shd, "estimated_edges": int(A.sum())})
                    removed = (An == 1) & (Abp == 0)
                    pruning.append({
                        "d": d, "s": s, "noise": noise, "rep": rep,
                        "notears_edges": int(An.sum()), "bp_edges": int(Abp.sum()),
                        "false_positive_removals": int(np.sum(removed & (B == 0))),
                        "true_positive_removals": int(np.sum(removed & (B == 1))),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out / "additional_simulation_metrics.csv", index=False)
    df.groupby("method", as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("estimated_edges", "mean")).to_csv(
            out / "additional_simulation_summary.csv", index=False)
    pd.DataFrame(pruning).to_csv(out / "pruning_diagnostics.csv", index=False)


def prepare_real_data(path):
    raw = pd.read_csv(path, na_values=["NA", "NaN", "null", ""])
    x = raw.copy()
    x["PRRS_binary"] = x["PRRSatPlacement"].astype(str).str.lower().eq("epidemic").astype(int)
    x["MYCO_binary"] = x["Mycoplasma_Status"].astype(str).str.lower().eq("endemic").astype(int)
    x["LateralPRRS_binary"] = x["LateralPRRS"].astype(str).str.lower().eq("yes").astype(int)
    quarter = pd.to_numeric(x["Year_Quarter"], errors="coerce")
    for q in [2, 3, 4]:
        x[f"Q{q}"] = quarter.eq(q).astype(int)
    return x[REAL_VARS].apply(pd.to_numeric, errors="coerce").dropna()


def edge_table(A, W, labels):
    return pd.DataFrame([{"from": labels[i], "to": labels[j], "weight": W[i, j]}
                         for i, j in np.argwhere(A == 1)])


def fit_real_variant(X, labels, name, out):
    t0 = time.perf_counter()
    W = notears_linear(X, lambda1=LAMBDA1)
    note_s = time.perf_counter() - t0
    An = adjacency(W)
    t0 = time.perf_counter()
    Abp = prune(X, An)
    prune_s = time.perf_counter() - t0
    edge_table(An, W, labels).to_csv(out / f"{name}_notears_edges.csv", index=False)
    edge_table(Abp, W, labels).to_csv(out / f"{name}_bp_edges.csv", index=False)
    return W, An, Abp, {"analysis": name, "n": X.shape[0], "d": X.shape[1],
                         "notears_edges": int(An.sum()), "bp_edges": int(Abp.sum()),
                         "notears_seconds": note_s, "pruning_seconds": prune_s}


def run_real_data(csv_path, out, B=100):
    df = prepare_real_data(csv_path)
    labels = list(df.columns)
    X = df.to_numpy(float)
    summaries = []
    W, An, Abp, row = fit_real_variant(X, labels, "original", out)
    summaries.append(row)
    _, _, _, row = fit_real_variant(StandardScaler().fit_transform(X), labels,
                                     "standardized", out)
    summaries.append(row)
    cont_labels = [c for c in labels if c not in BINARY_VARS]
    _, _, _, row = fit_real_variant(df[cont_labels].to_numpy(float), cont_labels,
                                     "continuous_only", out)
    summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / "realdata_sensitivity_summary.csv", index=False)

    indegree_binary = {b: int(An[:, labels.index(b)].sum()) for b in BINARY_VARS}
    pd.DataFrame({"variable": list(indegree_binary),
                  "initial_notears_indegree": list(indegree_binary.values())}).to_csv(
                      out / "binary_node_indegree_diagnostic.csv", index=False)

    mortality = labels.index("mortality_60days")
    pd.DataFrame([{"graph": "NOTEARS", "incoming": int(An[:, mortality].sum()),
                   "outgoing": int(An[mortality, :].sum())},
                  {"graph": "NOTEARS-BP", "incoming": int(Abp[:, mortality].sum()),
                   "outgoing": int(Abp[mortality, :].sum())}]).to_csv(
                       out / "mortality_direction_summary.csv", index=False)

    rng = np.random.default_rng(BASE_SEED)
    freq = np.zeros_like(Abp, dtype=float)
    for b in range(B):
        ix = rng.integers(0, X.shape[0], X.shape[0])
        Wb = notears_linear(X[ix], lambda1=LAMBDA1)
        freq += prune(X[ix], adjacency(Wb))
        print(f"bootstrap {b + 1}/{B}")
    freq /= B
    pd.DataFrame([{"threshold": 0.50, "edges": int(np.sum(freq >= 0.50))},
                  {"threshold": 0.70, "edges": int(np.sum(freq >= 0.70))}]).to_csv(
                       out / "bootstrap_stability_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--simulation-replicates", type=int, default=20)
    ap.add_argument("--real-data", type=Path)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--skip-simulations", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_simulations:
        run_additional_simulations(args.out, M=args.simulation_replicates)
    if args.real_data is not None:
        run_real_data(args.real_data, args.out, B=args.bootstrap)


if __name__ == "__main__":
    main()
