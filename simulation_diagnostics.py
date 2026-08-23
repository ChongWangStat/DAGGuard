#!/usr/bin/env python3
"""Targeted diagnostics for NOTEARS-BP simulations.

This script complements reproduce_simulations.py with diagnostics designed to
answer specific questions about the pruning mechanism rather than to create a
larger benchmark:

1. Equal-sparsity magnitude controls: if BP retains K edges, compare it with
   after-the-fact controls that retain exactly K NOTEARS candidate edges by raw
   or unit-normalized coefficient magnitude. Because K is supplied by BP,
   these are diagnostic controls, not practical competing estimators.
2. Varsortability audit: quantify how strongly marginal variances encode causal
   order in every primary simulation setting.
3. Standardized-data diagnostic: show that BP does not rescue an initialization
   that has already lost the relevant structure.
4. Modified-normal equal-sparsity check: verify that the mechanism is not
   specific to uniformly distributed structural coefficients.
5. Sample-size diagnostic: directly exercise the n-dependent BIC/partial-R2
   cutoff at n in {100, 500, 2000}.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from additional_noise_sensitivity import (
    adjacency,
    simulate_dag_seeded,
    simulate_weights_seeded,
    simulate_lsem_noise,
)
from local_bic_refinement import (
    exact_refine_dag,
    graph_metrics,
    greedy_refine_dag,
    initial_pruning_pressure as canonical_initial_pruning_pressure,
)
from reproduce_simulations import notears_linear

BASE_SEED = 12123
DEFAULT_N = 500


def metrics(T, A):
    result = graph_metrics(T, A)
    return dict(
        edges=result["edges"], tp=result["true_positives"],
        fp=result["false_positives"], fn=result["false_negatives"],
        fdr=result["fdr"], tpr=result["tpr"], shd=result["shd"],
    )


def varsortability(A, X):
    var = X.var(axis=0)
    Ak = A.astype(float).copy()
    num = den = 0.0
    for k in range(1, A.shape[0]):
        if k > 1:
            Ak = Ak @ A
        i, j = np.where(Ak > 0)
        w = Ak[i, j]
        inc = (var[i] < var[j]).astype(float) + 0.5 * (var[i] == var[j])
        num += float(np.dot(w, inc))
        den += float(w.sum())
    return num / den if den else np.nan


def top_k(W, candidate, k, sd=None):
    score = np.abs(W) if sd is None else np.abs(W) * sd[:, None] / sd[None, :]
    ranked = sorted(((score[u, v], int(u), int(v))
                     for u, v in np.argwhere(candidate == 1)), reverse=True)
    out = np.zeros_like(candidate)
    for _, u, v in ranked[:k]:
        out[u, v] = 1
    return out


def modnormal_weights(T, s, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, s, size=T.shape)
    z = np.where(z >= 0, z + 0.5, z - 0.5)
    W = np.zeros_like(z)
    W[T != 0] = z[T != 0]
    return W


def fit_candidate_and_bp(T, W, n, seed):
    X = simulate_lsem_noise(W, n, "normal", seed)
    Wh = notears_linear(X, lambda1=0.1)
    A0 = adjacency(Wh)
    exact = exact_refine_dag(X, A0)
    greedy = greedy_refine_dag(X, A0)
    if not exact.globally_optimal:
        raise RuntimeError("Exact local-BIC search was not certified")
    return X, Wh, A0, exact.adjacency, greedy.adjacency


def initial_pruning_pressure(X, A):
    """Fraction of current edges below the one-pass BIC partial-R2 cutoff."""
    summary, _ = canonical_initial_pruning_pressure(X, A)
    return summary["initial_pruning_pressure"]


def write_summary(df, groups, path):
    df.groupby(groups, as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"),
        fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(path, index=False)


def run_equal_sparsity_uniform(out, M):
    rows = []
    for s in [1, 4, 7]:
        si = [1, 4, 7, 10].index(s)
        for rep in range(M):
            seed = BASE_SEED + 1000 * si + rep
            T = simulate_dag_seeded(10, 20, seed)
            W = simulate_weights_seeded(T, s, seed + 100000)
            X, Wh, A0, Aexact, Agreedy = fit_candidate_and_bp(
                T, W, DEFAULT_N, seed + 200000)
            k = int(Aexact.sum())
            for name, A in [
                ("NOTEARS", A0),
                ("Local-BIC exact", Aexact),
                ("Local-BIC greedy", Agreedy),
                ("Equal-sparsity raw magnitude", top_k(Wh, A0, k)),
                ("Equal-sparsity unit-normalized magnitude",
                 top_k(Wh, A0, k, X.std(axis=0))),
            ]:
                rows.append(dict(s=s, rep=rep, method=name, **metrics(T, A)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "equal_sparsity_uniform_replicates.csv", index=False)
    write_summary(df, ["s", "method"], out / "equal_sparsity_uniform_summary.csv")


def run_equal_sparsity_modnormal(out, M):
    rows = []
    for rep in range(M):
        seed = BASE_SEED + 50000 + rep
        T = simulate_dag_seeded(10, 20, seed)
        W = modnormal_weights(T, 3, seed + 100000)
        X, Wh, A0, Aexact, Agreedy = fit_candidate_and_bp(
            T, W, DEFAULT_N, seed + 200000)
        k = int(Aexact.sum())
        for name, A in [
            ("NOTEARS", A0),
            ("Local-BIC exact", Aexact),
            ("Local-BIC greedy", Agreedy),
            ("Equal-sparsity raw magnitude", top_k(Wh, A0, k)),
            ("Equal-sparsity unit-normalized magnitude",
             top_k(Wh, A0, k, X.std(axis=0))),
        ]:
            rows.append(dict(d=10, kind="modnormal", s=3, n=500,
                             rep=rep, method=name, **metrics(T, A)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "equal_sparsity_modnormal_replicates.csv", index=False)
    write_summary(df, ["d", "kind", "s", "n", "method"],
                  out / "equal_sparsity_modnormal_summary.csv")


def run_varsortability(out, M):
    rows = []
    for kind, svals in [("uniform", [1, 4, 7, 10]),
                        ("modnormal", [1, 2, 3, 4])]:
        for d in [10, 20, 40]:
            for si, s in enumerate(svals):
                for rep in range(M):
                    seed = BASE_SEED + 1000 * si + rep
                    T = simulate_dag_seeded(d, 2 * d, seed)
                    W = (simulate_weights_seeded(T, s, seed + 100000)
                         if kind == "uniform"
                         else modnormal_weights(T, s, seed + 100000))
                    X = simulate_lsem_noise(W, DEFAULT_N, "normal", seed + 200000)
                    rows.append(dict(
                        weight_kind=kind, d=d, s=s, rep=rep,
                        varsortability=varsortability(T, X),
                        sd_ratio=X.std(axis=0).max() / X.std(axis=0).min()))
    df = pd.DataFrame(rows)
    df.to_csv(out / "varsortability_primary_replicates.csv", index=False)
    df.groupby(["weight_kind", "d", "s"], as_index=False).agg(
        varsortability_mean=("varsortability", "mean"),
        varsortability_sd=("varsortability", "std"),
        sd_ratio_median=("sd_ratio", "median"),
        sd_ratio_mean=("sd_ratio", "mean")
    ).to_csv(out / "varsortability_primary_summary.csv", index=False)


def run_standardized_diagnostic(out, M):
    rows = []
    diag = []
    for rep in range(M):
        seed = 17123 + rep
        T = simulate_dag_seeded(10, 20, seed)
        W = simulate_weights_seeded(T, 7, seed + 100000)
        X = simulate_lsem_noise(W, DEFAULT_N, "normal", seed + 200000)
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        Wh = notears_linear(Xs, lambda1=0.1)
        A0 = adjacency(Wh)
        Aexact = exact_refine_dag(Xs, A0).adjacency
        Agreedy = greedy_refine_dag(Xs, A0).adjacency
        for name, A in [("Standardized NOTEARS", A0),
                        ("Standardized local-BIC exact", Aexact),
                        ("Standardized local-BIC greedy", Agreedy)]:
            rows.append(dict(rep=rep, method=name, **metrics(T, A)))
        diag.append(dict(raw_varsortability=varsortability(T, X),
                         standardized_varsortability=varsortability(T, Xs)))
    df = pd.DataFrame(rows)
    write_summary(df, ["method"], out / "standardized_varsortability_summary.csv")
    pd.DataFrame(diag).to_csv(out / "standardized_varsortability_diagnostics.csv",
                              index=False)


def run_sample_size(out, M):
    rows = []
    for n in [100, 500, 2000]:
        for rep in range(M):
            seed = BASE_SEED + 7000 + rep
            T = simulate_dag_seeded(10, 20, seed)
            W = simulate_weights_seeded(T, 7, seed + 100000)
            _, _, A0, Aexact, Agreedy = fit_candidate_and_bp(
                T, W, n, seed + 200000)
            for name, A in [("NOTEARS", A0),
                            ("Local-BIC exact", Aexact),
                            ("Local-BIC greedy", Agreedy)]:
                row = dict(n=n, rep=rep, method=name, **metrics(T, A))
                row["partial_r2_cutoff"] = 1 - n ** (-1 / n)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sample_size_replicates.csv", index=False)
    df.groupby(["n", "method"], as_index=False).agg(
        partial_r2_cutoff=("partial_r2_cutoff", "first"),
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"),
        fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(out / "sample_size_diagnostic.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("results/simulation_diagnostics"))
    p.add_argument("--M", type=int, default=20,
                   help="Replicates for NOTEARS-based diagnostics")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_equal_sparsity_uniform(args.out, args.M)
    run_equal_sparsity_modnormal(args.out, args.M)
    run_varsortability(args.out, args.M)
    run_standardized_diagnostic(args.out, args.M)
    run_sample_size(args.out, args.M)


if __name__ == "__main__":
    main()
