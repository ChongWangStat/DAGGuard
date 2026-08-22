#!/usr/bin/env python3
"""Targeted simulation diagnostics added for the JDS revision.

The equal-sparsity control is an ablation, not a competing estimator: if BP
retains K edges, it keeps exactly K NOTEARS candidate edges by magnitude so
that graph size is held fixed. The varsortability diagnostics quantify how
strongly marginal variances encode causal order in the synthetic design.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from additional_validation_and_realdata import (
    adjacency, prune, simulate_dag_seeded, simulate_weights_seeded,
    simulate_lsem_noise,
)
from reproduce_simulations import notears_linear

BASE_SEED = 12123
N = 500


def metrics(T, A):
    tp = int(((T == 1) & (A == 1)).sum())
    fp = int(((T == 0) & (A == 1)).sum())
    fn = int(((T == 1) & (A == 0)).sum())
    est = int(A.sum())
    shd = int(np.abs(T - A).sum())
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] == 1 and A[i, j] == 0 and A[j, i] == 1:
                shd -= 1
    return dict(edges=est, tp=tp, fp=fp, fn=fn,
                fdr=fp / est if est else np.nan,
                tpr=tp / T.sum(), shd=shd)


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
        num += float(np.dot(w, inc)); den += float(w.sum())
    return num / den if den else np.nan


def top_k(W, candidate, k, sd=None):
    score = np.abs(W) if sd is None else np.abs(W) * sd[:, None] / sd[None, :]
    ranked = sorted(((score[u, v], int(u), int(v)) for u, v in np.argwhere(candidate == 1)), reverse=True)
    out = np.zeros_like(candidate)
    for _, u, v in ranked[:k]: out[u, v] = 1
    return out


def run_equal_sparsity(out, M):
    rows = []
    for s in [1, 4, 7]:
        si = [1, 4, 7, 10].index(s)
        for rep in range(M):
            seed = BASE_SEED + 1000 * si + rep
            T = simulate_dag_seeded(10, 20, seed)
            W = simulate_weights_seeded(T, s, seed + 100000)
            X = simulate_lsem_noise(W, N, "normal", seed + 200000)
            Wh = notears_linear(X, lambda1=0.1)
            A0 = adjacency(Wh); Abp = prune(X, A0); k = int(Abp.sum())
            for name, A in [
                ("NOTEARS", A0), ("NOTEARS-BP", Abp),
                ("Equal-sparsity |W|", top_k(Wh, A0, k)),
                ("Equal-sparsity standardized |W|", top_k(Wh, A0, k, X.std(axis=0))),
            ]:
                rows.append(dict(s=s, rep=rep, method=name, **metrics(T, A)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "equal_sparsity_ablation_replicates.csv", index=False)
    df.groupby(["s", "method"], as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(out / "equal_sparsity_ablation_summary.csv", index=False)


def run_varsortability(out, M):
    rows = []
    for kind, svals in [("uniform", [1, 4, 7, 10]), ("modnormal", [1, 2, 3, 4])]:
        for d in [10, 20, 40]:
            for si, s in enumerate(svals):
                for rep in range(M):
                    seed = BASE_SEED + 1000 * si + rep
                    T = simulate_dag_seeded(d, 2 * d, seed)
                    if kind == "uniform":
                        W = simulate_weights_seeded(T, s, seed + 100000)
                    else:
                        rng = np.random.default_rng(seed + 100000)
                        z = rng.normal(0, s, size=T.shape)
                        z = np.where(z >= 0, z + 0.5, z - 0.5)
                        W = np.zeros_like(z); W[T != 0] = z[T != 0]
                    X = simulate_lsem_noise(W, N, "normal", seed + 200000)
                    rows.append(dict(weight_kind=kind, d=d, s=s, rep=rep,
                                     varsortability=varsortability(T, X),
                                     sd_ratio=X.std(axis=0).max() / X.std(axis=0).min()))
    df = pd.DataFrame(rows)
    df.groupby(["weight_kind", "d", "s"], as_index=False).agg(
        varsortability_mean=("varsortability", "mean"),
        varsortability_sd=("varsortability", "std"),
        sd_ratio_median=("sd_ratio", "median"), sd_ratio_mean=("sd_ratio", "mean")
    ).to_csv(out / "varsortability_primary_summary.csv", index=False)


def run_standardized(out, M):
    rows = []; diag = []
    for rep in range(M):
        seed = 17123 + rep
        T = simulate_dag_seeded(10, 20, seed)
        W = simulate_weights_seeded(T, 7, seed + 100000)
        X = simulate_lsem_noise(W, N, "normal", seed + 200000)
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        Wh = notears_linear(Xs, lambda1=0.1)
        A0 = adjacency(Wh); Abp = prune(Xs, A0)
        for name, A in [("Standardized NOTEARS", A0), ("Standardized NOTEARS-BP", Abp)]:
            rows.append(dict(rep=rep, method=name, **metrics(T, A)))
        diag.append(dict(raw_varsortability=varsortability(T, X),
                         standardized_varsortability=varsortability(T, Xs)))
    df = pd.DataFrame(rows)
    df.groupby("method", as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(out / "standardized_varsortability_summary.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "standardized_varsortability_diagnostics.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/targeted_diagnostics"))
    p.add_argument("--M", type=int, default=20)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    run_equal_sparsity(args.out, args.M)
    run_varsortability(args.out, args.M)
    run_standardized(args.out, args.M)


if __name__ == "__main__": main()
