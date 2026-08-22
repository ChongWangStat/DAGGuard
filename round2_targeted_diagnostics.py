#!/usr/bin/env python3
"""Second-round targeted diagnostics for the NOTEARS-BP JDS revision.

This script extends the original equal-sparsity mechanism check to a
modified-normal coefficient regime and varies n to directly exercise the
sample-size-dependent BIC cutoff. The equal-sparsity controls are diagnostic
only: their edge count K is supplied after the fact by NOTEARS-BP.
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
from targeted_simulation_diagnostics import metrics, top_k

BASE_SEED = 12123


def seed_for(d, kind, s, rep):
    svals = {"uniform": [1, 4, 7, 10], "modnormal": [1, 2, 3, 4]}[kind]
    return BASE_SEED + 1000 * svals.index(s) + rep + 100000 * d + (50000 if kind == "modnormal" else 0)


def weights(T, kind, s, seed):
    if kind == "uniform":
        return simulate_weights_seeded(T, s, seed)
    rng = np.random.default_rng(seed)
    z = rng.normal(0, s, size=T.shape)
    z = np.where(z >= 0, z + 0.5, z - 0.5)
    W = np.zeros_like(z)
    W[T != 0] = z[T != 0]
    return W


def run_one(d, kind, s, rep, n):
    seed = seed_for(d, kind, s, rep)
    T = simulate_dag_seeded(d, 2 * d, seed)
    W = weights(T, kind, s, seed + 100000)
    X = simulate_lsem_noise(W, n, "normal", seed + 200000)
    Wh = notears_linear(X, lambda1=0.1)
    A0 = adjacency(Wh)
    Abp = prune(X, A0)
    k = int(Abp.sum())
    sd = X.std(axis=0)
    out = []
    for name, A in [
        ("NOTEARS", A0),
        ("NOTEARS-BP", Abp),
        ("Equal-sparsity raw magnitude", top_k(Wh, A0, k)),
        ("Equal-sparsity unit-normalized magnitude", top_k(Wh, A0, k, sd)),
    ]:
        out.append(dict(d=d, kind=kind, s=s, n=n, rep=rep, method=name, **metrics(T, A)))
    return out


def summarize(df, groups, path):
    df.groupby(groups, as_index=False).agg(
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/round2_diagnostics"))
    ap.add_argument("--M", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Why: test whether the equal-sparsity advantage is specific to uniform weights.
    rows = []
    for rep in range(args.M):
        rows.extend(run_one(10, "modnormal", 3, rep, 500))
    mod = pd.DataFrame(rows)
    mod.to_csv(args.out / "equal_sparsity_modnormal_replicates.csv", index=False)
    summarize(mod, ["d", "kind", "s", "n", "method"], args.out / "equal_sparsity_modnormal_summary.csv")

    # Why: empirically exercise the n-dependent BIC cutoff rather than cite only its formula.
    rows = []
    for n in [100, 500, 2000]:
        for rep in range(args.M):
            for row in run_one(10, "uniform", 7, rep, n):
                if row["method"] in {"NOTEARS", "NOTEARS-BP"}:
                    row["partial_r2_cutoff"] = 1 - n ** (-1 / n)
                    rows.append(row)
    ns = pd.DataFrame(rows)
    ns.to_csv(args.out / "sample_size_replicates.csv", index=False)
    ns.groupby(["n", "method"], as_index=False).agg(
        partial_r2_cutoff=("partial_r2_cutoff", "first"),
        fdr=("fdr", "mean"), tpr=("tpr", "mean"), shd=("shd", "mean"),
        edges=("edges", "mean"), tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean")
    ).to_csv(args.out / "sample_size_diagnostic.csv", index=False)


if __name__ == "__main__":
    main()
