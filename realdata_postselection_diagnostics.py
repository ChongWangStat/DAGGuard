#!/usr/bin/env python3
"""Targeted diagnostics for the proprietary swine-production application.

The script contains no commercial observations. Given an authorized local
copy of train2023cw_simple.csv, it reproduces the diagnostics added to the JDS
revision:
  * missingness and collinearity/VIF audit;
  * binary-node and Q2/Q3/Q4 diagnostics;
  * mortality-neighborhood partial-R2 and Delta-BIC evidence;
  * EBIC sensitivity;
  * fixed-candidate redundancy sensitivity;
  * arbitrary unit-change sensitivity;
  * optional row-bootstrap recurrence for NOTEARS and NOTEARS-BP.

Each diagnostic answers a specific alternative explanation for the real-data
result; none is presented as a substitute for the primary analysis.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler

from additional_validation_and_realdata import (
    BINARY_VARS, REAL_VARS, LAMBDA1, THRESHOLD,
)
from reproduce_simulations import notears_linear


def adjacency(W):
    A = (np.abs(W) > THRESHOLD).astype(int)
    np.fill_diagonal(A, 0)
    return A


def local_bic(X, child, parents, gamma=0.0):
    Xc = X - X.mean(axis=0, keepdims=True)
    y = Xc[:, child]
    parents = list(parents)
    if parents:
        Z = Xc[:, parents]
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        resid = y - Z @ beta
    else:
        resid = y
    n, d = X.shape
    rss = max(float(resid @ resid), np.finfo(float).eps)
    k = len(parents) + 1
    score = n * math.log(rss / n) + k * math.log(n)
    if gamma > 0:
        p = d - 1
        q = len(parents)
        logchoose = gammaln(p + 1) - gammaln(q + 1) - gammaln(p - q + 1)
        score += 2 * gamma * logchoose
    return score, rss


def prune(X, A0, gamma=0.0, record=False):
    A = A0.copy().astype(int)
    removed = []
    while A.sum():
        best = None
        for u, v in np.argwhere(A == 1):
            pb = list(np.where(A[:, v] == 1)[0])
            pa = [p for p in pb if p != u]
            bf, rssf = local_bic(X, v, pb, gamma)
            br, rssr = local_bic(X, v, pa, gamma)
            delta = br - bf
            pr2 = 1 - rssf / rssr
            item = (delta, int(u), int(v), pr2, bf, br)
            if best is None or delta < best[0]:
                best = item
        if best is None or best[0] >= 0:
            break
        delta, u, v, pr2, bf, br = best
        A[u, v] = 0
        if record:
            removed.append(dict(u=u, v=v, delta_bic=delta, partial_R2=pr2,
                                bic_before=bf, bic_after=br))
    return A, removed


def prep_with_missing(path):
    raw = pd.read_csv(path, na_values=["NA", "NaN", "null", ""])
    x = raw.copy()
    x["PRRS_binary"] = x["PRRSatPlacement"].astype(str).str.lower().eq("epidemic").astype(int)
    x["MYCO_binary"] = x["Mycoplasma_Status"].astype(str).str.lower().eq("endemic").astype(int)
    x["LateralPRRS_binary"] = x["LateralPRRS"].astype(str).str.lower().eq("yes").astype(int)
    quarter = pd.to_numeric(x["Year_Quarter"], errors="coerce")
    for q in [2, 3, 4]:
        x[f"Q{q}"] = quarter.eq(q).astype(int)
    numeric = x[REAL_VARS].apply(pd.to_numeric, errors="coerce")
    return raw, numeric, numeric.dropna().copy()


def collinearity_audit(df, out):
    cont = [v for v in REAL_VARS if v not in BINARY_VARS]
    corr = df[cont].corr()
    pairs = []
    for i in range(len(cont)):
        for j in range(i + 1, len(cont)):
            pairs.append((abs(corr.iloc[i, j]), corr.iloc[i, j], cont[i], cont[j]))
    pairs.sort(reverse=True)
    pd.DataFrame(pairs, columns=["abs_r", "r", "var1", "var2"]).to_csv(
        out / "top_correlations.csv", index=False)

    Xs = StandardScaler().fit_transform(df[cont].to_numpy(float))
    singular = np.linalg.svd(Xs, compute_uv=False)
    condition = float(singular[0] / singular[-1])
    vifs = []
    for j, name in enumerate(cont):
        y = Xs[:, j]
        Z = np.delete(Xs, j, axis=1)
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        resid = y - Z @ beta
        r2 = 1 - float(resid @ resid) / float(y @ y)
        vifs.append((name, r2, 1 / (1 - r2) if r2 < 1 else np.inf))
    vdf = pd.DataFrame(vifs, columns=["variable", "R2_from_other_continuous", "VIF"])
    vdf.sort_values("VIF", ascending=False).to_csv(out / "vif_continuous.csv", index=False)

    def projection_r2(yname, xnames):
        y = df[yname].to_numpy(float)
        Z = np.column_stack([np.ones(len(df)), df[xnames].to_numpy(float)])
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        resid = y - Z @ beta
        return 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())

    checks = [
        dict(diagnostic="max_abs_pairwise_correlation", value=pairs[0][0],
             detail=f"{pairs[0][2]} vs {pairs[0][3]}; signed r={pairs[0][1]:.6f}"),
        dict(diagnostic="standardized_continuous_condition_number", value=condition,
             detail="continuous-variable design matrix"),
        dict(diagnostic="max_VIF", value=float(vdf.VIF.max()),
             detail=str(vdf.sort_values('VIF', ascending=False).iloc[0].variable)),
        dict(diagnostic="parity_variable_correlation",
             value=float(df["Avg_parity_farrow"].corr(df["avg_parity_at_farrow"])),
             detail="Avg_parity_farrow vs avg_parity_at_farrow"),
        dict(diagnostic="R2_total_born_from_components",
             value=projection_r2("Total_born_avg", ["Born_alive_avg", "Stillborn_avg", "Mummies_avg"]),
             detail="Born_alive_avg + Stillborn_avg + Mummies_avg"),
        dict(diagnostic="R2_prenatal_losses_from_components",
             value=projection_r2("prenatal_losses_avg", ["Stillborn_avg", "Mummies_avg"]),
             detail="Stillborn_avg + Mummies_avg"),
    ]
    pd.DataFrame(checks).to_csv(out / "realdata_collinearity_audit.csv", index=False)


def mortality_diagnostics(X, A0, Afinal, W, labels, removed, out):
    idx = {name: i for i, name in enumerate(labels)}
    m = idx["mortality_60days"]
    remap = {(r["u"], r["v"]): r for r in removed}
    cutoff = 1 - X.shape[0] ** (-1 / X.shape[0])
    rows = []
    for u, v in np.argwhere(A0 == 1):
        if u != m and v != m:
            continue
        if Afinal[u, v]:
            pb = list(np.where(Afinal[:, v] == 1)[0])
            pa = [p for p in pb if p != u]
            bf, rssf = local_bic(X, v, pb)
            br, rssr = local_bic(X, v, pa)
            pr2, delta = 1 - rssf / rssr, br - bf
            status = "retained"
        else:
            rec = remap[(int(u), int(v))]
            pr2, delta, status = rec["partial_R2"], rec["delta_bic"], "removed"
        rows.append(dict(from_=labels[u], to=labels[v], weight=W[u, v], status=status,
                         partial_R2_relevant_step=pr2, delta_BIC_relevant_step=delta,
                         BIC_partial_R2_cutoff=cutoff))
    pd.DataFrame(rows).rename(columns={"from_": "from"}).to_csv(
        out / "mortality_partial_r2_diagnostics.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Authorized local train2023cw_simple.csv")
    ap.add_argument("--out", type=Path, default=Path("results/realdata_diagnostics"))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Optional row-bootstrap replicates; 0 skips this expensive check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw, numeric, df = prep_with_missing(args.data)
    numeric.isna().sum().sort_values(ascending=False).to_csv(
        args.out / "missingness_by_variable.csv", header=["missing_n"])
    collinearity_audit(df, args.out)

    X = df[REAL_VARS].to_numpy(float)
    labels = list(REAL_VARS)
    W = notears_linear(X, lambda1=LAMBDA1)
    A0 = adjacency(W)
    Abp, removed = prune(X, A0, record=True)

    indeg = [{"variable": b, "initial_notears_indegree": int(A0[:, labels.index(b)].sum())}
             for b in BINARY_VARS]
    pd.DataFrame(indeg).to_csv(args.out / "binary_node_indegree_diagnostic.csv", index=False)
    qedges = []
    for a in ["Q2", "Q3", "Q4"]:
        for b in ["Q2", "Q3", "Q4"]:
            if a != b and A0[labels.index(a), labels.index(b)]:
                qedges.append(dict(from_=a, to=b))
    pd.DataFrame(qedges).rename(columns={"from_": "from"}).to_csv(
        args.out / "quarter_within_block_edges.csv", index=False)

    mortality_diagnostics(X, A0, Abp, W, labels, removed, args.out)

    ebic = []
    m = labels.index("mortality_60days")
    for gamma in [0.0, 0.5, 1.0]:
        A, _ = prune(X, A0, gamma=gamma)
        ebic.append(dict(gamma=gamma, edges=int(A.sum()),
                         mortality_in=int(A[:, m].sum()), mortality_out=int(A[m, :].sum())))
    pd.DataFrame(ebic).to_csv(args.out / "ebic_sensitivity.csv", index=False)

    remove = ["Avg_parity_farrow", "productive_days_rate", "Total_born_avg", "prenatal_losses_avg"]
    keep = [v for v in labels if v not in remove]
    ix = [labels.index(v) for v in keep]
    Xr = X[:, ix]
    A0r = A0[np.ix_(ix, ix)]
    Abr, _ = prune(Xr, A0r)
    Abp_restricted = Abp[np.ix_(ix, ix)]
    e1 = set(map(tuple, np.argwhere(Abr == 1)))
    e2 = set(map(tuple, np.argwhere(Abp_restricted == 1)))
    jaccard = len(e1 & e2) / len(e1 | e2)
    mr = keep.index("mortality_60days")
    pd.DataFrame([dict(analysis="fixed-candidate redundancy-reduced pruning",
                       n=Xr.shape[0], d=Xr.shape[1], removed_variables="; ".join(remove),
                       restricted_candidate_edges=int(A0r.sum()), bp_edges=int(Abr.sum()),
                       original_bp_edges_on_retained_variables=int(Abp_restricted.sum()),
                       bp_jaccard_vs_original_restricted=jaccard,
                       mortality_in=int(Abr[:, mr].sum()), mortality_out=int(Abr[mr, :].sum()))]).to_csv(
                           args.out / "redundancy_pruning_sensitivity.csv", index=False)

    Xscaled = X.copy()
    for v in ["HeadIn", "final_inventory"]:
        Xscaled[:, labels.index(v)] /= 1000.0
    Ws = notears_linear(Xscaled, lambda1=LAMBDA1)
    A0s = adjacency(Ws)
    Abs, _ = prune(Xscaled, A0s)
    def jaccard_edges(A, B):
        a = set(map(tuple, np.argwhere(A == 1))); b = set(map(tuple, np.argwhere(B == 1)))
        return len(a & b) / len(a | b)
    pd.DataFrame([
        dict(metric="notears_edges_original", value=int(A0.sum())),
        dict(metric="notears_edges_scaled", value=int(A0s.sum())),
        dict(metric="notears_jaccard", value=jaccard_edges(A0, A0s)),
        dict(metric="bp_edges_original", value=int(Abp.sum())),
        dict(metric="bp_edges_scaled", value=int(Abs.sum())),
        dict(metric="bp_jaccard", value=jaccard_edges(Abp, Abs)),
    ]).to_csv(args.out / "unit_change_sensitivity.csv", index=False)

    if args.bootstrap > 0:
        rng = np.random.default_rng(12123)
        F0 = np.zeros_like(A0, float); Fb = np.zeros_like(Abp, float)
        for b in range(args.bootstrap):
            ixr = rng.integers(0, X.shape[0], X.shape[0])
            Wb = notears_linear(X[ixr], lambda1=LAMBDA1)
            A0b = adjacency(Wb); Abb, _ = prune(X[ixr], A0b)
            F0 += A0b; Fb += Abb
            print(f"bootstrap {b+1}/{args.bootstrap}")
        F0 /= args.bootstrap; Fb /= args.bootstrap
        rows = []
        for name, Aorig, F in [("NOTEARS", A0, F0), ("NOTEARS-BP", Abp, Fb)]:
            for th in [0.5, 0.7]:
                recur = int(((F >= th) & (Aorig == 1)).sum())
                rows.append(dict(method=name, threshold=th, original_edges=int(Aorig.sum()),
                                 original_edges_recur=recur, fraction=recur / Aorig.sum(),
                                 all_edges_above=int((F >= th).sum())))
        pd.DataFrame(rows).to_csv(args.out / "bootstrap_stability_comparison.csv", index=False)

    print(f"complete cases: {len(df)}/{len(raw)}")
    print(f"NOTEARS edges: {int(A0.sum())}; BP edges: {int(Abp.sum())}")


if __name__ == "__main__":
    main()
