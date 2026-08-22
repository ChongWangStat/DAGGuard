#!/usr/bin/env python3
"""Reproduce the NOTEARS-BP primary simulation design.

The script implements the manuscript design for d in {10, 20, 40} using the
same seed schedule, NOTEARS estimator, BIC pruning rule, GES/PC/LiNGAM
comparators, and directed-edge metrics. It also exports skeleton metrics so
that the adjacency recovery of equivalence-class methods can be inspected
without requiring all edges to be oriented.

Examples
--------
python reproduce_simulations.py --d 20 --weight-kind uniform --out results/d20_uniform
python reproduce_simulations.py --d 40 --weight-kind modnormal --out results/d40_modnormal
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import igraph as ig
import lingam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz
from joblib import Parallel, delayed
from scipy.stats import norm

BASE_SEED = 12123
LAMBDA1 = 0.1
THRESHOLD = 0.3
N_SAMPLES = 500
METHODS = ["GES", "PC", "LiNGAM", "NOTEARS", "NOTEARS-BP"]
S_VALUES = {
    "uniform": [1, 4, 7, 10],
    "modnormal": [1, 2, 3, 4],
}


def notears_linear(X, lambda1=0.0, max_iter=100, h_tol=1e-8,
                   rho_max=1e16, w_threshold=0.0):
    """Linear NOTEARS implementation used in the study."""
    n, d = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)

    def _loss(W):
        M = Xc @ W
        R = Xc - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * Xc.T @ R
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h0, G_h = _h(W)
        obj = loss + 0.5 * rho * h0 * h0 + alpha * h0 + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h0 + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1,
                                -G_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [
        (0, 0) if i == j else (0, None)
        for _ in range(2) for i in range(d) for j in range(d)
    ]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True,
                                bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def simulate_lsem(G: nx.DiGraph, n: int, noise_scale: float = 1.0):
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros((n, d))
    for j in nx.topological_sort(G):
        parents = list(G.predecessors(j))
        noise = np.random.normal(scale=noise_scale, size=n)
        X[:, j] = X[:, parents].dot(W[parents, j]) + noise if parents else noise
    return X


def simulate_dag(d: int, s0: int):
    """Generate an ER DAG with exactly s0 edges."""
    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    B_und = np.array(G_und.get_adjacency().data)
    B = _random_acyclic_orientation(B_und)
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    assert int(B_perm.sum()) == s0
    return B_perm


def simulate_parameter_uniform(B, s):
    W = np.zeros(B.shape)
    ranges = ((-s, -0.5), (0.5, s))
    signs = np.random.randint(len(ranges), size=B.shape)
    for i, (low, high) in enumerate(ranges):
        draws = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (signs == i) * draws
    return W


def simulate_parameter_modnormal(B, s):
    # scipy.stats.norm(scale=s) means standard deviation s, i.e. N(0, s^2).
    samples = norm.rvs(loc=0, scale=s, size=B.shape)
    modified = np.zeros(B.shape)
    modified[samples <= 0] = samples[samples <= 0] - 0.5
    modified[samples > 0] = samples[samples > 0] + 0.5
    W = np.zeros(B.shape)
    W[B != 0] = modified[B != 0]
    return W


def loglk_bic(X, G):
    """Global sum of local Gaussian BIC contributions for a DAG."""
    n, d = X.shape
    Xc = X - np.mean(X, axis=0, keepdims=True)
    logL = 0.0
    for i in range(d):
        parents = list(G.predecessors(i))
        if not parents:
            residual = Xc[:, i]
        else:
            beta = np.linalg.lstsq(Xc[:, parents], Xc[:, i], rcond=None)[0]
            residual = Xc[:, i] - Xc[:, parents] @ beta
        sigma2 = max(float((residual ** 2).mean()), 1e-300)
        logL += -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    k = np.count_nonzero(nx.to_numpy_array(G, dtype=int)) + d
    return -2 * logL + k * np.log(n)


def subgraph_with_parents(G, child):
    out = nx.DiGraph()
    out.add_nodes_from(G.nodes())
    for parent in G.predecessors(child):
        out.add_edge(parent, child)
    return out


def greedy_edge_removal(X, G):
    """Greedy backward deletion using decomposable Gaussian BIC."""
    Xc = X - np.mean(X, axis=0, keepdims=True)
    improved = True
    while improved and len(G.edges()) > 0:
        improved = False
        candidates = []
        for u, v in list(G.edges()):
            G_temp = G.copy()
            before = subgraph_with_parents(G_temp, v)
            G_temp.remove_edge(u, v)
            after = subgraph_with_parents(G_temp, v)
            try:
                delta = loglk_bic(Xc, after) - loglk_bic(Xc, before)
            except Exception:
                continue
            candidates.append(((u, v), delta))
        if not candidates:
            break
        edge, delta = min(candidates, key=lambda x: x[1])
        if delta < 0:
            G.remove_edge(*edge)
            improved = True
    return G


def directed_shd(A_true, A_est):
    n = A_true.shape[0]
    count = np.sum(np.abs(A_true - A_est))
    for i in range(n):
        for j in range(n):
            if (A_true[i, j] == 1 and A_est[i, j] == 0
                    and A_est[j, i] == 1):
                count -= 1
    return float(count)


def directed_metrics(A_true, A_est):
    est = np.sum(np.abs(A_est))
    true = np.sum(np.abs(A_true))
    fdr = np.sum(A_est - A_true == 1) / est if est else np.nan
    fnr = np.sum(A_est - A_true == -1) / true if true else np.nan
    return float(fdr), float(1 - fnr), directed_shd(A_true, A_est)


def skeleton(A):
    return ((A != 0) | (A.T != 0)).astype(int)


def skeleton_metrics(A_true, A_est):
    S_true = skeleton(A_true)
    S_est = skeleton(A_est)
    iu = np.triu_indices_from(S_true, k=1)
    truth = S_true[iu].astype(bool)
    est = S_est[iu].astype(bool)
    tp = int(np.sum(truth & est))
    fp = int(np.sum(~truth & est))
    fn = int(np.sum(truth & ~est))
    fdr = fp / (tp + fp) if tp + fp else np.nan
    tpr = tp / (tp + fn) if tp + fn else np.nan
    shd = fp + fn
    return float(fdr), float(tpr), float(shd)


def causal_learn_directed(graph_matrix):
    """Historical directed scoring used for PC/GES in the manuscript.

    causal-learn encodes i -> j by graph[j, i] = 1 and graph[i, j] = -1.
    Undirected (-1, -1) endpoints are not credited as directed edges here.
    """
    A = np.array(graph_matrix, dtype=int).copy()
    A[A == -1] = 0
    return A.T


def run_one(d, s, weight_kind, rep):
    seed = BASE_SEED + 1000 * S_VALUES[weight_kind].index(s) + rep
    np.random.seed(seed)
    random.seed(seed)

    B = simulate_dag(d, 2 * d)
    if weight_kind == "uniform":
        W = simulate_parameter_uniform(B, s)
    else:
        W = simulate_parameter_modnormal(B, s)
    X = simulate_lsem(nx.DiGraph(W), n=N_SAMPLES)

    W0 = notears_linear(X, lambda1=LAMBDA1)
    A_note = (np.abs(W0) > THRESHOLD).astype(int)
    A_bp = nx.to_numpy_array(
        greedy_edge_removal(X, nx.DiGraph(A_note)), dtype=int
    )

    ges_graph = ges(X)["G"]
    A_ges = causal_learn_directed(ges_graph.graph)

    # Effective PC configuration of the reported simulations: Fisher-Z at
    # alpha=0.05 with causal-learn's standard conditioning-depth search.
    # Earlier archival code passed non-API keyword names through **kwargs;
    # they did not impose a conditioning-depth cap.
    pc_graph = pc(
        X,
        alpha=0.05,
        indep_test=fisherz,
        show_progress=False,
    ).G
    A_pc = causal_learn_directed(pc_graph.graph)

    model = lingam.DirectLiNGAM()
    model.fit(X)
    A_lingam = (np.abs(model.adjacency_matrix_.T) > 0).astype(int)

    rows = []
    for method, A in [
        ("GES", A_ges),
        ("PC", A_pc),
        ("LiNGAM", A_lingam),
        ("NOTEARS", A_note),
        ("NOTEARS-BP", A_bp),
    ]:
        fdr, tpr, shd = directed_metrics(B, A)
        sfdr, stpr, sshd = skeleton_metrics(B, A)
        rows.append({
            "d": d,
            "weight_kind": weight_kind,
            "s": s,
            "rep": rep,
            "seed": seed,
            "method": method,
            "fdr": fdr,
            "tpr": tpr,
            "shd": shd,
            "skeleton_fdr": sfdr,
            "skeleton_tpr": stpr,
            "skeleton_shd": sshd,
            "true_edges": int(B.sum()),
            "estimated_edges": int(A.sum()),
        })
    return rows


def draw_metric(df, metric, title, ylabel, s_values, out_base):
    colors = {
        "GES": "orange",
        "PC": "green",
        "LiNGAM": "yellow",
        "NOTEARS": "blue",
        "NOTEARS-BP": "red",
    }
    offsets = {
        "GES": -0.40,
        "PC": -0.24,
        "LiNGAM": -0.08,
        "NOTEARS": 0.08,
        "NOTEARS-BP": 0.24,
    }
    fig, ax = plt.subplots(figsize=(18, 5))
    handles = []
    for method in METHODS:
        arrays = [
            df[(df.s == s) & (df.method == method)][metric].to_numpy()
            for s in s_values
        ]
        positions = np.arange(1, len(s_values) + 1) + offsets[method]
        bp = ax.boxplot(
            arrays,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            boxprops={"facecolor": colors[method], "color": "black"},
            medianprops={"color": "black"},
        )
        handles.append(bp["boxes"][0])
    ax.set_xlim(0.4, len(s_values) + 0.74)
    ax.set_xticks(np.arange(1, len(s_values) + 1))
    ax.set_xticklabels(s_values, fontsize=16)
    ax.set_xlabel("S", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.legend(handles, METHODS, loc="upper right")
    for i in range(len(s_values) - 1):
        ax.axvline(x=i + 1.4, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            f"{out_base}.{ext}",
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def draw_combined(df, s_values, out_base):
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    colors = {
        "GES": "orange",
        "PC": "green",
        "LiNGAM": "yellow",
        "NOTEARS": "blue",
        "NOTEARS-BP": "red",
    }
    offsets = {
        "GES": -0.40,
        "PC": -0.24,
        "LiNGAM": -0.08,
        "NOTEARS": 0.08,
        "NOTEARS-BP": 0.24,
    }
    for ax, metric, title, ylabel in zip(
        axes,
        ["fdr", "tpr", "shd"],
        ["False Discovery Rate", "True Positive Rate", "SHD"],
        ["FDR", "TPR", "SHD"],
    ):
        handles = []
        for method in METHODS:
            arrays = [
                df[(df.s == s) & (df.method == method)][metric].to_numpy()
                for s in s_values
            ]
            positions = np.arange(1, len(s_values) + 1) + offsets[method]
            bp = ax.boxplot(
                arrays,
                positions=positions,
                widths=0.15,
                patch_artist=True,
                boxprops={"facecolor": colors[method], "color": "black"},
                medianprops={"color": "black"},
            )
            handles.append(bp["boxes"][0])
        ax.set_xlim(0.4, len(s_values) + 0.74)
        ax.set_xticks(np.arange(1, len(s_values) + 1))
        ax.set_xticklabels(s_values, fontsize=16)
        ax.set_xlabel("S", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.legend(handles, METHODS, loc="upper right")
        for i in range(len(s_values) - 1):
            ax.axvline(x=i + 1.4, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            f"{out_base}.{ext}",
            dpi=300 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, choices=[10, 20, 40], required=True)
    parser.add_argument(
        "--weight-kind", choices=["uniform", "modnormal"], required=True
    )
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    s_values = S_VALUES[args.weight_kind]
    tasks = [(s, rep) for s in s_values for rep in range(args.M)]
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one)(args.d, s, args.weight_kind, rep)
        for s, rep in tasks
    )
    rows = [row for group in nested for row in group]
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "replicate_metrics.csv", index=False)

    summary = df.groupby(
        ["d", "weight_kind", "s", "method"], as_index=False
    ).agg(
        fdr_mean=("fdr", "mean"),
        tpr_mean=("tpr", "mean"),
        shd_mean=("shd", "mean"),
        fdr_sd=("fdr", "std"),
        tpr_sd=("tpr", "std"),
        shd_sd=("shd", "std"),
        skeleton_fdr_mean=("skeleton_fdr", "mean"),
        skeleton_tpr_mean=("skeleton_tpr", "mean"),
        skeleton_shd_mean=("skeleton_shd", "mean"),
        estimated_edges_mean=("estimated_edges", "mean"),
    )
    summary.to_csv(args.out / "summary.csv", index=False)

    config = {
        "d": args.d,
        "weight_kind": args.weight_kind,
        "M": args.M,
        "n_samples": N_SAMPLES,
        "lambda1": LAMBDA1,
        "threshold": THRESHOLD,
        "base_seed": BASE_SEED,
        "s_values": s_values,
        "true_edges": 2 * args.d,
        "pc": {
            "alpha": 0.05,
            "indep_test": "fisherz",
            "max_k": None,
            "note": "standard causal-learn conditioning-depth search",
        },
    }
    (args.out / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    draw_metric(df, "fdr", "False Discovery Rate", "FDR",
                s_values, args.out / "fdr")
    draw_metric(df, "tpr", "True Positive Rate", "TPR",
                s_values, args.out / "tpr")
    draw_metric(df, "shd", "SHD", "SHD",
                s_values, args.out / "shd")
    draw_combined(df, s_values, args.out / "combined")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
