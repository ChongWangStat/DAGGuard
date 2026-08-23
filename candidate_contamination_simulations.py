#!/usr/bin/env python3
"""Controlled fixed-candidate contamination experiments.

These experiments target the refinement problem directly.  A known linear DAG
is simulated, its candidate graph is contaminated in controlled ways, and the
same candidate is refined by exact local-BIC subset selection and greedy
deletion.  The design varies false positives, missing and reversible edges,
weak signals, density/maximum indegree, and a population-standardized regime
with low varsortability and heterogeneous error variances.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t

from local_bic_refinement import (
    candidate_indegree_summary,
    edge_jaccard,
    exact_refine_dag,
    graph_metrics,
    greedy_refine_dag,
    initial_pruning_pressure,
    total_gaussian_bic,
)

BASE_SEED = 48151623


@dataclass(frozen=True)
class Setting:
    name: str
    true_edges: int
    true_max_indegree: int
    candidate_max_indegree: int
    false_positive_ratio: float = 0.0
    missing_ratio: float = 0.0
    reversal_ratio: float = 0.0
    weak_fraction: float = 0.0
    error_regime: str = "homoskedastic"


SETTINGS = [
    Setting("clean_sparse", 20, 3, 4),
    Setting("fp_025", 20, 3, 4, false_positive_ratio=0.25),
    Setting("fp_050", 20, 3, 5, false_positive_ratio=0.50),
    Setting("fp_100", 20, 3, 6, false_positive_ratio=1.00),
    Setting("missing_010", 20, 3, 4, missing_ratio=0.10),
    Setting("reversal_010", 20, 3, 4, reversal_ratio=0.10),
    Setting("weak_fp", 20, 3, 5, false_positive_ratio=0.50, weak_fraction=0.30),
    Setting("dense_moderate", 40, 5, 7, false_positive_ratio=0.50),
    Setting("dense_high_indegree", 40, 7, 10, false_positive_ratio=1.00),
    Setting("lowvar_heterogeneous", 20, 3, 4, error_regime="lowvar_heterogeneous"),
    Setting("lowvar_weak_fp", 20, 3, 6, false_positive_ratio=0.50,
            weak_fraction=0.30, error_regime="lowvar_heterogeneous"),
    Setting("combined_contamination", 40, 5, 8, false_positive_ratio=0.50,
            missing_ratio=0.10, reversal_ratio=0.10, weak_fraction=0.30,
            error_regime="lowvar_heterogeneous"),
]


def generate_dag(d: int, edges: int, max_indegree: int, rng: np.random.Generator):
    for _ in range(1000):
        order = rng.permutation(d)
        pairs = [(int(order[i]), int(order[j]))
                 for j in range(1, d) for i in range(j)]
        rng.shuffle(pairs)
        A = np.zeros((d, d), dtype=int)
        for parent, child in pairs:
            if A[:, child].sum() < max_indegree:
                A[parent, child] = 1
                if A.sum() == edges:
                    return A, order
    raise RuntimeError("Unable to generate a DAG under the requested indegree cap")


def assign_weights(A: np.ndarray, weak_fraction: float, rng: np.random.Generator):
    W = np.zeros_like(A, dtype=float)
    edge_indices = np.argwhere(A == 1)
    weak_n = int(round(weak_fraction * len(edge_indices)))
    weak_rows = set(rng.choice(len(edge_indices), size=weak_n, replace=False).tolist()) \
        if weak_n else set()
    for index, (u, v) in enumerate(edge_indices):
        magnitude = rng.uniform(0.12, 0.30) if index in weak_rows else rng.uniform(0.50, 1.50)
        W[u, v] = rng.choice([-1.0, 1.0]) * magnitude
    return W


def population_standardize(W: np.ndarray):
    d = W.shape[0]
    transform = np.linalg.inv(np.eye(d) - W)
    covariance = transform.T @ transform
    scales = 1.0 / np.sqrt(np.diag(covariance))
    D = np.diag(scales)
    W_scaled = np.diag(1.0 / scales) @ W @ D
    error_scales = scales
    return W_scaled, error_scales


def simulate_sem(W: np.ndarray, n: int, error_scales: np.ndarray,
                 rng: np.random.Generator):
    X = np.zeros((n, W.shape[0]))
    for child in nx.topological_sort(nx.DiGraph(W)):
        parents = np.flatnonzero(W[:, child])
        error = rng.normal(0.0, error_scales[child], size=n)
        X[:, child] = error
        if len(parents):
            X[:, child] += X[:, parents] @ W[parents, child]
    return X


def contaminate_candidate(A_true: np.ndarray, setting: Setting,
                          rng: np.random.Generator):
    candidate = A_true.copy()
    true_edges = [tuple(map(int, edge)) for edge in np.argwhere(A_true == 1)]

    missing_target = int(round(setting.missing_ratio * len(true_edges)))
    if missing_target:
        for index in rng.choice(len(true_edges), size=missing_target, replace=False):
            candidate[true_edges[int(index)]] = 0

    reversal_target = int(round(setting.reversal_ratio * len(true_edges)))
    reversed_edges = 0
    remaining = [edge for edge in true_edges if candidate[edge] == 1]
    rng.shuffle(remaining)
    for u, v in remaining:
        if reversed_edges >= reversal_target:
            break
        trial = candidate.copy()
        trial[u, v] = 0
        if trial[:, u].sum() >= setting.candidate_max_indegree:
            continue
        trial[v, u] = 1
        if nx.is_directed_acyclic_graph(nx.DiGraph(trial)):
            candidate = trial
            reversed_edges += 1

    false_positive_target = int(round(setting.false_positive_ratio * len(true_edges)))
    added = 0
    for _ in range(10 if false_positive_target else 0):
        order = list(nx.topological_sort(nx.DiGraph(candidate)))
        positions = {node: index for index, node in enumerate(order)}
        eligible = []
        for u in range(candidate.shape[0]):
            for v in range(candidate.shape[0]):
                if u == v or positions[u] >= positions[v] or candidate[u, v]:
                    continue
                if A_true[u, v] or A_true[v, u]:
                    continue
                if candidate[:, v].sum() >= setting.candidate_max_indegree:
                    continue
                eligible.append((u, v))
        rng.shuffle(eligible)
        if not eligible:
            break
        for edge in eligible:
            candidate[edge] = 1
            added += 1
            if added >= false_positive_target:
                break
        if added >= false_positive_target:
            break

    if not nx.is_directed_acyclic_graph(nx.DiGraph(candidate)):
        raise RuntimeError("Candidate contamination created a cycle")
    return candidate, {
        "missing_achieved": int(np.sum((A_true == 1) & (candidate == 0))),
        "reversals_achieved": reversed_edges,
        "false_positive_additions_achieved": added,
    }


def varsortability(A: np.ndarray, X: np.ndarray) -> float:
    variances = X.var(axis=0)
    paths = A.astype(float).copy()
    numerator = denominator = 0.0
    for length in range(1, A.shape[0]):
        if length > 1:
            paths = paths @ A
        parents, children = np.where(paths > 0)
        weights = paths[parents, children]
        concordance = ((variances[parents] < variances[children]).astype(float)
                       + 0.5 * (variances[parents] == variances[children]))
        numerator += float(weights @ concordance)
        denominator += float(weights.sum())
    return numerator / denominator if denominator else np.nan


def run_one(setting: Setting, setting_index: int, rep: int, d: int, n: int):
    seed = BASE_SEED + 100_000 * setting_index + rep
    rng = np.random.default_rng(seed)
    truth, _ = generate_dag(d, setting.true_edges, setting.true_max_indegree, rng)
    W = assign_weights(truth, setting.weak_fraction, rng)
    if setting.error_regime == "lowvar_heterogeneous":
        W, error_scales = population_standardize(W)
    else:
        error_scales = np.ones(d)
    X = simulate_sem(W, n, error_scales, rng)
    candidate, achieved = contaminate_candidate(truth, setting, rng)

    exact = exact_refine_dag(X, candidate)
    greedy = greedy_refine_dag(X, candidate)
    if not exact.globally_optimal:
        raise RuntimeError(f"Uncertified exact search in {setting.name}, replicate {rep}")
    pressure, _ = initial_pruning_pressure(X, candidate)
    indegree = candidate_indegree_summary(candidate)
    candidate_bic = total_gaussian_bic(X, candidate)
    gap = greedy.total_bic - exact.total_bic
    common = {
        "setting": setting.name, "rep": rep, "seed": seed, "d": d, "n": n,
        **asdict(setting), **achieved,
        "varsortability": varsortability(truth, X),
        "error_scale_ratio": float(error_scales.max() / error_scales.min()),
        "candidate_edges": int(candidate.sum()),
        "candidate_max_indegree_achieved": indegree["maximum_indegree"],
        "candidate_enumeration_fits": indegree["enumeration_fits"],
        "initial_pruning_pressure": pressure["initial_pruning_pressure"],
        "candidate_bic": candidate_bic,
        "exact_bic": exact.total_bic,
        "greedy_bic": greedy.total_bic,
        "candidate_to_exact_bic_improvement": candidate_bic - exact.total_bic,
        "greedy_bic_gap": gap,
        "greedy_suboptimal": bool(gap > 1e-8),
        "exact_greedy_jaccard": edge_jaccard(exact.adjacency, greedy.adjacency),
        "exact_greedy_disagreement_edges": int(np.logical_xor(
            exact.adjacency, greedy.adjacency).sum()),
        "exact_runtime_seconds": exact.runtime_seconds,
        "greedy_runtime_seconds": greedy.runtime_seconds,
        "exact_score_evaluations": exact.score_evaluations,
        "greedy_score_evaluations": greedy.score_evaluations,
    }
    rows = []
    for method, A in [
        ("Candidate", candidate),
        ("Exact local BIC", exact.adjacency),
        ("Greedy local BIC", greedy.adjacency),
    ]:
        metrics = graph_metrics(truth, A)
        rows.append({"method": method, **common, **metrics})
    return rows


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().astype(float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z / denominator * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    low = max(0.0, float(center - half))
    high = min(1.0, float(center + half))
    if successes == 0 and low < 1e-15:
        low = 0.0
    if successes == total and 1 - high < 1e-15:
        high = 1.0
    return low, high


def summarize(df: pd.DataFrame, out: Path) -> None:
    method_summary = df.groupby(["setting", "method"], as_index=False).agg(
        replicates=("rep", "nunique"), fdr_mean=("fdr", "mean"),
        tpr_mean=("tpr", "mean"), shd_mean=("shd", "mean"),
        edges_mean=("edges", "mean"), tp_mean=("true_positives", "mean"),
        fp_mean=("false_positives", "mean"), fn_mean=("false_negatives", "mean"),
        reversals_mean=("reversals", "mean"), varsortability_mean=("varsortability", "mean"),
    )
    method_summary.to_csv(out / "contamination_method_summary.csv", index=False)

    wide = df.pivot(index=["setting", "rep"], columns="method",
                    values=["fdr", "tpr", "shd", "edges"])
    replicate = df[df.method == "Exact local BIC"].copy()
    pairs = []
    for setting, group in replicate.groupby("setting"):
        record = {"setting": setting, "replicates": len(group)}
        for metric in ["fdr", "tpr", "shd", "edges"]:
            difference = (wide.loc[setting, (metric, "Greedy local BIC")]
                          - wide.loc[setting, (metric, "Exact local BIC")])
            mean, low, high = mean_ci(difference)
            record[f"greedy_minus_exact_{metric}_mean"] = mean
            record[f"greedy_minus_exact_{metric}_ci_low"] = low
            record[f"greedy_minus_exact_{metric}_ci_high"] = high
        gap_mean, gap_low, gap_high = mean_ci(group["greedy_bic_gap"])
        record.update({
            "greedy_bic_gap_mean": gap_mean,
            "greedy_bic_gap_ci_low": gap_low,
            "greedy_bic_gap_ci_high": gap_high,
            "greedy_bic_gap_max": float(group.greedy_bic_gap.max()),
            "exact_greedy_jaccard_mean": float(group.exact_greedy_jaccard.mean()),
            "exact_runtime_seconds_mean": float(group.exact_runtime_seconds.mean()),
            "greedy_runtime_seconds_mean": float(group.greedy_runtime_seconds.mean()),
        })
        successes = int(group.greedy_suboptimal.sum())
        low, high = wilson_interval(successes, len(group))
        record.update({
            "greedy_suboptimal_count": successes,
            "greedy_suboptimal_frequency": successes / len(group),
            "greedy_suboptimal_wilson_low": low,
            "greedy_suboptimal_wilson_high": high,
        })
        pairs.append(record)
    pd.DataFrame(pairs).to_csv(out / "exact_greedy_paired_summary.csv", index=False)

    diagnostics = replicate.drop_duplicates(["setting", "rep"])
    correlations = []
    outcomes = [
        "candidate_to_exact_bic_improvement", "greedy_bic_gap",
        "exact_greedy_disagreement_edges",
    ]
    diagnostics = diagnostics.assign(
        exact_deletion_fraction=(diagnostics.candidate_edges - diagnostics.edges)
        / diagnostics.candidate_edges
    )
    outcomes.insert(0, "exact_deletion_fraction")
    for outcome in outcomes:
        x = diagnostics.initial_pruning_pressure.astype(float)
        y = diagnostics[outcome].astype(float)
        if x.nunique() < 2 or y.nunique() < 2:
            pearson_stat = pearson_p = spearman_stat = spearman_p = np.nan
        else:
            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            pearson_stat, pearson_p = pr.statistic, pr.pvalue
            spearman_stat, spearman_p = sr.statistic, sr.pvalue
        correlations.append({
            "outcome": outcome, "n": len(x), "pearson_r": pearson_stat,
            "pearson_p": pearson_p, "spearman_rho": spearman_stat,
            "spearman_p": spearman_p,
        })
    pd.DataFrame(correlations).to_csv(out / "initial_pruning_pressure_validation.csv", index=False)


def plot_paired_summary(out: Path) -> None:
    summary = pd.read_csv(out / "exact_greedy_paired_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary.greedy_suboptimal_frequency, color="#D55E00")
    axes[0].errorbar(
        x, summary.greedy_suboptimal_frequency,
        yerr=np.vstack([
            np.maximum(0.0, summary.greedy_suboptimal_frequency
                       - summary.greedy_suboptimal_wilson_low),
            np.maximum(0.0, summary.greedy_suboptimal_wilson_high
                       - summary.greedy_suboptimal_frequency),
        ]), fmt="none", ecolor="black", capsize=3,
    )
    axes[0].set_ylabel("Greedy suboptimality frequency")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x, [])
    axes[1].bar(x, summary.greedy_bic_gap_mean, color="#0072B2")
    axes[1].set_ylabel("Mean BIC(greedy) - BIC(exact)")
    axes[1].set_xticks(x, summary.setting, rotation=50, ha="right")
    axes[1].set_xlabel("Candidate-contamination setting")
    fig.savefig(out / "exact_greedy_contamination_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "exact_greedy_contamination_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/candidate_contamination"))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for setting_index, setting in enumerate(SETTINGS):
        for rep in range(args.replicates):
            rows.extend(run_one(setting, setting_index, rep, args.d, args.n))
            print(f"{setting.name}: {rep + 1}/{args.replicates}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "contamination_replicate_metrics.csv", index=False)
    summarize(df, args.out)
    plot_paired_summary(args.out)
    config = {
        "base_seed": BASE_SEED, "replicates": args.replicates, "d": args.d,
        "n": args.n, "settings": [asdict(setting) for setting in SETTINGS],
        "shd_convention": "one operation per reversal",
        "gumbel_used": False,
    }
    (args.out / "contamination_run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
