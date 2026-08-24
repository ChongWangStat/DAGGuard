"""Shared utilities for the seven-method DAGGuard benchmark."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import norm

BASE_SEED = 12123


def simulate_dag_seeded(d: int, num_edges: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=int)
    pairs = [(order[i], order[j]) for i in range(d) for j in range(i + 1, d)]
    chosen = rng.choice(len(pairs), size=min(num_edges, len(pairs)), replace=False)
    for idx in chosen:
        u, v = pairs[idx]
        adjacency[u, v] = 1
    return adjacency


def simulate_weights_seeded(adjacency: np.ndarray, s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros_like(adjacency, dtype=float)
    mask = adjacency != 0
    signs = rng.choice([-1, 1], size=adjacency.shape)
    magnitudes = rng.uniform(0.5, s, size=adjacency.shape)
    weights[mask] = signs[mask] * magnitudes[mask]
    return weights


def simulate_lsem_noise(weights: np.ndarray, n: int, noise: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, weights.shape[0]))
    graph = nx.DiGraph(weights)
    for child in nx.topological_sort(graph):
        parents = np.flatnonzero(weights[:, child])
        if noise == "normal":
            eps = rng.normal(0, 1, n)
        elif noise == "exponential":
            eps = rng.exponential(1, n) - 1
        elif noise == "gumbel":
            beta = np.sqrt(6.0) / np.pi
            eps = rng.gumbel(0, beta, n) - beta * np.euler_gamma
        else:
            raise ValueError(f"Unsupported noise distribution: {noise}")
        X[:, child] = (X[:, parents] @ weights[parents, child] if len(parents) else 0) + eps
    return X


def skeleton_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    true_skeleton = (np.asarray(truth) != 0) | (np.asarray(truth).T != 0)
    est_skeleton = (np.asarray(estimate) != 0) | (np.asarray(estimate).T != 0)
    upper = np.triu_indices_from(true_skeleton, k=1)
    t = true_skeleton[upper]
    e = est_skeleton[upper]
    tp = int(np.sum(t & e))
    fp = int(np.sum((~t) & e))
    fn = int(np.sum(t & (~e)))
    selected = tp + fp
    return {
        "fdr": fp / selected if selected else np.nan,
        "tpr": tp / (tp + fn) if tp + fn else np.nan,
        "shd": fp + fn,
        "edges": selected,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def fisherz_p(correlation: np.ndarray, n: int, x: int, y: int, conditioning=()) -> float:
    """Two-sided Gaussian Fisher-z conditional-independence p-value."""
    conditioning = tuple(conditioning)
    indices = [x, y, *conditioning]
    if len(conditioning) == 0:
        r = float(correlation[x, y])
    elif len(conditioning) == 1:
        z = conditioning[0]
        denominator = (1 - correlation[y, z] ** 2) * (1 - correlation[x, z] ** 2)
        r = 0.0 if denominator <= 0 else float(
            (correlation[x, y] - correlation[x, z] * correlation[y, z]) / np.sqrt(denominator)
        )
    else:
        precision = np.linalg.pinv(correlation[np.ix_(indices, indices)])
        denominator = precision[0, 0] * precision[1, 1]
        r = float(precision[0, 1] / np.sqrt(denominator)) if denominator > 0 else 0.0
        # The sign convention is immaterial for the two-sided p-value.
    r = float(np.clip(r, -1 + 1e-12, 1 - 1e-12))
    dof = n - len(conditioning) - 3
    if dof <= 0:
        return 1.0
    statistic = np.sqrt(dof) * np.arctanh(r)
    if not np.isfinite(statistic):
        statistic = 0.0
    return float(2 * norm.sf(abs(statistic)))
