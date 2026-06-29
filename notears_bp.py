"""
BIC-pruned NOTEARS (NOTEARS-BP).

This module implements the BIC-based pruning step used to refine an
initial directed acyclic graph (DAG), for example a graph estimated by
NOTEARS. The pruning step is deletion-only: it removes edges when doing
so improves a local BIC score.

Adjacency convention
--------------------
A[i, j] = 1 indicates a directed edge i -> j.

Data convention
---------------
X is an n by d numeric matrix, where rows are observations and columns
are variables.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class PruningResult:
    """Container for NOTEARS-BP pruning output."""
    adjacency: np.ndarray
    removed_edges: List[Tuple[int, int]]
    bic_history: List[float]
    total_bic_improvement: float
    n_bic_evaluations: int


def center_columns(X: np.ndarray) -> np.ndarray:
    """Return a copy of X with each column centered."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=0, keepdims=True)


def local_bic_score(X: np.ndarray, child: int, parents: List[int]) -> float:
    """
    Compute the local Gaussian BIC score for one child node.

    The local model is X_child ~ X_parents. The returned score is
    n * log(RSS / n) + k * log(n), up to constants that cancel in
    local BIC comparisons.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    y = X[:, child]

    if len(parents) == 0:
        resid = y - y.mean()
        k = 1
    else:
        design = X[:, parents]
        design = np.column_stack([np.ones(n), design])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        k = design.shape[1]

    rss = float(np.sum(resid ** 2))
    rss = max(rss, np.finfo(float).eps)
    return n * np.log(rss / n) + k * np.log(n)


def total_local_bic(X: np.ndarray, adjacency: np.ndarray) -> float:
    """Compute the sum of local BIC scores over all child nodes."""
    A = np.asarray(adjacency, dtype=int)
    d = A.shape[0]
    score = 0.0
    for child in range(d):
        parents = list(np.where(A[:, child] != 0)[0])
        score += local_bic_score(X, child, parents)
    return float(score)


def is_acyclic(adjacency: np.ndarray) -> bool:
    """Check whether a directed graph is acyclic using Kahn's algorithm."""
    A = np.asarray(adjacency, dtype=int)
    d = A.shape[0]
    indegree = A.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indegree[i] == 0]
    visited = 0

    while queue:
        node = queue.pop()
        visited += 1
        children = np.where(A[node, :] != 0)[0]
        for child in children:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return visited == d


def bic_prune_dag(
    X: np.ndarray,
    adjacency: np.ndarray,
    bic_tolerance: float = 0.0,
    check_acyclic: bool = True,
    center_data: bool = True,
    max_steps: Optional[int] = None,
) -> PruningResult:
    """
    Apply greedy BIC-based edge pruning to an initial DAG.

    At each iteration, the algorithm evaluates every current edge i -> j.
    It computes the local BIC difference for the child node j after removing
    i from the parent set. The edge with the most negative BIC difference is
    removed if the improvement is larger than `bic_tolerance`.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(adjacency, dtype=int).copy()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square 2D array.")
    if X.shape[1] != A.shape[0]:
        raise ValueError("The number of columns in X must match adjacency size.")
    if np.any(np.diag(A) != 0):
        raise ValueError("Self edges are not allowed; adjacency diagonal must be zero.")
    if bic_tolerance < 0:
        raise ValueError("bic_tolerance must be nonnegative.")
    if check_acyclic and not is_acyclic(A):
        raise ValueError("The input adjacency matrix is not acyclic.")

    if center_data:
        X = center_columns(X)

    removed_edges: List[Tuple[int, int]] = []
    bic_history = [total_local_bic(X, A)]
    n_bic_evaluations = 0
    steps = 0

    while True:
        edges = list(zip(*np.where(A != 0)))
        if not edges:
            break
        if max_steps is not None and steps >= max_steps:
            break

        best_edge = None
        best_delta = 0.0

        for parent, child in edges:
            current_parents = list(np.where(A[:, child] != 0)[0])
            before = local_bic_score(X, child, current_parents)

            reduced_parents = [p for p in current_parents if p != parent]
            after = local_bic_score(X, child, reduced_parents)

            delta = after - before
            n_bic_evaluations += 1

            if delta < best_delta:
                best_delta = delta
                best_edge = (int(parent), int(child))

        if best_edge is None or best_delta >= -bic_tolerance:
            break

        parent, child = best_edge
        A[parent, child] = 0
        removed_edges.append(best_edge)
        bic_history.append(bic_history[-1] + best_delta)
        steps += 1

    total_improvement = bic_history[0] - bic_history[-1]
    return PruningResult(
        adjacency=A,
        removed_edges=removed_edges,
        bic_history=bic_history,
        total_bic_improvement=float(total_improvement),
        n_bic_evaluations=n_bic_evaluations,
    )


def edge_counts(adjacency: np.ndarray) -> int:
    """Return the number of directed edges in an adjacency matrix."""
    return int(np.sum(np.asarray(adjacency) != 0))


def compare_to_truth(estimated: np.ndarray, truth: np.ndarray) -> dict:
    """Compute directed-edge diagnostics against a true DAG."""
    est = np.asarray(estimated, dtype=bool)
    tru = np.asarray(truth, dtype=bool)

    if est.shape != tru.shape:
        raise ValueError("estimated and truth must have the same shape.")

    tp = int(np.sum(est & tru))
    fp = int(np.sum(est & ~tru))
    fn = int(np.sum(~est & tru))
    shd = int(np.sum(est != tru))

    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)

    return {
        "edges": int(np.sum(est)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "fdr": fdr,
        "tpr": tpr,
        "shd_directed": shd,
    }
