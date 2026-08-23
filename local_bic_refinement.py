"""Local-BIC refinement of a fixed candidate directed acyclic graph.

This module is the single source of truth for local Gaussian BIC scores,
greedy and exact fixed-candidate refinement, initial pruning diagnostics, and
directed graph metrics.  The adjacency convention is ``A[parent, child] = 1``.

The score for child ``j`` and parent subset ``S`` is, up to constants that do
not depend on the graph,

    n log(RSS_j(S) / n) + (|S| + 1) log(n).

The additional parameter is the local error variance.  Data are centered once
and nonconstant predictor columns are normalized before least squares.  The
normalization does not change fitted column spaces and makes the documented
rank policy stable under changes of measurement units.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


DEFAULT_SCORE_TOLERANCE = 1e-10
DEFAULT_ENUMERATION_MAX_PARENTS = 15
DEFAULT_BRANCH_NODE_LIMIT = 2_000_000


@dataclass(frozen=True)
class LocalSelectionResult:
    """Best parent subset found for one child."""

    child: int
    candidate_parents: tuple[int, ...]
    selected_parents: tuple[int, ...]
    score: float
    method: str
    score_evaluations: int
    search_nodes: int
    globally_optimal: bool


@dataclass(frozen=True)
class RefinementResult:
    """Result of refining all child-specific parent sets."""

    adjacency: np.ndarray
    total_bic: float
    local_results: tuple[LocalSelectionResult, ...]
    removed_edges: tuple[tuple[int, int], ...]
    bic_history: tuple[float, ...]
    score_evaluations: int
    runtime_seconds: float
    method: str
    globally_optimal: bool


class _LocalScoreCache:
    """Cached local scores for one centered response and candidate design."""

    def __init__(self, X: np.ndarray, child: int, rank_tolerance: float | None):
        self.X = _center_columns(X)
        self.child = int(child)
        self.rank_tolerance = rank_tolerance
        self.cache: dict[tuple[int, ...], tuple[float, float]] = {}
        self.evaluations = 0

    def score_and_rss(self, parents: Iterable[int]) -> tuple[float, float]:
        key = tuple(sorted(int(p) for p in parents))
        if key in self.cache:
            return self.cache[key]
        n = self.X.shape[0]
        y = self.X[:, self.child]
        if key:
            design = self.X[:, key]
            norms = np.linalg.norm(design, axis=0)
            nonconstant = norms > np.finfo(float).eps
            if np.any(nonconstant):
                stable_design = design[:, nonconstant] / norms[nonconstant]
                coef, *_ = np.linalg.lstsq(
                    stable_design, y, rcond=self.rank_tolerance
                )
                residual = y - stable_design @ coef
            else:
                residual = y
        else:
            residual = y
        rss = max(float(residual @ residual), np.finfo(float).tiny)
        score = n * log(rss / n) + (len(key) + 1) * log(n)
        self.cache[key] = (float(score), rss)
        self.evaluations += 1
        return self.cache[key]


def _center_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two observations")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X - X.mean(axis=0, keepdims=True)


def _validate_adjacency(adjacency: np.ndarray, d: int | None = None) -> np.ndarray:
    A = (np.asarray(adjacency) != 0).astype(np.int8)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    if d is not None and A.shape != (d, d):
        raise ValueError("adjacency dimension does not match X")
    if np.any(np.diag(A)):
        raise ValueError("self edges are not allowed")
    return A


def is_acyclic(adjacency: np.ndarray) -> bool:
    """Return whether the directed adjacency matrix is acyclic."""
    A = _validate_adjacency(adjacency)
    indegree = A.sum(axis=0).astype(int)
    stack = [int(i) for i in np.flatnonzero(indegree == 0)]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(int(child))
    return visited == A.shape[0]


def gaussian_local_bic(
    X: np.ndarray,
    child: int,
    parents: Sequence[int],
    *,
    rank_tolerance: float | None = None,
) -> float:
    """Return the graph-dependent local Gaussian BIC contribution."""
    X = np.asarray(X, dtype=float)
    if not 0 <= child < X.shape[1]:
        raise IndexError("child is outside the column range of X")
    if child in parents:
        raise ValueError("a child cannot be its own parent")
    return _LocalScoreCache(X, child, rank_tolerance).score_and_rss(parents)[0]


def total_gaussian_bic(
    X: np.ndarray,
    adjacency: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> float:
    """Sum local Gaussian BIC contributions across children."""
    X = np.asarray(X, dtype=float)
    A = _validate_adjacency(adjacency, X.shape[1])
    return float(sum(
        gaussian_local_bic(
            X, child, np.flatnonzero(A[:, child]).tolist(),
            rank_tolerance=rank_tolerance,
        )
        for child in range(A.shape[0])
    ))


def deletion_diagnostics(
    X: np.ndarray,
    child: int,
    full_parents: Sequence[int],
    removed_parent: int,
    *,
    rank_tolerance: float | None = None,
) -> dict[str, float]:
    """Return BIC and partial-R2 algebra for deleting one current parent."""
    full = tuple(sorted(int(p) for p in full_parents))
    if removed_parent not in full:
        raise ValueError("removed_parent is not in full_parents")
    reduced = tuple(p for p in full if p != removed_parent)
    cache = _LocalScoreCache(X, child, rank_tolerance)
    full_score, rss_full = cache.score_and_rss(full)
    reduced_score, rss_reduced = cache.score_and_rss(reduced)
    n = np.asarray(X).shape[0]
    partial_r2 = 1.0 - rss_full / rss_reduced
    delta = reduced_score - full_score
    algebraic_delta = n * log(rss_reduced / rss_full) - log(n)
    return {
        "bic_full": full_score,
        "bic_reduced": reduced_score,
        "delta_bic": delta,
        "delta_bic_algebra": algebraic_delta,
        "rss_full": rss_full,
        "rss_reduced": rss_reduced,
        "partial_r2": partial_r2,
        "partial_r2_cutoff": bic_partial_r2_cutoff(n),
    }


def bic_partial_r2_cutoff(n: float) -> float:
    """BIC one-parameter deletion cutoff ``1 - n**(-1/n)``."""
    if n <= 1:
        raise ValueError("n must be greater than one")
    return float(1.0 - np.exp(-np.log(float(n)) / float(n)))


def _prefer(
    score: float,
    parents: tuple[int, ...],
    incumbent_score: float,
    incumbent_parents: tuple[int, ...],
    tolerance: float,
) -> bool:
    if score < incumbent_score - tolerance:
        return True
    if abs(score - incumbent_score) <= tolerance:
        return (len(parents), parents) < (len(incumbent_parents), incumbent_parents)
    return False


def _greedy_local(
    cache: _LocalScoreCache,
    child: int,
    candidate_parents: tuple[int, ...],
    tolerance: float,
) -> tuple[LocalSelectionResult, list[tuple[int, int]], list[float]]:
    current = candidate_parents
    current_score, _ = cache.score_and_rss(current)
    history = [current_score]
    removed: list[tuple[int, int]] = []
    while current:
        candidates = []
        for parent in current:
            reduced = tuple(p for p in current if p != parent)
            score, _ = cache.score_and_rss(reduced)
            candidates.append((score, reduced, parent))
        score, reduced, parent = min(candidates, key=lambda z: (z[0], len(z[1]), z[1]))
        if score >= current_score - tolerance:
            break
        current = reduced
        current_score = score
        removed.append((int(parent), int(child)))
        history.append(current_score)
    return (
        LocalSelectionResult(
            child=child,
            candidate_parents=candidate_parents,
            selected_parents=current,
            score=current_score,
            method="greedy",
            score_evaluations=cache.evaluations,
            search_nodes=0,
            globally_optimal=False,
        ),
        removed,
        history,
    )


def greedy_refine_dag(
    X: np.ndarray,
    adjacency: np.ndarray,
    *,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
    rank_tolerance: float | None = None,
) -> RefinementResult:
    """Greedily delete parents child by child until no deletion lowers BIC."""
    start = perf_counter()
    X = np.asarray(X, dtype=float)
    A0 = _validate_adjacency(adjacency, X.shape[1])
    if not is_acyclic(A0):
        raise ValueError("the fixed candidate graph must be acyclic")
    A = A0.copy()
    initial_total = total_gaussian_bic(X, A, rank_tolerance=rank_tolerance)
    total_history = [initial_total]
    total = initial_total
    local_results = []
    removed_edges = []
    evaluations = 0
    for child in range(A.shape[0]):
        parents = tuple(int(p) for p in np.flatnonzero(A0[:, child]))
        cache = _LocalScoreCache(X, child, rank_tolerance)
        local, removed, local_history = _greedy_local(
            cache, child, parents, score_tolerance
        )
        local_results.append(local)
        evaluations += cache.evaluations
        previous = local_history[0]
        for edge, score in zip(removed, local_history[1:]):
            A[edge] = 0
            total += score - previous
            total_history.append(float(total))
            previous = score
            removed_edges.append(edge)
    final_total = total_gaussian_bic(X, A, rank_tolerance=rank_tolerance)
    return RefinementResult(
        adjacency=A,
        total_bic=final_total,
        local_results=tuple(local_results),
        removed_edges=tuple(removed_edges),
        bic_history=tuple(total_history),
        score_evaluations=evaluations,
        runtime_seconds=perf_counter() - start,
        method="greedy",
        globally_optimal=False,
    )


def _enumerate_local(
    cache: _LocalScoreCache,
    child: int,
    parents: tuple[int, ...],
    tolerance: float,
) -> LocalSelectionResult:
    best_parents: tuple[int, ...] = ()
    best_score, _ = cache.score_and_rss(best_parents)
    for size in range(1, len(parents) + 1):
        for subset in combinations(parents, size):
            score, _ = cache.score_and_rss(subset)
            if _prefer(score, subset, best_score, best_parents, tolerance):
                best_score, best_parents = score, subset
    return LocalSelectionResult(
        child=child,
        candidate_parents=parents,
        selected_parents=best_parents,
        score=best_score,
        method="enumeration",
        score_evaluations=cache.evaluations,
        search_nodes=2 ** len(parents),
        globally_optimal=True,
    )


def _branch_and_bound_local(
    cache: _LocalScoreCache,
    child: int,
    parents: tuple[int, ...],
    tolerance: float,
    node_limit: int,
) -> LocalSelectionResult:
    greedy_cache = _LocalScoreCache(cache.X, child, cache.rank_tolerance)
    greedy, _, _ = _greedy_local(greedy_cache, child, parents, tolerance)
    best_parents = greedy.selected_parents
    best_score, _ = cache.score_and_rss(best_parents)
    empty_score, _ = cache.score_and_rss(())
    if _prefer(empty_score, (), best_score, best_parents, tolerance):
        best_score, best_parents = empty_score, ()

    y = cache.X[:, child]
    priority = sorted(
        parents,
        key=lambda p: (-abs(float(cache.X[:, p] @ y)), p),
    )
    nodes = 0
    complete = True

    def visit(included: tuple[int, ...], undecided: tuple[int, ...]) -> None:
        nonlocal best_score, best_parents, nodes, complete
        if not complete:
            return
        nodes += 1
        if nodes > node_limit:
            complete = False
            return

        included = tuple(sorted(included))
        included_score, _ = cache.score_and_rss(included)
        if _prefer(included_score, included, best_score, best_parents, tolerance):
            best_score, best_parents = included_score, included
        if not undecided:
            return

        optimistic_parents = tuple(sorted(included + undecided))
        _, optimistic_rss = cache.score_and_rss(optimistic_parents)
        n = cache.X.shape[0]
        # A descendant either selects no undecided parent (and therefore has
        # the already-computed score for ``included``) or selects at least one.
        # In the latter case, the full undecided design gives an optimistic RSS
        # while the model must pay for at least one additional coefficient.
        # Taking the smaller of these two quantities is a valid and sharper
        # lower bound than pairing full-model RSS with the included-only
        # penalty.
        any_addition_bound = (
            n * log(optimistic_rss / n) + (len(included) + 2) * log(n)
        )
        lower_bound = min(included_score, any_addition_bound)
        if lower_bound >= best_score - tolerance:
            return

        parent = undecided[0]
        rest = undecided[1:]
        # Sparse models are visited first to obtain a strong incumbent early.
        visit(included, rest)
        visit(tuple(sorted(included + (parent,))), rest)

    visit((), tuple(priority))
    return LocalSelectionResult(
        child=child,
        candidate_parents=parents,
        selected_parents=best_parents,
        score=best_score,
        method="branch-and-bound" if complete else "branch-and-bound-limited",
        score_evaluations=cache.evaluations,
        search_nodes=nodes,
        globally_optimal=complete,
    )


def exact_refine_dag(
    X: np.ndarray,
    adjacency: np.ndarray,
    *,
    enumeration_max_parents: int = DEFAULT_ENUMERATION_MAX_PARENTS,
    branch_node_limit: int = DEFAULT_BRANCH_NODE_LIMIT,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
    rank_tolerance: float | None = None,
) -> RefinementResult:
    """Optimize every child-specific parent subset independently.

    Enumeration is used through ``enumeration_max_parents`` candidate parents.
    Larger local problems use exact branch-and-bound.  If the explicit search
    node limit is reached, the best incumbent is returned with
    ``globally_optimal=False`` and method ``hybrid-limited``.  Consequently a
    caller can never mistake a resource-limited result for a certified optimum.
    """
    start = perf_counter()
    X = np.asarray(X, dtype=float)
    A0 = _validate_adjacency(adjacency, X.shape[1])
    if not is_acyclic(A0):
        raise ValueError("the fixed candidate graph must be acyclic")
    if enumeration_max_parents < 0 or branch_node_limit < 1:
        raise ValueError("search limits must be positive")

    A = np.zeros_like(A0)
    local_results = []
    evaluations = 0
    for child in range(A0.shape[0]):
        parents = tuple(int(p) for p in np.flatnonzero(A0[:, child]))
        cache = _LocalScoreCache(X, child, rank_tolerance)
        if len(parents) <= enumeration_max_parents:
            local = _enumerate_local(cache, child, parents, score_tolerance)
        else:
            local = _branch_and_bound_local(
                cache, child, parents, score_tolerance, branch_node_limit
            )
        A[list(local.selected_parents), child] = 1
        local_results.append(local)
        evaluations += cache.evaluations

    total = float(sum(result.score for result in local_results))
    certified = all(result.globally_optimal for result in local_results)
    methods = {result.method for result in local_results if result.candidate_parents}
    if not certified:
        method = "hybrid-limited"
    elif methods <= {"enumeration"}:
        method = "exact-enumeration"
    else:
        method = "exact-hybrid"
    removed = tuple(
        (int(parent), int(child))
        for parent, child in np.argwhere((A0 == 1) & (A == 0))
    )
    return RefinementResult(
        adjacency=A,
        total_bic=total,
        local_results=tuple(local_results),
        removed_edges=removed,
        bic_history=(total_gaussian_bic(X, A0, rank_tolerance=rank_tolerance), total),
        score_evaluations=evaluations,
        runtime_seconds=perf_counter() - start,
        method=method,
        globally_optimal=certified,
    )


def candidate_indegree_summary(adjacency: np.ndarray) -> dict[str, object]:
    """Return the full candidate indegree vector and exact-search burden."""
    A = _validate_adjacency(adjacency)
    indegrees = A.sum(axis=0).astype(int)
    return {
        "indegrees": indegrees,
        "maximum_indegree": int(indegrees.max(initial=0)),
        "mean_indegree": float(indegrees.mean()),
        "candidate_edges": int(A.sum()),
        "enumeration_fits": int(sum(1 << int(q) for q in indegrees)),
    }


def initial_pruning_pressure(
    X: np.ndarray,
    adjacency: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> tuple[dict[str, float], list[dict[str, float | int | bool]]]:
    """Evaluate every edge once in its child's initial candidate parent set."""
    X = np.asarray(X, dtype=float)
    A = _validate_adjacency(adjacency, X.shape[1])
    cutoff = bic_partial_r2_cutoff(X.shape[0])
    rows: list[dict[str, float | int | bool]] = []
    for child in range(A.shape[0]):
        parents = tuple(int(p) for p in np.flatnonzero(A[:, child]))
        for parent in parents:
            diag = deletion_diagnostics(
                X, child, parents, parent, rank_tolerance=rank_tolerance
            )
            rows.append({
                "parent": parent,
                "child": child,
                "partial_r2": diag["partial_r2"],
                "delta_bic": diag["delta_bic"],
                "cutoff": cutoff,
                "below_cutoff": bool(diag["partial_r2"] < cutoff),
            })
    below = sum(bool(row["below_cutoff"]) for row in rows)
    summary = {
        "candidate_edges": float(len(rows)),
        "edges_below_cutoff": float(below),
        "initial_pruning_pressure": float(below / len(rows)) if rows else 0.0,
        "partial_r2_cutoff": cutoff,
    }
    return summary, rows


def edge_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Jaccard similarity of two directed edge sets."""
    A = _validate_adjacency(first)
    B = _validate_adjacency(second, A.shape[0])
    union = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / union) if union else 1.0


def graph_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    """Canonical directed metrics with every reversal counted as one SHD move."""
    T = _validate_adjacency(truth)
    E = _validate_adjacency(estimate, T.shape[0])
    truth_edges = {(int(u), int(v)) for u, v in np.argwhere(T == 1)}
    estimate_edges = {(int(u), int(v)) for u, v in np.argwhere(E == 1)}
    missing = truth_edges - estimate_edges
    extra = estimate_edges - truth_edges
    reversals = sum((v, u) in extra for u, v in missing)
    tp = len(truth_edges & estimate_edges)
    fp = len(extra)
    fn = len(missing)
    estimated = len(estimate_edges)
    return {
        "edges": estimated,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "reversals": int(reversals),
        "fdr": float(fp / estimated) if estimated else 0.0,
        "tpr": float(tp / len(truth_edges)) if truth_edges else 0.0,
        "shd": int(fp + fn - reversals),
    }
