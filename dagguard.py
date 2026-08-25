"""Public DAGGuard API.

DAGGuard refines an already learned directed acyclic graph (DAG) by deleting
candidate edges using decomposable Gaussian BIC. Two complementary algorithms
are exposed:

- ``method='greedy'``: repeated best single-parent deletion;
- ``method='exact'``: exact child-wise best-subset search using enumeration
  and branch-and-bound.

The public API enforces the regular local-regression conditions used by the
Gaussian-BIC theory. The numerical engine remains in ``local_bic_refinement.py``
for backward compatibility with the original NOTEARS-BP reproducibility
commit. New code should import from this module.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from local_bic_refinement import (
    RefinementResult,
    bic_partial_r2_cutoff,
    candidate_indegree_summary,
    deletion_diagnostics,
    edge_jaccard,
    exact_refine_dag,
    gaussian_local_bic,
    graph_metrics,
    greedy_refine_dag,
    initial_pruning_pressure,
    is_acyclic,
    total_gaussian_bic,
)

__all__ = [
    "RefinementResult",
    "refine_dag",
    "dagguard_exact",
    "dagguard_greedy",
    "pruning_pressure",
    "exact_refine_dag",
    "greedy_refine_dag",
    "gaussian_local_bic",
    "total_gaussian_bic",
    "deletion_diagnostics",
    "bic_partial_r2_cutoff",
    "candidate_indegree_summary",
    "edge_jaccard",
    "graph_metrics",
    "is_acyclic",
]


def _validate_candidate_full_rank(X, candidate_adjacency, rank_tolerance=None):
    """Validate regular Gaussian-BIC conditions for every candidate regression.

    The conventional local BIC used by DAGGuard penalizes the nominal number of
    selected parents. To keep that score aligned with regular Gaussian BIC, the
    public API requires each child's centered full candidate-parent design to
    have full column rank, to satisfy ``q_j < n - 1``, and to have strictly
    positive, numerically nondegenerate full-model residual variance. Every
    deletion subset is then also nonsaturated and full rank.

    The rank check uses the same scale-stabilization principle as the score
    engine: centered nonconstant predictor columns are normalized before the
    SVD. This makes the numerical validation invariant to changes of measurement
    units. When ``rank_tolerance`` is supplied, it is interpreted as a relative
    singular-value cutoff, matching NumPy least-squares ``rcond`` semantics.
    """
    X = np.asarray(X, dtype=float)
    A = (np.asarray(candidate_adjacency) != 0).astype(np.int8)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional numeric array")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two observations")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != X.shape[1]:
        raise ValueError("candidate_adjacency must be square and match X")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    if rank_tolerance is not None and rank_tolerance < 0:
        raise ValueError("rank_tolerance must be nonnegative")

    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    eps = np.finfo(float).eps
    residual_ratio_floor = 100.0 * eps

    for child in range(A.shape[0]):
        y = Xc[:, child]
        tss = float(y @ y)
        if tss == 0.0:
            raise ValueError(
                "candidate child response is constant after centering for child "
                f"{child}; conventional Gaussian BIC requires positive residual "
                "variance"
            )

        parents = np.flatnonzero(A[:, child])
        q = int(len(parents))
        if q >= n - 1:
            raise ValueError(
                "candidate local regression is saturated for child "
                f"{child}: q_j={q} candidate parents with n={n}; require "
                "q_j < n - 1 for conventional Gaussian BIC"
            )

        if q == 0:
            full_rss = tss
        else:
            design = Xc[:, parents]
            norms = np.linalg.norm(design, axis=0)
            if np.any(norms <= eps):
                raise ValueError(
                    "candidate parent design is rank deficient for child "
                    f"{child}: at least one candidate parent is constant after "
                    "centering; remove redundant predictors before Gaussian-BIC "
                    "refinement"
                )
            stable_design = design / norms
            singular = np.linalg.svd(stable_design, compute_uv=False)
            if singular.size == 0:
                rank = 0
            elif rank_tolerance is None:
                cutoff = max(stable_design.shape) * eps * singular[0]
                rank = int(np.sum(singular > cutoff))
            else:
                rank = int(np.sum(singular > float(rank_tolerance) * singular[0]))
            if rank < q:
                raise ValueError(
                    "candidate parent design is rank deficient for child "
                    f"{child}: rank {rank} < {q} candidate parents; remove "
                    "redundant predictors before Gaussian-BIC refinement"
                )

            coef, *_ = np.linalg.lstsq(
                stable_design, y, rcond=rank_tolerance
            )
            residual = y - stable_design @ coef
            full_rss = float(residual @ residual)

        if full_rss <= 0.0 or full_rss / tss <= residual_ratio_floor:
            raise ValueError(
                "candidate local regression is degenerate for child "
                f"{child}: full-model residual variance is numerically zero; "
                "conventional Gaussian BIC requires strictly positive residual "
                "variance"
            )


def refine_dag(
    X,
    candidate_adjacency,
    *,
    method: Literal["exact", "greedy"] = "exact",
    enumeration_max_parents: int = 15,
    branch_node_limit: int = 2_000_000,
    score_tolerance: float = 1e-10,
    rank_tolerance: float | None = None,
) -> RefinementResult:
    """Refine a fixed candidate DAG by local Gaussian BIC.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_variables)
        Numeric analysis matrix.
    candidate_adjacency : array-like, shape (d, d)
        Directed candidate adjacency with ``A[parent, child] = 1``. The graph
        must be acyclic. DAGGuard only deletes edges; it never adds or reverses
        an edge. For conventional Gaussian BIC, each child's centered candidate
        parent design must have full column rank, ``q_j < n - 1``, and positive
        full-model residual variance.
    method : {"exact", "greedy"}
        Exact best-subset refinement or fast greedy deletion.
    enumeration_max_parents, branch_node_limit : int
        Exact-search controls. They are ignored by the greedy method.
    score_tolerance : float
        Numerical tolerance for score comparisons. ``globally_optimal=True``
        records that the exact search established the minimum score to this
        tolerance without hitting the branch-node limit; it does not assert a
        unique representative when distinct subsets are numerically tied.
    rank_tolerance : float or None
        Relative singular-value cutoff used by the scale-stabilized public rank
        check and passed as ``rcond`` to the least-squares engine.

    Returns
    -------
    RefinementResult
        Selected adjacency, score, runtime, search diagnostics, and exact-search
        status where applicable.
    """
    _validate_candidate_full_rank(X, candidate_adjacency, rank_tolerance)
    method = str(method).lower()
    if method == "exact":
        return exact_refine_dag(
            X,
            candidate_adjacency,
            enumeration_max_parents=enumeration_max_parents,
            branch_node_limit=branch_node_limit,
            score_tolerance=score_tolerance,
            rank_tolerance=rank_tolerance,
        )
    if method == "greedy":
        return greedy_refine_dag(
            X,
            candidate_adjacency,
            score_tolerance=score_tolerance,
            rank_tolerance=rank_tolerance,
        )
    raise ValueError("method must be 'exact' or 'greedy'")


def pruning_pressure(X, candidate_adjacency, *, rank_tolerance: float | None = None):
    """Return DAGGuard's one-pass initial pruning-pressure diagnostic."""
    _validate_candidate_full_rank(X, candidate_adjacency, rank_tolerance)
    return initial_pruning_pressure(
        X, candidate_adjacency, rank_tolerance=rank_tolerance
    )


def dagguard_exact(X, candidate_adjacency, **kwargs) -> RefinementResult:
    """Convenience wrapper for exact DAGGuard refinement."""
    return refine_dag(X, candidate_adjacency, method="exact", **kwargs)


def dagguard_greedy(X, candidate_adjacency, **kwargs) -> RefinementResult:
    """Convenience wrapper for fast DAGGuard-Greedy refinement."""
    return refine_dag(X, candidate_adjacency, method="greedy", **kwargs)
