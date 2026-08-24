"""Public DAGGuard API.

DAGGuard refines an already learned directed acyclic graph (DAG) by deleting
candidate edges using decomposable Gaussian BIC.  Two complementary algorithms
are exposed:

- ``method='greedy'``: repeated best single-parent deletion;
- ``method='exact'``: certified child-wise best-subset search using enumeration
  and branch-and-bound.

The numerical engine remains in ``local_bic_refinement.py`` for backward
compatibility with the original NOTEARS-BP reproducibility commit.  New code
should import from this module.
"""
from __future__ import annotations

from typing import Literal

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
        Directed candidate adjacency with ``A[parent, child] = 1``.  The graph
        must be acyclic. DAGGuard only deletes edges; it never adds or reverses
        an edge.
    method : {"exact", "greedy"}
        Exact certified best-subset refinement or fast greedy deletion.
    enumeration_max_parents, branch_node_limit : int
        Exact-search controls. They are ignored by the greedy method.
    score_tolerance, rank_tolerance : float or None
        Numerical controls passed to the underlying scoring/search engine.

    Returns
    -------
    RefinementResult
        Selected adjacency, score, runtime, search diagnostics, and exactness
        certificate where applicable.
    """
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
    return initial_pruning_pressure(
        X, candidate_adjacency, rank_tolerance=rank_tolerance
    )


def dagguard_exact(X, candidate_adjacency, **kwargs) -> RefinementResult:
    """Convenience wrapper for certified DAGGuard-Exact refinement."""
    return refine_dag(X, candidate_adjacency, method="exact", **kwargs)


def dagguard_greedy(X, candidate_adjacency, **kwargs) -> RefinementResult:
    """Convenience wrapper for fast DAGGuard-Greedy refinement."""
    return refine_dag(X, candidate_adjacency, method="greedy", **kwargs)
