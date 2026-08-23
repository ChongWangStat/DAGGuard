"""
Synthetic example for exact and greedy local-BIC refinement.

This example generates a true DAG, simulates linear-Gaussian data,
adds false-positive edges to create an initial estimated DAG, and then
uses exact local subset selection and greedy deletion to remove unsupported
edges from the same fixed candidate DAG.

Run:
    python example_simulation.py
"""

import numpy as np

from local_bic_refinement import (
    exact_refine_dag,
    graph_metrics,
    greedy_refine_dag,
    is_acyclic,
)


def random_dag(d: int, expected_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random DAG by sampling edges in a random topological order."""
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=int)
    p = expected_edges / (d * (d - 1) / 2)

    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                parent = order[a]
                child = order[b]
                A[parent, child] = 1

    return A


def topological_order(A: np.ndarray) -> list[int]:
    """Return one topological order for a DAG."""
    A = np.asarray(A, dtype=int)
    indegree = A.sum(axis=0).astype(int)
    queue = list(np.where(indegree == 0)[0])
    order = []

    while queue:
        node = queue.pop(0)
        order.append(int(node))
        for child in np.where(A[node, :] != 0)[0]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))

    if len(order) != A.shape[0]:
        raise ValueError("A is not acyclic.")
    return order


def simulate_linear_sem(
    A: np.ndarray,
    n: int,
    rng: np.random.Generator,
    weight_low: float = 0.5,
    weight_high: float = 2.0,
) -> np.ndarray:
    """Simulate data from a linear SEM using a DAG adjacency matrix."""
    d = A.shape[0]
    W = np.zeros((d, d), dtype=float)

    for i, j in zip(*np.where(A != 0)):
        sign = rng.choice([-1.0, 1.0])
        W[i, j] = sign * rng.uniform(weight_low, weight_high)

    X = np.zeros((n, d), dtype=float)
    noise = rng.normal(size=(n, d))

    for j in topological_order(A):
        parents = np.where(A[:, j] != 0)[0]
        X[:, j] = noise[:, j]
        if len(parents) > 0:
            X[:, j] += X[:, parents] @ W[parents, j]

    return X


def add_false_positive_edges(
    A: np.ndarray,
    n_extra: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add false-positive edges while preserving acyclicity.

    This mimics an initial DAG estimate that contains extra edges.
    """
    d = A.shape[0]
    A_initial = A.copy()
    candidates = [(i, j) for i in range(d) for j in range(d) if i != j and A_initial[i, j] == 0]
    rng.shuffle(candidates)

    added = 0
    for i, j in candidates:
        A_initial[i, j] = 1
        if is_acyclic(A_initial):
            added += 1
        else:
            A_initial[i, j] = 0
        if added >= n_extra:
            break

    return A_initial


def main() -> None:
    rng = np.random.default_rng(123)

    d = 10
    n = 500
    expected_edges = 2 * d

    A_true = random_dag(d=d, expected_edges=expected_edges, rng=rng)
    X = simulate_linear_sem(A_true, n=n, rng=rng)

    A_initial = add_false_positive_edges(A_true, n_extra=10, rng=rng)
    exact = exact_refine_dag(X, A_initial)
    greedy = greedy_refine_dag(X, A_initial)

    print("Fixed-candidate local-BIC refinement example")
    print("--------------------------------------------")
    print(f"True edges:       {int(A_true.sum())}")
    print(f"Candidate edges:  {int(A_initial.sum())}")
    print(f"Exact edges:      {int(exact.adjacency.sum())}")
    print(f"Greedy edges:     {int(greedy.adjacency.sum())}")
    print(f"Exact certified:  {exact.globally_optimal}")
    print(f"Greedy BIC gap:   {greedy.total_bic - exact.total_bic:.6f}")
    print()

    print("Initial graph diagnostics")
    print(graph_metrics(A_true, A_initial))
    print()

    print("Exact-refinement diagnostics")
    print(graph_metrics(A_true, exact.adjacency))
    print()

    print("Greedy-refinement diagnostics")
    print(graph_metrics(A_true, greedy.adjacency))


if __name__ == "__main__":
    main()
