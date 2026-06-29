"""
Synthetic example for NOTEARS-BP.

This example generates a true DAG, simulates linear-Gaussian data,
adds false-positive edges to create an initial estimated DAG, and then
uses BIC pruning to remove unsupported edges.

Run:
    python example_simulation.py
"""

import numpy as np

from notears_bp import bic_prune_dag, compare_to_truth, edge_counts, is_acyclic


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
    result = bic_prune_dag(X, A_initial)

    print("Synthetic NOTEARS-BP example")
    print("----------------------------")
    print(f"True edges:       {edge_counts(A_true)}")
    print(f"Initial edges:    {edge_counts(A_initial)}")
    print(f"Pruned edges:     {edge_counts(result.adjacency)}")
    print(f"Removed edges:    {len(result.removed_edges)}")
    print(f"BIC evaluations:  {result.n_bic_evaluations}")
    print(f"BIC improvement:  {result.total_bic_improvement:.3f}")
    print()

    print("Initial graph diagnostics")
    print(compare_to_truth(A_initial, A_true))
    print()

    print("Pruned graph diagnostics")
    print(compare_to_truth(result.adjacency, A_true))


if __name__ == "__main__":
    main()
