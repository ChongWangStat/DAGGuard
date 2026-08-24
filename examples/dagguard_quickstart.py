"""Minimal DAGGuard example using a fixed candidate DAG."""
import numpy as np

from dagguard import edge_jaccard, refine_dag

rng = np.random.default_rng(20260824)
n = 500
x0 = rng.normal(size=n)
x1 = 0.9 * x0 + rng.normal(size=n)
x2 = 0.8 * x1 + rng.normal(size=n)
x3 = rng.normal(size=n)
X = np.column_stack([x0, x1, x2, x3])

# Candidate contains the true chain plus two extra edges.
candidate = np.zeros((4, 4), dtype=int)
candidate[0, 1] = 1
candidate[1, 2] = 1
candidate[0, 2] = 1  # extra candidate edge
candidate[3, 2] = 1  # extra candidate edge

exact = refine_dag(X, candidate, method="exact")
greedy = refine_dag(X, candidate, method="greedy")

print("candidate edges:", int(candidate.sum()))
print("exact edges:", int(exact.adjacency.sum()), "certified:", exact.globally_optimal)
print("greedy edges:", int(greedy.adjacency.sum()))
print("exact/greedy Jaccard:", edge_jaccard(exact.adjacency, greedy.adjacency))
print("BIC exact:", round(exact.total_bic, 3))
print("BIC greedy:", round(greedy.total_bic, 3))
