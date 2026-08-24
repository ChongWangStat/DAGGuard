"""Li and Wang (2009) PC-FDR benchmark implementation."""
from __future__ import annotations

import itertools
import time
import numpy as np

from .common import fisherz_p


def fdr_stepup(pvalues, q: float, *, by: bool = False) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    sorted_p = p[order]
    multiplicity = m * (np.sum(1 / np.arange(1, m + 1)) if by else 1.0)
    keep = 0
    for rank in range(1, m + 1):
        if multiplicity / rank * sorted_p[rank - 1] <= q:
            keep = rank
    reject = np.zeros(m, dtype=bool)
    if keep:
        reject[order[:keep]] = True
    return reject


def pc_fdr_skeleton(X, q: float = 0.05, *, by: bool = False, heuristic: bool = False):
    """Algorithm 3 style PC-FDR skeleton with Gaussian Fisher-z CI tests.

    The primary benchmark uses ``heuristic=False`` and ``by=False``. Optional
    switches are retained only for sensitivity checks documented in the audit.
    """
    start = time.perf_counter()
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    correlation = np.corrcoef(X, rowvar=False)
    graph = np.ones((d, d), dtype=bool)
    np.fill_diagonal(graph, False)
    pmax = {(i, j): -1.0 for i in range(d) for j in range(i + 1, d)}
    tests = 0
    depth = 0
    while True:
        any_pair = False
        for a in range(d):
            for b in list(np.flatnonzero(graph[a])):
                if not graph[a, b]:
                    continue
                neighbors = [v for v in np.flatnonzero(graph[a]) if v != b]
                if len(neighbors) < depth:
                    continue
                any_pair = True
                for conditioning in itertools.combinations(neighbors, depth):
                    p = fisherz_p(correlation, n, a, b, conditioning)
                    tests += 1
                    key = (min(a, b), max(a, b))
                    if p > pmax.get(key, -1):
                        pmax[key] = p
                        relevant = (
                            [key for key in pmax if graph[key[0], key[1]]]
                            if heuristic
                            else [(i, j) for i in range(d) for j in range(i + 1, d)]
                        )
                        if relevant and all(pmax.get(edge, -1) >= 0 for edge in relevant):
                            reject_nonedge = fdr_stepup([pmax[edge] for edge in relevant], q, by=by)
                            for idx, rejected in enumerate(reject_nonedge):
                                if not rejected:
                                    u, v = relevant[idx]
                                    graph[u, v] = graph[v, u] = False
                            if not graph[a, b]:
                                break
        depth += 1
        possible = any(
            len([v for v in np.flatnonzero(graph[a]) if v != b]) >= depth
            for a in range(d)
            for b in np.flatnonzero(graph[a])
        )
        if not possible or not any_pair or depth > d - 2:
            break
    return graph.astype(int), time.perf_counter() - start, tests
