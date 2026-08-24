import unittest
import numpy as np

from dagguard import refine_dag


class DagGuardApiTests(unittest.TestCase):
    def test_exact_and_greedy_return_subgraphs(self):
        rng = np.random.default_rng(7)
        n = 250
        x0 = rng.normal(size=n)
        x1 = 0.8 * x0 + rng.normal(size=n)
        x2 = 0.7 * x1 + rng.normal(size=n)
        X = np.column_stack([x0, x1, x2])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 1] = 1
        candidate[0, 2] = 1
        candidate[1, 2] = 1

        exact = refine_dag(X, candidate, method="exact")
        greedy = refine_dag(X, candidate, method="greedy")
        self.assertTrue(exact.globally_optimal)
        self.assertTrue(np.all(exact.adjacency <= candidate))
        self.assertTrue(np.all(greedy.adjacency <= candidate))
        self.assertLessEqual(exact.total_bic, greedy.total_bic + 1e-8)

    def test_unknown_method_rejected(self):
        X = np.eye(3)
        A = np.zeros((3, 3), dtype=int)
        with self.assertRaises(ValueError):
            refine_dag(X, A, method="unknown")


if __name__ == "__main__":
    unittest.main()
