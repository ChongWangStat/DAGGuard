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

    def test_duplicate_candidate_parent_is_rejected(self):
        rng = np.random.default_rng(11)
        x0 = rng.normal(size=200)
        x1 = x0.copy()
        y = 0.7 * x0 + rng.normal(size=200)
        X = np.column_stack([x0, x1, y])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            refine_dag(X, candidate, method="exact")

    def test_near_collinearity_is_handled_when_full_rank(self):
        rng = np.random.default_rng(13)
        x0 = rng.normal(size=300)
        x1 = x0 + 1e-5 * rng.normal(size=300)
        y = 0.5 * x0 + 0.5 * x1 + rng.normal(size=300)
        X = np.column_stack([x0, x1, y])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        result = refine_dag(X, candidate, method="exact")
        self.assertTrue(result.globally_optimal)
        self.assertTrue(np.all(result.adjacency <= candidate))

    def test_near_tie_is_deterministic(self):
        rng = np.random.default_rng(19)
        n = 350
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        # Symmetric weak signals produce nearly competitive one-parent models.
        y = 0.12 * x0 + 0.12 * x1 + rng.normal(size=n)
        X = np.column_stack([x0, x1, y])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        first = refine_dag(X, candidate, method="exact", score_tolerance=1e-8)
        second = refine_dag(X, candidate, method="exact", score_tolerance=1e-8)
        self.assertTrue(first.globally_optimal)
        np.testing.assert_array_equal(first.adjacency, second.adjacency)
        self.assertAlmostEqual(first.total_bic, second.total_bic, places=10)


if __name__ == "__main__":
    unittest.main()
