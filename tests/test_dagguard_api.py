import unittest
import numpy as np

from dagguard import pruning_pressure, refine_dag


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

    def test_near_tie_is_deterministic(self):
        rng = np.random.default_rng(19)
        n = 350
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
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

    def test_public_exact_is_scale_invariant(self):
        X, candidate = self._scale_test_problem()
        scaled = X * np.array([1e-8, 1e7, 1e3, 1e-4])
        base = refine_dag(X, candidate, method="exact")
        changed = refine_dag(scaled, candidate, method="exact")
        np.testing.assert_array_equal(base.adjacency, changed.adjacency)
        self.assertTrue(base.globally_optimal)
        self.assertTrue(changed.globally_optimal)

    def test_public_greedy_is_scale_invariant(self):
        X, candidate = self._scale_test_problem()
        scaled = X * np.array([1e-8, 1e7, 1e3, 1e-4])
        base = refine_dag(X, candidate, method="greedy")
        changed = refine_dag(scaled, candidate, method="greedy")
        np.testing.assert_array_equal(base.adjacency, changed.adjacency)

    def test_public_pruning_pressure_is_scale_invariant(self):
        X, candidate = self._scale_test_problem()
        scaled = X * np.array([1e-8, 1e7, 1e3, 1e-4])
        base_summary, base_rows = pruning_pressure(X, candidate)
        changed_summary, changed_rows = pruning_pressure(scaled, candidate)
        self.assertEqual(base_summary["edges_below_cutoff"], changed_summary["edges_below_cutoff"])
        self.assertAlmostEqual(
            base_summary["initial_pruning_pressure"],
            changed_summary["initial_pruning_pressure"],
            places=12,
        )
        self.assertEqual(
            [(r["parent"], r["child"], r["below_cutoff"]) for r in base_rows],
            [(r["parent"], r["child"], r["below_cutoff"]) for r in changed_rows],
        )

    def test_saturated_candidate_is_rejected(self):
        rng = np.random.default_rng(31)
        n = 5
        parents = rng.normal(size=(n, n - 1))
        y = rng.normal(size=n)
        X = np.column_stack([parents, y])
        child = n - 1
        candidate = np.zeros((n, n), dtype=int)
        candidate[: n - 1, child] = 1
        with self.assertRaisesRegex(ValueError, "saturated"):
            refine_dag(X, candidate, method="exact")

    def test_constant_response_is_rejected(self):
        rng = np.random.default_rng(37)
        x0 = rng.normal(size=100)
        y = np.ones(100)
        X = np.column_stack([x0, y])
        candidate = np.zeros((2, 2), dtype=int)
        candidate[0, 1] = 1
        with self.assertRaisesRegex(ValueError, "constant after centering"):
            refine_dag(X, candidate, method="exact")

    def test_exact_fit_is_rejected(self):
        rng = np.random.default_rng(41)
        n = 120
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        y = 2.0 * x0 - 0.5 * x1
        X = np.column_stack([x0, x1, y])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        with self.assertRaisesRegex(ValueError, "residual variance is numerically zero"):
            refine_dag(X, candidate, method="exact")

    def test_near_zero_full_model_rss_is_rejected(self):
        rng = np.random.default_rng(43)
        n = 150
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        y = 1.5 * x0 - 0.3 * x1 + 1e-9 * rng.normal(size=n)
        X = np.column_stack([x0, x1, y])
        candidate = np.zeros((3, 3), dtype=int)
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        with self.assertRaisesRegex(ValueError, "residual variance is numerically zero"):
            refine_dag(X, candidate, method="exact")

    @staticmethod
    def _scale_test_problem():
        rng = np.random.default_rng(29)
        n = 400
        x0 = rng.normal(size=n)
        x1 = 0.8 * x0 + rng.normal(size=n)
        x2 = -0.6 * x0 + 0.7 * x1 + rng.normal(size=n)
        x3 = 0.9 * x2 + 0.08 * x0 + rng.normal(size=n)
        X = np.column_stack([x0, x1, x2, x3])
        candidate = np.zeros((4, 4), dtype=int)
        candidate[0, 1] = 1
        candidate[0, 2] = 1
        candidate[1, 2] = 1
        candidate[0, 3] = 1
        candidate[1, 3] = 1
        candidate[2, 3] = 1
        return X, candidate


if __name__ == "__main__":
    unittest.main()
