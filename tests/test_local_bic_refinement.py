import unittest

import numpy as np

from additional_noise_sensitivity import simulate_lsem_noise
from local_bic_refinement import (
    bic_partial_r2_cutoff,
    deletion_diagnostics,
    exact_refine_dag,
    gaussian_local_bic,
    graph_metrics,
    greedy_refine_dag,
    total_gaussian_bic,
)


class LocalBICRefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260823)
        self.X = rng.normal(size=(250, 5))
        self.X[:, 4] = 1.2 * self.X[:, 0] - 0.8 * self.X[:, 2] + rng.normal(
            scale=0.8, size=250
        )
        self.A = np.zeros((5, 5), dtype=int)
        self.A[:4, 4] = 1

    def test_bic_deletion_algebra(self):
        diagnostic = deletion_diagnostics(self.X, 4, [0, 1, 2, 3], 1)
        self.assertAlmostEqual(
            diagnostic["delta_bic"], diagnostic["delta_bic_algebra"], places=10
        )
        self.assertAlmostEqual(
            diagnostic["partial_r2"],
            1 - diagnostic["rss_full"] / diagnostic["rss_reduced"],
            places=12,
        )

    def test_partial_r2_cutoff_matches_bic_decision(self):
        for parent in range(4):
            diagnostic = deletion_diagnostics(self.X, 4, [0, 1, 2, 3], parent)
            self.assertEqual(
                diagnostic["delta_bic"] < 0,
                diagnostic["partial_r2"] < bic_partial_r2_cutoff(len(self.X)),
            )

    def test_greedy_accepts_only_monotone_deletions(self):
        result = greedy_refine_dag(self.X, self.A)
        self.assertTrue(all(
            after < before - 1e-10
            for before, after in zip(result.bic_history, result.bic_history[1:])
        ))
        self.assertAlmostEqual(
            result.total_bic,
            total_gaussian_bic(self.X, result.adjacency),
            places=9,
        )

    def test_fixed_candidate_scale_invariance(self):
        scales = np.array([-3.0, 0.01, 100.0, -0.4, 8.0])
        for refiner in [greedy_refine_dag, exact_refine_dag]:
            original = refiner(self.X, self.A)
            rescaled = refiner(self.X * scales, self.A)
            np.testing.assert_array_equal(original.adjacency, rescaled.adjacency)

    def test_exact_and_greedy_agree_in_regular_case(self):
        exact = exact_refine_dag(self.X, self.A)
        greedy = greedy_refine_dag(self.X, self.A)
        self.assertTrue(exact.globally_optimal)
        np.testing.assert_array_equal(exact.adjacency, greedy.adjacency)
        self.assertAlmostEqual(exact.total_bic, greedy.total_bic, places=9)

    def test_constructed_greedy_failure(self):
        covariance = np.array([
            [1.0, -0.6308680699527369, -0.010056489454507466],
            [-0.6308680699527369, 1.0, 0.4741748172508156],
            [-0.010056489454507466, 0.4741748172508156, 1.0],
        ])
        beta = np.array([-2.7889506515605027, -0.3887820519872834, 0.5171986784279542])
        rng = np.random.default_rng(43)
        parents = rng.multivariate_normal(np.zeros(3), covariance, size=30)
        response = parents @ beta + rng.normal(scale=0.5, size=30)
        X = np.column_stack([parents, response])
        A = np.zeros((4, 4), dtype=int)
        A[:3, 3] = 1
        exact = exact_refine_dag(X, A)
        greedy = greedy_refine_dag(X, A)
        self.assertGreater(greedy.total_bic - exact.total_bic, 2.0)
        np.testing.assert_array_equal(exact.adjacency[:, 3], [1, 0, 0, 0])
        np.testing.assert_array_equal(greedy.adjacency[:, 3], [1, 1, 1, 0])

    def test_branch_and_bound_matches_enumeration(self):
        enumeration = exact_refine_dag(self.X, self.A, enumeration_max_parents=20)
        branch = exact_refine_dag(self.X, self.A, enumeration_max_parents=0)
        self.assertTrue(branch.globally_optimal)
        np.testing.assert_array_equal(enumeration.adjacency, branch.adjacency)
        self.assertAlmostEqual(enumeration.total_bic, branch.total_bic, places=9)

    def test_resource_limit_is_never_mislabeled_exact(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 9))
        X[:, 8] = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(
            scale=0.4, size=100
        )
        A = np.zeros((9, 9), dtype=int)
        A[:8, 8] = 1
        result = exact_refine_dag(
            X, A, enumeration_max_parents=0, branch_node_limit=1
        )
        self.assertFalse(result.globally_optimal)
        self.assertEqual(result.method, "hybrid-limited")

    def test_shd_counts_reversal_as_one(self):
        truth = np.zeros((3, 3), dtype=int)
        truth[0, 1] = 1
        truth[1, 2] = 1
        estimate = np.zeros((3, 3), dtype=int)
        estimate[1, 0] = 1
        estimate[1, 2] = 1
        result = graph_metrics(truth, estimate)
        self.assertEqual(result["reversals"], 1)
        self.assertEqual(result["shd"], 1)

    def test_gumbel_noise_is_unit_variance_and_deterministic(self):
        W = np.zeros((1, 1))
        first = simulate_lsem_noise(W, 300_000, "gumbel", 19)[:, 0]
        second = simulate_lsem_noise(W, 300_000, "gumbel", 19)[:, 0]
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(first.mean()), 0.0, delta=0.01)
        self.assertAlmostEqual(float(first.var()), 1.0, delta=0.02)

    def test_local_score_is_deterministic(self):
        first = gaussian_local_bic(self.X, 4, [0, 2])
        second = gaussian_local_bic(self.X, 4, [2, 0])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
