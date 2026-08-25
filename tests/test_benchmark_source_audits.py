import unittest

import numpy as np
from scipy.stats import norm

from benchmarks.seven_method.common import fisherz_p
from benchmarks.seven_method.pc_fdr import fdr_stepup
from benchmarks.seven_method.pcp_faithful import by_fdr
from benchmarks.seven_method.wang_full import local_disc_bic
from dagguard import exact_refine_dag, greedy_refine_dag


class BenchmarkSourceAuditTests(unittest.TestCase):
    def test_fisher_z_matches_published_gaussian_formula(self):
        correlation = np.array([[1.0, 0.30], [0.30, 1.0]])
        n = 100
        observed = fisherz_p(correlation, n, 0, 1, ())
        statistic = np.sqrt(n - 3) * np.arctanh(0.30)
        expected = 2.0 * norm.sf(abs(statistic))
        self.assertAlmostEqual(observed, expected, places=14)

    def test_pc_fdr_stepup_matches_bh_rule(self):
        pvalues = np.array([0.001, 0.010, 0.200])
        rejected = fdr_stepup(pvalues, 0.05, by=False)
        np.testing.assert_array_equal(rejected, [True, True, False])

    def test_pc_p_by_estimator_matches_official_formula(self):
        pvalues = np.array([0.01, 0.02, 0.20])
        alpha = 0.05
        harmonic = 1.0 + 1.0 / 2.0 + 1.0 / 3.0
        expected = len(pvalues) * alpha * harmonic / 2.0
        self.assertAlmostEqual(by_fdr(pvalues, alpha), expected, places=14)

    def test_wang_discrete_bic_parameter_count(self):
        # A binary parent perfectly predicts a binary child. The saturated
        # conditional log-likelihood is zero and there are two free conditional
        # Bernoulli parameters, giving BIC = -0.5 * 2 * log(4).
        D = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=int)
        observed = local_disc_bic(D, child=1, parents=(0,))
        expected = -np.log(4.0)
        self.assertAlmostEqual(observed, expected, places=14)

    def test_all_public_refinement_entrypoints_reject_saturation(self):
        rng = np.random.default_rng(20260824)
        n = 5
        parents = rng.normal(size=(n, n - 1))
        y = rng.normal(size=n)
        X = np.column_stack([parents, y])
        candidate = np.zeros((n, n), dtype=int)
        candidate[: n - 1, n - 1] = 1
        for refiner in (exact_refine_dag, greedy_refine_dag):
            with self.assertRaisesRegex(ValueError, "saturated"):
                refiner(X, candidate)


if __name__ == "__main__":
    unittest.main()
