# DAGGuard v1.0.0 release candidate

This repository state accompanies the JDS submission **“DAGGuard: Exact and Greedy BIC Refinement of Learned DAGs with an Application to Commercial Swine Production.”**

## Reproducibility scope

- Validated public refinement API: `dagguard.py`.
- Historical numerical engine and pinned application audit trail: `local_bic_refinement.py` and commit `509cb29c24967d12b99e2b53641349ce7bb470ed`.
- Controlled candidate-contamination simulations: 1,200 replicates.
- NOTEARS end-to-end Gaussian/non-Gaussian simulations: 240 datasets.
- NOTEARS penalty/threshold sensitivity addressing dependence on upstream tuning.
- Seven-method benchmark provenance and non-row-level summaries: `results/seven_method_benchmark/`.
- External-method source audit: `benchmarks/seven_method/SOURCE_AUDIT.md`.
- Proprietary swine observations are not distributed; the authorized workflow verifies the pinned input SHA-256.
- `synthetic_application_twin.py` provides a public 37-variable end-to-end example without confidential observations.
- `reproduce_submission.sh` is the staged reproduction entry point.

## Numerical policy

For conventional Gaussian BIC, every public refinement entry point requires each centered full candidate-parent design to have full column rank, `q_j < n - 1`, and strictly positive, numerically nondegenerate full-model residual variance. Rank-deficient, saturated, constant-response, and degenerate candidate regressions are rejected rather than silently scored. `globally_optimal=True` means that the exact search established the minimum objective value within the documented numerical tolerance and did not hit its search limit; it does not imply uniqueness among numerically tied subsets.

The deterministic test suite covers the public numerical policy, exact-versus-greedy invariants, branch-and-bound certification, and targeted source-audit checks for the comparator calculations.

A GitHub tag/release should point to the final merged submission commit after the final acceptance-focused pull request passes CI and is merged.
