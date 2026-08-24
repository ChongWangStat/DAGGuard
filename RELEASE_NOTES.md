# DAGGuard v1.0.0 release candidate

This repository state accompanies the JDS submission **“DAGGuard: Exact and Greedy BIC Refinement of Learned DAGs with an Application to Commercial Swine Production.”**

## Reproducibility scope

- Public refinement API: `dagguard.py`.
- Historical numerical engine and pinned application audit trail: `local_bic_refinement.py` and commit `509cb29c24967d12b99e2b53641349ce7bb470ed`.
- Controlled candidate-contamination simulations: 1,200 replicates.
- NOTEARS end-to-end Gaussian/non-Gaussian simulations: 240 datasets.
- NOTEARS penalty/threshold sensitivity addressing dependence on upstream tuning.
- Seven-method benchmark provenance and non-row-level summaries: `results/seven_method_benchmark/`.
- Proprietary swine observations are not distributed; the authorized workflow verifies the pinned input SHA-256.
- `synthetic_application_twin.py` provides a public 37-variable end-to-end example without confidential observations.
- `reproduce_submission.sh` is the staged reproduction entrypoint.

## Numerical policy

The public API requires each centered full candidate-parent design to have full column rank for conventional Gaussian BIC. Rank-deficient candidates are rejected. `globally_optimal=True` means that the exact search established the minimum objective value within the documented numerical tolerance and did not hit its search limit; it does not imply uniqueness among numerically tied subsets.

A GitHub tag/release should point to the final merged submission commit after the referee-revision branch is approved.
