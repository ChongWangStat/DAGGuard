# NOTEARS-BP

BIC-pruned NOTEARS (NOTEARS-BP) is the motivating application of a deletion-only BIC refinement operator, denoted BP, to a thresholded NOTEARS candidate DAG. BP itself is defined for any fixed candidate DAG: it removes an edge only when deletion lowers the decomposable local Gaussian Bayesian Information Criterion (BIC) for the child node.

This repository accompanies **“BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.”** The method was motivated by a commercial swine-production analysis in which the initial 190-edge NOTEARS graph was too dense for useful scientific interpretation. The manuscript uses paired simulations and targeted diagnostics to separate the contribution of BIC-guided edge selection from simply returning fewer edges, makes the scale structure of common synthetic benchmarks explicit, and tests whether the same pruning operator can refine a second continuous-optimization learner.

The commercial swine-production records are proprietary and are not included.

## Repository contents

- `notears_bp.py` — core NOTEARS-BP pruning implementation.
- `example_simulation.py` — small runnable synthetic example.
- `reproduce_simulations.py` — primary simulation design for `d = 10, 20, 40`, including the corrected PC configuration and both directed and skeleton metrics.
- `simulation_diagnostics.py` — equal-sparsity controls, varsortability audit, standardized-data check, modified-normal check, sample-size diagnostic, and compact DAGMA-to-BP generality diagnostic.
- `additional_validation_and_realdata.py` — additional Gaussian/non-Gaussian validation plus the authorized-data workflow for the primary, standardized, continuous-only, runtime, and bootstrap analyses.
- `realdata_postselection_diagnostics.py` — authorized-data diagnostics for initial pruning pressure, partial-R2 evidence, EBIC sensitivity, collinearity/redundancy, unit changes, and bootstrap recurrence.
- `REAL_DATA_SCHEMA.md` — names, types, and construction notes for the 37 application variables without row-level observations.
- `results/` — non-row-level derived summaries used to verify numerical statements in the manuscript and supplement.
- `requirements.txt` — pinned Python dependencies, including DAGMA for the generality diagnostic.
- `.github/workflows/reproduce-primary.yml` — lightweight reproducibility smoke test.
- `LICENSE` — MIT license.

## Installation

Python 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
```

## Quick start

```bash
python example_simulation.py
```

## Fixed-DAG BP interpretation

For deleting one current parent,

```text
Delta BIC = n * log(RSS_reduced / RSS_full) - log(n).
```

Equivalently, deletion occurs when the parent's partial R-squared is below

```text
1 - n^(-1/n).
```

Thus BP uses incremental conditional fit and a sample-size-adaptive BIC penalty rather than another fixed cutoff on raw edge magnitude. For a **fixed candidate DAG**, its deletion decisions are invariant to arbitrary nonzero rescaling of individual variables. This is not a claim that a full NOTEARS-plus-threshold-plus-BP pipeline is scale invariant: changing units can change the upstream candidate graph.

BP is also a no-op on a fixed DAG that is already locally optimal with respect to every single-edge deletion under the same BIC. This gives a practical scope rule: BP is useful when an upstream estimator supplies a candidate graph whose final selection step is not already deletion-optimal under that score.

## Primary simulation design

The manuscript considers `d = 10, 20, 40`, `n = 500`, and exactly `2*d` directed edges. NOTEARS uses `lambda1 = 0.1`, and its weighted estimate is thresholded at `0.3` to define the candidate DAG before BIC pruning.

```bash
python reproduce_simulations.py --d 20 --weight-kind uniform --M 20 --n-jobs 2 --out results/d20_uniform
python reproduce_simulations.py --d 20 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d20_modnormal
```

The same script supports `d=10` and `d=40`. The `d=40` configuration uses the same simulation design but is substantially more computationally intensive, so it is not included in routine GitHub Actions checks. In the modified-normal regime, SciPy uses `scale=s`, so the pre-shift draw is mathematically `N(0, s^2)`.

### PC configuration and equivalence-class metrics

The reported PC analysis used Fisher's Z test with `alpha=0.05` and causal-learn's standard conditioning-depth search:

```python
pc(X, alpha=0.05, indep_test=fisherz, show_progress=False)
```

The public reproduction script also exports skeleton metrics so PC/GES can be inspected without requiring every CPDAG edge to be oriented. DirectLiNGAM is not treated as a primary comparator in the Gaussian-error simulations because its identifying assumption is non-Gaussianity.

## Why the targeted simulation diagnostics are included

These diagnostics answer specific alternative explanations; they are not intended as a larger benchmark for its own sake.

### Equal-sparsity magnitude-pruning ablation

A smaller graph can have fewer false positives simply because it contains fewer edges. To isolate **which edges are removed** from **how many edges are removed**, the equal-sparsity controls use an after-the-fact rule: if BP retains `K` edges in a replicate, the control keeps exactly `K` NOTEARS candidate edges ranked by raw coefficient magnitude, with a second control using unit-normalized magnitude. These controls are not practical estimators because `K` is supplied by BP; their purpose is to hold graph size fixed.

At `d=10`, uniform `s=7`, all equal-sparsity graphs contain 18.45 edges on average, but BP has FDR 0.104 and SHD 5.45 versus FDR 0.177 and SHD 8.15 for raw-magnitude ranking, while retaining more true edges. A modified-normal check gives the same qualitative result.

### Varsortability and sample-size diagnostics

Mean path-based varsortability ranges from 0.971 to 0.998 across the 24 primary settings. The manuscript therefore emphasizes the paired NOTEARS-versus-BP change conditional on the same candidate graph rather than using absolute NOTEARS performance as a general causal-discovery claim. In a standardized high-varsortability setting, NOTEARS performs poorly and BP makes no additional deletion, illustrating that BP cannot recover structure that the upstream candidate graph has already lost.

The exact one-edge BIC cutoff is `1 - n^(-1/n)`. Varying `n = 100, 500, 2000` at `d=10`, uniform `s=7`, directly exercises that sample-size dependence.

### Generality beyond NOTEARS

To test whether the pruning operator is NOTEARS-specific, `simulation_diagnostics.py` applies the identical BP rule to DAGMA candidate graphs in two compact `d=10`, `n=500` settings. This is a generality diagnostic, not a broad DAGMA benchmark. With uniform `s=7`, DAGMA averaged FDR 0.133, TPR 0.970, SHD 5.20, and 24.0 edges; DAGMA-BP averaged FDR 0.005, TPR 0.970, SHD 0.70, and 19.5 edges. With modified-normal `s=3`, FDR decreased from 0.026 to 0.000 and SHD from 1.40 to 0.80 with TPR unchanged at 0.960.

Run the targeted simulation diagnostics with:

```bash
python simulation_diagnostics.py --M 20 --dagma-replicates 10 --out results/simulation_diagnostics
```

## Initial pruning-pressure diagnostic

Before running the greedy path, users can evaluate every candidate edge once using its child's initial parent set and compute the fraction whose partial R-squared is below the BIC cutoff. This **initial pruning pressure** is a pre-check, not an exact prediction of final sparsity because the conditional regressions change after deletions.

For the swine candidate graph, 106 of 190 edges (55.8%) were initially below the cutoff `0.003065`; iterative BP ultimately removed 98 edges and retained 92. For the standardized 20-edge candidate graph, none of the initial edges fell below the cutoff and BP made no deletion.

## Application diagnostics and proprietary data

The application contains six binary indicators. In the reported 190-edge thresholded candidate DAG all six had in-degree zero, and there were no within-block Q2-Q4 edges. Thus every local BP comparison had a continuous child and the Gaussian BIC was never used as a binary-response likelihood on the reported pruning path. This does not validate the mixed-variable NOTEARS initialization; the manuscript therefore also reports a continuous-variable-only sensitivity analysis and discusses mixed-data DAG methods explicitly.

The public derived results include mortality-neighborhood partial-R2 and Delta-BIC values, EBIC sensitivity, collinearity and redundancy diagnostics, fixed-candidate redundancy sensitivity, arbitrary unit-change sensitivity, NOTEARS and NOTEARS-BP bootstrap recurrence summaries, and original-scale/standardized/continuous-only sensitivity results.

The row-level data are not distributed. `REAL_DATA_SCHEMA.md` documents the analysis variables, while `additional_validation_and_realdata.py` and `realdata_postselection_diagnostics.py` run when an authorized local data file is supplied.

## Citation

Wang M, Liu P, Magalhães ES, Wang C. *BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.*
