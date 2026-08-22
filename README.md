# NOTEARS-BP

BIC-pruned NOTEARS (NOTEARS-BP) is a deletion-only, post-estimation refinement of a candidate directed acyclic graph (DAG). Starting from a thresholded NOTEARS graph, it removes an edge only when deletion lowers the local Gaussian Bayesian Information Criterion (BIC) for the child node.

This repository accompanies **“BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.”** The method was motivated by a commercial swine-production analysis in which the initial 190-edge NOTEARS graph was too dense for useful scientific interpretation. The manuscript then uses targeted simulations and diagnostics to ask whether the improvement is attributable to the BIC selection rule rather than simply to returning fewer edges, and to make the scale structure of the synthetic benchmarks explicit.

The commercial swine-production records are proprietary and are not included.

## Repository contents

- `notears_bp.py` — core NOTEARS-BP implementation.
- `example_simulation.py` — small runnable synthetic example.
- `reproduce_simulations.py` — primary simulation design for `d = 10, 20, 40`, with corrected PC configuration and both directed and skeleton metrics.
- `targeted_simulation_diagnostics.py` — original equal-sparsity magnitude ablation, varsortability audit, and standardized-data diagnostic.
- `round2_targeted_diagnostics.py` — second-round diagnostic extension: a modified-normal equal-sparsity check and an `n = 100, 500, 2000` sample-size diagnostic for the BIC cutoff.
- `additional_validation_and_realdata.py` — additional Gaussian/non-Gaussian validation and proprietary-data workflow; the real-data portion runs only when the authorized input file is supplied locally.
- `realdata_postselection_diagnostics.py` — authorized-data diagnostics for partial-R2 evidence, EBIC sensitivity, collinearity/redundancy, unit changes, and bootstrap recurrence.
- `REAL_DATA_SCHEMA.md` — names, types, and construction notes for the 37 analysis variables without row-level observations.
- `results/` — non-row-level derived summaries used to check the numerical statements in the manuscript and supplement.
- `requirements.txt` — pinned Python dependencies matching the recovered analysis environment.
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

## Primary simulation design

The manuscript considers `d = 10, 20, 40`, `n = 500`, and exactly `2*d` directed edges. NOTEARS uses `lambda1 = 0.1`, and its weighted estimate is thresholded at `0.3` to define the candidate DAG before BIC pruning.

```bash
python reproduce_simulations.py --d 20 --weight-kind uniform --M 20 --n-jobs 2 --out results/d20_uniform
python reproduce_simulations.py --d 20 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d20_modnormal
```

The same script supports `d=10` and `d=40`. The `d=40` setting is substantially more computationally intensive and is therefore not included in routine GitHub Actions checks. In the modified-normal regime, SciPy uses `scale=s`, so the pre-shift draw is mathematically `N(0, s^2)`.

### PC configuration and equivalence-class metrics

The reported PC analysis used Fisher's Z test with `alpha=0.05` and causal-learn's standard conditioning-depth search:

```python
pc(X, alpha=0.05, indep_test=fisherz, show_progress=False)
```

An earlier archival script passed non-API keyword names through `**kwargs`; those keywords did not impose a conditioning-depth cap. The public reproduction script records the effective configuration rather than claiming such a cap. It also exports skeleton metrics so PC/GES can be inspected without requiring every CPDAG edge to be oriented.

DirectLiNGAM is not treated as a primary comparator in the Gaussian-error simulations because its identifying assumption is non-Gaussianity.

## Why the targeted diagnostics are included

The revision adds diagnostics for specific alternative explanations; they are not intended as a larger benchmark for its own sake.

### 1. Equal-sparsity magnitude-pruning ablation

A smaller graph can have fewer false positives simply because it contains fewer edges. To isolate **which edges are removed** from **how many edges are removed**, the equal-sparsity controls use an after-the-fact rule: if BP retains `K` edges in a replicate, the control keeps exactly `K` NOTEARS candidate edges ranked by raw coefficient magnitude, with a second control using unit-normalized magnitude. These controls are **not practical estimators** because `K` is supplied by BP. Their only purpose is to hold graph size fixed.

At `d=10`, uniform `s=7`, all equal-sparsity graphs contain 18.45 edges on average, but BP has FDR 0.104 and SHD 5.45 versus FDR 0.177 and SHD 8.15 for raw-magnitude ranking; BP also has higher TPR (0.820 versus 0.755). To verify that this mechanism is not specific to the uniform coefficient distribution, `round2_targeted_diagnostics.py` repeats the same check with modified-normal weights (`d=10`, `s=3`): all diagnostic graphs contain 17.30 edges, while BP has FDR 0.075 and SHD 5.45 versus 0.114 and 6.90 for raw-magnitude ranking. Derived summaries are in `results/equal_sparsity_ablation_summary.csv` and `results/equal_sparsity_modnormal_summary.csv`.

### 2. Varsortability audit

Synthetic additive-noise DAG benchmarks can reveal causal order through marginal variances. The repository therefore reports path-based varsortability for every primary setting instead of treating the benchmark as artifact-free. Mean varsortability ranges from 0.971 to 0.998 across the 24 graph-size/weight settings. The manuscript consequently interprets the **paired NOTEARS-versus-BP change conditional on the same candidate graph**, rather than using absolute NOTEARS performance as a general causal-discovery claim.

A standardized `d=10`, uniform-`s=7` diagnostic reduces mean marginal varsortability from about 0.995 to 0.566. Standardized NOTEARS then performs poorly and BP makes no additional deletions, which is an important negative result: BP does not recover structure that the initial graph failed to include.

### 3. Sample-size adaptivity diagnostic

The exact one-edge BIC rule has cutoff `1 - n^(-1/n)` on partial R-squared, so the cutoff changes with sample size. `round2_targeted_diagnostics.py` therefore varies `n = 100, 500, 2000` at `d=10`, uniform `s=7` rather than relying only on the formula. The cutoff decreases from 0.0450 to 0.00379. Across all three sample sizes BP reduces FDR and SHD relative to its paired NOTEARS candidate graph with essentially unchanged TPR. The purpose is to demonstrate the claimed sample-size adaptivity, not to introduce a new broad benchmark. See `results/sample_size_diagnostic.csv`.

### 4. Real-data post-selection checks

The public derived results make several alternative explanations directly checkable without disclosing commercial rows:

- mortality-neighborhood partial-R2 and Delta-BIC values;
- EBIC sensitivity at `gamma = 0.5` and `1`;
- collinearity and redundancy diagnostics;
- a fixed-candidate redundancy sensitivity;
- arbitrary unit-change sensitivity illustrating that conditional scale invariance of BP is **not** full-pipeline scale invariance;
- NOTEARS and NOTEARS-BP bootstrap recurrence summaries;
- continuous-only and standardized real-data sensitivity analyses.

The unit-change diagnostic also reports edge-set Jaccard similarities. Dividing `HeadIn` and `final_inventory` by 1,000 changes NOTEARS from 190 to 156 edges and BP from 92 to 90, but the edge-set Jaccards versus the original fits are only 0.498 and 0.529. Similar edge counts therefore should not be read as set-level invariance.

## Statistical interpretation

For deleting one current parent, the local BIC difference is

```text
Delta BIC = n * log(RSS_reduced / RSS_full) - log(n).
```

Equivalently, deletion occurs when the parent's partial R-squared is below

```text
1 - n^(-1/n).
```

Thus the pruning decision uses incremental conditional fit and a sample-size-adaptive BIC penalty rather than another fixed cutoff on raw NOTEARS coefficient magnitude.

For a **fixed candidate DAG**, the pruning path is invariant to arbitrary nonzero rescaling of individual variables. This is a property of the pruning stage only: rescaling can change the upstream NOTEARS optimization or which coefficients cross the initial candidate threshold, and BP cannot restore an excluded edge.

## Mixed-variable application

The application contains six binary indicators. In the reported **190-edge thresholded candidate DAG** all six had in-degree zero, and there were no within-block Q2-Q4 edges. Thus every local BP comparison had a continuous child and the Gaussian BIC was never used as a binary-response likelihood on the reported pruning path. This diagnostic is computed from the thresholded candidate adjacency, not the much denser unthresholded coefficient matrix. It does not validate the mixed-variable NOTEARS initialization; the manuscript therefore also reports a continuous-variable-only sensitivity analysis and discusses mixed-data DAG methods explicitly.

## Reproducing the targeted simulation diagnostics

```bash
python targeted_simulation_diagnostics.py --M 20 --out results/targeted_diagnostics
python round2_targeted_diagnostics.py --M 20 --out results/round2_diagnostics
```

## Data availability

The row-level swine-production data are proprietary and are not distributed. `REAL_DATA_SCHEMA.md` documents the analysis variables, `additional_validation_and_realdata.py` and `realdata_postselection_diagnostics.py` provide authorized-data workflows, and `results/` contains only shareable derived summaries.

## Citation

Wang M, Liu P, Magalhães ES, Wang C. *BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.*
