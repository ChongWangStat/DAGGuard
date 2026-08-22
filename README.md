# NOTEARS-BP

BIC-pruned NOTEARS (NOTEARS-BP) is a post-estimation pruning procedure for refining an initial directed acyclic graph (DAG), such as one estimated by NOTEARS. Starting from a thresholded NOTEARS graph, the method greedily removes directed edges when deletion improves the local Gaussian Bayesian Information Criterion (BIC) for the child node.

This repository accompanies the manuscript **“BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.”** The method was motivated by a commercial swine-production analysis in which the initial NOTEARS graph was too dense for useful scientific interpretation. The manuscript then uses simulations to test whether BIC pruning improves graph recovery more generally.

The commercial swine-production records are proprietary and are not included.

## Repository contents

- `notears_bp.py` — core NOTEARS-BP pruning implementation.
- `example_simulation.py` — small runnable synthetic example.
- `reproduce_simulations.py` — parameterized implementation of the primary manuscript simulation design for `d = 10, 20, 40`.
- `additional_validation_and_realdata.py` — additional simulation, sensitivity, bootstrap, and real-data workflow; the real-data portion runs when the proprietary input file is supplied locally.
- `REAL_DATA_SCHEMA.md` — analysis-variable names, types, and construction notes without row-level data.
- `results/` — shareable derived summaries used to check reported simulation and application results.
- `requirements.txt` — pinned Python dependencies matching the recovered analysis environment.
- `.github/workflows/reproduce-primary.yml` — lightweight reproducibility smoke test.
- `LICENSE` — MIT license.

## Installation

Python 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
```

## Quick start

For a minimal demonstration of BIC pruning:

```bash
python example_simulation.py
```

The example generates a true DAG, adds false-positive edges to create an initial graph, applies NOTEARS-BP, and reports edge-count diagnostics before and after pruning.

## Reproducing the primary simulation design

The primary simulation script is parameterized by graph size `d`, weight regime, number of replicates, and number of parallel workers. The manuscript considers `d = 10, 20, 40`. Each simulated DAG contains **exactly `2*d` directed edges**, each dataset has `n = 500` observations, NOTEARS uses `lambda1 = 0.1`, and the estimated weighted graph is thresholded at `0.3` before BIC pruning.

Uniform-weight examples:

```bash
python reproduce_simulations.py --d 10 --weight-kind uniform --M 20 --n-jobs 2 --out results/d10_uniform
python reproduce_simulations.py --d 20 --weight-kind uniform --M 20 --n-jobs 2 --out results/d20_uniform
python reproduce_simulations.py --d 40 --weight-kind uniform --M 20 --n-jobs 2 --out results/d40_uniform
```

Modified-normal examples:

```bash
python reproduce_simulations.py --d 10 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d10_modnormal
python reproduce_simulations.py --d 20 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d20_modnormal
python reproduce_simulations.py --d 40 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d40_modnormal
```

In the modified-normal regime, SciPy uses `scale=s`, so the pre-shift draw is mathematically `N(0, s^2)`.

For a short code check, reduce the number of replicates:

```bash
python reproduce_simulations.py --d 10 --weight-kind uniform --M 1 --n-jobs 1 --out results/smoke_test
```

The `d = 40` configuration uses exactly the same code and simulation design as the smaller graph sizes, but it is substantially more computationally intensive and individual optimization times can vary greatly. For that reason, `d = 40` is **not included in routine GitHub Actions checks**; it can be run locally or on suitable high-performance computing resources.

The script writes replicate-level directed metrics, skeleton-level metrics, summary statistics, run configuration information, and PDF/SVG/PNG figures. Directed PC/GES metrics reproduce the manuscript convention in which unresolved CPDAG edges are not credited as oriented edges. The exported skeleton metrics provide an additional orientation-agnostic diagnostic for those equivalence-class methods.

### PC configuration

The reported PC analysis used Fisher's Z test with `alpha=0.05` and causal-learn's standard conditioning-depth search. The reproduction script states this explicitly using the current API:

```python
pc(X, alpha=0.05, indep_test=fisherz, show_progress=False)
```

An earlier archival script passed non-API keyword names through `**kwargs`; those keywords did not impose a conditioning-depth cap. The code here records the effective configuration used by the analysis rather than claiming such a cap.

## Additional validation and application workflow

`additional_validation_and_realdata.py` contains the additional Gaussian/non-Gaussian simulations, standardized-NOTEARS and HC+BIC comparisons, pruning diagnostics, real-data sensitivity analyses, runtime summaries, and bootstrap stability workflow used in the manuscript. It does **not** contain the commercial records. The real-data analysis expects the authorized input file to be supplied locally.

The public `results/` directory contains only non-row-level derived summaries. It is intended to let readers verify the numerical statements in the article without exposing proprietary observations.

## Method summary

Given data matrix `X` and an initial DAG adjacency matrix `A`, NOTEARS-BP:

1. centers the data;
2. computes local BIC scores for child-node regressions;
3. evaluates deletion of each existing edge;
4. removes the edge giving the largest BIC decrease;
5. repeats until no single-edge deletion improves BIC.

Because NOTEARS-BP only deletes edges from the initial graph, it preserves acyclicity but cannot add a true edge missed by NOTEARS or reverse an incorrectly oriented edge.

For a fixed initial DAG, the Gaussian pruning rule is invariant to nonzero rescaling of individual variables. This property applies to the **pruning stage only**; rescaling may change the initial NOTEARS graph.

## Mixed-variable application

The swine application includes six binary indicators. In the reported 190-edge initial NOTEARS graph, all six binary nodes had in-degree zero. Therefore every edge evaluated by NOTEARS-BP had a continuous child, and the Gaussian local BIC was never used as a binary-response likelihood on the reported pruning path. This does not remove the mixed-data working-model concern for the NOTEARS initialization; the manuscript therefore also reports a continuous-variable-only sensitivity analysis.

The pruning algorithm itself only requires a decomposable local score, so type-specific local scores (for example Gaussian BIC for continuous children and logistic BIC for binary children) are a natural extension when a candidate graph contains edges into discrete child nodes.

## Data availability

The row-level swine-production data are proprietary and are not distributed in this repository. `REAL_DATA_SCHEMA.md` documents the analysis variables, and shareable derived summaries are provided in `results/`.

## Citation

If you use this code, please cite the associated manuscript:

Wang M, Liu P, Magalhães ES, Wang C. *BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.*
