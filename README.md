# NOTEARS-BP

BIC-pruned NOTEARS (NOTEARS-BP) is a post-estimation pruning procedure for refining an initial directed acyclic graph (DAG), such as one estimated by NOTEARS. Starting from a thresholded NOTEARS graph, the method greedily removes directed edges when deletion improves the local Gaussian Bayesian Information Criterion (BIC) for the child node.

This repository accompanies the manuscript **“BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.”** It provides the NOTEARS-BP implementation, a small synthetic example, and a parameterized script for reproducing the manuscript simulation design.

The commercial swine-production records used in the application are not included because of privacy and commercial restrictions.

## Repository contents

- `notears_bp.py` — core NOTEARS-BP pruning implementation
- `example_simulation.py` — small runnable synthetic example
- `reproduce_primary_simulations.py` — parameterized simulation script used for the manuscript design
- `requirements.txt` — Python dependencies matching the recovered analysis environment
- `.github/workflows/reproduce-primary.yml` — lightweight reproducibility smoke test
- `LICENSE` — MIT license

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

## Reproducing the simulation design

The manuscript simulation script is parameterized by graph size `d`, weight regime, number of replicates, and number of parallel workers. The manuscript considers `d = 10, 20, 40`. Each simulated DAG contains **exactly `2*d` directed edges**, each dataset has `n = 500` observations, NOTEARS uses `lambda1 = 0.1`, and the estimated weighted graph is thresholded at `0.3` before BIC pruning.

Uniform-weight examples:

```bash
python reproduce_primary_simulations.py --d 10 --weight-kind uniform --M 20 --n-jobs 2 --out results/d10_uniform
python reproduce_primary_simulations.py --d 20 --weight-kind uniform --M 20 --n-jobs 2 --out results/d20_uniform
python reproduce_primary_simulations.py --d 40 --weight-kind uniform --M 20 --n-jobs 2 --out results/d40_uniform
```

Modified-normal examples:

```bash
python reproduce_primary_simulations.py --d 10 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d10_modnormal
python reproduce_primary_simulations.py --d 20 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d20_modnormal
python reproduce_primary_simulations.py --d 40 --weight-kind modnormal --M 20 --n-jobs 2 --out results/d40_modnormal
```

For a short code check, reduce the number of replicates, for example:

```bash
python reproduce_primary_simulations.py --d 10 --weight-kind uniform --M 1 --n-jobs 1 --out results/smoke_test
```

The `d = 40` configuration uses the same code and simulation design as the smaller graph sizes, but it is substantially more computationally intensive and individual replicates can have highly variable optimization times. For that reason, `d = 40` is **not included in routine GitHub Actions checks**. Users wishing to reproduce the full `d = 40` setting should run the same script on suitable local or high-performance computing resources.

The simulation output directory contains replicate-level metrics, summary statistics, run configuration information, and PDF/SVG/PNG figures for FDR, TPR, and structural Hamming distance (SHD).

## Method summary

Given a data matrix `X` and an initial DAG adjacency matrix `A`, NOTEARS-BP:

1. centers the data;
2. computes local Gaussian BIC scores for child-node regressions;
3. evaluates deletion of each existing edge;
4. removes the edge giving the largest BIC decrease;
5. repeats until no single-edge deletion improves BIC.

Because NOTEARS-BP only deletes edges from the initial graph, it preserves acyclicity but cannot add a true edge missed by NOTEARS or reverse an incorrectly oriented edge.

## Programmatic use

```python
from notears_bp import bic_prune_dag

result = bic_prune_dag(X, A_initial)
A_pruned = result.adjacency
removed_edges = result.removed_edges
```

`X` is an `n x d` data matrix and `A_initial` is a `d x d` binary adjacency matrix where `A_initial[i, j] = 1` denotes the edge `i -> j`.

## Data availability

The row-level swine-production data are proprietary and are not distributed in this repository. The public code is sufficient to run the synthetic examples and the parameterized simulation study without access to those records.

## Citation

If you use this code, please cite the associated manuscript:

Wang M, Liu P, Magalhães ES, Wang C. *BIC-Pruned NOTEARS for Sparse DAG Refinement with an Application to Swine Production.*
