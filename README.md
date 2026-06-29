# NOTEARS-BP

BIC-pruned NOTEARS (NOTEARS-BP) is a post-estimation pruning procedure for refining an initial directed acyclic graph (DAG), such as one estimated by NOTEARS. The method greedily removes directed edges when their deletion improves a local Bayesian Information Criterion (BIC) score.

This repository provides a clean implementation of the BIC-based pruning step and a small synthetic-data example. The swine production data used in the manuscript are not included because of privacy and commercial restrictions.

## Files

- `notears_bp.py`: core NOTEARS-BP pruning implementation
- `example_simulation.py`: runnable synthetic-data example
- `requirements.txt`: Python package requirements
- `LICENSE`: MIT license

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

Run the synthetic example:

```bash
python example_simulation.py
```

The example generates a true DAG, adds false-positive edges to create an initial DAG, applies BIC pruning, and prints edge-count diagnostics before and after pruning.

## Method summary

Given data matrix `X` and an initial DAG adjacency matrix `A`, NOTEARS-BP:

1. Centers the data.
2. Computes local BIC scores for each child node regression.
3. Iteratively considers removing each existing edge.
4. Removes the edge that gives the largest BIC decrease.
5. Stops when no single-edge deletion improves BIC.

Because NOTEARS-BP only deletes edges from the initial graph, it cannot add true edges missed by the initial estimator or reverse incorrectly oriented edges.

## Example use

```python
from notears_bp import bic_prune_dag

result = bic_prune_dag(X, A_initial)
A_pruned = result.adjacency
removed_edges = result.removed_edges
```

`X` is an `n x d` data matrix, and `A_initial` is a `d x d` binary adjacency matrix where `A_initial[i, j] = 1` indicates edge `i -> j`.

## Citation

If you use this code, please cite the associated manuscript:

Wang M, Liu P, Magalhães ES, Wang C. BIC-Pruned NOTEARS for Sparse DAG Refinement in Heterogeneous-Scale Data: An Application to Swine Production Systems.
