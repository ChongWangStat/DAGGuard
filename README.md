# DAGGuard

**False-discovery-aware refinement of learned directed acyclic graphs**

DAGGuard is a post-learning refinement method for an already oriented candidate DAG. Its fixed-candidate objective is defined conditional on the supplied data and graph: DAGGuard only decides which candidate edges to retain and never adds an omitted edge or reverses an upstream orientation. In the accompanying paper, all end-to-end experiments and the commercial application use NOTEARS-generated candidates; finite-sample performance after other upstream DAG learners has not been established here.

The accompanying manuscript is **“DAGGuard: False-Discovery-Aware Refinement of Learned DAGs with an Application to Commercial Swine Production.”**

## Why DAGGuard?

A learned DAG can be overconnected or contain weakly supported edges. DAGGuard separates candidate generation from post-learning assessment:

```text
DAG learning (NOTEARS in this study) -> fixed candidate DAG -> DAGGuard -> refined DAG
```

For Gaussian local BIC,

```text
BIC_j(S) = n log(RSS_j(S)/n) + (|S| + 1) log(n)
```

and the deletion-subgraph problem separates exactly by child. DAGGuard therefore provides two complementary algorithms:

- **DAGGuard-Greedy**: fast repeated best single-edge deletion.
- **DAGGuard-Exact**: certified child-wise best-subset selection using enumeration and branch-and-bound.

The one-edge BIC rule has the exact partial-R-squared interpretation

```text
delete edge iff partial R^2 < 1 - n^(-1/n).
```

DAGGuard is **false-discovery-aware**, not a finite-sample nominal FDR procedure. Its formal guarantees are conditional on a fixed or independently learned candidate. The current end-to-end empirical validation is specifically for NOTEARS-generated candidates.

## Installation

Python 3.12 was used for the reported analyses.

```bash
python -m pip install -r requirements.txt
```

## Public API

New code should import from `dagguard.py`:

```python
from dagguard import dagguard_exact, dagguard_greedy, pruning_pressure

exact = dagguard_exact(X, candidate)
greedy = dagguard_greedy(X, candidate)

print(exact.globally_optimal)
print(exact.total_bic)
```

Run the minimal example:

```bash
python -m examples.dagguard_quickstart
```

`local_bic_refinement.py` remains the tested numerical engine for backward compatibility with the earlier reproducibility commit.

## Main empirical results

### Controlled fixed-candidate experiments

Across 1,200 controlled candidate-contamination experiments, DAGGuard removed nearly all added false positives when the candidate was an overconnected screen. Under 100% added false-positive contamination, candidate FDR was 0.500 and DAGGuard-Exact FDR was 0.013 with TPR 1.000. DAGGuard-Greedy and DAGGuard-Exact returned the same edge set in 1,192/1,200 replicates.

### Seven-method end-to-end benchmark

Across 240 common Gaussian, centered-exponential, and centered-Gumbel simulation datasets using NOTEARS candidates for the DAGGuard rows, the pooled skeleton false-discovery proportion decreased from 0.148 for NOTEARS to 0.066 for DAGGuard-Exact while pooled TPR changed only from 0.860 to 0.856. The repository includes transparent comparator code and provenance for Wang et al. (2026), Li & Wang PC-FDR, PC-p, and ordinary PC.

Primary summary: `results/seven_method_benchmark/simulation_primary_seven_methods.csv`.

### Commercial swine application

The authorized analysis uses 2,556 complete lots and 37 variables. The pinned NOTEARS candidate has 185 edges; DAGGuard-Exact retains 87 and DAGGuard-Greedy 89. PRRS is adjacent to 60-day nursery mortality in all seven benchmark methods, while MYCO and third-quarter placement are supported by six. Four outgoing NOTEARS mortality relationships are removed by both DAGGuard variants and are absent from all four comparator graphs.

The proprietary row-level data are not distributed. The real-data workflow records a SHA256 hash and exports only non-row-level summaries.

## Repository map

- `dagguard.py` - public DAGGuard API.
- `local_bic_refinement.py` - backward-compatible numerical engine: local BIC, exact search, greedy search, pruning pressure, and graph metrics.
- `candidate_contamination_simulations.py` - 12-setting fixed-candidate experiment.
- `reproduce_simulations.py` - primary NOTEARS simulation workflow.
- `additional_noise_sensitivity.py` - Gaussian/exponential/Gumbel sensitivity analysis.
- `realdata_postselection_diagnostics.py` - authorized swine-data analysis.
- `benchmarks/seven_method/` - audited competitor implementations and real-data benchmark runner.
- `results/seven_method_benchmark/` - audited benchmark summary tables (no proprietary observations).
- `tests/` - deterministic unit tests.
- `REAL_DATA_SCHEMA.md` - construction of the 37 application variables.

## Reproduce the main DAGGuard analyses

```bash
python candidate_contamination_simulations.py \
  --replicates 100 --d 20 --n 500 \
  --out results/candidate_contamination

python additional_noise_sensitivity.py \
  --simulation-replicates 20 \
  --out results/additional_noise_sensitivity
```

Authorized real-data analysis:

```bash
python realdata_postselection_diagnostics.py \
  --data /authorized/path/train2023cw_simple.csv \
  --out results/swine_application \
  --primary-only
```

## Seven-method benchmark provenance

The benchmark distinguishes the methods' inferential targets. PC-family procedures are compared by skeleton adjacency because they need not return a uniquely oriented DAG. The Wang et al. (2026) structural method is transparently adapted to continuous variables by empirical tertiles and is not represented as official author software. PC-p is a source-audited Python port of the authors' official MATLAB code because MATLAB/Octave was unavailable in the benchmark runtime.

See `benchmarks/seven_method/README.md` and `results/seven_method_benchmark/method_implementation_provenance.csv`.

## Tests

```bash
python -m unittest discover -s tests -v
python -m examples.dagguard_quickstart
```

## Data availability

The commercial swine observations are proprietary and are not included. Requests for access remain subject to the applicable data-use agreements. Only non-row-level derived summaries are public.

## Historical reproducibility

This repository was renamed from `NOTEARS-BP` to `DAGGuard` on August 24, 2026. The application results used in the manuscript remain tied to historical commit `509cb29c24967d12b99e2b53641349ce7bb470ed`; preserving that commit keeps the analysis audit trail intact. GitHub redirects the earlier repository URL to the current DAGGuard repository.

## Citation

Wang M, Liu P, Magalhaes ES, Wang C. *DAGGuard: False-Discovery-Aware Refinement of Learned DAGs with an Application to Commercial Swine Production.* Journal of Data Science, submitted.
