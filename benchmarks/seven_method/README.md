# Seven-method benchmark

This directory contains the comparator implementations used for the DAGGuard JDS benchmark. The common simulation endpoint is skeleton adjacency because PC, PC-FDR, and PC-p do not necessarily return a uniquely oriented DAG.

Primary methods:

1. NOTEARS (existing project implementation)
2. NOTEARS + DAGGuard-Greedy
3. NOTEARS + DAGGuard-Exact
4. Wang et al. (2026) hybrid structural pipeline, transparently adapted to continuous variables by empirical-tertile discretization
5. Li & Wang (2009) PC-FDR
6. Strobl, Spirtes & Visweswaran (2019) PC-p
7. ordinary PC

`pcp_faithful.py` is a Python port audited against the authors' official MATLAB source (`ericstrobl/PCp`, master tree SHA `42a179d7305641dbe6f8926e46a67ba622c66524`). It is labeled as a port, not official author software.

The Wang et al. implementation reproduces structural Steps 1-3 with the published threshold setting A `(0.008, 0.005, 0.009)` and empirical-tertile discretization for continuous inputs. Step 4 only orients the resulting skeleton and was not used for the common skeleton benchmark because the publication does not fully specify a generic multilevel adaptation of that orientation rule. All four threshold settings tied as optimal in their application were checked as sensitivity analyses.

For the proprietary swine data, run only with an authorized local copy:

```bash
python -m benchmarks.seven_method.swine_benchmark \
  --data /authorized/path/train2023cw_simple.csv \
  --method pcfdr05
```

The script writes only non-row-level adjacency matrices and summaries. The expected SHA256 for the analysis file used in the manuscript is `b933fd66f49fd381bb9698ee2b3f5835d0db8d01820a01d5e45c9ac3a7bf5156`.

See `results/seven_method_benchmark/` for the audited primary summary tables included with the repository.
