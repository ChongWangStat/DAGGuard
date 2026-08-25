# Seven-method benchmark

This directory contains the comparator implementations used for the DAGGuard JDS benchmark. The common simulation endpoint is skeleton adjacency because PC, PC-FDR, PC-p, and the adapted Wang procedure do not necessarily return a uniquely oriented DAG comparable to NOTEARS/DAGGuard.

Primary methods:

1. NOTEARS (project implementation checked against the public `xunzheng/notears` linear source)
2. NOTEARS + DAGGuard-Greedy
3. NOTEARS + DAGGuard-Exact
4. Wang et al. (2026) hybrid structural pipeline, transparently adapted to continuous variables by empirical-tertile discretization
5. Li & Wang (2009) PC-FDR, independently implemented from published Algorithm 3
6. Strobl, Spirtes & Visweswaran (2019) PC-p, source-audited Python port of the authors' MATLAB code
7. ordinary PC

`SOURCE_AUDIT.md` records the source-by-source audit, including which components are direct source ports, which are independent implementations, and which are adaptations. This distinction is intentional: none of the independently implemented or adapted comparators is represented as official author software.

`pcp_faithful.py` was audited against the authors' official MATLAB repository (`ericstrobl/PCp`, master tree SHA `42a179d7305641dbe6f8926e46a67ba622c66524`). The port preserves the official adaptive PC threshold, PC-stable neighborhood snapshots, p-value propagation, and BY FDR procedure. A rare conflict-indexing ambiguity in the MATLAB source is documented in `SOURCE_AUDIT.md`; no orientation conflicts occurred in the 240 primary simulation runs, so it cannot affect the primary benchmark table.

The Wang et al. implementation reproduces the published structural Steps 1-3 as a generic continuous-data adaptation using empirical-tertile discretization. The primary threshold setting is A `(0.008, 0.005, 0.009)` and all four source-paper settings tied as optimal in their application are checked as sensitivities. Source-application root-node labels are not transferred to the generic simulation benchmark. Step 4 only orients the retained skeleton and was not used for the common skeleton endpoint because the publication does not fully specify a generic multilevel adaptation of that orientation rule.

For the proprietary swine data, run only with an authorized local copy:

```bash
python -m benchmarks.seven_method.swine_benchmark \
  --data /authorized/path/train2023cw_simple.csv \
  --method pcfdr05
```

The script writes only non-row-level adjacency matrices and summaries. The expected SHA256 for the analysis file used in the manuscript is `b933fd66f49fd381bb9698ee2b3f5835d0db8d01820a01d5e45c9ac3a7bf5156`.

See `SOURCE_AUDIT.md` and `results/seven_method_benchmark/` for the audit record and primary summary tables.
