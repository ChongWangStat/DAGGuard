# External-method source audit

This benchmark distinguishes direct source ports from published-parameter adaptations. The purpose of this file is to make the external-method comparisons auditable without implying that independently implemented or adapted methods are official author software.

## NOTEARS

The project linear NOTEARS implementation was checked against the public `xunzheng/notears` linear least-squares implementation. The objective, positive/negative variable split for the L1 penalty, matrix-exponential acyclicity function and gradient, augmented-Lagrangian updates, zero-diagonal bounds, `h_tol=1e-8`, and `rho_max=1e16` agree in substance. The reported primary settings `lambda1=0.1` and coefficient threshold `0.3` match the standard public example. The project adds a deterministic safeguard that drops threshold-passing edges only if a residual numerical cycle would otherwise remain; this has no effect when the thresholded solution is already acyclic.

## Li and Wang (2009) PC-FDR

`pc_fdr.py` is an independent implementation of the paper's Algorithm 3 for the skeleton. The source audit checked the following features against Algorithm 3 and its Gaussian appendix: ordered-pair testing with current neighborhoods, maximum conditional-independence p-value (`pmax`) updates, repeated FDR application after valid p-values have been accumulated, theorem-version retention of `pmax` values for removed edges, and Fisher-z testing with the `sqrt(n-|C|-3)` factor. The primary benchmark uses the paper's practical `H*=H` step-up choice at `q=0.05`; `q=0.10` and a Benjamini-Yekutieli sensitivity are reported separately. Finite-sample FDP in these simulations is treated as an empirical operating characteristic, not as a test of the paper's asymptotic guarantee.

## Strobl, Spirtes, and Visweswaran (2019) PC-p

`pcp_faithful.py` is a source-audited Python port, not official author software. The audit used the authors' public MATLAB repository `ericstrobl/PCp`, master tree SHA `42a179d7305641dbe6f8926e46a67ba622c66524`, and checked the workflow corresponding to `PC_with_pval.m`, `get_skeleton_stable.m`, `get_v_structures2.m`, `clamp_edges.m`, `orientation_rules.m`, `control_FDR.m`, `binary_search.m`, `get_BY_FDR.m`, and `gaussCItest.m`. The port preserves the official adaptive initial PC threshold, PC-stable neighborhood snapshots, p-value propagation, and Benjamini-Yekutieli FDR step.

One rare source-code ambiguity is documented rather than hidden: the MATLAB conflict-handling code contains expressions that can be read literally as linear indexing although the surrounding logic indicates edge-coordinate indexing. The primary Python port uses the coordinate interpretation and exposes `literal_conflict_indexing=True` for audit purposes. No orientation conflicts occurred in any of the 240 primary simulation runs, so this ambiguity cannot affect the primary simulation table.

Earlier exploratory PC-p results from an incomplete port were discarded and are not part of the manuscript or archived primary results.

## Wang et al. (2026) hybrid structural pipeline

No official public implementation was located. The benchmark therefore labels this method as a transparent published-parameter adaptation rather than an exact reproduction. `wang_full.py`/`wang_sensitivity.py` implement the published structural Steps 1-3: mutual-information skeleton screening, conditional-mutual-information collider identification and pruning, and local discrete-BIC parent pruning. The primary thresholds are the paper's setting A `(0.008, 0.005, 0.009)`; the other three settings tied for best BIC in the source application are included as sensitivity analyses.

The source application is discrete and includes domain-specific root-node restrictions. For the generic continuous simulation benchmark, variables are discretized by empirical tertiles and no source-application root labels are transferred. The paper does not fully formalize the phrase describing exclusion of collider-related nodes in the Step-3 conditioning set, so the implementation uses a conservative documented interpretation. Step 4 is not benchmarked because it only orients the retained skeleton and the common endpoint is skeleton adjacency; its state-wise orientation rule is also not sufficiently specified for a generic continuous-variable adaptation. These choices are limitations of comparability, not claimed features of the authors' original implementation.

## Ordinary PC

The ordinary-PC baseline uses an original-style ordered-pair skeleton search with immediate graph updates and two-sided Gaussian Fisher-z tests at `alpha=0.05`. It is presented as a conventional PC skeleton baseline, not as a reproduction of a particular software package.

## Common endpoint and data checks

PC-family methods and the adapted Wang procedure need not produce the same type of fully oriented DAG as NOTEARS/DAGGuard. The primary seven-method comparison therefore uses skeleton adjacency for all methods. Simulation comparisons use the same 240 `(d, s, noise, rep, seed)` keys. On the authorized commercial data, the comparator methods were rerun from the pinned input bytes and their adjacency matrices matched the archived results entry-for-entry.
