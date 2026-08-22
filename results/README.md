# Shareable derived results

These files contain non-row-level summaries derived from the analyses reported in the manuscript and supplement. The proprietary swine-production observations are not included.

The directory is organized around the specific questions raised by the validation analyses:

- `equal_sparsity_ablation_summary.csv` — original diagnostic ablation holding graph size fixed under uniform weights. If BP retains K edges, the controls retain exactly K NOTEARS candidate edges by raw or unit-normalized coefficient magnitude. This is not a practical competing method; it isolates edge choice from the trivial effect of returning fewer edges.
- `equal_sparsity_modnormal_summary.csv` — second coefficient-regime version of the same equal-sparsity mechanism check (`d=10`, modified-normal `s=3`), included to verify that the finding is not specific to uniform weights.
- `sample_size_diagnostic.csv` — targeted `n = 100, 500, 2000` diagnostic showing the exact partial-R2 BIC cutoff and paired NOTEARS/BP performance. It exists because the method is described as sample-size adaptive and that property should be exercised empirically.
- `varsortability_primary_summary.csv` — path-based varsortability and marginal SD-ratio summaries for every primary simulation setting, included so that the unusually strong variance-ordering signal in the benchmark is explicit.
- `standardized_varsortability_diagnostic.csv` — targeted standardized-data experiment showing that BP does not rescue a poor standardized NOTEARS initialization.
- `additional_simulation_summary.csv` — overall means from the Gaussian/non-Gaussian validation experiment.
- `pruning_diagnostics_summary.csv` — average initial/final edge counts and false-positive/true-positive removals in that experiment.
- `mortality_partial_r2_diagnostics.csv` — edge-specific partial-R2 and Delta-BIC evidence for the eight edges incident to 60-day mortality in the initial application graph.
- `realdata_collinearity_audit.csv` — non-row-level correlation, condition-number, VIF, parity, compositional, and seasonal-block diagnostics.
- `redundancy_pruning_sensitivity.csv` — fixed-candidate sensitivity after removing four highly redundant representations; designed to isolate the pruning stage from a new NOTEARS optimization.
- `ebic_sensitivity.csv` — local EBIC penalty sensitivity at gamma 0, 0.5, and 1.
- `unit_change_sensitivity.csv` — full-pipeline sensitivity after expressing two inventory variables in thousands. The edge-set Jaccards show that similar edge counts do not imply unit invariance of the complete pipeline.
- `bootstrap_stability_comparison.csv` — recurrence of edges in the original NOTEARS and NOTEARS-BP fits across row-bootstrap re-estimation.
- `realdata_sensitivity_summary.csv` — original-scale, standardized, and continuous-only application results.
- `realdata_runtime_summary.csv` — runtime, BIC-evaluation, and memory summary for the primary application analysis.
- `binary_node_indegree_diagnostic.csv` — in-degrees of the six binary indicator nodes in the 190-edge **thresholded** NOTEARS candidate DAG.
- `mortality_direction_summary.csv` — incoming/outgoing mortality edge counts before and after BP pruning.

The values are supplied so that the numerical statements in the article can be checked without disclosing proprietary observations. The real-data bootstrap is a row bootstrap because the de-identified analysis matrix does not contain a reliable farm/site identifier for cluster resampling; it should not be interpreted as cluster-robust uncertainty quantification.
