# Shareable derived results

These files contain non-row-level summaries derived from the analyses reported in the manuscript. The proprietary swine-production observations are not included.

- `additional_simulation_summary.csv`: overall means for NOTEARS, NOTEARS-BP, standardized NOTEARS, and HC+BIC in the additional Gaussian/non-Gaussian validation experiment.
- `pruning_diagnostics_summary.csv`: average initial/final edge counts and false-positive/true-positive removals in that experiment.
- `realdata_sensitivity_summary.csv`: original-scale, standardized, and continuous-only real-data sensitivity results.
- `realdata_runtime_summary.csv`: runtime, BIC-evaluation, and memory summary for the primary real-data analysis.
- `realdata_stability_summary.csv`: numbers of BP edges exceeding bootstrap selection-frequency thresholds 0.50 and 0.70.
- `binary_node_indegree_diagnostic.csv`: initial NOTEARS in-degrees for the six binary indicator nodes.
- `mortality_direction_summary.csv`: incoming/outgoing mortality edge counts before and after BP pruning.

The values are supplied to make the numerical statements in the article directly checkable without disclosing proprietary observations.
