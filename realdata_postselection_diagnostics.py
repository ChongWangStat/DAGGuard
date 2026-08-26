#!/usr/bin/env python3
"""Authorized-data diagnostics for the swine-production application.

No row-level observations are distributed. Given an authorized local copy of
``train2023cw_simple.csv``, this script reproduces the exact and greedy
fixed-candidate refinements and exports only non-row-level summaries.

Possible temporal or operational grouping fields are audited but never used
automatically. Effective-sample-size results are explicitly illustrative
cutoff sensitivities, not cluster-corrected likelihood or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler

from local_bic_refinement import (
    bic_partial_r2_cutoff,
    candidate_indegree_summary,
    deletion_diagnostics,
    edge_jaccard,
    exact_refine_dag,
    gaussian_local_bic,
    greedy_refine_dag,
    initial_pruning_pressure,
    total_gaussian_bic,
)
from reproduce_simulations import LAMBDA1, THRESHOLD, notears_linear, thresholded_candidate_dag


BINARY_VARS = [
    "PRRS_binary", "MYCO_binary", "LateralPRRS_binary", "Q2", "Q3", "Q4"
]
REAL_VARS = [
    "PRRS_binary", "MYCO_binary", "LateralPRRS_binary", "Q2", "Q3", "Q4",
    "Avg_parity_farrow", "Litters_female_year", "mated_inventory_20wks",
    "PWMFyear", "nonproductive_days", "number_services", "wean_to_service",
    "abortions_rate", "Total_born_avg", "Stillborn_avg", "Mummies_avg",
    "prenatal_losses_avg", "Born_alive_avg", "Gestation_days",
    "Interval_farrows", "Pre_weaning_mortality", "PWSow",
    "productive_days_rate", "services_per_inventory_N_rate", "repeats__rate",
    "gilts_bred_rate", "Last_week_wean_bred_rate", "pregnant_105days_rate",
    "Cull_rate_annual", "Sow_Death_rate", "avg_parity_at_farrow",
    "Lactation_days", "final_inventory", "Farrowing__rate", "HeadIn",
    "mortality_60days",
]


def adjacency(W: np.ndarray) -> np.ndarray:
    return thresholded_candidate_dag(W, THRESHOLD)[0]


def prep_with_missing(path: Path):
    raw = pd.read_csv(path, na_values=["NA", "NaN", "null", ""])
    x = raw.copy()
    x["PRRS_binary"] = x["PRRSatPlacement"].astype(str).str.lower().eq("epidemic").astype(int)
    x["MYCO_binary"] = x["Mycoplasma_Status"].astype(str).str.lower().eq("endemic").astype(int)
    x["LateralPRRS_binary"] = x["LateralPRRS"].astype(str).str.lower().eq("yes").astype(int)
    quarter = pd.to_numeric(x["Year_Quarter"], errors="coerce")
    for q in [2, 3, 4]:
        x[f"Q{q}"] = quarter.eq(q).astype(int)
    numeric = x[REAL_VARS].apply(pd.to_numeric, errors="coerce")
    return raw, numeric, numeric.dropna().copy()


def audit_grouping_fields(raw: pd.DataFrame, out: Path) -> None:
    """Inventory, but do not authorize, name-matched grouping/time fields."""
    keywords = ("farm", "site", "source", "lot", "batch", "date", "week", "month", "year")
    measurement_notes = {
        "Litters_female_year": "Annualized reproductive-performance measure, not an identifier.",
        "PWMFyear": "Annualized production measure, not an identifier or calendar year.",
        "Last_week_wean_bred_rate": "One-week performance rate, not a week index or date.",
        "SowFarmMed": "Medication category (CTC, Linco, or Mixed), not a sow-farm identifier.",
    }
    rows = []
    for column in raw.columns:
        if any(word in column.lower() for word in keywords):
            values = raw[column]
            if column == "Year_Quarter":
                note = (
                    "Four-level quarter-of-year seasonal covariate; it has no year, date, "
                    "farm, site, lot, or batch identity and does not define temporal blocks."
                )
            else:
                note = measurement_notes.get(
                    column,
                    "Name matched the audit search, but the field is not an authorized grouping identifier.",
                )
            rows.append({
                "column": column,
                "nonmissing_n": int(values.notna().sum()),
                "distinct_nonmissing": int(values.nunique(dropna=True)),
                "dtype": str(values.dtype),
                "authorized_for_grouping": False,
                "note": note,
            })
    pd.DataFrame(rows, columns=[
        "column", "nonmissing_n", "distinct_nonmissing", "dtype",
        "authorized_for_grouping", "note",
    ]).to_csv(out / "available_grouping_field_audit.csv", index=False)


def write_indegree_distribution(A0: np.ndarray, labels: list[str], out: Path) -> dict:
    summary = candidate_indegree_summary(A0)
    indegrees = summary["indegrees"]
    pd.DataFrame({
        "variable": labels,
        "candidate_indegree": indegrees,
        "local_enumeration_fits": [1 << int(q) for q in indegrees],
    }).sort_values(["candidate_indegree", "variable"], ascending=[False, True]).to_csv(
        out / "candidate_indegree_distribution.csv", index=False)
    pd.DataFrame([{
        key: value for key, value in summary.items() if key != "indegrees"
    }]).to_csv(out / "candidate_indegree_summary.csv", index=False)
    return summary


def write_local_comparison(exact, greedy, labels: list[str], out: Path) -> None:
    rows = []
    for er, gr in zip(exact.local_results, greedy.local_results):
        rows.append({
            "child": labels[er.child],
            "candidate_indegree": len(er.candidate_parents),
            "exact_indegree": len(er.selected_parents),
            "greedy_indegree": len(gr.selected_parents),
            "candidate_parents": "; ".join(labels[p] for p in er.candidate_parents),
            "exact_parents": "; ".join(labels[p] for p in er.selected_parents),
            "greedy_parents": "; ".join(labels[p] for p in gr.selected_parents),
            "exact_local_bic": er.score,
            "greedy_local_bic": gr.score,
            "greedy_local_bic_gap": gr.score - er.score,
            "edge_sets_equal": set(er.selected_parents) == set(gr.selected_parents),
            "exact_method": er.method,
            "exact_certified": er.globally_optimal,
            "exact_score_evaluations": er.score_evaluations,
            "exact_search_nodes": er.search_nodes,
        })
    pd.DataFrame(rows).to_csv(out / "exact_greedy_local_comparison.csv", index=False)


def write_graph_outputs(
    candidate: np.ndarray, exact: np.ndarray, greedy: np.ndarray,
    weights: np.ndarray, labels: list[str], out: Path,
) -> None:
    """Write labeled, non-row-level graph summaries for manuscript verification."""
    for name, matrix in [
        ("candidate_adjacency", candidate),
        ("exact_adjacency", exact),
        ("greedy_adjacency", greedy),
    ]:
        pd.DataFrame(matrix.astype(int), index=labels, columns=labels).to_csv(
            out / f"{name}.csv", index_label="from"
        )
    rows = []
    for parent, child in np.argwhere(candidate == 1):
        rows.append({
            "from": labels[int(parent)], "to": labels[int(child)],
            "candidate_weight": float(weights[parent, child]),
            "exact_retained": bool(exact[parent, child]),
            "greedy_retained": bool(greedy[parent, child]),
        })
    pd.DataFrame(rows).sort_values(["to", "from"]).to_csv(
        out / "candidate_edge_refinement_status.csv", index=False
    )


def write_initial_pressure(
    X: np.ndarray, A0: np.ndarray, exact, greedy, labels: list[str], analysis: str, out: Path
) -> dict:
    summary, rows = initial_pruning_pressure(X, A0)
    edge_rows = []
    for row in rows:
        edge_rows.append({
            "analysis": analysis,
            "from": labels[int(row["parent"])],
            "to": labels[int(row["child"])],
            **{k: v for k, v in row.items() if k not in {"parent", "child"}},
        })
    pd.DataFrame(edge_rows).to_csv(out / f"{analysis}_initial_pruning_pressure_edges.csv", index=False)
    candidate_bic = total_gaussian_bic(X, A0)
    summary.update({
        "analysis": analysis,
        "n": X.shape[0],
        "actual_exact_deletion_fraction": (A0.sum() - exact.adjacency.sum()) / A0.sum(),
        "candidate_to_exact_bic_improvement": candidate_bic - exact.total_bic,
        "greedy_bic_gap": greedy.total_bic - exact.total_bic,
        "exact_greedy_disagreement_edges": int(np.logical_xor(exact.adjacency, greedy.adjacency).sum()),
        "exact_greedy_jaccard": edge_jaccard(exact.adjacency, greedy.adjacency),
    })
    return summary


def mortality_diagnostics(
    X: np.ndarray, A0: np.ndarray, exact_A: np.ndarray, greedy_A: np.ndarray,
    W: np.ndarray, labels: list[str], out: Path
) -> pd.DataFrame:
    mortality = labels.index("mortality_60days")
    rows = []
    for u, v in np.argwhere(A0 == 1):
        if u != mortality and v != mortality:
            continue
        selected = tuple(int(p) for p in np.flatnonzero(exact_A[:, v]))
        if exact_A[u, v]:
            diag = deletion_diagnostics(X, int(v), selected, int(u))
            comparison = "delete from exact model"
            delta = diag["delta_bic"]
        else:
            augmented = tuple(sorted(selected + (int(u),)))
            diag = deletion_diagnostics(X, int(v), augmented, int(u))
            comparison = "add to exact model"
            delta = diag["bic_full"] - diag["bic_reduced"]
        rows.append({
            "from": labels[u], "to": labels[v], "candidate_weight": W[u, v],
            "exact_status": "retained" if exact_A[u, v] else "removed",
            "greedy_status": "retained" if greedy_A[u, v] else "removed",
            "comparison": comparison, "partial_r2": diag["partial_r2"],
            "bic_cost_of_changing_exact_decision": delta,
            "bic_partial_r2_cutoff": bic_partial_r2_cutoff(X.shape[0]),
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "mortality_edge_diagnostics.csv", index=False)
    return result


def effective_n_sensitivity(mortality: pd.DataFrame, n: int, out: Path) -> None:
    rows = []
    for divisor in [1, 2, 3, 5, 10]:
        effective_n = max(2.0, n / divisor)
        cutoff = bic_partial_r2_cutoff(effective_n)
        for row in mortality.itertuples(index=False):
            rows.append({
                "effective_n_divisor": divisor,
                "illustrative_effective_n": effective_n,
                "from": row[0], "to": row[1],
                "partial_r2_from_full_sample_fit": row.partial_r2,
                "illustrative_cutoff": cutoff,
                "would_retain_by_cutoff": bool(row.partial_r2 >= cutoff),
                "interpretation": "Sensitivity only; not cluster-corrected inference.",
            })
    pd.DataFrame(rows).to_csv(out / "mortality_effective_sample_size_sensitivity.csv", index=False)


def local_ebic(X: np.ndarray, child: int, parents: list[int], gamma: float) -> float:
    score = gaussian_local_bic(X, child, parents)
    if gamma <= 0:
        return score
    p = X.shape[1] - 1
    q = len(parents)
    logchoose = gammaln(p + 1) - gammaln(q + 1) - gammaln(p - q + 1)
    return float(score + 2 * gamma * logchoose)


def greedy_ebic_sensitivity(X: np.ndarray, A0: np.ndarray, gamma: float) -> np.ndarray:
    """Explicitly segregated EBIC sensitivity; not part of the primary method."""
    A = A0.copy().astype(int)
    for child in range(A.shape[0]):
        current = tuple(int(p) for p in np.flatnonzero(A[:, child]))
        score = local_ebic(X, child, list(current), gamma)
        while current:
            candidates = []
            for parent in current:
                reduced = tuple(p for p in current if p != parent)
                candidates.append((local_ebic(X, child, list(reduced), gamma), reduced))
            new_score, reduced = min(candidates, key=lambda z: (z[0], len(z[1]), z[1]))
            if new_score >= score - 1e-10:
                break
            current, score = reduced, new_score
        A[:, child] = 0
        A[list(current), child] = 1
    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Authorized local train2023cw_simple.csv")
    parser.add_argument("--out", type=Path, default=Path("results/realdata_diagnostics"))
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Optional row-bootstrap replicates; zero skips this check")
    parser.add_argument(
        "--primary-only", action="store_true",
        help="Run the primary candidate/exact/greedy analysis and skip secondary full-pipeline sensitivities",
    )
    parser.add_argument("--enumeration-max-parents", type=int, default=15)
    parser.add_argument("--branch-node-limit", type=int, default=2_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw, numeric, df = prep_with_missing(args.data)
    numeric.isna().sum().sort_values(ascending=False).to_csv(
        args.out / "missingness_by_variable.csv", header=["missing_n"])
    audit_grouping_fields(raw, args.out)

    X = df[REAL_VARS].to_numpy(float)
    labels = list(REAL_VARS)
    W = notears_linear(X, lambda1=LAMBDA1)
    A0 = adjacency(W)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    pd.DataFrame(A0.astype(int), index=labels, columns=labels).to_csv(
        args.out / "candidate_adjacency.csv"
    )
    pd.DataFrame(W, index=labels, columns=labels).to_csv(
        args.out / "candidate_weight_matrix.csv"
    )
    package_versions = {}
    for package in [
        "numpy", "scipy", "pandas", "scikit-learn", "networkx",
        "matplotlib", "joblib", "causal-learn", "python-igraph",
    ]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not installed"
    (args.out / "application_run_config.json").write_text(
        json.dumps({
            "python": platform.python_version(),
            "packages": package_versions,
            "source_file": args.data.name,
            "source_sha256": data_hash,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "lambda1": LAMBDA1,
            "candidate_weight_threshold": THRESHOLD,
            "enumeration_max_parents": args.enumeration_max_parents,
            "branch_node_limit": args.branch_node_limit,
            "candidate_edges": int(A0.sum()),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exact = exact_refine_dag(
        X, A0,
        enumeration_max_parents=args.enumeration_max_parents,
        branch_node_limit=args.branch_node_limit,
    )
    greedy = greedy_refine_dag(X, A0)
    write_graph_outputs(A0, exact.adjacency, greedy.adjacency, W, labels, args.out)
    write_indegree_distribution(A0, labels, args.out)
    write_local_comparison(exact, greedy, labels, args.out)

    pressure_rows = [write_initial_pressure(
        X, A0, exact, greedy, labels, "original_scale", args.out
    )]

    pd.DataFrame([{
        "analysis": "original_scale", "n": X.shape[0], "d": X.shape[1],
        "candidate_edges": int(A0.sum()), "exact_edges": int(exact.adjacency.sum()),
        "greedy_edges": int(greedy.adjacency.sum()),
        "candidate_bic": total_gaussian_bic(X, A0), "exact_bic": exact.total_bic,
        "greedy_bic": greedy.total_bic,
        "greedy_bic_gap": greedy.total_bic - exact.total_bic,
        "exact_greedy_jaccard": edge_jaccard(exact.adjacency, greedy.adjacency),
        "exact_method": exact.method, "exact_certified": exact.globally_optimal,
        "exact_seconds": exact.runtime_seconds, "greedy_seconds": greedy.runtime_seconds,
        "exact_score_evaluations": exact.score_evaluations,
        "greedy_score_evaluations": greedy.score_evaluations,
    }]).to_csv(args.out / "exact_greedy_global_comparison.csv", index=False)
    pd.DataFrame([{
        "source_file": args.data.name,
        "sha256": data_hash,
        "source_rows": len(raw),
        "source_columns": len(raw.columns),
        "complete_case_rows": len(df),
        "analysis_variables": len(labels),
        "excluded_incomplete_rows": len(raw) - len(df),
    }]).to_csv(args.out / "data_provenance_summary.csv", index=False)

    indeg = [{"variable": b, "initial_notears_indegree": int(A0[:, labels.index(b)].sum())}
             for b in BINARY_VARS]
    pd.DataFrame(indeg).to_csv(args.out / "binary_node_indegree_diagnostic.csv", index=False)
    qedges = [{"from": a, "to": b} for a in ["Q2", "Q3", "Q4"]
              for b in ["Q2", "Q3", "Q4"]
              if a != b and A0[labels.index(a), labels.index(b)]]
    pd.DataFrame(qedges, columns=["from", "to"]).to_csv(
        args.out / "quarter_within_block_edges.csv", index=False)

    mortality = mortality_diagnostics(
        X, A0, exact.adjacency, greedy.adjacency, W, labels, args.out
    )
    effective_n_sensitivity(mortality, X.shape[0], args.out)

    if args.primary_only:
        pd.DataFrame(pressure_rows).to_csv(
            args.out / "initial_pruning_pressure_validation.csv", index=False
        )
        print(f"complete cases: {len(df)}/{len(raw)}")
        print(
            f"candidate edges: {int(A0.sum())}; exact: {int(exact.adjacency.sum())}; "
            f"greedy: {int(greedy.adjacency.sum())}; certified={exact.globally_optimal}"
        )
        return

    Xs = StandardScaler().fit_transform(X)
    A0s = adjacency(notears_linear(Xs, lambda1=LAMBDA1))
    exact_s = exact_refine_dag(Xs, A0s)
    greedy_s = greedy_refine_dag(Xs, A0s)
    pressure_rows.append(write_initial_pressure(
        Xs, A0s, exact_s, greedy_s, labels, "standardized", args.out
    ))
    pd.DataFrame(pressure_rows).to_csv(
        args.out / "initial_pruning_pressure_validation.csv", index=False
    )

    m = labels.index("mortality_60days")
    ebic_rows = []
    for gamma in [0.5, 1.0]:
        A = greedy_ebic_sensitivity(X, A0, gamma)
        ebic_rows.append({
            "gamma": gamma, "search": "greedy sensitivity only", "edges": int(A.sum()),
            "mortality_in": int(A[:, m].sum()), "mortality_out": int(A[m, :].sum()),
        })
    pd.DataFrame(ebic_rows).to_csv(args.out / "ebic_greedy_sensitivity.csv", index=False)

    Xscaled = X.copy()
    for variable in ["HeadIn", "final_inventory"]:
        Xscaled[:, labels.index(variable)] /= 1000.0
    A0scaled = adjacency(notears_linear(Xscaled, lambda1=LAMBDA1))
    exact_scaled = exact_refine_dag(Xscaled, A0scaled)
    pd.DataFrame([
        {"metric": "candidate_edges_original", "value": int(A0.sum())},
        {"metric": "candidate_edges_scaled", "value": int(A0scaled.sum())},
        {"metric": "candidate_jaccard", "value": edge_jaccard(A0, A0scaled)},
        {"metric": "exact_edges_original", "value": int(exact.adjacency.sum())},
        {"metric": "exact_edges_scaled", "value": int(exact_scaled.adjacency.sum())},
        {"metric": "exact_jaccard", "value": edge_jaccard(exact.adjacency, exact_scaled.adjacency)},
    ]).to_csv(args.out / "full_pipeline_unit_change_sensitivity.csv", index=False)

    if args.bootstrap > 0:
        rng = np.random.default_rng(12123)
        candidate_frequency = np.zeros_like(A0, float)
        exact_frequency = np.zeros_like(A0, float)
        for b in range(args.bootstrap):
            row_indices = rng.integers(0, X.shape[0], X.shape[0])
            Wb = notears_linear(X[row_indices], lambda1=LAMBDA1)
            A0b = adjacency(Wb)
            exact_b = exact_refine_dag(X[row_indices], A0b)
            candidate_frequency += A0b
            exact_frequency += exact_b.adjacency
            print(f"bootstrap {b + 1}/{args.bootstrap}")
        candidate_frequency /= args.bootstrap
        exact_frequency /= args.bootstrap
        rows = []
        for name, fitted, frequency in [
            ("NOTEARS candidate", A0, candidate_frequency),
            ("Exact local-BIC", exact.adjacency, exact_frequency),
        ]:
            for threshold in [0.5, 0.7]:
                recurrent = int(((frequency >= threshold) & (fitted == 1)).sum())
                rows.append({
                    "method": name, "threshold": threshold,
                    "fitted_edges": int(fitted.sum()), "fitted_edges_recurrent": recurrent,
                    "fraction": recurrent / fitted.sum(),
                    "all_edges_above_threshold": int((frequency >= threshold).sum()),
                    "resampling_unit": "row",
                    "limitation": "Does not address within-system clustering.",
                })
        pd.DataFrame(rows).to_csv(args.out / "row_bootstrap_stability_comparison.csv", index=False)

    print(f"complete cases: {len(df)}/{len(raw)}")
    print(
        f"candidate edges: {int(A0.sum())}; exact: {int(exact.adjacency.sum())}; "
        f"greedy: {int(greedy.adjacency.sum())}; certified={exact.globally_optimal}"
    )


if __name__ == "__main__":
    main()
