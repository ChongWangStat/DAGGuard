#!/usr/bin/env python3
"""Build manuscript tables and vector figures from final analysis CSV files.

The script deliberately fails on count, score, or edge-set inconsistencies.  It
also writes a manifest containing source hashes and the numeric values used in
every plot so rendered manuscript assets can be audited against their inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np
import pandas as pd


SETTING_ORDER = [
    "clean_sparse", "fp_025", "fp_050", "fp_100", "missing_010",
    "reversal_010", "weak_fp", "dense_moderate", "dense_high_indegree",
    "lowvar_heterogeneous", "lowvar_weak_fp", "combined_contamination",
]
SETTING_LABELS = {
    "clean_sparse": "Clean sparse",
    "fp_025": "False positives 25\\%",
    "fp_050": "False positives 50\\%",
    "fp_100": "False positives 100\\%",
    "missing_010": "Missing 10\\%",
    "reversal_010": "Reversed 10\\%",
    "weak_fp": "Weak edges + false positives",
    "dense_moderate": "Dense, moderate indegree",
    "dense_high_indegree": "Dense, high indegree",
    "lowvar_heterogeneous": "Low-varsortability heterogeneous",
    "lowvar_weak_fp": "Low-varsortability weak + FP",
    "combined_contamination": "Combined contamination",
}
PLOT_LABELS = {
    key: value.replace("\\%", "%").replace("Low-varsortability", "Low-var.")
    for key, value in SETTING_LABELS.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
        ("&", r"\&"), ("#", r"\#"), ("$", r"\$"),
    ]:
        text = text.replace(old, new)
    return text


def load_contamination(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = results / "candidate_contamination"
    raw = pd.read_csv(folder / "contamination_replicate_metrics.csv")
    paired = pd.read_csv(folder / "exact_greedy_paired_summary.csv").set_index("setting")
    method = pd.read_csv(folder / "contamination_method_summary.csv")
    assert set(raw["setting"]) == set(SETTING_ORDER)
    counts = raw.groupby(["setting", "method"])["rep"].nunique()
    assert (counts == 100).all(), counts
    exact_rows = raw[raw.method.eq("Exact local BIC")]
    assert len(exact_rows) == 1200
    assert exact_rows["exact_bic"].le(exact_rows["greedy_bic"] + 1e-8).all()
    recalculated = exact_rows.groupby("setting")["greedy_suboptimal"].mean()
    np.testing.assert_allclose(
        recalculated.loc[paired.index], paired["greedy_suboptimal_frequency"], atol=1e-12
    )
    return raw, paired, method


def write_contamination_table(paired: pd.DataFrame, method: pd.DataFrame, target: Path) -> None:
    exact = method[method.method.eq("Exact local BIC")].set_index("setting")
    lines = []
    for setting in SETTING_ORDER:
        e, p = exact.loc[setting], paired.loc[setting]
        label = SETTING_LABELS[setting]
        frequency = 100 * p.greedy_suboptimal_frequency
        low, high = 100 * p.greedy_suboptimal_wilson_low, 100 * p.greedy_suboptimal_wilson_high
        gap = p.greedy_bic_gap_mean
        gap_low, gap_high = p.greedy_bic_gap_ci_low, p.greedy_bic_gap_ci_high
        lines.append(
            f"{label} & {e.fdr_mean:.3f} & {e.tpr_mean:.3f} & {e.shd_mean:.2f} & "
            f"{frequency:.0f} [{low:.1f}, {high:.1f}] & "
            f"{gap:.3f} [{gap_low:.3f}, {gap_high:.3f}]\\\\"
        )
    target.write_text("\n".join(lines) + "\n\\bottomrule%\n", encoding="utf-8")


def write_contamination_supplement(
    raw: pd.DataFrame, paired: pd.DataFrame, method: pd.DataFrame, target: Path
) -> None:
    exact = method[method.method.eq("Exact local BIC")].set_index("setting")
    greedy = method[method.method.eq("Greedy local BIC")].set_index("setting")
    candidate = method[method.method.eq("Candidate")].set_index("setting")
    diagnostics = raw[raw.method.eq("Exact local BIC")].groupby("setting").agg(
        candidate_edges=("candidate_edges", "mean"),
        candidate_qmax=("candidate_max_indegree_achieved", "mean"),
        enumeration_fits=("candidate_enumeration_fits", "mean"),
        error_scale_ratio=("error_scale_ratio", "mean"),
        exact_runtime=("exact_runtime_seconds", "mean"),
        greedy_runtime=("greedy_runtime_seconds", "mean"),
    )
    lines = []
    for setting in SETTING_ORDER:
        c, e, g, p, d = (
            candidate.loc[setting], exact.loc[setting], greedy.loc[setting],
            paired.loc[setting], diagnostics.loc[setting],
        )
        lines.append(
            f"{SETTING_LABELS[setting]} & {d.candidate_edges:.1f} & {d.candidate_qmax:.2f} & "
            f"{c.fdr_mean:.3f} & {e.fdr_mean:.3f} & {g.fdr_mean:.3f} & "
            f"{e.tpr_mean:.3f} & {g.tpr_mean:.3f} & {e.shd_mean:.2f} & {g.shd_mean:.2f} & "
            f"{p.exact_greedy_jaccard_mean:.4f}\\\\"
        )
    target.write_text("\n".join(lines) + "\n\\bottomrule%\n", encoding="utf-8")

    diagnostic_lines = []
    for setting in SETTING_ORDER:
        e, p, d = exact.loc[setting], paired.loc[setting], diagnostics.loc[setting]
        diagnostic_lines.append(
            f"{SETTING_LABELS[setting]} & {e.varsortability_mean:.3f} & "
            f"{d.error_scale_ratio:.2f} & {d.candidate_qmax:.2f} & "
            f"{d.enumeration_fits:.0f} & {d.exact_runtime:.4f} & {d.greedy_runtime:.4f} & "
            f"{100*p.greedy_suboptimal_frequency:.1f} & {p.greedy_bic_gap_max:.3f}\\\\"
        )
    (target.parent / "table_contamination_diagnostics_rows.tex").write_text(
        "\n".join(diagnostic_lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )


def plot_contamination(paired: pd.DataFrame, target: Path) -> dict[str, object]:
    plot = paired.loc[SETTING_ORDER].copy()
    labels = [PLOT_LABELS[s] for s in SETTING_ORDER]
    x = np.arange(len(plot))
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.7), sharex=True)
    frequency = 100 * plot.greedy_suboptimal_frequency.to_numpy(float)
    yerr_frequency = np.vstack([
        frequency - 100 * plot.greedy_suboptimal_wilson_low.to_numpy(float),
        100 * plot.greedy_suboptimal_wilson_high.to_numpy(float) - frequency,
    ])
    axes[0].errorbar(x, frequency, yerr=yerr_frequency, fmt="o", color="#2266aa",
                     ecolor="#7aa6d8", capsize=3)
    axes[0].set_ylabel("Greedy suboptimal (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.25)

    gap = plot.greedy_bic_gap_mean.to_numpy(float)
    yerr_gap = np.vstack([
        gap - plot.greedy_bic_gap_ci_low.to_numpy(float),
        plot.greedy_bic_gap_ci_high.to_numpy(float) - gap,
    ])
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].errorbar(x, gap, yerr=yerr_gap, fmt="o", color="#b4472d",
                     ecolor="#d79b8c", capsize=3)
    axes[1].set_ylabel("Greedy minus exact BIC")
    axes[1].set_xticks(x, labels, rotation=48, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return {
        "settings": SETTING_ORDER,
        "suboptimal_percent": frequency.tolist(),
        "mean_bic_gap": gap.tolist(),
    }


def write_additional_noise_table(results: Path, supplement_dir: Path) -> dict[str, object] | None:
    folder = results / "additional_noise_sensitivity"
    summary_path = folder / "additional_simulation_summary.csv"
    raw_path = folder / "additional_simulation_metrics.csv"
    config_path = folder / "additional_simulation_run_config.json"
    if not (summary_path.exists() and raw_path.exists() and config_path.exists()):
        return None
    summary = pd.read_csv(summary_path).set_index("method")
    raw = pd.read_csv(raw_path)
    assert np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()
    assert raw["estimated_edges"].between(0, raw["d"] * (raw["d"] - 1) / 2).all()
    expected = {
        "NOTEARS": "NOTEARS candidate",
        "LOCAL_BIC_EXACT": "Exact local BIC",
        "LOCAL_BIC_GREEDY": "Greedy local BIC",
        "STD_NOTEARS": "Standardized NOTEARS",
        "STD_LOCAL_BIC_EXACT": "Standardized exact local BIC",
    }
    assert set(expected).issubset(summary.index)
    setting_counts = raw.groupby(["method", "noise", "d", "s"])["rep"].nunique()
    assert setting_counts.nunique() == 1
    assert int(setting_counts.iloc[0]) == 20
    if "exact_certified" in raw:
        assert raw["exact_certified"].all()
    if "standardized_exact_certified" in raw:
        assert raw["standardized_exact_certified"].all()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(config["gumbel_scale"], np.sqrt(6.0) / np.pi)
    assert config["gumbel_variance"] == 1.0
    lines = []
    for method, label in expected.items():
        row = summary.loc[method]
        suboptimal = (
            f"{100 * row.greedy_suboptimal_frequency:.1f}"
            if method in {"LOCAL_BIC_EXACT", "LOCAL_BIC_GREEDY"} else "--"
        )
        lines.append(
            f"{label} & {row.fdr:.3f} & {row.tpr:.3f} & {row.shd:.2f} & "
            f"{row.edges:.2f} & {row.fp:.2f} & {suboptimal}\\\\"
        )
    (supplement_dir / "table_additional_noise_rows.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )
    return {"replicates_per_setting": int(setting_counts.iloc[0]),
            "gumbel_scale": config["gumbel_scale"],
            "summary_rows": len(summary)}


def read_adjacency(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, index_col=0)
    assert list(frame.index) == list(frame.columns)
    return list(frame.columns), frame.to_numpy(int)


def load_swine(results: Path) -> dict[str, object] | None:
    folder = results / "swine_application"
    required = [
        folder / "candidate_adjacency.csv", folder / "exact_adjacency.csv",
        folder / "greedy_adjacency.csv", folder / "candidate_edge_refinement_status.csv",
        folder / "exact_greedy_global_comparison.csv",
        folder / "exact_greedy_local_comparison.csv", folder / "mortality_edge_diagnostics.csv",
    ]
    if not all(path.exists() for path in required):
        return None
    labels, candidate = read_adjacency(required[0])
    exact_labels, exact = read_adjacency(required[1])
    greedy_labels, greedy = read_adjacency(required[2])
    assert labels == exact_labels == greedy_labels
    global_row = pd.read_csv(required[4]).iloc[0]
    assert str(global_row.exact_certified).lower() == "true", (
        "The manuscript labels the application result exact, but the saved "
        "search was not certified."
    )
    assert float(global_row.exact_bic) <= float(global_row.greedy_bic) + 1e-8
    assert int(candidate.sum()) == int(global_row.candidate_edges)
    assert int(exact.sum()) == int(global_row.exact_edges)
    assert int(greedy.sum()) == int(global_row.greedy_edges)
    assert np.all(exact <= candidate) and np.all(greedy <= candidate)
    edges = pd.read_csv(required[3])
    assert len(edges) == int(candidate.sum())
    assert int(edges.exact_retained.sum()) == int(exact.sum())
    assert int(edges.greedy_retained.sum()) == int(greedy.sum())
    local = pd.read_csv(required[5])
    assert len(local) == len(labels)
    np.testing.assert_allclose(
        local.greedy_local_bic_gap.sum(), global_row.greedy_bic_gap, atol=1e-6
    )
    return {
        "folder": folder, "labels": labels, "candidate": candidate,
        "exact": exact, "greedy": greedy, "edges": edges,
        "global": global_row, "local": local,
        "mortality": pd.read_csv(required[6]),
    }


def plot_swine_adjacency(swine: dict[str, object], target: Path) -> dict[str, object]:
    labels = swine["labels"]
    edges = swine["edges"]
    signs = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: i for i, label in enumerate(labels)}
    for row in edges.itertuples(index=False):
        signs[index[row[0]], index[row[1]]] = 1 if row.candidate_weight > 0 else -1
    matrices = [swine["candidate"], swine["exact"], swine["greedy"]]
    titles = ["Candidate", "Exact local BIC", "Greedy local BIC"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), constrained_layout=True)
    cmap = ListedColormap(["#2b6cb0", "#ffffff", "#c53030"])
    for ax, adjacency, title in zip(axes, matrices, titles):
        signed = signs * adjacency
        # Draw cell polygons rather than embedding a bitmap so the full
        # adjacency display remains a true vector graphic in the PDF.
        ax.pcolormesh(
            np.arange(signed.shape[1] + 1),
            np.arange(signed.shape[0] + 1),
            signed,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            shading="flat",
            edgecolors="none",
            antialiased=False,
        )
        ax.set_xlim(0, signed.shape[1])
        ax.set_ylim(signed.shape[0], 0)
        ax.set_title(f"{title} ({int(adjacency.sum())} edges)", fontsize=9)
        ax.set_xlabel("Child")
        ax.set_ylabel("Parent")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return {title: int(matrix.sum()) for title, matrix in zip(titles, matrices)}


def plot_mortality(swine: dict[str, object], target: Path) -> dict[str, object]:
    mortality = swine["mortality"].copy()
    exact = swine["exact"]
    labels = swine["labels"]
    mortality_name = "mortality_60days"
    m = labels.index(mortality_name)
    incoming = int(exact[:, m].sum())
    outgoing = int(exact[m, :].sum())
    graph = nx.DiGraph()
    graph.add_node(mortality_name)
    for row in mortality.itertuples(index=False):
        graph.add_edge(row[0], row[1], retained=row.exact_status == "retained",
                       partial_r2=float(row.partial_r2))
    others = sorted(set(graph.nodes) - {mortality_name})
    left = [node for node in others if graph.has_edge(node, mortality_name)]
    right = [node for node in others if graph.has_edge(mortality_name, node)]
    positions = {mortality_name: (0, 0)}
    for nodes, x in [(left, -1.4), (right, 1.4)]:
        ys = np.linspace(1.2, -1.2, max(len(nodes), 1))
        positions.update({node: (x, y) for node, y in zip(nodes, ys)})
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    retained_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["retained"]]
    removed_edges = [(u, v) for u, v, data in graph.edges(data=True) if not data["retained"]]
    nx.draw_networkx_nodes(graph, positions, node_size=1450, node_color="#e8eef6", ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[mortality_name], node_size=1850,
                           node_color="#f6d7cc", ax=ax)
    nx.draw_networkx_labels(graph, positions,
                           labels={n: n.replace("_", "\n") for n in graph.nodes},
                           font_size=7, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=retained_edges, width=1.8,
                           edge_color="#176b3a", arrows=True, arrowsize=14, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=removed_edges, width=1.3,
                           edge_color="#9a9a9a", style="dashed", arrows=True,
                           arrowsize=14, ax=ax)
    edge_labels = {
        (u, v): f"$R_p^2$={data['partial_r2']:.4f}"
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels,
                                 font_size=6, rotate=False, ax=ax)
    ax.text(0.01, 0.01, "Solid green: retained by exact refinement; dashed gray: removed",
            transform=ax.transAxes, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return {"candidate_incident_edges": int(len(mortality)),
            "exact_incoming": incoming, "exact_outgoing": outgoing}


def write_swine_tables(swine: dict[str, object], main_dir: Path, supplement_dir: Path) -> None:
    global_row = swine["global"]
    indegree = pd.read_csv(swine["folder"] / "candidate_indegree_distribution.csv")
    pressure = pd.read_csv(
        swine["folder"] / "initial_pruning_pressure_validation.csv"
    )
    pressure = pressure[pressure.analysis.eq("original_scale")].iloc[0]
    labels = swine["labels"]
    mortality_index = labels.index("mortality_60days")
    candidate = swine["candidate"]
    exact = swine["exact"]
    greedy = swine["greedy"]
    effective = pd.read_csv(
        swine["folder"] / "mortality_effective_sample_size_sensitivity.csv"
    )

    def tex_integer(value: int) -> str:
        return f"{int(value):,}".replace(",", "{,}")

    macros = {
        "SwineCandidateEdges": tex_integer(global_row.candidate_edges),
        "SwineExactEdges": tex_integer(global_row.exact_edges),
        "SwineGreedyEdges": tex_integer(global_row.greedy_edges),
        "SwineMaximumIndegree": tex_integer(indegree.candidate_indegree.max()),
        "SwineEnumerationBurden": tex_integer(indegree.local_enumeration_fits.sum()),
        "SwineExactGreedyBICGap": f"{float(global_row.greedy_bic_gap):.2f}",
        "SwineExactGreedyJaccard": f"{float(global_row.exact_greedy_jaccard):.3f}",
        "SwineExactRuntimeMinutes": f"{float(global_row.exact_seconds) / 60:.1f}",
        "SwineGreedyRuntimeSeconds": f"{float(global_row.greedy_seconds):.2f}",
        "SwineExactEvaluations": tex_integer(global_row.exact_score_evaluations),
        "SwineGreedyEvaluations": tex_integer(global_row.greedy_score_evaluations),
        "SwineCandidateToExactBICImprovement": (
            f"{float(global_row.candidate_bic - global_row.exact_bic):.2f}"
        ),
        "SwinePressureEdges": tex_integer(pressure.edges_below_cutoff),
        "SwinePressurePercent": f"{100 * float(pressure.initial_pruning_pressure):.1f}",
        "SwineExactDeletionPercent": (
            f"{100 * float(pressure.actual_exact_deletion_fraction):.1f}"
        ),
        "SwinePartialRSquaredCutoff": f"{float(pressure.partial_r2_cutoff):.6f}",
        "SwineDisagreementEdges": tex_integer(
            pressure.exact_greedy_disagreement_edges
        ),
        "SwineCandidateMortalityIncoming": tex_integer(candidate[:, mortality_index].sum()),
        "SwineCandidateMortalityOutgoing": tex_integer(candidate[mortality_index, :].sum()),
        "SwineExactMortalityIncoming": tex_integer(exact[:, mortality_index].sum()),
        "SwineExactMortalityOutgoing": tex_integer(exact[mortality_index, :].sum()),
        "SwineGreedyMortalityIncoming": tex_integer(greedy[:, mortality_index].sum()),
        "SwineGreedyMortalityOutgoing": tex_integer(greedy[mortality_index, :].sum()),
        "SwineSuboptimalChildren": tex_integer(
            (swine["local"].greedy_local_bic_gap > 1e-8).sum()
        ),
        "SwineEffectiveFiveRetained": tex_integer(
            effective.loc[
                effective.effective_n_divisor.eq(5), "would_retain_by_cutoff"
            ].sum()
        ),
        "SwineEffectiveTenRetained": tex_integer(
            effective.loc[
                effective.effective_n_divisor.eq(10), "would_retain_by_cutoff"
            ].sum()
        ),
    }
    (main_dir / "swine_result_macros.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
        ) + "\n",
        encoding="utf-8",
    )

    (main_dir / "table_swine_global_rows.tex").write_text(
        f"Candidate & {int(global_row.candidate_edges)} & "
        f"{float(global_row.candidate_bic):.3f} & -- & -- & --\\\\\n"
        f"Exact local BIC & {int(global_row.exact_edges)} & "
        f"{float(global_row.exact_bic):.3f} & 1.0000 & "
        f"{float(global_row.exact_seconds):.3f} & "
        f"{int(global_row.exact_score_evaluations)}\\\\\n"
        f"Greedy local BIC & {int(global_row.greedy_edges)} & "
        f"{float(global_row.greedy_bic):.3f} & "
        f"{float(global_row.exact_greedy_jaccard):.4f} & "
        f"{float(global_row.greedy_seconds):.3f} & "
        f"{int(global_row.greedy_score_evaluations)}\\\\\n"
        "\\bottomrule%\n", encoding="utf-8"
    )
    indegree_lines = [
        f"{latex_escape(row.variable)} & {int(row.candidate_indegree)} & "
        f"{int(row.local_enumeration_fits):,}\\\\"
        for row in indegree.itertuples(index=False)
    ]
    (supplement_dir / "table_swine_indegree_rows.tex").write_text(
        "\n".join(indegree_lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )
    local_lines = []
    for row in swine["local"].itertuples(index=False):
        method = "B\\&B" if row.exact_method == "branch-and-bound" else "Enumeration"
        local_lines.append(
            f"{latex_escape(row.child)} & {int(row.candidate_indegree)} & "
            f"{int(row.exact_indegree)} & {int(row.greedy_indegree)} & "
            f"{float(row.greedy_local_bic_gap):.4f} & {method}\\\\"
        )
    (supplement_dir / "table_swine_local_rows.tex").write_text(
        "\n".join(local_lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )
    mortality_lines = [
        f"{latex_escape(row[0])} & {latex_escape(row[1])} & {latex_escape(row.exact_status)} & "
        f"{latex_escape(row.greedy_status)} & {float(row.partial_r2):.6f} & "
        f"{float(row.bic_cost_of_changing_exact_decision):.3f}\\\\"
        for row in swine["mortality"].itertuples(index=False)
    ]
    (supplement_dir / "table_mortality_edges_rows.tex").write_text(
        "\n".join(mortality_lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )

    effective_lines = []
    for divisor, group in effective.groupby("effective_n_divisor", sort=True):
        assert group.illustrative_cutoff.nunique() == 1
        assert group.illustrative_effective_n.nunique() == 1
        effective_lines.append(
            f"{int(divisor)} & {group.illustrative_effective_n.iloc[0]:.1f} & "
            f"{group.illustrative_cutoff.iloc[0]:.6f} & "
            f"{int(group.would_retain_by_cutoff.sum())} & {len(group)}\\\\"
        )
    (supplement_dir / "table_mortality_effective_n_rows.tex").write_text(
        "\n".join(effective_lines) + "\n\\bottomrule%\n", encoding="utf-8"
    )

    sensitivity_lines = [
        f"Original-scale exact local BIC & {int(swine['candidate'].sum())} & "
        f"{int(exact.sum())} & 1.000 & {int(exact[:, mortality_index].sum())}/"
        f"{int(exact[mortality_index, :].sum())}\\\\"
    ]
    optional_rows = 0
    pressure_path = swine["folder"] / "initial_pruning_pressure_validation.csv"
    if pressure_path.exists():
        pressure = pd.read_csv(pressure_path)
        standardized_rows = pressure[pressure.analysis.eq("standardized")]
        if len(standardized_rows):
            standardized = standardized_rows.iloc[0]
            standardized_selected = int(round(
                standardized.candidate_edges
                * (1 - standardized.actual_exact_deletion_fraction)
            ))
            sensitivity_lines.append(
                f"Standardized full pipeline, exact local BIC & "
                f"{int(standardized.candidate_edges)} & {standardized_selected} & -- & --\\\\"
            )
            optional_rows += 1
    redundancy_path = swine["folder"] / "redundancy_exact_refinement_sensitivity.csv"
    if redundancy_path.exists():
        redundancy = pd.read_csv(redundancy_path).iloc[0]
        sensitivity_lines.append(
            f"Redundancy-reduced fixed candidate, exact local BIC & "
            f"{int(redundancy.restricted_candidate_edges)} & {int(redundancy.exact_edges)} & "
            f"{redundancy.jaccard_vs_original_restricted:.3f} & "
            f"{int(redundancy.mortality_in)}/{int(redundancy.mortality_out)}\\\\"
        )
        optional_rows += 1
    ebic_path = swine["folder"] / "ebic_greedy_sensitivity.csv"
    if ebic_path.exists():
        for row in pd.read_csv(ebic_path).itertuples(index=False):
            sensitivity_lines.append(
                f"Segregated greedy EBIC, $\\gamma={row.gamma:g}$ & "
                f"{int(swine['candidate'].sum())} & {int(row.edges)} & -- & "
                f"{int(row.mortality_in)}/{int(row.mortality_out)}\\\\"
            )
            optional_rows += 1
    scaled_path = swine["folder"] / "full_pipeline_unit_change_sensitivity.csv"
    if scaled_path.exists():
        scaled_frame = pd.read_csv(scaled_path)
        scaled = dict(zip(scaled_frame.metric, scaled_frame.value))
        sensitivity_lines.append(
            f"Full pipeline after two unit changes, exact local BIC & "
            f"{int(scaled['candidate_edges_scaled'])} & "
            f"{int(scaled['exact_edges_scaled'])} & "
            f"{scaled['exact_jaccard']:.3f} & --\\\\"
        )
        optional_rows += 1
    sensitivity_target = supplement_dir / "table_swine_sensitivity_rows.tex"
    if optional_rows:
        sensitivity_target.write_text(
            "\n".join(sensitivity_lines) + "\n\\bottomrule%\n", encoding="utf-8"
        )
    elif sensitivity_target.exists():
        sensitivity_target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--manuscript-root", type=Path,
        default=Path("../manuscript/NOTEARS_BP_JDS_Final_Source"),
    )
    args = parser.parse_args()
    main_dir = args.manuscript_root / "main"
    supplement_dir = args.manuscript_root / "supplement"
    main_dir.mkdir(parents=True, exist_ok=True)
    supplement_dir.mkdir(parents=True, exist_ok=True)

    raw, paired, method = load_contamination(args.results)
    write_contamination_table(paired, method, main_dir / "table_contamination_rows.tex")
    write_contamination_supplement(
        raw, paired, method, supplement_dir / "table_contamination_full_rows.tex"
    )
    plotted = plot_contamination(paired, main_dir / "exact_greedy_contamination_comparison.pdf")
    manifest: dict[str, object] = {
        "contamination_replicate_rows": len(raw),
        "contamination_plot_values": plotted,
    }
    additional = write_additional_noise_table(args.results, supplement_dir)
    if additional is not None:
        manifest["additional_noise"] = additional

    swine = load_swine(args.results)
    if swine is not None:
        write_swine_tables(swine, main_dir, supplement_dir)
        manifest["swine_adjacency_plot"] = plot_swine_adjacency(
            swine, supplement_dir / "swine_adjacency_matrices.pdf"
        )
        manifest["swine_mortality_plot"] = plot_mortality(
            swine, main_dir / "swine_mortality_neighborhood.pdf"
        )
        manifest["swine_global"] = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in swine["global"].to_dict().items()
        }

    source_paths = sorted(args.results.glob("**/*.csv"))
    manifest["source_csv_sha256"] = {
        str(path.relative_to(args.results)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    (args.manuscript_root / "manuscript_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote manuscript assets; swine outputs included={swine is not None}")


if __name__ == "__main__":
    main()
