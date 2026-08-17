#!/usr/bin/env python3
"""Merge all d=40 simulation settings, validate completeness, and draw figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["GES", "PC", "LiNGAM", "NOTEARS", "NOTEARS-BP"]
SETTINGS = {
    "uniform": [1, 4, 7, 10],
    "modnormal": [1, 2, 3, 4],
}
TOTAL_REPLICATES = 20


def find_metric_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("replicate_metrics.csv"))
        files.extend(root.rglob("replicate_metrics_rep*.csv"))
    unique = sorted(set(path.resolve() for path in files))
    if not unique:
        raise FileNotFoundError("No replicate-metric CSV files were found")
    return unique


def validate_complete(df: pd.DataFrame) -> None:
    required = {
        "d", "weight_kind", "s", "rep", "seed", "method",
        "fdr", "tpr", "shd", "true_edges", "estimated_edges",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not (df["d"] == 40).all():
        raise ValueError("Merged input contains rows with d != 40")
    if df.duplicated(["weight_kind", "s", "rep", "method"]).any():
        dup = df.loc[
            df.duplicated(["weight_kind", "s", "rep", "method"], keep=False),
            ["weight_kind", "s", "rep", "method"],
        ]
        raise ValueError(f"Duplicate result rows detected:\n{dup.to_string(index=False)}")

    observed = set(
        df[["weight_kind", "s", "rep", "method"]]
        .itertuples(index=False, name=None)
    )
    expected = {
        (kind, s, rep, method)
        for kind, s_values in SETTINGS.items()
        for s in s_values
        for rep in range(TOTAL_REPLICATES)
        for method in METHODS
    }
    missing_rows = sorted(expected.difference(observed))
    extra_rows = sorted(observed.difference(expected))
    if missing_rows or extra_rows:
        message = [
            f"Expected {len(expected)} rows and observed {len(observed)} unique rows."
        ]
        if missing_rows:
            message.append(f"First missing rows: {missing_rows[:10]}")
        if extra_rows:
            message.append(f"First unexpected rows: {extra_rows[:10]}")
        raise ValueError(" ".join(message))
    if len(df) != len(expected):
        raise ValueError(f"Expected {len(expected)} total rows, obtained {len(df)}")
    if not (df["true_edges"] == 80).all():
        raise ValueError("At least one d=40 replicate does not contain exactly 80 true edges")


def draw_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    s_values: list[int],
    out_base: Path,
) -> None:
    colors = {
        "GES": "orange",
        "PC": "green",
        "LiNGAM": "yellow",
        "NOTEARS": "blue",
        "NOTEARS-BP": "red",
    }
    offsets = {
        "GES": -0.40,
        "PC": -0.24,
        "LiNGAM": -0.08,
        "NOTEARS": 0.08,
        "NOTEARS-BP": 0.24,
    }
    fig, ax = plt.subplots(figsize=(18, 5))
    handles = []
    for method in METHODS:
        values = [
            df[(df["s"] == s) & (df["method"] == method)][metric].to_numpy()
            for s in s_values
        ]
        positions = np.arange(1, len(s_values) + 1) + offsets[method]
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            boxprops={"facecolor": colors[method], "color": "black"},
            medianprops={"color": "black"},
        )
        handles.append(bp["boxes"][0])
    ax.set_xlim(0.4, len(s_values) + 0.74)
    ax.set_xticks(np.arange(1, len(s_values) + 1))
    ax.set_xticklabels(s_values, fontsize=16)
    ax.set_xlabel("S", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(handles, METHODS, loc="upper right")
    ax.set_title(title, fontsize=16)
    for index in range(len(s_values) - 1):
        ax.axvline(x=index + 1.4, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            f"{out_base}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def draw_combined(df: pd.DataFrame, s_values: list[int], out_base: Path) -> None:
    colors = {
        "GES": "orange",
        "PC": "green",
        "LiNGAM": "yellow",
        "NOTEARS": "blue",
        "NOTEARS-BP": "red",
    }
    offsets = {
        "GES": -0.40,
        "PC": -0.24,
        "LiNGAM": -0.08,
        "NOTEARS": 0.08,
        "NOTEARS-BP": 0.24,
    }
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    specifications = [
        ("fdr", "False Discovery Rate", "FDR"),
        ("tpr", "True Positive Rate", "TPR"),
        ("shd", "SHD", "SHD"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specifications):
        handles = []
        for method in METHODS:
            values = [
                df[(df["s"] == s) & (df["method"] == method)][metric].to_numpy()
                for s in s_values
            ]
            positions = np.arange(1, len(s_values) + 1) + offsets[method]
            bp = ax.boxplot(
                values,
                positions=positions,
                widths=0.15,
                patch_artist=True,
                boxprops={"facecolor": colors[method], "color": "black"},
                medianprops={"color": "black"},
            )
            handles.append(bp["boxes"][0])
        ax.set_xlim(0.4, len(s_values) + 0.74)
        ax.set_xticks(np.arange(1, len(s_values) + 1))
        ax.set_xticklabels(s_values, fontsize=16)
        ax.set_xlabel("S", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16)
        ax.legend(handles, METHODS, loc="upper right")
        for index in range(len(s_values) - 1):
            ax.axvline(x=index + 1.4, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            f"{out_base}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Root containing prior complete settings or new replicate chunks; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = find_metric_files(args.input_root)
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    validate_complete(merged)
    merged = merged.sort_values(["weight_kind", "s", "rep", "method"])

    args.out.mkdir(parents=True, exist_ok=True)
    merged.drop(columns="source_file").to_csv(
        args.out / "d40_all_replicate_metrics.csv", index=False
    )

    audit = {
        "input_files": [str(path) for path in files],
        "row_count": int(len(merged)),
        "settings": SETTINGS,
        "replicates_per_setting": TOTAL_REPLICATES,
        "methods": METHODS,
    }
    (args.out / "merge_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    for kind, s_values in SETTINGS.items():
        panel = merged[merged["weight_kind"] == kind].drop(columns="source_file")
        panel_dir = args.out / f"d40_{kind}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(panel_dir / "replicate_metrics.csv", index=False)
        summary = (
            panel.groupby(["d", "weight_kind", "s", "method"], as_index=False)
            .agg(
                fdr_mean=("fdr", "mean"),
                tpr_mean=("tpr", "mean"),
                shd_mean=("shd", "mean"),
                fdr_sd=("fdr", "std"),
                tpr_sd=("tpr", "std"),
                shd_sd=("shd", "std"),
                estimated_edges_mean=("estimated_edges", "mean"),
            )
        )
        summary.to_csv(panel_dir / "summary.csv", index=False)
        draw_metric(panel, "fdr", "False Discovery Rate", "FDR", s_values, panel_dir / "fdr")
        draw_metric(panel, "tpr", "True Positive Rate", "TPR", s_values, panel_dir / "tpr")
        draw_metric(panel, "shd", "SHD", "SHD", s_values, panel_dir / "shd")
        draw_combined(panel, s_values, panel_dir / "combined")
        print(f"Completed d=40 {kind} panel with {len(panel)} rows")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
