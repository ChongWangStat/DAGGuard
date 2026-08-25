#!/usr/bin/env python3
"""Reproduce the external-method rows of the seven-method simulation benchmark.

The 240 datasets are regenerated from the same deterministic seeds used by
``additional_noise_sensitivity.py``. This script reruns the four external
comparators (Wang adaptation, PC-FDR, PC-p, ordinary PC), writes replicate-level
metrics, and verifies the manuscript's primary and sensitivity summaries.

NOTEARS/DAGGuard rows are produced by the separate pinned end-to-end workflow;
this audit deliberately does not refit NOTEARS, which is substantially more
expensive and already has archived replicate-level outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    BASE_SEED,
    simulate_dag_seeded,
    simulate_lsem_noise,
    simulate_weights_seeded,
    skeleton_metrics,
)
from .pc_fdr import pc_fdr_skeleton
from .pc_original import pc_original_skeleton
from .pcp_faithful import pc_p_skeleton_faithful
from .wang_sensitivity import THRESHOLDS, wang_full_skeleton_bins


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ARCHIVE = REPO_ROOT / "results" / "seven_method_benchmark" / "simulation_primary_seven_methods.csv"

METHOD_CONFIGS = (
    ("Wang et al. adapted", "wang", (.008, .005, .009)),
    ("PC-FDR (q=.05)", "pcfdr", (.05, False)),
    ("PC-FDR (q=.10)", "pcfdr", (.10, False)),
    ("PC-FDR BY (q=.05)", "pcfdr", (.05, True)),
    ("PC-p (q=.05)", "pcp", .05),
    ("PC-p (q=.10)", "pcp", .10),
    ("Ordinary PC (alpha=.05)", "pc", .05),
)

EXPECTED_SENSITIVITY = {
    "PC-FDR (q=.05)": (0.191, 0.200, 0.467, 19.50),
    "PC-FDR (q=.10)": (0.195, 0.205, 0.477, 19.39),
    "PC-FDR BY (q=.05)": (0.190, 0.199, 0.448, 19.91),
    "PC-p (q=.05)": (0.097, 0.104, 0.366, 20.28),
    "PC-p (q=.10)": (0.106, 0.113, 0.381, 20.02),
}

EXPECTED_WANG = {
    "A": (0.531, 0.317, 31.27, 20.31),
    "B": (0.531, 0.316, 31.24, 20.20),
    "C": (0.533, 0.323, 31.36, 20.72),
    "D": (0.532, 0.321, 31.32, 20.60),
}


def _dataset(d: int, s: float, noise: str, rep: int, n: int = 500):
    noise_index = ["normal", "exponential", "gumbel"].index(noise)
    seed = BASE_SEED + 100000 * d + 1000 * int(10 * s) + 10 * noise_index + rep
    truth = simulate_dag_seeded(d, 2 * d, seed)
    weights = simulate_weights_seeded(truth, s, seed + 1)
    X = simulate_lsem_noise(weights, n, noise, seed + 2)
    return seed, truth, X


def _row(method: str, d: int, s: float, noise: str, rep: int,
         seed: int, truth: np.ndarray, estimate: np.ndarray, runtime: float):
    return {
        "method": method,
        "d": d,
        "s": s,
        "noise": noise,
        "rep": rep,
        "seed": seed,
        "runtime_seconds": runtime,
        **skeleton_metrics(truth, estimate),
    }


def regenerate(replicates: int = 20, n: int = 500):
    rows = []
    wang_rows = []
    for d in (10, 20):
        for s in (2, 5):
            for noise in ("normal", "exponential", "gumbel"):
                for rep in range(replicates):
                    seed, truth, X = _dataset(d, s, noise, rep, n)

                    for label, kind, params in METHOD_CONFIGS:
                        if kind == "wang":
                            estimate, runtime, _ = wang_full_skeleton_bins(X, *params, bins=3)
                        elif kind == "pcfdr":
                            q, by = params
                            estimate, runtime, _ = pc_fdr_skeleton(X, q=q, by=by, heuristic=False)
                        elif kind == "pcp":
                            estimate, runtime, _ = pc_p_skeleton_faithful(X, q=params)
                        elif kind == "pc":
                            estimate, runtime, _ = pc_original_skeleton(X, alpha=params)
                        else:
                            raise AssertionError(kind)
                        rows.append(_row(label, d, s, noise, rep, seed, truth, estimate, runtime))

                    for idx, thresholds in enumerate(THRESHOLDS):
                        estimate, runtime, _ = wang_full_skeleton_bins(X, *thresholds, bins=3)
                        wang_rows.append(_row(chr(ord("A") + idx), d, s, noise, rep,
                                              seed, truth, estimate, runtime))
    return pd.DataFrame(rows), pd.DataFrame(wang_rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method, g in df.groupby("method", sort=False):
        selected = int(g["edges"].sum())
        fp = int(g["fp"].sum())
        tp = int(g["tp"].sum())
        fn = int(g["fn"].sum())
        out.append({
            "method": method,
            "n": len(g),
            "mean_fdp": float(g["fdr"].mean()),
            "pooled_fdp": fp / selected if selected else np.nan,
            "pooled_tpr": tp / (tp + fn) if tp + fn else np.nan,
            "mean_shd": float(g["shd"].mean()),
            "mean_adjacencies": float(g["edges"].mean()),
        })
    return pd.DataFrame(out)


def _assert_close(name: str, observed, expected, digits):
    failures = []
    for value, target, ndigits in zip(observed, expected, digits):
        if round(float(value), ndigits) != round(float(target), ndigits):
            failures.append((value, target, ndigits))
    if failures:
        raise AssertionError(f"{name} mismatch: {failures}")


def check_manuscript(primary: pd.DataFrame, wang: pd.DataFrame):
    # Primary external comparator rows are checked against the committed seven-method table.
    archive = pd.read_csv(PRIMARY_ARCHIVE)
    mapping = {
        "Wang et al. adapted": "Wang et al. (2026) hybrid structural pipeline, tertile-adapted (published setting A)",
        "PC-FDR (q=.05)": "Li & Wang (2009) PC-FDR, q=0.05",
        "PC-p (q=.05)": "Strobl, Spirtes & Visweswaran (2019) PC-p, q=0.05",
        "Ordinary PC (alpha=.05)": "Ordinary PC, alpha=0.05",
    }
    for ours, archived in mapping.items():
        got = primary.loc[primary.method == ours].iloc[0]
        want = archive.loc[archive.method == archived].iloc[0]
        _assert_close(
            ours,
            (got.pooled_fdp, got.pooled_tpr, got.mean_shd, got.mean_adjacencies),
            (want.pooled_skeleton_false_discovery_proportion,
             want.pooled_skeleton_tpr, want.mean_skeleton_shd,
             want.mean_selected_adjacencies),
            (12, 12, 12, 12),
        )

    for method, expected in EXPECTED_SENSITIVITY.items():
        got = primary.loc[primary.method == method].iloc[0]
        _assert_close(method,
                      (got.mean_fdp, got.pooled_fdp, got.pooled_tpr, got.mean_shd),
                      expected, (3, 3, 3, 2))

    for setting, expected in EXPECTED_WANG.items():
        got = wang.loc[wang.method == setting].iloc[0]
        _assert_close("Wang " + setting,
                      (got.pooled_fdp, got.pooled_tpr, got.mean_shd, got.mean_adjacencies),
                      expected, (3, 3, 2, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-manuscript", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw, wang_raw = regenerate(replicates=args.replicates)
    summary = summarize(raw)
    wang_summary = summarize(wang_raw)
    raw.to_csv(args.out / "external_comparator_replicates.csv", index=False)
    summary.to_csv(args.out / "external_comparator_summary.csv", index=False)
    wang_raw.to_csv(args.out / "wang_threshold_replicates.csv", index=False)
    wang_summary.to_csv(args.out / "wang_threshold_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(wang_summary.to_string(index=False))
    if args.check_manuscript:
        if args.replicates != 20:
            raise ValueError("Manuscript check requires 20 replicates per setting")
        check_manuscript(summary, wang_summary)


if __name__ == "__main__":
    main()
