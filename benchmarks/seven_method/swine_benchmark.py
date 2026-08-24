"""Run benchmark competitors on the authorized commercial-swine analysis matrix.

This script never writes row-level data. It writes only adjacency matrices and
non-row-level summaries. The proprietary input file is not distributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pc_fdr import pc_fdr_skeleton
from .pc_original import pc_original_skeleton
from .pcp_faithful import pc_p_skeleton_faithful
from .wang_sensitivity import wang_full_skeleton_bins

REAL_VARS = [
    "PRRS_binary", "MYCO_binary", "LateralPRRS_binary", "Q2", "Q3", "Q4",
    "Avg_parity_farrow", "Litters_female_year", "mated_inventory_20wks", "PWMFyear",
    "nonproductive_days", "number_services", "wean_to_service", "abortions_rate",
    "Total_born_avg", "Stillborn_avg", "Mummies_avg", "prenatal_losses_avg",
    "Born_alive_avg", "Gestation_days", "Interval_farrows", "Pre_weaning_mortality", "PWSow",
    "productive_days_rate", "services_per_inventory_N_rate", "repeats__rate", "gilts_bred_rate",
    "Last_week_wean_bred_rate", "pregnant_105days_rate", "Cull_rate_annual", "Sow_Death_rate",
    "avg_parity_at_farrow", "Lactation_days", "final_inventory", "Farrowing__rate", "HeadIn",
    "mortality_60days",
]
EXPECTED_SHA256 = "b933fd66f49fd381bb9698ee2b3f5835d0db8d01820a01d5e45c9ac3a7bf5156"


def prepare(path: Path):
    raw = pd.read_csv(path, na_values=["NA", "NaN", "null", ""])
    x = raw.copy()
    x["PRRS_binary"] = x["PRRSatPlacement"].astype(str).str.lower().eq("epidemic").astype(int)
    x["MYCO_binary"] = x["Mycoplasma_Status"].astype(str).str.lower().eq("endemic").astype(int)
    x["LateralPRRS_binary"] = x["LateralPRRS"].astype(str).str.lower().eq("yes").astype(int)
    quarter = pd.to_numeric(x["Year_Quarter"], errors="coerce")
    for q in [2, 3, 4]:
        x[f"Q{q}"] = quarter.eq(q).astype(int)
    df = x[REAL_VARS].apply(pd.to_numeric, errors="coerce").dropna().copy()
    return raw, df, df.to_numpy(float), list(REAL_VARS)


def edge_count(S):
    S = np.asarray(S) != 0
    return int(np.triu(S | S.T, 1).sum())


def mortality_neighbors(S, labels):
    m = labels.index("mortality_60days")
    return [labels[i] for i in range(len(labels)) if i != m and (S[i, m] or S[m, i])]


def run_one(X, method):
    if method == "pc":
        S, runtime, tests = pc_original_skeleton(X, alpha=0.05)
        return S, runtime, {"alpha": 0.05, "tests": tests}
    if method == "pcfdr05":
        S, runtime, tests = pc_fdr_skeleton(X, q=0.05, by=False, heuristic=False)
        return S, runtime, {"q": 0.05, "tests": tests, "by": False}
    if method == "pcfdr10":
        S, runtime, tests = pc_fdr_skeleton(X, q=0.10, by=False, heuristic=False)
        return S, runtime, {"q": 0.10, "tests": tests, "by": False}
    if method == "pcp05":
        S, runtime, details = pc_p_skeleton_faithful(X, q=0.05, return_details=True)
        return S, runtime, {"q": 0.05, **details}
    if method == "pcp10":
        S, runtime, details = pc_p_skeleton_faithful(X, q=0.10, return_details=True)
        return S, runtime, {"q": 0.10, **details}
    if method == "wang":
        S, runtime, details = wang_full_skeleton_bins(
            X, eps_skeleton=0.008, eps_collider=0.005, eps_prune=0.009, bins=3
        )
        return S, runtime, details
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/seven_method_benchmark/swine"))
    ap.add_argument("--method", choices=["pc", "pcfdr05", "pcfdr10", "pcp05", "pcp10", "wang"], required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    raw, df, X, labels = prepare(args.data)
    S, runtime, details = run_one(X, args.method)
    pd.DataFrame(S, index=labels, columns=labels).to_csv(args.out / f"{args.method}_adjacency.csv")
    summary = {
        "method": args.method,
        "source_sha256": digest,
        "source_matches_pinned_application": digest == EXPECTED_SHA256,
        "source_rows": len(raw),
        "complete_case_rows": len(df),
        "variables": len(labels),
        "selected_adjacencies": edge_count(S),
        "mortality_neighbors": "; ".join(mortality_neighbors(S, labels)),
        "runtime_seconds": runtime,
        **details,
    }
    pd.DataFrame([summary]).to_csv(args.out / f"{args.method}_summary.csv", index=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
