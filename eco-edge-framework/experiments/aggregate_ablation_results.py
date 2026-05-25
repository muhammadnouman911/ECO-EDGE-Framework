"""
Aggregate EdgeCloudSim ablation logs (50 iterations) into Table tab:ablation.

Expects logs:
  SIMRESULT_TWO_TIER_WITH_EO_<POLICY>_1000DEVICES_ALL_APPS_GENERIC.log
under EdgeCloudSim/scripts/sample_app1/output/ite<N>/

Usage:
  py -3 experiments/aggregate_ablation_results.py --output-dir ../EdgeCloudSim/scripts/sample_app1/output
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import statistics
from typing import Dict, List, Tuple

# Five leave-one-out ablations (Improvement 1 / Table tab:ablation)
POLICIES = [
    ("ECO_EDGE_FIT", r"\textbf{Full \ecoedge{}}"),
    ("ECO_EDGE_NO_SIMCRIT", "w/o ActSimSecCrit"),
    ("ECO_EDGE_NO_ELASTIC", "w/o Elastic Intelligence"),
    ("ECO_EDGE_NO_AFFINITY", "w/o Role Affinity"),
    ("ECO_EDGE_NO_SECURITY", "w/o Security Critic"),
]

DEVICES = 1000


def parse_log(path: str) -> Dict[str, float]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if len(lines) < 6:
        raise ValueError(f"Unexpected log format: {path}")
    delim = ";" if ";" in lines[0] else "\t"
    gen = lines[0].split(delim)
    completed = float(gen[0])
    failed = float(gen[1])
    uncompleted = float(gen[2])
    service_time = float(gen[4])
    failed_cap = float(gen[9]) if len(gen) > 9 else 0.0
    total = completed + failed + uncompleted
    success_rate = (completed / total * 100.0) if total > 0 else 0.0
    perf = lines[5].split(delim)
    ethink = float(perf[2]) if len(perf) > 2 else 0.0
    return {
        "success_rate": success_rate,
        "latency_s": service_time,
        "ethink_j": ethink,
        "failed_capacity": failed_cap,
    }


def compute_eta(success_pct: float, latency_s: float, ethink_j: float, failed_cap: float) -> float:
    """Consistent with paper Eq. 18 proxy (1000-device snapshot)."""
    s_norm = success_pct / 100.0
    lat_improv = max(0.1, 4.8 / max(latency_s, 0.1))
    energy_saved = max(0.05, 1.0 - ethink_j / 400000.0)
    delta_b = 0.4 * s_norm + 0.3 * (lat_improv / 3.0) + 0.3 * energy_saved
    cost = ethink_j / 1e5 + failed_cap / 50000.0 + 0.12
    return delta_b / cost


def aggregate_policy(output_dirs: List[str], policy: str) -> Dict[str, Tuple[float, float]]:
    values: Dict[str, List[float]] = {
        "success_rate": [],
        "latency_s": [],
        "ethink_j": [],
        "failed_capacity": [],
        "eta_a": [],
    }
    pattern = f"SIMRESULT_TWO_TIER_WITH_EO_{policy}_{DEVICES}DEVICES_ALL_APPS_GENERIC.log"
    for d in output_dirs:
        path = os.path.join(d, pattern)
        if not os.path.isfile(path):
            continue
        m = parse_log(path)
        m["eta_a"] = compute_eta(
            m["success_rate"], m["latency_s"], m["ethink_j"], m["failed_capacity"]
        )
        for k, v in m.items():
            values[k].append(v)

    out = {}
    for k, arr in values.items():
        if not arr:
            out[k] = (float("nan"), float("nan"))
        elif len(arr) == 1:
            out[k] = (arr[0], 0.0)
        else:
            out[k] = (statistics.mean(arr), statistics.stdev(arr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default="../EdgeCloudSim/scripts/sample_app1/output",
        help="Parent folder containing ite1..ite50 subfolders",
    )
    args = ap.parse_args()

    ite_dirs = sorted(glob.glob(os.path.join(args.output_dir, "ite*")))
    if not ite_dirs:
        ite_dirs = [args.output_dir]

    print("LaTeX rows for Table~\\ref{tab:ablation} (mean ± std, 1000 devices):\n")
    print(
        r"\textbf{Configuration} & \textbf{Success (\%)} & "
        r"\textbf{Avg.\ Latency (s)} & \textbf{Cap.\ Failures} & $\boldsymbol{\eta_A}$ \\"
    )
    print(r"\midrule")

    for policy, label in POLICIES:
        agg = aggregate_policy(ite_dirs, policy)
        if math.isnan(agg["success_rate"][0]):
            print(f"% missing logs for {policy}")
            continue
        sr_m, sr_s = agg["success_rate"]
        lat_m, lat_s = agg["latency_s"]
        fail_m, fail_s = agg["failed_capacity"]
        eta_m, eta_s = agg["eta_a"]
        fail_str = f"{fail_m:,.0f}$\\pm${fail_s:,.0f}".replace(",", "{,}")
        print(
            f"{label} & "
            f"{sr_m:.1f}$\\pm${sr_s:.1f} & "
            f"{lat_m:.2f}$\\pm${lat_s:.2f} & "
            f"{fail_str} & "
            f"{eta_m:.2f}$\\pm${eta_s:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
