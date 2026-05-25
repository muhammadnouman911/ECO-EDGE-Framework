"""
Compute Agentic Efficiency (eta_A) from benchmark_results.json and plot comparison.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
PAPER_SCALE_POINTS = [200, 400, 600, 800, 1000]


def compute_delta_bsys(success, lat_improv, energy_saved, w1=0.4, w2=0.3, w3=0.3):
    return w1 * np.array(success) + w2 * (np.array(lat_improv) / 3.0) + w3 * np.array(
        energy_saved
    )


def compute_eta(delta_b, ethink, overhead):
    cost = np.array(ethink) / 1e5 + np.array(overhead) / 1000.0
    return np.array(delta_b) / (cost + 1e-9)


def _interp_series(scales_all, values_by_scale: dict, target_scales):
    xs = sorted(int(k) for k in values_by_scale.keys())
    ys = [values_by_scale[str(x)] for x in xs]
    return [float(np.interp(t, xs, ys)) for t in target_scales]


def load_metrics():
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        eco = data["eco_edge"]
        dqn = data["dqn"]
        return {
            "from_simulation": True,
            "eco_success": _interp_series(
                eco, {k: eco[k]["success_rate"] for k in eco}, PAPER_SCALE_POINTS
            ),
            "eco_lat_improv": _interp_series(
                eco,
                {k: eco[k]["latency_improv_vs_next_fit"] for k in eco},
                PAPER_SCALE_POINTS,
            ),
            "eco_energy_saved": _interp_series(
                eco, {k: eco[k]["energy_saved_fraction"] for k in eco}, PAPER_SCALE_POINTS
            ),
            "eco_ethink": _interp_series(
                eco, {k: eco[k]["ethink_j"] for k in eco}, PAPER_SCALE_POINTS
            ),
            "eco_overhead": _interp_series(
                eco, {k: eco[k]["orchestration_overhead"] for k in eco}, PAPER_SCALE_POINTS
            ),
            "dqn_success": _interp_series(
                dqn, {k: dqn[k]["success_rate"] for k in dqn}, PAPER_SCALE_POINTS
            ),
            "dqn_lat_improv": _interp_series(
                dqn,
                {k: dqn[k]["latency_improv_vs_next_fit"] for k in dqn},
                PAPER_SCALE_POINTS,
            ),
            "dqn_energy_saved": _interp_series(
                dqn, {k: dqn[k]["energy_saved_fraction"] for k in dqn}, PAPER_SCALE_POINTS
            ),
            "dqn_ethink": _interp_series(
                dqn, {k: dqn[k].get("ethink_j", 0) for k in dqn}, PAPER_SCALE_POINTS
            ),
            "dqn_overhead": _interp_series(
                dqn, {k: dqn[k]["orchestration_overhead"] for k in dqn}, PAPER_SCALE_POINTS
            ),
        }

    # Fallback example curves from paper draft
    return {
        "from_simulation": False,
        "eco_success": [0.97, 0.96, 0.95, 0.94, 0.93],
        "eco_lat_improv": [2.1, 2.3, 2.5, 2.7, 2.9],
        "eco_energy_saved": [0.30, 0.32, 0.35, 0.37, 0.40],
        "eco_ethink": [5000, 12000, 35000, 80000, 150000],
        "eco_overhead": [100, 200, 350, 500, 700],
        "dqn_success": [0.88, 0.84, 0.79, 0.74, 0.70],
        "dqn_lat_improv": [1.3, 1.4, 1.5, 1.5, 1.6],
        "dqn_energy_saved": [0.15, 0.17, 0.18, 0.19, 0.20],
        "dqn_ethink": [0, 0, 0, 0, 0],
        "dqn_overhead": [500, 900, 1400, 1900, 2500],
    }


def main() -> None:
    m = load_metrics()
    network_scales = PAPER_SCALE_POINTS

    eco_delta = compute_delta_bsys(
        m["eco_success"], m["eco_lat_improv"], m["eco_energy_saved"]
    )
    dqn_delta = compute_delta_bsys(
        m["dqn_success"], m["dqn_lat_improv"], m["dqn_energy_saved"]
    )
    eco_eta = compute_eta(eco_delta, m["eco_ethink"], m["eco_overhead"])
    dqn_eta = compute_eta(dqn_delta, m["dqn_ethink"], m["dqn_overhead"])

    plt.figure(figsize=(8, 5))
    plt.plot(
        network_scales,
        eco_eta,
        "g-o",
        linewidth=2,
        label="Eco-Edge (w/ ActSimSecCrit)",
    )
    plt.plot(
        network_scales, dqn_eta, "r--s", linewidth=2, label="DQN Baseline"
    )
    plt.xlabel("Number of Edge Devices")
    plt.ylabel(r"Agentic Efficiency ($\eta_A$)")
    plt.title("Agentic Efficiency vs Network Scale")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_framework = os.path.join(os.path.dirname(__file__), "figure_eta_comparison.png")
    out_paper = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "ECO-EDGE", "figure_eta_comparison.png"
    )
    plt.savefig(out_framework, dpi=300)
    try:
        plt.savefig(out_paper, dpi=300)
    except OSError:
        pass
    print("Eco-Edge eta_A:", np.round(eco_eta, 3))
    print("DQN eta_A:", np.round(dqn_eta, 3))
    if not m["from_simulation"]:
        print("(Used fallback constants; run run_dqn_benchmark.py for measured values.)")


if __name__ == "__main__":
    main()
