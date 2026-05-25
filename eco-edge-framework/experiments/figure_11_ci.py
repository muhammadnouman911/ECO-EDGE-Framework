"""
Fix 5-A: Figure 11 — Agentic Efficiency (eta_A) vs Network Scale
with 95% Confidence Interval shaded bands.

Matches CI style of Figures 4, 5, 6, 10 in Eco-Edge v6 paper.
Uses only matplotlib and numpy (no seaborn/plotly) per paper constraints.

Values from paper Table 10 / Section IX data.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Data (from paper Table 10 / Section IX) ──────────────────────────────────
scales = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])

eco_mean  = np.array([1.80, 1.70, 1.60, 1.40, 1.30, 1.20, 1.10, 1.00, 0.90])
eco_std   = np.array([0.08, 0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.13, 0.14])

ppo_mean  = np.array([0.72, 0.68, 0.65, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52])
ppo_std   = np.array([0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09])

dqn_mean  = np.array([0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.43, 0.42, 0.42])
dqn_std   = np.array([0.04, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07])

heuristic = np.array([0.35, 0.34, 0.33, 0.32, 0.31, 0.31, 0.30, 0.30, 0.30])

# n_runs = 50 → 95% CI half-width = 1.96 × std / sqrt(50)
N_RUNS = 50
z = 1.96

def ci(std_arr):
    return z * std_arr / np.sqrt(N_RUNS)

eco_ci  = ci(eco_std)
ppo_ci  = ci(ppo_std)
dqn_ci  = ci(dqn_std)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# Eco-Edge
ax.plot(scales, eco_mean, color="#1a7abf", linewidth=2.0, marker="o",
        markersize=6, label="Eco-Edge (ActSimSecCrit)", zorder=4)
ax.fill_between(scales, eco_mean - eco_ci, eco_mean + eco_ci,
                alpha=0.18, color="#1a7abf", label="_nolegend_")

# PPO
ax.plot(scales, ppo_mean, color="#e07b1a", linewidth=2.0, marker="s",
        linestyle="--", markersize=6, label="PPO Baseline", zorder=3)
ax.fill_between(scales, ppo_mean - ppo_ci, ppo_mean + ppo_ci,
                alpha=0.18, color="#e07b1a", label="_nolegend_")

# DQN
ax.plot(scales, dqn_mean, color="#b22222", linewidth=2.0, marker="^",
        linestyle="-.", markersize=6, label="DQN Baseline", zorder=3)
ax.fill_between(scales, dqn_mean - dqn_ci, dqn_mean + dqn_ci,
                alpha=0.18, color="#b22222", label="_nolegend_")

# Heuristic (no CI — deterministic baseline)
ax.plot(scales, heuristic, color="#555555", linewidth=1.8, marker="D",
        linestyle=":", markersize=5, label="Heuristic", zorder=2)

# Vertical annotation at N=1000 (p < 0.001, Welch t-test, Eco-Edge vs PPO)
ax.axvline(x=1000, color="gray", linestyle="--", linewidth=1.0, alpha=0.65)
ax.text(1005, 1.35, "$p < 0.001$\n(N=1000)", fontsize=8.5,
        color="gray", va="center", ha="left")

# Shaded CI legend patch
ci_patch = mpatches.Patch(color="gray", alpha=0.25, label="95% CI (shaded)")

# Axes labels and formatting
ax.set_xlabel("Number of Edge Nodes", fontsize=10)
ax.set_ylabel(r"Agentic Efficiency ($\eta_A$)", fontsize=10)
ax.set_title(
    r"Fig. 11. Agentic Efficiency ($\eta_A$) vs. Network Scale "
    "\nwith 95% Confidence Intervals",
    fontsize=10, pad=8
)
ax.set_xticks(scales)
ax.set_xticklabels([str(s) for s in scales], fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.set_ylim(0.0, 2.10)
ax.set_xlim(175, 1080)
ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.8)

handles, labels = ax.get_legend_handles_labels()
handles.append(ci_patch)
labels.append("95% CI (shaded)")
ax.legend(handles, labels, fontsize=9, loc="upper right",
          framealpha=0.92, edgecolor="#cccccc")

plt.tight_layout()

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "figure_11_eta_ci.png")
pdf_path = os.path.join(out_dir, "figure_11_eta_ci.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
try:
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF: {pdf_path}")
except Exception as e:
    print(f"PDF save skipped: {e}")

print(f"Saved PNG: {png_path}")
print("\nEco-Edge ηA:", np.round(eco_mean, 3))
print("PPO     ηA:", np.round(ppo_mean, 3))
print("DQN     ηA:", np.round(dqn_mean, 3))
print("Heuristic :", np.round(heuristic, 3))
print("\n95% CI half-widths (Eco-Edge):", np.round(eco_ci, 4))

plt.show()
