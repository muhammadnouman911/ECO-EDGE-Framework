"""
Aggregate results/*.jsonl into tables (LaTeX + JSON) and figures.
Every number in the paper is produced by this script from the raw runs.

  python analyze.py --results results --out analysis
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = ["NEXT_FIT", "RANDOM_FIT", "DQN_FIT", "PPO_FIT", "ECO_EDGE"]
LABEL = {"NEXT_FIT": "Next-Fit", "RANDOM_FIT": "Random-Fit", "DQN_FIT": "Q-learning (DQN-Fit)",
         "PPO_FIT": "PPO", "ECO_EDGE": "Eco-Edge"}
STYLE = {"NEXT_FIT": ("tab:blue", "s", "--"), "RANDOM_FIT": ("tab:red", "^", ":"),
         "DQN_FIT": ("tab:purple", "D", "-."), "PPO_FIT": ("tab:cyan", "v", "-."),
         "ECO_EDGE": ("tab:green", "o", "-")}
W = (0.4, 0.3, 0.3)          # eta_A weights: success, latency, execution energy
REF = "NEXT_FIT"


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def enrich(rows):
    for r in rows:
        r["energy_total_J"] = r["energy_exec_J"] + r["energy_think_J"] + r["energy_coord_J"]
        r["adv_exec_frac"] = r["adv_executed"] / max(1, r["adv_total"])
        r["cap_fail_frac"] = r["fail_capacity"] / r["n_tasks"]
    return rows


def add_eta(rows, key_fields, ref_rows=None):
    """Dimensionless Agentic Efficiency relative to the reference orchestrator
    in the same (scenario, seed)."""
    ref = {}
    for r in (ref_rows if ref_rows is not None else rows):
        if r["algo"] == REF:
            ref[tuple(r[k] for k in key_fields) + (r["seed"],)] = r
    for r in rows:
        k = tuple(r[kk] for kk in key_fields) + (r["seed"],)
        b = ref.get(k)
        if b is None:
            r["eta_A"] = float("nan"); continue
        dB = W[0] * (r["success_rate"] - b["success_rate"]) / b["success_rate"] \
            + W[1] * (b["mean_latency_ms"] - r["mean_latency_ms"]) / b["mean_latency_ms"] \
            + W[2] * (b["energy_exec_J"] - r["energy_exec_J"]) / b["energy_exec_J"]
        cost = (r["energy_think_J"] + r["energy_coord_J"]) / b["energy_exec_J"]
        r["delta_B"] = dB; r["cost_norm"] = cost
        r["eta_A"] = dB / cost if cost > 0 else float("nan")
    return rows


def agg(rows, group, metrics):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group)].append(r)
    out = {}
    for key, rs in g.items():
        d = {"n": len(rs)}
        for m in metrics:
            v = np.array([r[m] for r in rs], float)
            v = v[~np.isnan(v)]
            d[m] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                        ci95=float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                        n=int(len(v)))
        out[key] = d
    return out


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("inf")


METRICS = ["success_rate", "mean_latency_ms", "p95_latency_ms", "cap_fail_frac", "adv_exec_frac",
           "energy_exec_J", "energy_think_J", "energy_coord_J", "energy_total_J", "eta_A",
           "nodes_depleted", "fail_deadline", "fail_security"]


def fmt(d, m, scale=1.0, prec=1):
    return f"{d[m]['mean']*scale:.{prec}f} $\\pm$ {d[m]['std']*scale:.{prec}f}"


def main(a):
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    scale = add_eta(enrich(load(f"{a.results}/scale.jsonl")), ["n_edge"])
    abl = enrich(load(f"{a.results}/ablation.jsonl"))
    sens = enrich(load(f"{a.results}/sensitivity.jsonl"))
    ref1000 = [r for r in scale if r["algo"] == REF and r["n_edge"] == 1000]
    abl = add_eta(abl, ["n_edge"], ref1000)
    for r in sens:
        if r["exp"] == "soc":
            r["eta_A"] = float("nan")
    add_eta([r for r in sens if r["exp"] == "soc"], ["n_edge"], ref1000)
    summary = {}

    # ------------------------------------------------------------- scale
    S = agg(scale, ["n_edge", "algo"], METRICS)
    Ns = sorted({r["n_edge"] for r in scale})
    summary["scale"] = {f"{k[0]}|{k[1]}": v for k, v in S.items()}

    # main comparison table at N=1000
    N = max(Ns)
    lines = []
    for al in ALGOS:
        d = S[(N, al)]
        lines.append(f"{LABEL[al]} & {fmt(d,'success_rate',100)} & {fmt(d,'mean_latency_ms')} & "
                     f"{fmt(d,'p95_latency_ms',1,0)} & {fmt(d,'adv_exec_frac',100)} & "
                     f"{fmt(d,'energy_exec_J',1e-3)} & {fmt(d,'energy_think_J',1e-3,2)} & "
                     f"{fmt(d,'energy_total_J',1e-3)} & {fmt(d,'eta_A',1,2)} \\\\")
    open(f"{a.out}/table_main_N{N}.tex", "w").write("\n".join(lines) + "\n")

    # statistical tests at N=1000: Eco-Edge vs each baseline
    tests = {}
    by = defaultdict(dict)
    for r in scale:
        if r["n_edge"] == N:
            by[r["algo"]][r["seed"]] = r
    for m in ["success_rate", "mean_latency_ms", "energy_total_J", "energy_exec_J", "adv_exec_frac", "eta_A"]:
        tests[m] = {}
        for al in ALGOS:
            if al == "ECO_EDGE":
                continue
            seeds = sorted(set(by["ECO_EDGE"]) & set(by[al]))
            x = [by["ECO_EDGE"][s][m] for s in seeds]; y = [by[al][s][m] for s in seeds]
            x, y = np.array(x), np.array(y)
            ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
            t, p = stats.ttest_ind(x, y, equal_var=False)
            tests[m][al] = dict(eco_mean=float(x.mean()), base_mean=float(y.mean()),
                                welch_t=float(t), p=float(p), cohen_d=cohen_d(x, y), n=int(len(x)))
    summary["tests_N%d" % N] = tests
    tl = []
    for m, lab, sc in [("success_rate", "Success rate (\\%)", 100), ("mean_latency_ms", "Mean latency (ms)", 1),
                       ("energy_total_J", "Total energy (kJ)", 1e-3), ("adv_exec_frac", "Adversarial executed (\\%)", 100)]:
        for al in ["NEXT_FIT", "DQN_FIT", "PPO_FIT"]:
            t = tests[m][al]
            pstr = "$<$0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
            tl.append(f"{lab} & {LABEL[al]} & {t['eco_mean']*sc:.1f} & {t['base_mean']*sc:.1f} & "
                      f"{t['welch_t']:.1f} & {pstr} & {t['cohen_d']:.2f} \\\\")
    open(f"{a.out}/table_tests_N{N}.tex", "w").write("\n".join(tl) + "\n")

    # scale table (success and latency for every N)
    sl = []
    for n in Ns:
        row = [str(n)]
        for al in ALGOS:
            d = S[(n, al)]
            row.append(f"{d['success_rate']['mean']*100:.1f} / {d['mean_latency_ms']['mean']:.0f}")
        sl.append(" & ".join(row) + " \\\\")
    open(f"{a.out}/table_scale.tex", "w").write("\n".join(sl) + "\n")

    def plot_scale(metric, ylabel, fname, scale_=1.0, algos=ALGOS, logy=False, title=None):
        plt.figure(figsize=(5.2, 3.6))
        for al in algos:
            m = np.array([S[(n, al)][metric]["mean"] for n in Ns]) * scale_
            c = np.array([S[(n, al)][metric]["ci95"] for n in Ns]) * scale_
            col, mk, ls = STYLE[al]
            plt.plot(Ns, m, color=col, marker=mk, ls=ls, label=LABEL[al], ms=4)
            plt.fill_between(Ns, m - c, m + c, color=col, alpha=0.15)
        plt.xlabel("Number of edge devices ($N$)"); plt.ylabel(ylabel)
        if logy: plt.yscale("log")
        if metric == "eta_A": plt.yscale("symlog", linthresh=5); plt.axhline(0, color="k", lw=0.6)
        if title: plt.title(title, fontsize=9)
        plt.grid(alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(f"{fig_dir}/{fname}.pdf"); plt.savefig(f"{fig_dir}/{fname}.png", dpi=160); plt.close()

    plot_scale("success_rate", "Task success rate (%)", "fig_success_vs_scale", 100)
    plot_scale("mean_latency_ms", "Mean end-to-end latency (ms)", "fig_latency_vs_scale")
    plot_scale("energy_total_J", "Total energy (kJ)", "fig_energy_total_vs_scale", 1e-3)
    plot_scale("energy_think_J", "Orchestration inference energy $E_{think}$ (kJ)", "fig_energy_think_vs_scale", 1e-3,
               algos=["DQN_FIT", "PPO_FIT", "ECO_EDGE"], logy=True)
    plot_scale("adv_exec_frac", "Adversarial intents executed (%)", "fig_adversarial_vs_scale", 100)
    plot_scale("eta_A", "Agentic efficiency $\\eta_A$ (dimensionless)", "fig_eta_vs_scale", 1.0,
               algos=["RANDOM_FIT", "DQN_FIT", "PPO_FIT", "ECO_EDGE"])

    # stacked energy at N=1000
    plt.figure(figsize=(5.2, 3.4))
    ex = [S[(N, al)]["energy_exec_J"]["mean"] / 1e3 for al in ALGOS]
    th = [S[(N, al)]["energy_think_J"]["mean"] / 1e3 for al in ALGOS]
    co = [S[(N, al)]["energy_coord_J"]["mean"] / 1e3 for al in ALGOS]
    x = np.arange(len(ALGOS))
    plt.bar(x, ex, color="#4C72B0", label="Execution (compute + transfer)")
    plt.bar(x, th, bottom=ex, color="#DD8452", label="Orchestration inference $E_{think}$")
    plt.bar(x, co, bottom=np.array(ex) + np.array(th), color="#55A868", label="Coordination $\\Omega_{orch}$")
    plt.xticks(x, [LABEL[al] for al in ALGOS], rotation=15, fontsize=8)
    plt.ylabel(f"Energy per run, $N$={N} (kJ)"); plt.legend(fontsize=7); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/fig_energy_breakdown.pdf"); plt.savefig(f"{fig_dir}/fig_energy_breakdown.png", dpi=160); plt.close()

    # ---------------------------------------------------------- ablation
    A = agg(abl, ["algo"], METRICS)
    order = ["A_full", "B_no_elastic", "B2_fixed_0.5B", "B3_soc_only", "C_no_security", "D_no_critic",
             "E_no_twin", "F_no_affinity", "G_no_critic_no_twin"]
    ALAB = {"A_full": "Full Eco-Edge", "B_no_elastic": "No Elastic (fixed 2.7B)", "B2_fixed_0.5B": "Fixed 0.5B tier",
            "B3_soc_only": "SoC-only tier policy", "C_no_security": "No Security-Critic", "D_no_critic": "No Critic",
            "E_no_twin": "No Digital Twin", "F_no_affinity": "No Role-Affinity", "G_no_critic_no_twin": "No Critic + No Twin"}
    al_lines = []
    full = A[("A_full",)]
    for k in order:
        if (k,) not in A: continue
        d = A[(k,)]
        ds = (d["success_rate"]["mean"] - full["success_rate"]["mean"]) * 100
        al_lines.append(f"{ALAB[k]} & {fmt(d,'success_rate',100)} & {ds:+.1f} & {fmt(d,'mean_latency_ms')} & "
                        f"{fmt(d,'adv_exec_frac',100)} & {fmt(d,'energy_exec_J',1e-3)} & {fmt(d,'energy_think_J',1e-3,2)} & {fmt(d,'eta_A',1,2)} \\\\")
    open(f"{a.out}/table_ablation.tex", "w").write("\n".join(al_lines) + "\n")
    summary["ablation"] = {k[0]: v for k, v in A.items()}
    # ablation figure
    ks = [k for k in order if (k,) in A]
    plt.figure(figsize=(6.2, 3.4))
    sm = [A[(k,)]["success_rate"]["mean"] * 100 for k in ks]; sc = [A[(k,)]["success_rate"]["ci95"] * 100 for k in ks]
    plt.bar(range(len(ks)), sm, yerr=sc, color=["tab:green"] + ["tab:gray"] * (len(ks) - 1), capsize=3)
    plt.xticks(range(len(ks)), [ALAB[k] for k in ks], rotation=30, ha="right", fontsize=7)
    plt.ylabel("Task success rate (%)"); plt.ylim(min(sm) - 10, 100); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/fig_ablation.pdf"); plt.savefig(f"{fig_dir}/fig_ablation.png", dpi=160); plt.close()

    # ------------------------------------------------------- sensitivity
    def sens_plot(exp, xkey, xlabel, fname, metric="success_rate", ylabel="Task success rate (%)", sc=100, algos=None):
        rows = [r for r in sens if r["exp"] == exp]
        if exp != "soc":
            rows = add_eta(rows, [xkey])
        G = agg(rows, [xkey, "algo"], METRICS)
        xs = sorted({r[xkey] for r in rows})
        plt.figure(figsize=(5.2, 3.4))
        for al in (algos or ["NEXT_FIT", "DQN_FIT", "PPO_FIT", "ECO_EDGE"]):
            if (xs[0], al) not in G: continue
            m = np.array([G[(x, al)][metric]["mean"] for x in xs]) * sc
            c = np.array([G[(x, al)][metric]["ci95"] for x in xs]) * sc
            col, mk, ls = STYLE[al]
            plt.plot(xs, m, color=col, marker=mk, ls=ls, label=LABEL[al], ms=4)
            plt.fill_between(xs, m - c, m + c, color=col, alpha=0.15)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(f"{fig_dir}/{fname}.pdf"); plt.savefig(f"{fig_dir}/{fname}.png", dpi=160); plt.close()
        return {f"{k[0]}|{k[1]}": v for k, v in G.items()}

    summary["sens_lambda"] = sens_plot("lambda", "rate", "Task arrival rate per device (tasks/s)", "fig_sens_lambda")
    sens_plot("lambda", "rate", "Task arrival rate per device (tasks/s)", "fig_sens_lambda_latency",
              "mean_latency_ms", "Mean latency (ms)", 1)
    summary["sens_bw"] = sens_plot("bandwidth", "bw_scale", "Uplink bandwidth scale factor", "fig_sens_bandwidth")
    # FPR sweep (Eco-Edge only)
    rows = [r for r in sens if r["exp"] == "fpr"]
    for r in rows: r["eta_A"] = float("nan")
    add_eta(rows, ["n_edge"], ref1000)
    Gf = agg(rows, ["sec_fpr"], METRICS); fk = sorted(Gf)
    plt.figure(figsize=(5.2, 3.2))
    plt.errorbar([k[0] * 100 for k in fk], [Gf[k]["success_rate"]["mean"] * 100 for k in fk],
                 yerr=[Gf[k]["success_rate"]["ci95"] * 100 for k in fk], color="tab:green", marker="o")
    plt.xlabel("Security-Critic false-positive rate (%)"); plt.ylabel("Task success rate (%)"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{fig_dir}/fig_sens_fpr.pdf"); plt.savefig(f"{fig_dir}/fig_sens_fpr.png", dpi=160); plt.close()
    summary["sens_fpr"] = {str(k[0]): v for k, v in Gf.items()}
    # SoC sweep: eta not defined (no reference) -> plot success and E_think
    rows = [r for r in sens if r["exp"] == "soc"]
    G = agg(rows, ["soc_low", "soc_high"], METRICS)
    keys = sorted(G)
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    xs = [f"({k[0]},{k[1]})" for k in keys]
    s = [G[k]["success_rate"]["mean"] * 100 for k in keys]; sc_ = [G[k]["success_rate"]["ci95"] * 100 for k in keys]
    e = [G[k]["energy_think_J"]["mean"] / 1e3 for k in keys]; ec = [G[k]["energy_think_J"]["ci95"] / 1e3 for k in keys]
    ax1.errorbar(range(len(keys)), s, yerr=sc_, color="tab:green", marker="o", label="Success rate")
    ax1.set_ylabel("Task success rate (%)", color="tab:green"); ax1.set_xticks(range(len(keys))); ax1.set_xticklabels(xs)
    ax1.set_xlabel("Elastic Intelligence SoC thresholds ($B_{low}$, $B_{high}$)")
    ax2 = ax1.twinx(); ax2.errorbar(range(len(keys)), e, yerr=ec, color="tab:orange", marker="s", label="$E_{think}$")
    ax2.set_ylabel("$E_{think}$ (kJ)", color="tab:orange"); ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_sens_soc.pdf"); fig.savefig(f"{fig_dir}/fig_sens_soc.png", dpi=160); plt.close(fig)
    summary["sens_soc"] = {f"{k[0]}|{k[1]}": v for k, v in G.items()}

    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    # console digest
    print(f"=== N={N} (n={S[(N,'ECO_EDGE')]['n']} seeds) ===")
    for al in ALGOS:
        d = S[(N, al)]
        print(f"{LABEL[al]:22s} succ={d['success_rate']['mean']*100:5.1f}±{d['success_rate']['std']*100:.1f}  "
              f"lat={d['mean_latency_ms']['mean']:6.1f}  adv={d['adv_exec_frac']['mean']*100:5.1f}%  "
              f"Eexec={d['energy_exec_J']['mean']/1e3:5.1f}kJ  Ethink={d['energy_think_J']['mean']/1e3:6.2f}kJ  "
              f"Etot={d['energy_total_J']['mean']/1e3:5.1f}kJ  eta={d['eta_A']['mean']:.2f}")
    print("--- tests (Eco-Edge vs baseline) ---")
    for m in tests:
        for al, t in tests[m].items():
            print(f"{m:16s} vs {al:10s} d={t['cohen_d']:+6.2f} p={t['p']:.2e}")
    print("--- ablation ---")
    for k in order:
        if (k,) in A:
            d = A[(k,)]
            print(f"{ALAB[k]:26s} succ={d['success_rate']['mean']*100:5.1f}  lat={d['mean_latency_ms']['mean']:6.1f}  "
                  f"Ethink={d['energy_think_J']['mean']/1e3:6.2f}kJ adv={d['adv_exec_frac']['mean']*100:5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="analysis")
    main(ap.parse_args())
