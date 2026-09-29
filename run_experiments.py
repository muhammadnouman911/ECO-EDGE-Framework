"""
Run all experiments reported in the paper and write results as JSON lines.

  python run_experiments.py --exp scale      # 100..1000 devices x 50 seeds x 5 orchestrators
  python run_experiments.py --exp ablation   # component ablation at 1000 devices
  python run_experiments.py --exp sensitivity# lambda / bandwidth / SoC-threshold sweeps
  python run_experiments.py --exp all
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import ecoedge_sim as E

SEEDS = list(range(1, 51))
SCALES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ALGOS = ["NEXT_FIT", "RANDOM_FIT", "DQN_FIT", "PPO_FIT", "ECO_EDGE"]

ABLATIONS = {
    "A_full": {},
    "B_no_elastic": dict(use_elastic=False, fixed_tier="2.7B"),
    "B2_fixed_0.5B": dict(use_elastic=False, fixed_tier="0.5B"),
    "B3_soc_only": dict(dl_fast=0.0, dl_mid=0.0),
    "C_no_security": dict(use_security=False),
    "D_no_critic": dict(use_critic=False),
    "E_no_twin": dict(use_twin=False),
    "F_no_affinity": dict(use_affinity=False),
    "G_no_critic_no_twin": dict(use_critic=False, use_twin=False),
}


def load_ppo(path):
    w = np.load(path)
    return {k: w[k] for k in w.files}


def emit(f, res, extra=None):
    d = E.result_to_dict(res)
    if extra:
        d.update(extra)
    f.write(json.dumps(d) + "\n"); f.flush()


def done_keys(path, keys):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s.add(tuple(d.get(k) for k in keys))
    return s


def exp_scale(out, ppo, seeds, scales=SCALES, mode="a"):
    t0 = time.time(); n = 0
    done = done_keys(out, ("n_edge", "seed", "algo"))
    with open(out, "a") as f:
        for N in scales:
            for seed in seeds:
                cfg = E.Config(n_edge=N)
                for algo in ALGOS:
                    if (N, seed, algo) in done:
                        continue
                    emit(f, E.run(algo, cfg, seed, ppo_weights=ppo), {"exp": "scale"}); n += 1
            print(f"[scale] N={N} done ({n} runs, {time.time()-t0:.0f}s)", flush=True)


def exp_ablation(out, seeds, N=1000):
    t0 = time.time()
    done = done_keys(out, ("algo", "seed"))
    with open(out, "a") as f:
        for name, kw in ABLATIONS.items():
            for seed in seeds:
                if (name, seed) in done:
                    continue
                cfg = E.Config(n_edge=N, **kw)
                emit(f, E.run("ECO_EDGE", cfg, seed, label=name), {"exp": "ablation"})
            print(f"[ablation] {name} done ({time.time()-t0:.0f}s)", flush=True)


def exp_sensitivity(out, ppo, seeds, N=1000):
    t0 = time.time()
    algos = ["NEXT_FIT", "DQN_FIT", "PPO_FIT", "ECO_EDGE"]
    done = done_keys(out, ("exp", "seed", "algo", "rate", "bw_scale", "soc_low"))
    done_fpr = done_keys(out, ("exp", "seed", "algo", "rate", "bw_scale", "soc_low", "sec_fpr"))
    with open(out, "a") as f:
        for rate in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            for seed in seeds:
                cfg = E.Config(n_edge=N, task_rate_per_device=rate)
                for algo in algos:
                    if ("lambda", seed, algo, rate, None, None) in done: continue
                    emit(f, E.run(algo, cfg, seed, ppo_weights=ppo), {"exp": "lambda", "rate": rate})
            print(f"[sens-lambda] rate={rate} ({time.time()-t0:.0f}s)", flush=True)
        for bws in [0.1, 0.25, 0.5, 0.75, 1.0]:
            for seed in seeds:
                cfg = E.Config(n_edge=N, bw_scale=bws)
                for algo in algos:
                    if ("bandwidth", seed, algo, None, bws, None) in done: continue
                    emit(f, E.run(algo, cfg, seed, ppo_weights=ppo), {"exp": "bandwidth", "bw_scale": bws})
            print(f"[sens-bw] bw_scale={bws} ({time.time()-t0:.0f}s)", flush=True)
        for fpr in [0.0, 0.05, 0.10, 0.20]:
            for seed in seeds:
                if ("fpr", seed, "ECO_EDGE", None, None, None, fpr) in done_fpr: continue
                cfg = E.Config(n_edge=N, sec_fpr=fpr)
                emit(f, E.run("ECO_EDGE", cfg, seed), {"exp": "fpr", "sec_fpr": fpr})
            print(f"[sens-fpr] fpr={fpr} ({time.time()-t0:.0f}s)", flush=True)
        for lo, hi in [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8), (0.5, 0.9)]:
            for seed in seeds:
                if ("soc", seed, "ECO_EDGE", None, None, lo) in done: continue
                cfg = E.Config(n_edge=N, soc_low=lo, soc_high=hi)
                emit(f, E.run("ECO_EDGE", cfg, seed), {"exp": "soc", "soc_low": lo, "soc_high": hi})
            print(f"[sens-soc] ({lo},{hi}) ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--ppo", default="ppo_weights.npz")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scales", default=",".join(map(str, SCALES)))
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    seeds = SEEDS[: a.seeds]
    ppo = load_ppo(a.ppo) if os.path.exists(a.ppo) else None
    if a.exp in ("scale", "all"):
        exp_scale(f"{a.outdir}/scale.jsonl", ppo, seeds,
                  [int(x) for x in a.scales.split(",")], "a")
    if a.exp in ("ablation", "all"):
        exp_ablation(f"{a.outdir}/ablation.jsonl", seeds)
    if a.exp in ("sensitivity", "all"):
        exp_sensitivity(f"{a.outdir}/sensitivity.jsonl", ppo, seeds)
