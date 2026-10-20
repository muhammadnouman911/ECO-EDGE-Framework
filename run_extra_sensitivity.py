"""Extra sensitivity sweeps (OJ-COMS revision): digital-twin state error and SLM grounding accuracy.
Usage: python run_extra_sensitivity.py results/sensitivity_extra.jsonl [seeds=20]
"""
import sys, json, time
import ecoedge_sim as E

out = sys.argv[1]; seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N = 1000
MU_SETS = {"mu_low": (0.50, 0.70, 0.85), "mu_default": (0.65, 0.82, 0.95), "mu_high": (0.80, 0.90, 0.98), "mu_perfect": (1.0, 1.0, 1.0)}
with open(out, "a") as f:
    for tn in (0.0, 0.05, 0.10, 0.20, 0.40):
        t0 = time.time()
        for seed in range(1, seeds + 1):
            r = E.run("ECO_EDGE", E.Config(n_edge=N, twin_noise=tn), seed)
            d = E.result_to_dict(r); d.update(exp="twin_noise", twin_noise=tn); f.write(json.dumps(d) + "\n"); f.flush()
        print(f"[twin_noise={tn}] done ({time.time()-t0:.0f}s)", flush=True)
    base = {k: dict(v) for k, v in E.TIERS.items()}
    for name, accs in MU_SETS.items():
        t0 = time.time()
        for tier, acc in zip(E.TIER_ORDER, accs):
            E.TIERS[tier]["acc"] = acc
        for seed in range(1, seeds + 1):
            r = E.run("ECO_EDGE", E.Config(n_edge=N), seed)
            d = E.result_to_dict(r); d.update(exp="mu", mu_set=name, mu=list(accs)); f.write(json.dumps(d) + "\n"); f.flush()
        print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)
    for k, v in base.items(): E.TIERS[k].update(v)
