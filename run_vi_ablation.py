import sys, json, time, os
repo, out, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, repo)
import ecoedge_sim as E
with open(out, "a") as f:
    for seed in range(1, seeds + 1):
        t0 = time.time()
        cfg = E.Config(n_edge=1000, soc_low=0.124, soc_high=0.70)
        r = E.run("ECO_EDGE", cfg, seed, label="H_vi_optimal")
        d = E.result_to_dict(r); d["exp"] = "ablation"
        f.write(json.dumps(d) + "\n"); f.flush()
        print(seed, f"{r.success_rate:.4f}", f"{r.energy_think_J/1e3:.2f}", f"{time.time()-t0:.0f}s", flush=True)
