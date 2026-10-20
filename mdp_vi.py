"""Value iteration for the Eco-Edge tier-selection MDP (paper Section IV-C).

State  s = (B, kappa): battery SoC B in [0,1] (discretised), deadline class kappa in {fast, mid, slow}.
Action a in {0.5B, 1.1B, 2.7B}.
Reward R(s,a) = w1 * mu(a) * P[tau_think(a) <= lambda * tau_max | kappa] - w2 * E_think(a) / (B + eps)
Transition B' = max(0, B - (E_exec + E_think(a) + E_coord) / E_cap); kappa' ~ task mix (i.i.d.).
B = 0 is absorbing (node offline, no further reward).
"""
import json, sys
import numpy as np

TIERS = [dict(name="0.5B", E=0.15, tau=22.0, mu=0.65),
         dict(name="1.1B", E=0.31, tau=45.0, mu=0.82),
         dict(name="2.7B", E=0.84, tau=120.0, mu=0.95)]
# Deadline classes from the simulator task mix: steering 40% U(50,150), security 20% U(300,600), sensing 40% U(100,300)
# fast: tau_max < 200, mid: 200 <= tau_max < 400, slow: >= 400
KAPPA = ["fast", "mid", "slow"]
LAM = 0.30            # inference latency may consume at most 30% of the deadline
W1 = 1.0
EPS = 0.05
E_EXEC = 0.6          # J, mean task-execution energy charged to the origin per decision (Table 6: 29.7 kJ / 49,750 requests)
E_COORD = 0.002       # J, 1 KB at 1 W over 10 Mbps
E_CAP = 3.0 * 3600    # J, mean battery budget (1-5 Wh)
NB = 400              # SoC bins

rng = np.random.default_rng(0)
M = 400000
typ = rng.choice(3, size=M, p=[0.4, 0.2, 0.4])
lo = np.array([50, 300, 100])[typ]; hi = np.array([150, 600, 300])[typ]
dl = rng.uniform(lo, hi)
kap = np.where(dl < 200, 0, np.where(dl < 400, 1, 2))
P_KAPPA = np.bincount(kap, minlength=3) / M
# g(a, kappa) = mu(a) * P[tau_think(a) <= LAM * tau_max | kappa]
G = np.array([[t["mu"] * np.mean(t["tau"] <= LAM * dl[kap == k]) for k in range(3)] for t in TIERS])

def greedy_w2_for_upper_threshold(B_high=0.7):
    # calibrate w2 so that, for the slow class, greedy switches 2.7B -> 1.1B exactly at B_high
    dg = G[2, 2] - G[1, 2]; dE = TIERS[2]["E"] - TIERS[1]["E"]
    return W1 * dg * (B_high + EPS) / dE

def reward(Bv, a, k, w2):
    return W1 * G[a, k] - w2 * TIERS[a]["E"] / (Bv + EPS)

def greedy_policy(w2):
    B = (np.arange(NB) + 0.5) / NB
    pol = np.zeros((NB, 3), dtype=int)
    for k in range(3):
        R = np.stack([reward(B, a, k, w2) for a in range(3)], 1)
        pol[:, k] = R.argmax(1)
    return pol

def value_iteration(w2, gamma, tol=1e-9):
    B = (np.arange(NB) + 0.5) / NB
    V = np.zeros(NB)           # V(B) averaged over kappa; B=0 absorbing with V=0
    Q = np.zeros((NB, 3, 3))
    for it in range(100000):
        Vn = np.zeros(NB)
        for a, t in enumerate(TIERS):
            drain = (E_EXEC + t["E"] + E_COORD) / E_CAP
            Bn = B - drain
            idx = np.floor(Bn * NB).astype(int)
            Vnext = np.where(idx < 0, 0.0, V[np.clip(idx, 0, NB - 1)])
            for k in range(3):
                Q[:, k, a] = reward(B, a, k, w2) + gamma * Vnext
        Vn = (Q.max(2) * P_KAPPA).sum(1)
        if np.abs(Vn - V).max() < tol:
            V = Vn; break
        V = Vn
    pol = Q.argmax(2)
    return V, Q, pol, it

def policy_value(pol, w2, gamma, tol=1e-9):
    B = (np.arange(NB) + 0.5) / NB
    V = np.zeros(NB)
    for it in range(100000):
        Vn = np.zeros(NB)
        for k in range(3):
            for a, t in enumerate(TIERS):
                m = pol[:, k] == a
                drain = (E_EXEC + t["E"] + E_COORD) / E_CAP
                idx = np.floor((B - drain) * NB).astype(int)
                Vnext = np.where(idx < 0, 0.0, V[np.clip(idx, 0, NB - 1)])
                Vn[m] += P_KAPPA[k] * (reward(B[m], a, k, w2) + gamma * Vnext[m])
        if np.abs(Vn - V).max() < tol:
            return Vn
        V = Vn
    return V

def thresholds(pol):
    """Return, per class, the SoC levels at which the chosen tier changes."""
    B = (np.arange(NB) + 0.5) / NB
    out = {}
    for k in range(3):
        ch = [(round(float(B[i]), 3), TIERS[pol[i - 1, k]]["name"], TIERS[pol[i, k]]["name"])
              for i in range(1, NB) if pol[i, k] != pol[i - 1, k]]
        out[KAPPA[k]] = ch
    return out

if __name__ == "__main__":
    out = sys.argv[1]
    w2 = greedy_w2_for_upper_threshold(0.7)
    res = {"lambda": LAM, "w1": W1, "w2": float(w2), "eps": EPS, "E_exec": E_EXEC, "E_cap": E_CAP,
           "P_kappa": P_KAPPA.tolist(), "G": G.tolist(), "nbins": NB}
    gp = greedy_policy(w2)
    res["greedy_thresholds"] = thresholds(gp)
    # paper's configured policy (0.3, 0.7) with deadline cap
    B = (np.arange(NB) + 0.5) / NB
    cfg_pol = np.zeros((NB, 3), dtype=int)
    for k in range(3):
        socT = np.where(B > 0.7, 2, np.where(B >= 0.3, 1, 0))
        cfg_pol[:, k] = np.minimum(socT, k)
    for gamma in (0.99, 0.999, 0.9999):
        V, Q, pol, it = value_iteration(w2, gamma)
        Vg = policy_value(gp, w2, gamma); Vc = policy_value(cfg_pol, w2, gamma)
        res[f"gamma_{gamma}"] = {
            "iters": it,
            "vi_thresholds": thresholds(pol),
            "policy_diff_vs_greedy_frac": float((pol != gp).mean()),
            "policy_diff_vs_configured_frac": float((pol != cfg_pol).mean()),
            "value_gap_greedy_pct": float(np.mean((V - Vg) / np.abs(V)) * 100),
            "value_gap_configured_pct": float(np.mean((V - Vc) / np.abs(V)) * 100),
            "max_value_gap_greedy_pct": float(np.max((V - Vg) / np.abs(V)) * 100),
            "max_value_gap_configured_pct": float(np.max((V - Vc) / np.abs(V)) * 100),
        }
        np.save(out.replace(".json", f"_policy_g{gamma}.npy"), pol)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
