"""
PPO_FIT baseline: clipped policy-gradient actor-critic trained on the
Eco-Edge simulator.  Pure-numpy implementation (no torch dependency) using
the hyper-parameters reported in the paper's PPO table:

  policy MLP [128,128] (tanh), lr 3e-4, clip 0.2, GAE lambda 0.95,
  n_steps 2048, batch 64, gamma 0.99, 10 epochs, entropy coef 0.01,
  reward r = w_s*success - w_tau*(tau/tau_max) - w_e*energy,
  (w_s, w_tau, w_e) = (1.0, 0.4, 0.3)

Usage:  python ppo_train.py --steps 300000 --n-edge 200 --out ppo_weights.npz
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ecoedge_sim as E


# ---------------------------------------------------------------------------
class MLP:
    """2-hidden-layer tanh MLP with manual backprop and Adam."""

    def __init__(self, d_in, d_out, hidden=128, rng=None, out_scale=0.01):
        rng = rng or np.random.default_rng(0)
        s1 = np.sqrt(2.0 / (d_in + hidden)); s2 = np.sqrt(2.0 / (2 * hidden))
        self.p = {
            "W1": rng.normal(0, s1, (d_in, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, s2, (hidden, hidden)), "b2": np.zeros(hidden),
            "Wp": rng.normal(0, out_scale, (hidden, d_out)), "bp": np.zeros(d_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        out = h2 @ p["Wp"] + p["bp"]
        return out, (X, h1, h2)

    def backward(self, cache, dout):
        X, h1, h2 = cache
        p = self.p
        g = {}
        g["Wp"] = h2.T @ dout; g["bp"] = dout.sum(0)
        dh2 = (dout @ p["Wp"].T) * (1 - h2 ** 2)
        g["W2"] = h1.T @ dh2; g["b2"] = dh2.sum(0)
        dh1 = (dh2 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"] = X.T @ dh1; g["b1"] = dh1.sum(0)
        return g

    def adam(self, g, lr, b1=0.9, b2=0.999, eps=1e-8, max_norm=0.5):
        self.t += 1
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        if norm > max_norm:
            g = {k: v * (max_norm / norm) for k, v in g.items()}
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
class TaskEnv:
    """Sequential decision environment over one simulation run."""

    def __init__(self, cfg: E.Config, seed: int):
        self.cfg = cfg
        self.env = E.Env(cfg, seed)
        self.T = self.env.tasks
        self.k = 0
        self.snap = None; self.snap_slot = -1
        self.w = (1.0, 0.4, 0.3)
        self.DECISION_MS = E.PPOAgent.DECISION_MS

    def _snapshot(self, slot, t):
        if self.snap_slot != slot:
            self.snap = np.array([self.env.qlen(s, t) for s in range(self.env.S)])
            self.snap_slot = slot

    def _skip_offline(self):
        while self.k < self.T["n"] and self.env.offline[int(self.T["origin"][self.k])]:
            self.k += 1

    def obs(self):
        self._skip_offline()
        if self.k >= self.T["n"]:
            return None
        k = self.k; t = self.T["arr"][k]
        self._snapshot(int(t // self.cfg.slot_ms), t)
        o = int(self.T["origin"][k])
        self.cands = E._candidates(self.env, o)
        return E._features(self.env, k, self.cands, self.snap, t + self.DECISION_MS)

    def step(self, a):
        k = self.k; t = self.T["arr"][k]
        self.env.energy_think += E.PPOAgent.DECISION_J
        e0 = self.env.energy_exec
        code, lat = self.env.execute(k, self.cands[a], t + self.DECISION_MS)
        ws, wt, we = self.w
        r = ws * (1.0 if code == E.FAIL_NONE else 0.0) \
            - wt * min(lat / self.T["deadline"][k], 3.0) \
            - we * min((self.env.energy_exec - e0) / 5.0, 3.0)
        self.k += 1
        self.succ = getattr(self, "succ", 0) + (code == E.FAIL_NONE)
        return r, self.k >= self.T["n"]


def train(args):
    rng = np.random.default_rng(args.seed)
    cfg = E.Config(n_edge=args.n_edge)
    probe = E.Env(cfg, 0)
    d_in = E.obs_dim(probe); n_act = len(E._candidates(probe, 0))
    actor = MLP(d_in, n_act, rng=rng); critic = MLP(d_in, 1, rng=rng, out_scale=1.0)
    lr, clip, gamma, lam, n_steps, batch, epochs, ent_coef = 3e-4, 0.2, 0.99, 0.95, 2048, 64, 10, 0.01

    ep_seed = 1000
    env = TaskEnv(cfg, ep_seed); ob = env.obs()
    total, t0, log = 0, time.time(), []
    ep_ret, ep_rets = 0.0, []
    while total < args.steps:
        O = np.zeros((n_steps, d_in)); A = np.zeros(n_steps, int); R = np.zeros(n_steps)
        D = np.zeros(n_steps); V = np.zeros(n_steps + 1); LP = np.zeros(n_steps)
        for i in range(n_steps):
            lg, _ = actor.forward(ob[None]); pi = softmax(lg)[0]
            a = int(rng.choice(n_act, p=pi))
            v, _ = critic.forward(ob[None])
            r, done = env.step(a)
            O[i], A[i], R[i], V[i], LP[i] = ob, a, r, v[0, 0], np.log(pi[a] + 1e-12)
            ep_ret += r
            if done:
                ep_rets.append((ep_ret, env.succ / env.T["n"])); ep_ret = 0.0
                ep_seed += 1; env = TaskEnv(cfg, ep_seed); ob = env.obs(); D[i] = 1.0
            else:
                ob = env.obs()
        V[n_steps] = critic.forward(ob[None])[0][0, 0]
        # GAE
        adv = np.zeros(n_steps); last = 0.0
        for i in reversed(range(n_steps)):
            nonterm = 1.0 - D[i]
            delta = R[i] + gamma * V[i + 1] * nonterm - V[i]
            last = delta + gamma * lam * nonterm * last
            adv[i] = last
        ret = adv + V[:n_steps]
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        # PPO epochs
        for _ in range(epochs):
            perm = rng.permutation(n_steps)
            for j in range(0, n_steps, batch):
                idx = perm[j:j + batch]
                lg, ca = actor.forward(O[idx]); pi = softmax(lg)
                logp = np.log(pi[np.arange(len(idx)), A[idx]] + 1e-12)
                ratio = np.exp(logp - LP[idx])
                a_ = advn[idx]
                unclipped = ratio * a_; clipped = np.clip(ratio, 1 - clip, 1 + clip) * a_
                use_unclipped = unclipped <= clipped
                # d(-min)/dlogp
                dlogp = -np.where(use_unclipped, ratio * a_, 0.0) / len(idx)
                # entropy bonus gradient: H = -sum pi log pi; dH/dlogits = -pi*(log pi + H)
                H = -(pi * np.log(pi + 1e-12)).sum(1)
                dlogits = np.zeros_like(pi)
                dlogits[np.arange(len(idx)), A[idx]] += dlogp
                dlogits -= pi * dlogp[:, None]                    # softmax Jacobian
                dlogits += -ent_coef * (-pi * (np.log(pi + 1e-12) + H[:, None])) / len(idx)
                actor.adam(actor.backward(ca, dlogits), lr)
                v, cc = critic.forward(O[idx])
                dv = (v[:, 0] - ret[idx])[:, None] / len(idx)      # 0.5*(v-R)^2
                critic.adam(critic.backward(cc, dv), lr)
        total += n_steps
        if ep_rets:
            log.append((total, float(np.mean([e[0] for e in ep_rets[-3:]])), float(np.mean([e[1] for e in ep_rets[-3:]]))))
            print(f"steps={total:7d}  ep_return={log[-1][1]:8.1f}  ep_success={log[-1][2]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(args.out, **actor.p)
    json.dump({"steps": args.steps, "n_edge": args.n_edge, "seed": args.seed, "log": log,
               "hyper": dict(lr=lr, clip=clip, gamma=gamma, gae_lambda=lam, n_steps=n_steps,
                             batch=batch, epochs=epochs, ent_coef=ent_coef, net_arch=[128, 128])},
              open(args.out.replace(".npz", "_log.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-edge", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ppo_weights.npz")
    train(ap.parse_args())
