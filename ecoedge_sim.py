"""
ECO-EDGE discrete-event simulator (mechanistic version).

All outcomes (success, latency, capacity failures, energy) emerge from
queueing, transfer, compute, deadline and battery mechanics.  No orchestrator
is assigned a hard-coded "success probability".

Orchestrators implemented:
  ECO_EDGE   role-affinity ranking + digital-twin prediction + critic
             re-planning + security gate + battery-aware elastic tier
  NEXT_FIT   centralised round-robin over all nodes with free capacity
  RANDOM_FIT centralised uniform choice among nodes with free capacity
  DQN_FIT    tabular Q-learning over discretised local state (online)
  PPO_FIT    clipped policy-gradient actor-critic (numpy implementation,
             trained offline with ppo_train.py, weights loaded here)

Ablation flags allow removing individual Eco-Edge components.
"""
from __future__ import annotations

import math
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Constants: task profiles, SLM tiers
# ----------------------------------------------------------------------------
N_TYPES = 3
TYPE_NAMES = ["traffic_steering", "security_critical", "routine_sensor"]
TYPE_PROB = [0.4, 0.2, 0.4]
# (data_MB lo, hi), (compute GFLOP lo, hi), (deadline ms lo, hi)
TYPE_PROFILE = {
    0: ((0.02, 0.2), (0.02, 0.2), (50, 150)),
    1: ((0.2, 1.0), (0.5, 2.0), (300, 600)),
    2: ((0.01, 0.1), (0.01, 0.1), (100, 300)),
}

# SLM tiers used by Elastic Intelligence.  acc = intent-grounding accuracy
# mu_acc(a) of each tier (values from the prototype's slm_config profile);
# with probability 1-acc the SLM mis-grounds the request and the planner
# works from a distorted deadline estimate.  Energy (J/inference) and latency
# (ms) are the Jetson-Orin-Nano-class figures used in the paper's hardware
# table; obs_noise is the relative error of the tier's load estimate (proxy
# for reasoning quality; smaller model -> noisier state assessment).
TIER_ORDER = ["0.5B", "1.1B", "2.7B"]
TIERS = {
    "0.5B": dict(energy=0.15, latency=22.0, noise=0.25, acc=0.65),
    "1.1B": dict(energy=0.31, latency=45.0, noise=0.12, acc=0.82),
    "2.7B": dict(energy=0.84, latency=120.0, noise=0.05, acc=0.95),
}

FAIL_NONE, FAIL_DEADLINE, FAIL_CAPACITY, FAIL_SECURITY, FAIL_OFFLINE, FAIL_REJECT = range(6)
FAIL_NAMES = ["ok", "deadline", "capacity", "security_block", "node_offline", "critic_reject"]


@dataclass
class Config:
    n_edge: int = 200
    slot_ms: float = 200.0
    n_slots: int = 500
    task_rate_per_device: float = 0.5      # tasks/s per device
    k_neighbors: int = 6
    edge_qmax: int = 4
    mec_qmax: int = 16
    n_mec: int = 4                         # fixed MEC infrastructure (scale = device density)
    ap_bw_mbps: float = 500.0              # shared uplink capacity per MEC access point
    battery_fraction: float = 0.4
    p_adv: float = 0.02                    # fraction of adversarial intents
    sec_detect: float = 0.80               # measured: rule-based Security-Critic, 55 adversarial prompts (44/55)
    sec_fpr: float = 0.00                  # measured: 0/30 benign prompts flagged (see sensitivity sweep)
    bw_scale: float = 1.0                  # multiplies uplink bandwidth (sensitivity)
    soc_low: float = 0.3                   # elastic-intelligence thresholds
    soc_high: float = 0.7
    max_replans: int = 3
    dl_fast: float = 200.0                 # deadline (ms) below which only the 0.5B tier is used
    dl_mid: float = 400.0                  # deadline (ms) below which at most the 1.1B tier is used
    scrutiny_ms: float = 45.0              # extra delay for benign requests flagged by the security gate
    # Eco-Edge ablation switches
    use_elastic: bool = True
    use_security: bool = True
    use_critic: bool = True
    use_twin: bool = True
    use_affinity: bool = True
    fixed_tier: str = "2.7B"               # tier when elastic is off


@dataclass
class Result:
    algo: str
    n_edge: int
    seed: int
    n_tasks: int
    success: int
    fail_deadline: int
    fail_capacity: int
    fail_security: int
    fail_offline: int
    fail_reject: int
    adv_executed: int
    adv_total: int
    mean_latency_ms: float        # completed tasks
    p95_latency_ms: float
    energy_exec_J: float          # compute + communication of tasks
    energy_think_J: float         # SLM / policy inference energy
    energy_coord_J: float         # coordination messaging energy
    nodes_depleted: int
    tier_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self):
        return self.success / self.n_tasks


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class Env:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        N = cfg.n_edge
        self.N = N
        # --- edge nodes ---
        self.cpu = r.uniform(2.0, 8.0, N)                 # GFLOPS
        self.bw = r.uniform(10.0, 100.0, N) * cfg.bw_scale  # Mbps uplink / local link
        self.p_act = r.uniform(5.0, 15.0, N)              # W while computing
        self.p_tx = r.uniform(0.5, 2.0, N)                # W while transmitting
        self.spec = r.integers(0, N_TYPES, N)             # accelerator specialisation
        self.is_batt = r.random(N) < cfg.battery_fraction
        self.cap_J = r.uniform(1.0, 5.0, N) * 3600.0      # residual budget 1-5 Wh
        self.soc = np.where(self.is_batt, r.uniform(0.05, 1.0, N), 1.0)
        self.offline = np.zeros(N, dtype=bool)
        self.pos = r.uniform(0, 1000, (N, 2))
        # --- MEC servers ---
        M = cfg.n_mec
        self.M = M
        self.mec_pos = r.uniform(0, 1000, (M, 2))
        self.mec_cpu = r.uniform(50.0, 100.0, M)          # effective GFLOPS per task
        self.mec_p = r.uniform(80.0, 150.0, M)
        d = np.linalg.norm(self.pos[:, None, :] - self.mec_pos[None, :, :], axis=2)
        order = np.argsort(d, axis=1)
        self.home_mec = order[:, 0]
        self.backup_mec = order[:, 1] if M > 1 else order[:, 0]
        # --- neighbours ---
        de = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        np.fill_diagonal(de, np.inf)
        k = min(cfg.k_neighbors, N - 1)
        self.nbrs = np.argsort(de, axis=1)[:, :k]
        # --- servers: index 0..N-1 edge, N..N+M-1 MEC ---
        S = N + M
        self.S = S
        self.busy_until = np.zeros(S)
        self.fin = [deque() for _ in range(S)]           # finish times of queued tasks
        self.qmax = np.array([cfg.edge_qmax] * N + [cfg.mec_qmax] * M)
        self.energy_exec = 0.0
        self.energy_think = 0.0
        self.energy_coord = 0.0
        self.tier_counts = {t: 0 for t in TIERS}
        # tasks
        self.tasks = self._gen_tasks()

    # ------------------------------------------------------------------
    def _gen_tasks(self):
        cfg, r = self.cfg, self.rng
        T = cfg.n_slots * cfg.slot_ms
        lam = cfg.task_rate_per_device * T / 1000.0       # expected tasks per device
        counts = r.poisson(lam, self.N)
        total = int(counts.sum())
        origin = np.repeat(np.arange(self.N), counts)
        arr = r.uniform(0, T, total)
        idx = np.argsort(arr)
        origin, arr = origin[idx], arr[idx]
        ttype = r.choice(N_TYPES, total, p=TYPE_PROB)
        data = np.empty(total); comp = np.empty(total); dl = np.empty(total)
        for t in range(N_TYPES):
            m = ttype == t
            (dlo, dhi), (clo, chi), (llo, lhi) = TYPE_PROFILE[t]
            n = int(m.sum())
            data[m] = r.uniform(dlo, dhi, n)
            comp[m] = r.uniform(clo, chi, n)
            dl[m] = r.uniform(llo, lhi, n)
        adv = r.random(total) < cfg.p_adv
        comp[adv] *= 10.0                                  # resource-exhaustion intent
        return dict(origin=origin, arr=arr, type=ttype, data=data, comp=comp,
                    deadline=dl, adv=adv, n=total)

    # ------------------------------------------------------------------
    def qlen(self, s: int, t: float) -> int:
        q = self.fin[s]
        while q and q[0] <= t:
            q.popleft()
        return len(q)

    def proc_ms(self, s: int, comp: float, ttype: int) -> float:
        if s < self.N:
            f = 1.0 if self.spec[s] == ttype else 1.6      # no matching accelerator
            return comp / self.cpu[s] * 1000.0 * f
        return comp / self.mec_cpu[s - self.N] * 1000.0 * 0.8

    def transfer_ms(self, origin: int, s: int, data_mb: float, global_route: bool = False):
        """Transfer delay (ms) and energy (J) from origin to server s."""
        if s == origin:
            return 0.0, 0.0
        bits = data_mb * 8e6
        if s >= self.N:                                   # uplink to MEC via shared AP
            share = self.cfg.ap_bw_mbps / (1.0 + len(self.fin[s]))
            b = min(self.bw[origin], share)
            t = bits / (b * 1e6) * 1000.0 + 5.0
            return t, self.p_tx[origin] * t / 1000.0
        if global_route:                                  # via backhaul: up + down
            t = bits / (self.bw[origin] * 1e6) * 1000.0 + bits / (self.bw[s] * 1e6) * 1000.0 + 10.0
            return t, self.p_tx[origin] * t / 1000.0
        b = min(self.bw[origin], self.bw[s])              # direct local link
        t = bits / (b * 1e6) * 1000.0 + 2.0
        return t, self.p_tx[origin] * t / 1000.0

    def predict(self, origin: int, s: int, k: int, t_ready: float, qlen_obs: int,
                global_route: bool = False) -> Tuple[float, float, float]:
        """Predicted (latency_ms, proc_ms, transfer_ms) for placing task k on s."""
        tr, _ = self.transfer_ms(origin, s, self.tasks["data"][k], global_route)
        proc = self.proc_ms(s, self.tasks["comp"][k], self.tasks["type"][k])
        wait = max(0.0, self.busy_until[s] - (t_ready + tr))
        return tr + wait + proc, proc, tr

    def execute(self, k: int, s: int, t_ready: float, global_route: bool = False):
        """Place task k on server s.  Returns (fail_code, latency_ms)."""
        T = self.tasks
        origin = T["origin"][k]
        if s < self.N and self.offline[s]:
            return FAIL_OFFLINE, 0.0
        tr, e_tr = self.transfer_ms(origin, s, T["data"][k], global_route)
        self.energy_exec += e_tr
        if self.is_batt[origin]:
            self._drain(origin, e_tr)
        t_arr = t_ready + tr
        if self.qlen(s, t_arr) >= self.qmax[s]:
            return FAIL_CAPACITY, 0.0
        proc = self.proc_ms(s, T["comp"][k], T["type"][k])
        start = max(t_arr, self.busy_until[s])
        fin = start + proc
        self.busy_until[s] = fin
        self.fin[s].append(fin)
        p = self.p_act[s] if s < self.N else self.mec_p[s - self.N]
        e = p * proc / 1000.0
        self.energy_exec += e
        if s < self.N and self.is_batt[s]:
            self._drain(s, e)
        lat = fin - T["arr"][k]
        return (FAIL_NONE if lat <= T["deadline"][k] else FAIL_DEADLINE), lat

    def _drain(self, i: int, e_J: float):
        self.soc[i] -= e_J / self.cap_J[i]
        if self.soc[i] <= 0.0:
            self.soc[i] = 0.0
            self.offline[i] = True

    def coord_msg(self, i: int, n_bytes: float):
        t = n_bytes * 8 / (self.bw[i] * 1e6)
        self.energy_coord += self.p_tx[i] * t


# ----------------------------------------------------------------------------
# Orchestrators
# ----------------------------------------------------------------------------
def _sim(spec_match: bool, is_mec: bool) -> float:
    if is_mec:
        return 0.8
    return 1.0 if spec_match else 0.3


class EcoEdge:
    name = "ECO_EDGE"

    def __init__(self, env: Env, cfg: Config):
        self.env, self.cfg = env, cfg
        self.rng = np.random.default_rng(env.rng.integers(1 << 30))

    def tier_for(self, i: int, deadline: float) -> str:
        """Elastic Intelligence: tier = min(SoC-permitted tier, deadline-permitted tier)."""
        if not self.cfg.use_elastic:
            return self.cfg.fixed_tier
        # deadline-permitted tier: inference latency must leave slack for execution
        if deadline < self.cfg.dl_fast:
            dl_tier = 0
        elif deadline < self.cfg.dl_mid:
            dl_tier = 1
        else:
            dl_tier = 2
        if not self.env.is_batt[i]:
            soc_tier = 2
        else:
            b = self.env.soc[i]
            soc_tier = 2 if b > self.cfg.soc_high else (1 if b >= self.cfg.soc_low else 0)
        return TIER_ORDER[min(dl_tier, soc_tier)]

    def decide(self, k: int, slot: int, t: float):
        env, cfg, T = self.env, self.cfg, self.env.tasks
        o = int(T["origin"][k])
        if env.offline[o]:
            return FAIL_OFFLINE, 0.0
        # --- Act: one SLM inference per orchestration request ---
        tier = self.tier_for(o, T["deadline"][k])
        env.energy_think += TIERS[tier]["energy"]
        if env.is_batt[o]:
            env._drain(o, TIERS[tier]["energy"])
        env.tier_counts[tier] += 1
        t_ready = t + TIERS[tier]["latency"]
        env.coord_msg(o, 1024)                  # compressed state vector to regional twin
        dl_perceived = T["deadline"][k]
        if self.rng.random() > TIERS[tier]["acc"]:       # intent mis-grounding
            dl_perceived *= self.rng.uniform(0.5, 2.0)
        # --- Security gate (metadata/embedding checks, no extra inference) ---
        if cfg.use_security:
            if T["adv"][k]:
                if self.rng.random() < cfg.sec_detect:
                    return FAIL_SECURITY, 0.0
            elif self.rng.random() < cfg.sec_fpr:
                t_ready += cfg.scrutiny_ms        # false positive: enhanced scrutiny delay
        # --- candidate set: local, neighbours, home + backup MEC ---
        cands = [o] + [int(x) for x in env.nbrs[o]] + [env.N + int(env.home_mec[o])]
        if env.M > 1:
            cands.append(env.N + int(env.backup_mec[o]))
        noise = TIERS[tier]["noise"]
        ttype = int(T["type"][k])
        scored = []
        for s in cands:
            if s < env.N and env.offline[s]:
                continue
            is_mec = s >= env.N
            # observed queue: twin gives exact regional state; otherwise a noisy
            # tier-dependent estimate of the slot-start snapshot
            q_true = env.qlen(s, t_ready)
            if cfg.use_twin:
                q_obs = q_true
                busy_obs = env.busy_until[s]
            else:
                q_obs = max(0.0, q_true * (1.0 + self.rng.normal(0, noise)))
                busy_obs = env.busy_until[s] * (1.0 + self.rng.normal(0, noise))
            avail = 1.0 - min(1.0, q_obs / env.qmax[s])
            hops = 0 if s == o else (2 if is_mec else 1)
            if cfg.use_affinity:
                score = 0.4 * _sim(env.spec[s] == ttype if not is_mec else False, is_mec) \
                    + 0.4 * avail + 0.2 / (1.0 + hops)
            else:
                score = avail + 0.001 / (1.0 + hops)     # least-loaded, nearest tie-break
            tr, _ = env.transfer_ms(o, s, T["data"][k])
            proc = env.proc_ms(s, T["comp"][k], ttype)
            wait = max(0.0, busy_obs - (t_ready + tr))
            pred_lat = tr + wait + proc
            scored.append((score, s, pred_lat, q_obs))
        if not scored:
            return FAIL_OFFLINE, 0.0
        scored.sort(key=lambda x: -x[0])
        # --- critic: accept first candidate whose predicted KPIs satisfy constraints
        chosen = scored[0][1]
        if cfg.use_critic:
            dl = dl_perceived
            accepted = None
            for score, s, pred_lat, q_obs in scored[: cfg.max_replans + 1]:
                if q_obs < env.qmax[s] and (t_ready - T["arr"][k]) + pred_lat <= dl:
                    accepted = s
                    break
            if accepted is None:
                # no feasible placement within R re-plans: fall back to the
                # candidate with the lowest predicted latency that has capacity
                feas = [x for x in scored if x[3] < env.qmax[x[1]]]
                if not feas:
                    return FAIL_REJECT, 0.0
                accepted = min(feas, key=lambda x: x[2])[1]
            chosen = accepted
        return env.execute(k, chosen, t_ready)


class NextFit:
    name = "NEXT_FIT"
    DECISION_MS = 1.0
    DECISION_J = 0.001

    def __init__(self, env: Env, cfg: Config):
        self.env, self.cfg = env, cfg
        self.ptr = 0
        self.snap = None
        self.snap_slot = -1

    def _snapshot(self, slot, t):
        if self.snap_slot != slot:
            self.snap = np.array([self.env.qlen(s, t) for s in range(self.env.S)])
            self.snap_slot = slot

    def decide(self, k, slot, t):
        env, T = self.env, self.env.tasks
        o = int(T["origin"][k])
        if env.offline[o]:
            return FAIL_OFFLINE, 0.0
        self._snapshot(slot, t)
        env.energy_think += self.DECISION_J
        env.coord_msg(o, 2048)                       # telemetry + assignment
        S = env.S
        for step in range(S):
            s = (self.ptr + step) % S
            if s < env.N and env.offline[s]:
                continue
            if self.snap[s] < env.qmax[s]:
                self.ptr = (s + 1) % S
                self.snap[s] += 0                    # stale snapshot: not updated
                return env.execute(k, s, t + self.DECISION_MS, global_route=(s != o and s < env.N))
        return FAIL_CAPACITY, 0.0


class RandomFit(NextFit):
    name = "RANDOM_FIT"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.rng = np.random.default_rng(env.rng.integers(1 << 30))

    def decide(self, k, slot, t):
        env, T = self.env, self.env.tasks
        o = int(T["origin"][k])
        if env.offline[o]:
            return FAIL_OFFLINE, 0.0
        self._snapshot(slot, t)
        env.energy_think += self.DECISION_J
        env.coord_msg(o, 2048)
        ok = np.flatnonzero((self.snap < env.qmax) & ~np.concatenate([env.offline, np.zeros(env.M, bool)]))
        if len(ok) == 0:
            return FAIL_CAPACITY, 0.0
        s = int(self.rng.choice(ok))
        return env.execute(k, s, t + self.DECISION_MS, global_route=(s != o and s < env.N))


def _candidates(env: Env, o: int) -> List[int]:
    c = [o] + [int(x) for x in env.nbrs[o]] + [env.N + int(env.home_mec[o])]
    if env.M > 1:
        c.append(env.N + int(env.backup_mec[o]))
    return c


def _features(env: Env, k: int, cands: List[int], snap: np.ndarray, t_ready: float) -> np.ndarray:
    """Observation used by learning-based baselines (same information as a
    slot-start telemetry snapshot; no digital twin)."""
    T = env.tasks
    o = int(T["origin"][k]); ttype = int(T["type"][k]); dl = T["deadline"][k]
    f = []
    for s in cands:
        q = snap[s] / env.qmax[s]
        proc = env.proc_ms(s, T["comp"][k], ttype) / dl
        tr, _ = env.transfer_ms(o, s, T["data"][k])
        f += [min(q, 1.0), min(proc, 3.0), min(tr / dl, 3.0), float(s >= env.N)]
    onehot = [0.0] * N_TYPES; onehot[ttype] = 1.0
    f += onehot + [dl / 500.0, env.soc[o], float(T["adv"][k]) * 0.0]
    return np.asarray(f, dtype=np.float64)


N_CAND = 8   # 1 local + 6 neighbours + 1 home MEC (+1 backup when M>1 -> 9)
FEAT_PER_CAND = 4


def obs_dim(env: Env) -> int:
    return len(_candidates(env, 0)) * FEAT_PER_CAND + N_TYPES + 3


class TabularQ:
    """Tabular Q-learning scheduler (DQN_FIT): discretised local state, online."""
    name = "DQN_FIT"
    DECISION_MS = 2.0
    DECISION_J = 0.005

    def __init__(self, env: Env, cfg: Config):
        self.env, self.cfg = env, cfg
        self.rng = np.random.default_rng(env.rng.integers(1 << 30))
        self.q: Dict[Tuple, np.ndarray] = {}
        self.alpha, self.gamma = 0.1, 0.95
        self.eps, self.eps_min, self.eps_decay = 1.0, 0.01, 0.995
        self.snap = None; self.snap_slot = -1
        self.n_actions = 4   # local, best neighbour, home MEC, least-loaded candidate

    def _snapshot(self, slot, t):
        if self.snap_slot != slot:
            self.snap = np.array([self.env.qlen(s, t) for s in range(self.env.S)])
            self.snap_slot = slot

    def decide(self, k, slot, t):
        env, T = self.env, self.env.tasks
        o = int(T["origin"][k])
        if env.offline[o]:
            return FAIL_OFFLINE, 0.0
        self._snapshot(slot, t)
        env.energy_think += self.DECISION_J
        env.coord_msg(o, 2048)
        cands = _candidates(env, o)
        nb = cands[1:-1] if env.M > 1 else cands[1:]
        nb = [s for s in nb if s < env.N] or [o]
        best_nb = min(nb, key=lambda s: self.snap[s])
        mec = env.N + int(env.home_mec[o])
        least = min(cands, key=lambda s: self.snap[s] / env.qmax[s])
        actions = [o, best_nb, mec, least]
        ttype = int(T["type"][k])
        b = lambda x, m: min(2, int(3 * x / m))
        state = (ttype, b(T["deadline"][k], 500), b(self.snap[o], env.qmax[o]),
                 b(self.snap[best_nb], env.qmax[best_nb]), b(self.snap[mec], env.qmax[mec]),
                 int(env.spec[o] == ttype))
        qv = self.q.setdefault(state, np.zeros(self.n_actions))
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(self.n_actions))
        else:
            a = int(np.argmax(qv))
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        s = actions[a]
        e0 = env.energy_exec
        code, lat = env.execute(k, s, t + self.DECISION_MS)
        r = (1.0 if code == FAIL_NONE else 0.0) - 0.4 * min(lat / T["deadline"][k], 3.0) \
            - 0.3 * min((env.energy_exec - e0) / 5.0, 3.0)
        qv[a] += self.alpha * (r - qv[a])           # single-step (contextual) update
        return code, lat


class PPOAgent:
    """Clipped policy-gradient scheduler (PPO_FIT).  Inference only; weights
    trained with ppo_train.py (numpy implementation of PPO)."""
    name = "PPO_FIT"
    DECISION_MS = 2.0
    DECISION_J = 0.005

    def __init__(self, env: Env, cfg: Config, weights: dict):
        self.env, self.cfg = env, cfg
        self.w = weights
        self.snap = None; self.snap_slot = -1
        self.rng = np.random.default_rng(env.rng.integers(1 << 30))

    def _snapshot(self, slot, t):
        if self.snap_slot != slot:
            self.snap = np.array([self.env.qlen(s, t) for s in range(self.env.S)])
            self.snap_slot = slot

    def logits(self, x):
        w = self.w
        h1 = np.tanh(x @ w["W1"] + w["b1"])
        h2 = np.tanh(h1 @ w["W2"] + w["b2"])
        return h2 @ w["Wp"] + w["bp"]

    def decide(self, k, slot, t):
        env, T = self.env, self.env.tasks
        o = int(T["origin"][k])
        if env.offline[o]:
            return FAIL_OFFLINE, 0.0
        self._snapshot(slot, t)
        env.energy_think += self.DECISION_J
        env.coord_msg(o, 2048)
        cands = _candidates(env, o)
        x = _features(env, k, cands, self.snap, t + self.DECISION_MS)
        lg = self.logits(x)
        a = int(np.argmax(lg))                      # deterministic policy at evaluation
        return env.execute(k, cands[a], t + self.DECISION_MS)


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def make_orchestrator(algo: str, env: Env, cfg: Config, ppo_weights=None):
    if algo == "ECO_EDGE":
        return EcoEdge(env, cfg)
    if algo == "NEXT_FIT":
        return NextFit(env, cfg)
    if algo == "RANDOM_FIT":
        return RandomFit(env, cfg)
    if algo == "DQN_FIT":
        return TabularQ(env, cfg)
    if algo == "PPO_FIT":
        assert ppo_weights is not None, "PPO weights required"
        return PPOAgent(env, cfg, ppo_weights)
    raise ValueError(algo)


def run(algo: str, cfg: Config, seed: int, ppo_weights=None, label: Optional[str] = None) -> Result:
    env = Env(cfg, seed)
    orch = make_orchestrator(algo, env, cfg, ppo_weights)
    T = env.tasks
    n = T["n"]
    codes = np.zeros(n, dtype=np.int8)
    lats = np.zeros(n)
    adv_exec = 0
    for k in range(n):
        t = T["arr"][k]
        slot = int(t // cfg.slot_ms)
        code, lat = orch.decide(k, slot, t)
        codes[k] = code
        lats[k] = lat
        if T["adv"][k] and code in (FAIL_NONE, FAIL_DEADLINE):
            adv_exec += 1
    ok = codes == FAIL_NONE
    comp_lat = lats[(codes == FAIL_NONE) | (codes == FAIL_DEADLINE)]
    return Result(
        algo=label or algo, n_edge=cfg.n_edge, seed=seed, n_tasks=int(n),
        success=int(ok.sum()),
        fail_deadline=int((codes == FAIL_DEADLINE).sum()),
        fail_capacity=int((codes == FAIL_CAPACITY).sum()),
        fail_security=int((codes == FAIL_SECURITY).sum()),
        fail_offline=int((codes == FAIL_OFFLINE).sum()),
        fail_reject=int((codes == FAIL_REJECT).sum()),
        adv_executed=int(adv_exec), adv_total=int(T["adv"].sum()),
        mean_latency_ms=float(comp_lat.mean()) if len(comp_lat) else float("nan"),
        p95_latency_ms=float(np.percentile(comp_lat, 95)) if len(comp_lat) else float("nan"),
        energy_exec_J=float(env.energy_exec),
        energy_think_J=float(env.energy_think),
        energy_coord_J=float(env.energy_coord),
        nodes_depleted=int(env.offline.sum()),
        tier_counts=dict(env.tier_counts),
    )


def result_to_dict(r: Result) -> dict:
    d = asdict(r)
    d["success_rate"] = r.success_rate
    return d


if __name__ == "__main__":
    import time
    import sys
    for N in [100, 400, 1000]:
      cfg = Config(n_edge=N)
      for algo in ["NEXT_FIT", "RANDOM_FIT", "DQN_FIT", "ECO_EDGE"]:
        t0 = time.time()
        r = run(algo, cfg, seed=1)
        print(N, end=' ')
        print(f"{algo:10s} n={r.n_tasks} succ={r.success_rate:.3f} lat={r.mean_latency_ms:.1f}ms "
              f"cap={r.fail_capacity} dl={r.fail_deadline} sec={r.fail_security} adv_exec={r.adv_executed}/{r.adv_total} "
              f"Eexec={r.energy_exec_J:.0f}J Ethink={r.energy_think_J:.1f}J Ecoord={r.energy_coord_J:.2f}J "
              f"dead={r.nodes_depleted} [{time.time()-t0:.1f}s]")
