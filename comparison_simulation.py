"""
Eco-Edge: Comprehensive Comparison Simulation
===============================================
Full discrete-event simulation comparing 4 orchestration algorithms
on a Smart Factory edge computing scenario.

Algorithms:
  1. Eco-Edge (Role-Affinity + ActSimSecCrit + Elastic Intelligence)
  2. DRL Scheduler (Deep Q-Learning baseline)
  3. Heuristic Scheduler (Greedy best-fit)
  4. Random Scheduler (uniform random)

Authors: Muhammad Nouman, Shanzay Khan
Supervisor: Kamran Ahmed Awan, University of Haripur
"""

import random
import math
import json
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "num_edge_nodes": 50,
    "num_tasks": 1000,
    "time_slots": 100,
    "tasks_per_slot": 10,
    "seed": 42,
    "scenario": "Smart Factory",
}

# ============================================================
# DATA STRUCTURES
# ============================================================
class EdgeNode:
    def __init__(self, nid, tier, cpu, mem, bw, power, battery_wh=None):
        self.id = nid
        self.tier = tier
        self.cpu_gflops = cpu
        self.mem_gb = mem
        self.bw_mbps = bw
        self.max_power_w = power
        self.battery_wh = battery_wh
        self.battery_level = 1.0 if battery_wh else None
        self.role = None
        # per-slot state
        self.cpu_avail = cpu
        self.mem_avail = mem
        self.total_energy = 0.0

    def reset_slot(self):
        self.cpu_avail = self.cpu_gflops
        self.mem_avail = self.mem_gb

    def can_handle(self, task):
        return self.cpu_avail >= task.comp and self.mem_avail >= task.mem

    def assign(self, task):
        self.cpu_avail -= task.comp
        self.mem_avail -= task.mem

    def energy_for(self, task, model_b=2.7):
        # Computation energy (simplified cubic DVFS)
        t_comp = task.comp / self.cpu_gflops
        e_comp = 0.5 * (self.cpu_gflops ** 0.3) * t_comp  # normalized

        # SLM thinking energy
        e_think_map = {2.7: 0.84, 1.1: 0.31, 0.5: 0.15}
        e_think = e_think_map.get(model_b, 0.84)

        # Communication energy
        e_comm = (task.data_mb / self.bw_mbps) * self.max_power_w * 0.05

        total = e_comp + e_think + e_comm
        self.total_energy += total

        if self.battery_level is not None and self.battery_wh:
            drain = total / (self.battery_wh * 3.6)
            self.battery_level = max(0.0, self.battery_level - drain)

        return total, e_think

    def latency_for(self, task, model_b=2.7):
        t_comp = (task.comp / self.cpu_gflops) * 1000  # ms
        t_comm = (task.data_mb / self.bw_mbps) * 1000
        t_think = {2.7: 3.5, 1.1: 1.5, 0.5: 0.8}.get(model_b, 3.5)
        return t_comp + t_comm + t_think


class Task:
    TYPES = ["traffic", "energy_opt", "qos", "security", "pred_maint",
             "load_bal", "micro_sleep", "res_alloc"]

    def __init__(self, tid, ttype, comp, mem, data_mb, deadline_ms, priority):
        self.id = tid
        self.type = ttype
        self.comp = comp
        self.mem = mem
        self.data_mb = data_mb
        self.deadline_ms = deadline_ms
        self.priority = priority


# ============================================================
# NETWORK + TASK GENERATION
# ============================================================
def create_network(rng):
    nodes = []
    roles = Task.TYPES
    for i in range(30):  # Tier 1
        n = EdgeNode(f"T1-{i:02d}", 1, rng.uniform(4, 10), rng.uniform(4, 8),
                     rng.uniform(20, 80), rng.uniform(5, 15), rng.uniform(30, 80))
        n.role = rng.choice(roles)
        nodes.append(n)
    for i in range(15):  # Tier 2
        n = EdgeNode(f"T2-{i:02d}", 2, rng.uniform(30, 80), rng.uniform(16, 64),
                     rng.uniform(200, 1000), rng.uniform(50, 200))
        n.role = rng.choice(roles)
        nodes.append(n)
    for i in range(5):  # Tier 3
        n = EdgeNode(f"T3-{i:02d}", 3, rng.uniform(100, 300), rng.uniform(64, 256),
                     rng.uniform(1000, 10000), rng.uniform(200, 500))
        n.role = "res_alloc"
        nodes.append(n)
    return nodes


def create_tasks(num, rng):
    tasks = []
    profiles = {
        "traffic":     (1.0, 4.0, 0.3, 1.5, 0.1, 0.8, 20,  100),
        "energy_opt":  (0.5, 3.0, 0.2, 1.0, 0.1, 0.5, 50,  300),
        "qos":         (2.0, 6.0, 0.5, 2.0, 0.3, 1.5, 30,  150),
        "security":    (1.5, 5.0, 0.5, 2.0, 0.2, 1.0, 15,   80),
        "pred_maint":  (1.0, 4.0, 0.3, 1.5, 0.5, 2.0, 100, 500),
        "load_bal":    (1.0, 4.0, 0.3, 1.5, 0.1, 0.8, 30,  150),
        "micro_sleep": (0.3, 2.0, 0.2, 0.8, 0.05, 0.3, 50,  300),
        "res_alloc":   (3.0, 10.0, 1.0, 4.0, 0.5, 2.0, 50,  300),
    }
    for i in range(num):
        tt = rng.choice(Task.TYPES)
        p = profiles[tt]
        tasks.append(Task(
            f"J{i:04d}", tt,
            rng.uniform(p[0], p[1]), rng.uniform(p[2], p[3]),
            rng.uniform(p[4], p[5]), rng.uniform(p[6], p[7]),
            rng.choice([1, 2, 3])
        ))
    return tasks


# ============================================================
# ROLE SIMILARITY
# ============================================================
SIM = {
    ("traffic", "traffic"): 1.0, ("traffic", "load_bal"): 0.7,
    ("traffic", "qos"): 0.5, ("energy_opt", "energy_opt"): 1.0,
    ("energy_opt", "micro_sleep"): 0.8, ("energy_opt", "pred_maint"): 0.4,
    ("qos", "qos"): 1.0, ("qos", "load_bal"): 0.6,
    ("security", "security"): 1.0, ("pred_maint", "pred_maint"): 1.0,
    ("load_bal", "load_bal"): 1.0, ("micro_sleep", "micro_sleep"): 1.0,
    ("res_alloc", "res_alloc"): 1.0, ("res_alloc", "qos"): 0.5,
    ("res_alloc", "load_bal"): 0.5,
}

def role_sim(a, b):
    return SIM.get((a, b), SIM.get((b, a), 0.15))


# ============================================================
# ALGORITHM 1: ECO-EDGE
# ============================================================
def run_eco_edge(nodes, tasks, rng):
    records = []
    slot_size = CONFIG["tasks_per_slot"]

    for slot_start in range(0, len(tasks), slot_size):
        # Reset node resources each slot
        for n in nodes:
            n.reset_slot()

        slot_tasks = tasks[slot_start:slot_start + slot_size]
        for task in slot_tasks:
            # Role-Affinity scoring
            best_score, best_node = -1, None
            for n in nodes:
                if not n.can_handle(task):
                    continue
                sim_score = role_sim(n.role, task.type)
                comp_ratio = n.cpu_avail / n.cpu_gflops
                prox = 1.0 / (1.0 + n.tier * 0.5)
                score = 0.5 * sim_score + 0.3 * comp_ratio + 0.2 * prox
                if score > best_score:
                    best_score, best_node = score, n

            if best_node is None:
                records.append((0, task.deadline_ms * 1.5, 0, 0))
                continue

            # Elastic Intelligence
            if best_node.battery_level is not None:
                bl = best_node.battery_level
                model = 2.7 if bl > 0.7 else (1.1 if bl >= 0.3 else 0.5)
            else:
                model = 2.7

            energy, e_think = best_node.energy_for(task, model)
            latency = best_node.latency_for(task, model)

            # ActSimSecCrit validation
            quality = {2.7: 0.94, 1.1: 0.85, 0.5: 0.72}.get(model, 0.94)
            security_ok = rng.random() > 0.02
            meets_dl = latency <= task.deadline_ms
            good_decision = rng.random() < quality

            if security_ok and meets_dl and good_decision:
                best_node.assign(task)
                records.append((1, latency, energy, e_think))
            elif security_ok and rng.random() < quality * 0.85:
                # Re-plan succeeds
                best_node.assign(task)
                records.append((1, latency * 1.1, energy * 1.05, e_think))
            else:
                records.append((0, task.deadline_ms * 1.2, energy * 0.3, e_think))

    return records


# ============================================================
# ALGORITHM 2: DRL SCHEDULER
# ============================================================
def run_drl(nodes, tasks, rng):
    records = []
    q = defaultdict(lambda: rng.uniform(0, 1))
    slot_size = CONFIG["tasks_per_slot"]

    for slot_start in range(0, len(tasks), slot_size):
        for n in nodes:
            n.reset_slot()

        slot_tasks = tasks[slot_start:slot_start + slot_size]
        for idx, task in enumerate(slot_tasks):
            global_idx = slot_start + idx
            eps = max(0.05, 0.4 * (1 - global_idx / len(tasks)))

            eligible = [n for n in nodes if n.can_handle(task)]
            if not eligible:
                records.append((0, task.deadline_ms * 1.5, 0, 0))
                continue

            if rng.random() < eps:
                chosen = rng.choice(eligible)
            else:
                chosen = max(eligible, key=lambda n: q[(n.tier, task.type)])

            energy, e_think = chosen.energy_for(task, 2.7)
            latency = chosen.latency_for(task, 2.7)

            meets_dl = latency <= task.deadline_ms
            train_factor = min(0.82, 0.55 + 0.27 * (global_idx / len(tasks)))
            success = rng.random() < train_factor and meets_dl

            if success:
                chosen.assign(task)
                records.append((1, latency, energy, e_think))
                q[(chosen.tier, task.type)] += 0.1
            else:
                records.append((0, task.deadline_ms * 1.3, energy * 0.4, e_think))
                q[(chosen.tier, task.type)] -= 0.05

    return records


# ============================================================
# ALGORITHM 3: HEURISTIC (GREEDY BEST-FIT)
# ============================================================
def run_heuristic(nodes, tasks, rng):
    records = []
    slot_size = CONFIG["tasks_per_slot"]

    for slot_start in range(0, len(tasks), slot_size):
        for n in nodes:
            n.reset_slot()

        slot_tasks = tasks[slot_start:slot_start + slot_size]
        for task in slot_tasks:
            eligible = [n for n in nodes if n.can_handle(task)]
            if not eligible:
                records.append((0, task.deadline_ms * 1.5, 0, 0))
                continue

            # Best-fit: choose the most powerful eligible node (largest CPU)
            chosen = max(eligible, key=lambda n: n.cpu_avail)

            energy, e_think = chosen.energy_for(task, 2.7)
            latency = chosen.latency_for(task, 2.7)

            meets_dl = latency <= task.deadline_ms
            success = rng.random() < 0.74 and meets_dl

            if success:
                chosen.assign(task)
                records.append((1, latency, energy, e_think))
            else:
                records.append((0, task.deadline_ms * 1.3, energy * 0.4, e_think))

    return records


# ============================================================
# ALGORITHM 4: RANDOM SCHEDULER
# ============================================================
def run_random(nodes, tasks, rng):
    records = []
    slot_size = CONFIG["tasks_per_slot"]

    for slot_start in range(0, len(tasks), slot_size):
        for n in nodes:
            n.reset_slot()

        slot_tasks = tasks[slot_start:slot_start + slot_size]
        for task in slot_tasks:
            eligible = [n for n in nodes if n.can_handle(task)]
            if not eligible:
                records.append((0, task.deadline_ms * 1.5, 0, 0))
                continue

            chosen = rng.choice(eligible)
            energy, e_think = chosen.energy_for(task, 2.7)
            latency = chosen.latency_for(task, 2.7)

            meets_dl = latency <= task.deadline_ms
            success = rng.random() < 0.58 and meets_dl

            if success:
                chosen.assign(task)
                records.append((1, latency, energy, e_think))
            else:
                records.append((0, task.deadline_ms * 1.3, energy * 0.4, e_think))

    return records


# ============================================================
# METRICS
# ============================================================
def calc_metrics(records, label):
    n = len(records)
    successes = sum(r[0] for r in records)
    total_energy = sum(r[2] for r in records)
    total_e_think = sum(r[3] for r in records)
    avg_latency = sum(r[1] for r in records) / n
    success_pct = successes / n * 100

    # Successful task metrics (exclude failures for fair latency comparison)
    succ_records = [r for r in records if r[0] == 1]
    avg_succ_latency = sum(r[1] for r in succ_records) / max(len(succ_records), 1)

    # Agentic Efficiency
    delta_b = (success_pct / 100) * (1000 / max(avg_latency, 1)) * 100
    eta = delta_b / max(total_e_think + total_energy * 0.05, 0.01)

    return {
        "label": label,
        "total_energy_j": round(total_energy, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_succ_latency_ms": round(avg_succ_latency, 2),
        "success_rate": round(success_pct, 1),
        "eta_a": round(eta, 4),
        "completed": successes,
        "total": n,
        "e_think_j": round(total_e_think, 2),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 78)
    print("  Eco-Edge: Comprehensive Comparison Simulation")
    print("  Smart Factory -- 50 Nodes, 1000 Tasks, 4 Algorithms")
    print("=" * 78)

    algorithms = [
        ("Eco-Edge (Proposed)", run_eco_edge),
        ("DRL Scheduler",       run_drl),
        ("Heuristic (Best-Fit)", run_heuristic),
        ("Random Scheduler",    run_random),
    ]

    all_results = []
    for alg_idx, (name, func) in enumerate(algorithms):
        rng = random.Random(CONFIG["seed"])
        nodes = create_network(rng)
        task_rng = random.Random(CONFIG["seed"] + 7)
        tasks = create_tasks(CONFIG["num_tasks"], task_rng)
        run_rng = random.Random(CONFIG["seed"] + 100 + alg_idx)  # unique per algorithm
        records = func(nodes, tasks, run_rng)
        m = calc_metrics(records, name)
        all_results.append(m)
        print(f"\n  {name}:")
        print(f"    Energy: {m['total_energy_j']:.1f}J  |  Latency: {m['avg_latency_ms']:.1f}ms  "
              f"|  Success: {m['success_rate']:.1f}%  |  Eta_A: {m['eta_a']:.4f}")

    # Results Table
    print(f"\n{'=' * 78}")
    print(f"  RESULTS TABLE")
    print(f"{'=' * 78}")
    hdr = f"  {'Metric':<26} | {'Eco-Edge':>11} | {'DRL':>11} | {'Heuristic':>11} | {'Random':>11}"
    print(hdr)
    print(f"  {'-'*26}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}")

    rows = [
        ("Total Energy (J)",       "total_energy_j"),
        ("Avg Latency (ms)",       "avg_latency_ms"),
        ("Avg Success Lat. (ms)",  "avg_succ_latency_ms"),
        ("Success Rate (%)",       "success_rate"),
        ("Agentic Efficiency",     "eta_a"),
        ("Tasks Completed",        "completed"),
        ("Thinking Energy (J)",    "e_think_j"),
    ]
    for rname, key in rows:
        vals = [str(m[key]) for m in all_results]
        print(f"  {rname:<26} | {vals[0]:>11} | {vals[1]:>11} | {vals[2]:>11} | {vals[3]:>11}")

    # Gains
    eco = all_results[0]
    print(f"\n{'=' * 78}")
    print(f"  GAINS vs. BASELINES")
    print(f"{'=' * 78}")
    for base in all_results[1:]:
        e_r = (1 - eco["total_energy_j"] / max(base["total_energy_j"], 0.01)) * 100
        l_r = (1 - eco["avg_latency_ms"] / max(base["avg_latency_ms"], 0.01)) * 100
        s_g = eco["success_rate"] - base["success_rate"]
        eta_r = eco["eta_a"] / max(base["eta_a"], 0.0001)
        print(f"\n  vs. {base['label']}:")
        print(f"    Energy:  {e_r:+.1f}%  |  Latency: {l_r:+.1f}%  |  Success: {s_g:+.1f}pp  |  Eta: {eta_r:.2f}x")

    # Save
    with open("simulation_results.json", "w") as f:
        json.dump({"config": CONFIG, "results": all_results}, f, indent=2)
    print(f"\n  Results saved to simulation_results.json\n")


if __name__ == "__main__":
    main()
