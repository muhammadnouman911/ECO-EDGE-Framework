"""
Eco-Edge: Role-Affinity Task Scheduling Simulation
====================================================
Demonstrates the decentralized role-affinity scheduling algorithm
that assigns tasks to the most suitable SLM agents based on
semantic similarity, compute availability, and network distance.

Authors: Muhammad Nouman, Shanzay Khan
Supervisor: Kamran Ahmed Awan
University of Haripur
"""

import random
import math

# ============================================================
# Agent Definitions (Thinking Units)
# ============================================================
AGENTS = [
    {
        "id": "U1", "name": "Traffic-Steering Agent",
        "tier": 1, "role": "traffic_management",
        "compute_total": 8.0,  # GFLOPS
        "compute_avail": 6.5,
        "location": (0, 0),
        "model": "TinyLlama-1.1B",
    },
    {
        "id": "U2", "name": "Energy-Monitor Agent",
        "tier": 1, "role": "energy_optimization",
        "compute_total": 4.0,
        "compute_avail": 3.2,
        "location": (2, 1),
        "model": "SmolLM-0.5B",
    },
    {
        "id": "U3", "name": "QoS-Planner Agent",
        "tier": 2, "role": "qos_planning",
        "compute_total": 50.0,
        "compute_avail": 35.0,
        "location": (5, 3),
        "model": "Phi-2-2.7B",
    },
    {
        "id": "U4", "name": "Security-Monitor Agent",
        "tier": 2, "role": "security_monitoring",
        "compute_total": 50.0,
        "compute_avail": 42.0,
        "location": (5, 5),
        "model": "Phi-2-2.7B",
    },
    {
        "id": "U5", "name": "Load-Balancer Agent",
        "tier": 1, "role": "load_balancing",
        "compute_total": 8.0,
        "compute_avail": 5.0,
        "location": (1, 3),
        "model": "TinyLlama-1.1B",
    },
    {
        "id": "U6", "name": "Predictive-Maintenance Agent",
        "tier": 1, "role": "predictive_maintenance",
        "compute_total": 6.0,
        "compute_avail": 4.8,
        "location": (3, 0),
        "model": "SmolLM-0.5B",
    },
    {
        "id": "U7", "name": "Global-Orchestrator Agent",
        "tier": 3, "role": "global_orchestration",
        "compute_total": 200.0,
        "compute_avail": 150.0,
        "location": (10, 10),
        "model": "Phi-3-3.8B",
    },
    {
        "id": "U8", "name": "Micro-Sleep Agent",
        "tier": 1, "role": "energy_optimization",
        "compute_total": 4.0,
        "compute_avail": 3.5,
        "location": (1, 2),
        "model": "SmolLM-0.5B",
    },
]

# ============================================================
# Task Definitions
# ============================================================
TASKS = [
    {"id": "T1", "name": "Route traffic away from congested node",
     "type": "traffic_management", "compute_req": 2.0, "origin": (0, 1), "deadline_ms": 50},
    {"id": "T2", "name": "Reduce energy on idle edge nodes",
     "type": "energy_optimization", "compute_req": 1.5, "origin": (2, 0), "deadline_ms": 200},
    {"id": "T3", "name": "Ensure video stream QoS for user cluster",
     "type": "qos_planning", "compute_req": 10.0, "origin": (4, 2), "deadline_ms": 100},
    {"id": "T4", "name": "Detect anomalous traffic pattern",
     "type": "security_monitoring", "compute_req": 8.0, "origin": (5, 4), "deadline_ms": 30},
    {"id": "T5", "name": "Balance workload across edge cluster",
     "type": "load_balancing", "compute_req": 3.0, "origin": (1, 2), "deadline_ms": 75},
    {"id": "T6", "name": "Predict hardware failure on sensor node",
     "type": "predictive_maintenance", "compute_req": 2.5, "origin": (3, 1), "deadline_ms": 500},
    {"id": "T7", "name": "Activate micro-sleep on 12 idle nodes",
     "type": "energy_optimization", "compute_req": 1.0, "origin": (1, 1), "deadline_ms": 300},
    {"id": "T8", "name": "Cross-domain resource reallocation",
     "type": "global_orchestration", "compute_req": 25.0, "origin": (8, 8), "deadline_ms": 1000},
    {"id": "T9", "name": "Emergency traffic rerouting",
     "type": "traffic_management", "compute_req": 3.5, "origin": (0, 2), "deadline_ms": 20},
    {"id": "T10", "name": "Optimize uplink power allocation",
     "type": "energy_optimization", "compute_req": 2.0, "origin": (2, 2), "deadline_ms": 150},
]

# ============================================================
# Role Similarity Matrix (semantic similarity between roles)
# ============================================================
ROLE_SIMILARITY = {
    ("traffic_management", "traffic_management"): 1.0,
    ("traffic_management", "load_balancing"): 0.7,
    ("traffic_management", "qos_planning"): 0.5,
    ("energy_optimization", "energy_optimization"): 1.0,
    ("energy_optimization", "predictive_maintenance"): 0.4,
    ("qos_planning", "qos_planning"): 1.0,
    ("qos_planning", "traffic_management"): 0.5,
    ("qos_planning", "load_balancing"): 0.6,
    ("security_monitoring", "security_monitoring"): 1.0,
    ("load_balancing", "load_balancing"): 1.0,
    ("load_balancing", "traffic_management"): 0.7,
    ("predictive_maintenance", "predictive_maintenance"): 1.0,
    ("global_orchestration", "global_orchestration"): 1.0,
    ("global_orchestration", "qos_planning"): 0.5,
    ("global_orchestration", "traffic_management"): 0.4,
    ("global_orchestration", "energy_optimization"): 0.4,
    ("global_orchestration", "load_balancing"): 0.4,
    ("global_orchestration", "security_monitoring"): 0.3,
}


def get_similarity(role1, role2):
    """Get semantic similarity between two roles (symmetric)."""
    return ROLE_SIMILARITY.get((role1, role2),
           ROLE_SIMILARITY.get((role2, role1), 0.1))


def euclidean_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_affinity(task, agent, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Compute the Role-Affinity score (Eq. 3 in the paper).
    Affinity(t_j, U_i) = α·Sim(R_i, t_j) + β·(C_avail/C_total) + γ·(1/d)
    """
    # Semantic similarity between agent role and task type
    sim = get_similarity(agent["role"], task["type"])

    # Compute availability ratio
    compute_ratio = agent["compute_avail"] / agent["compute_total"]

    # Network distance (add 1 to avoid division by zero)
    dist = euclidean_distance(task["origin"], agent["location"]) + 1.0
    proximity = 1.0 / dist

    # Weighted affinity score
    affinity = alpha * sim + beta * compute_ratio + gamma * proximity

    return affinity, sim, compute_ratio, proximity


def role_affinity_scheduling(tasks, agents, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Algorithm 2: Role-Affinity Task Scheduling.
    Assigns each task to the best-matching available agent.
    """
    # Deep copy agents to track available compute
    agents_state = []
    for a in agents:
        agents_state.append({**a, "compute_avail": a["compute_avail"]})

    assignments = []

    for task in tasks:
        best_score = -1
        best_agent = None
        all_scores = []

        for agent in agents_state:
            if agent["compute_avail"] < task["compute_req"]:
                all_scores.append((agent["id"], -1, "INSUFFICIENT"))
                continue

            score, sim, comp, prox = compute_affinity(task, agent, alpha, beta, gamma)
            all_scores.append((agent["id"], score, f"sim={sim:.2f} comp={comp:.2f} prox={prox:.2f}"))

            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            best_agent["compute_avail"] -= task["compute_req"]
            assignments.append({
                "task": task,
                "agent": best_agent,
                "score": best_score,
                "all_scores": all_scores,
            })
        else:
            assignments.append({
                "task": task,
                "agent": None,
                "score": 0,
                "all_scores": all_scores,
            })

    return assignments


def random_scheduling(tasks, agents):
    """Baseline: Random task assignment (no intelligence)."""
    agents_state = [{**a, "compute_avail": a["compute_avail"]} for a in agents]
    assignments = []

    for task in tasks:
        eligible = [a for a in agents_state if a["compute_avail"] >= task["compute_req"]]
        if eligible:
            chosen = random.choice(eligible)
            chosen["compute_avail"] -= task["compute_req"]
            assignments.append({"task": task, "agent": chosen, "score": 0})
        else:
            assignments.append({"task": task, "agent": None, "score": 0})

    return assignments


def print_assignments(title, assignments):
    """Print formatted assignment results."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'Task':>4} | {'Task Description':<42} | {'Assigned Agent':<24} | {'Score':>5}")
    print(f"{'-' * 4}-+-{'-' * 42}-+-{'-' * 24}-+-{'-' * 5}")

    total_sim = 0
    assigned_count = 0

    for a in assignments:
        task_name = a["task"]["name"][:40]
        if a["agent"]:
            agent_name = f"{a['agent']['id']}: {a['agent']['name']}"[:22]
            score = f"{a['score']:.3f}" if a['score'] > 0 else "rand"
            assigned_count += 1
            total_sim += get_similarity(a["agent"]["role"], a["task"]["type"])
        else:
            agent_name = "UNASSIGNED"
            score = "N/A"

        print(f"{a['task']['id']:>4} | {task_name:<42} | {agent_name:<24} | {score:>5}")

    avg_sim = total_sim / assigned_count if assigned_count > 0 else 0

    print(f"\n  Assigned: {assigned_count}/{len(assignments)} tasks")
    print(f"  Avg Role-Task Similarity: {avg_sim:.2%}")

    return assigned_count, avg_sim


def main():
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║   Eco-Edge: Role-Affinity Task Scheduling Simulation                    ║")
    print("║   Decentralized Task-to-Agent Assignment                                ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    print(f"\n  Network Configuration:")
    print(f"  ├── Agents: {len(AGENTS)}")
    for a in AGENTS:
        print(f"  │   {a['id']}: {a['name']:<28} [Tier-{a['tier']}] {a['model']:<16} "
              f"Compute: {a['compute_avail']:.1f}/{a['compute_total']:.1f} GFLOPS")
    print(f"  └── Tasks: {len(TASKS)}")
    for t in TASKS:
        print(f"      {t['id']}: {t['name']:<42} [{t['type']}] {t['compute_req']:.1f} GFLOPS, "
              f"deadline: {t['deadline_ms']}ms")

    # ----- Role-Affinity Scheduling (Eco-Edge) -----
    assignments_ra = role_affinity_scheduling(TASKS, AGENTS)
    ra_count, ra_sim = print_assignments(
        "ROLE-AFFINITY SCHEDULING (Eco-Edge Algorithm 2)", assignments_ra
    )

    # Show detailed scoring for first 3 tasks
    print(f"\n  Detailed Affinity Scores (first 3 tasks):")
    for a in assignments_ra[:3]:
        print(f"\n  Task {a['task']['id']}: {a['task']['name']}")
        print(f"    {'Agent':>4} | {'Score':>8} | Details")
        print(f"    {'-' * 4}-+-{'-' * 8}-+-{'-' * 40}")
        for agent_id, score, details in a["all_scores"]:
            marker = " ◄── SELECTED" if a["agent"] and agent_id == a["agent"]["id"] else ""
            if score < 0:
                print(f"    {agent_id:>4} |     ---  | {details}")
            else:
                print(f"    {agent_id:>4} | {score:>7.3f}  | {details}{marker}")

    # ----- Random Scheduling (Baseline) -----
    random.seed(42)
    assignments_rnd = random_scheduling(TASKS, AGENTS)
    rnd_count, rnd_sim = print_assignments(
        "RANDOM SCHEDULING (Baseline)", assignments_rnd
    )

    # ----- Comparison -----
    sim_improvement = ((ra_sim - rnd_sim) / rnd_sim) * 100 if rnd_sim > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"  COMPARATIVE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  ┌───────────────────────────┬────────────────┬────────────────┬──────────┐")
    print(f"  │ Metric                    │ Role-Affinity  │ Random         │ Gain     │")
    print(f"  ├───────────────────────────┼────────────────┼────────────────┼──────────┤")
    print(f"  │ Tasks Assigned            │ {ra_count:>10d}/10 │ {rnd_count:>10d}/10 │   --     │")
    print(f"  │ Avg Role-Task Similarity  │ {ra_sim:>13.1%} │ {rnd_sim:>13.1%} │ {sim_improvement:>+5.1f}%  │")
    print(f"  └───────────────────────────┴────────────────┴────────────────┴──────────┘")
    print(f"\n  Key Insight: Role-Affinity scheduling achieves {sim_improvement:.0f}% higher")
    print(f"  role-task matching, ensuring tasks go to specialized agents.")
    print(f"  This directly improves decision quality and reduces rework.\n")


if __name__ == "__main__":
    main()
