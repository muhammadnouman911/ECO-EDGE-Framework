"""
ECO-EDGE: Role-Affinity Task Scheduler
========================================
Computes semantic affinity scores between tasks and agents/nodes,
then assigns tasks to the best-matching node.

Implements Algorithm 1 from the ECO-EDGE paper:
  Affinity(t_j, U_i) = α·Sim(R_i, t_j) + β·(C_avail/C_total) + γ·(1/distance)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple


# Role specialization keywords
ROLE_SPECIALIZATIONS = {
    "traffic_steering": ["traffic", "steering", "routing", "bandwidth", "network", "qos"],
    "sleep_scheduling": ["sleep", "energy", "power", "idle", "shutdown", "micro-sleep"],
    "workload_migration": ["migrate", "offload", "transfer", "consolidate", "workload", "balance"],
    "security_monitoring": ["security", "anomaly", "attack", "threat", "monitor", "defense"],
    "latency_optimization": ["latency", "delay", "real-time", "fast", "deadline", "response"],
}


def compute_role_similarity(task_description: str, role: str) -> float:
    """
    Semantic similarity between a task description and a role specialization.
    Uses keyword overlap as a lightweight proxy for embedding cosine similarity.
    """
    keywords = ROLE_SPECIALIZATIONS.get(role, [])
    if not keywords:
        return 0.1

    task_lower = task_description.lower()
    matches = sum(1 for kw in keywords if kw in task_lower)
    return min(1.0, matches / max(len(keywords) * 0.5, 1))


def compute_affinity(
    task: Dict[str, Any],
    node: Dict[str, Any],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """
    Compute the role-affinity score between a task and a node.

    Parameters:
        task: {"description": str, "type": str}
        node: {"role": str, "available_compute": float, "total_compute": float, "distance": float}
        alpha, beta, gamma: weighting coefficients (must sum to 1.0)

    Returns:
        Affinity score in [0, 1].
    """
    sim = compute_role_similarity(task.get("description", ""), node.get("role", ""))
    capacity_ratio = node.get("available_compute", 0) / max(node.get("total_compute", 1), 0.01)
    distance = max(node.get("distance", 1), 0.1)

    score = alpha * sim + beta * capacity_ratio + gamma * (1.0 / distance)
    return round(min(1.0, max(0.0, score)), 4)


def schedule_tasks(
    tasks: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Assign tasks to nodes using role-affinity scheduling.

    Returns:
        List of (task, best_node, affinity_score) tuples.
    """
    assignments = []

    for task in tasks:
        best_node = None
        best_score = -1.0

        for node in nodes:
            score = compute_affinity(task, node, alpha, beta, gamma)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is not None:
            assignments.append((task, best_node, best_score))

    return assignments
