"""
ECO-EDGE: Planner Agent
========================
Generates candidate orchestration actions based on the parsed intent
and the current state of the edge environment.
"""

from __future__ import annotations

import random
import json
import re
import sys
import os
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestration.slm_config import get_slm


def generate_plan_rules(parsed_intent: Dict[str, Any], env_snapshot: Dict[str, Any], attempt: int = 1) -> Dict[str, Any]:
    """Fallback rule-based plan generator."""
    goals = parsed_intent.get("goals", [])
    constraints = parsed_intent.get("constraints", [])
    sub_tasks = parsed_intent.get("sub_tasks", [])
    total_nodes = env_snapshot.get("total_nodes", 27)
    active_nodes = env_snapshot.get("active_nodes", 27)

    sub_actions: List[Dict[str, Any]] = []
    rationale_parts: List[str] = []

    # --- Sleep scheduling ---
    if "sleep_scheduling" in sub_tasks or "identify_idle_nodes" in sub_tasks:
        # More conservative on retries
        sleep_fraction = max(0.1, 0.3 - (attempt - 1) * 0.08)
        n_sleep = max(1, int(active_nodes * sleep_fraction))
        target_nodes = [f"edge-{i:03d}" for i in random.sample(range(active_nodes), min(n_sleep, active_nodes))]
        sub_actions.append({
            "type": "sleep_scheduling",
            "target_nodes": target_nodes,
            "description": f"Put {n_sleep} idle edge nodes to micro-sleep",
        })
        rationale_parts.append(f"Micro-sleep {n_sleep} nodes to cut energy")

    # --- Workload consolidation / migration ---
    if "workload_consolidation" in sub_tasks or "workload_migration" in sub_tasks:
        tier2_start = env_snapshot.get("nodes_by_tier", {}).get(1, 20)
        src = f"edge-{random.randint(0, max(0, tier2_start - 1)):03d}"
        dst = f"mec-{tier2_start + random.randint(0, 3):03d}"
        sub_actions.append({
            "type": "workload_migration",
            "source_node": src,
            "destination_node": dst,
            "target_nodes": [src, dst],
            "description": f"Consolidate workloads from {src} to {dst}",
        })
        rationale_parts.append(f"Migrate load {src} -> {dst}")

    # --- Traffic steering ---
    if "traffic_steering" in sub_tasks:
        steer_nodes = [f"edge-{random.randint(0, 10):03d}" for _ in range(3)]
        sub_actions.append({
            "type": "traffic_steering",
            "target_nodes": steer_nodes,
            "description": "Re-route traffic away from overloaded nodes",
        })
        rationale_parts.append("Steer traffic from hot-spots")

    # --- Build composite action ---
    if len(sub_actions) > 1:
        action = {
            "type": "composite",
            "sub_actions": sub_actions,
            "target_nodes": list({n for sa in sub_actions for n in sa.get("target_nodes", [])}),
            "description": " + ".join(rationale_parts),
        }
    elif sub_actions:
        action = sub_actions[0]
    else:
        action = {
            "type": "traffic_steering",
            "target_nodes": [f"edge-{i:03d}" for i in range(3)],
            "description": "Default: re-route traffic for optimization",
        }

    # Extract targets from intent
    energy_target = None
    latency_limit = None
    for g in goals:
        if g["metric"] == "energy_reduction":
            energy_target = g["target_pct"]
    for c in constraints:
        if c["metric"] == "latency":
            latency_limit = c["max_ms"]

    return {
        "action": action,
        "energy_target_pct": energy_target,
        "latency_limit_ms": latency_limit,
        "attempt": attempt,
        "rationale": " | ".join(rationale_parts) or "General optimization",
    }


def generate_plan(parsed_intent: Dict[str, Any], env_snapshot: Dict[str, Any], attempt: int = 1) -> Dict[str, Any]:
    """
    Generate a candidate orchestration plan.
    Tries SLM first, falls back to rules.
    """
    slm = get_slm()
    if not slm:
        return generate_plan_rules(parsed_intent, env_snapshot, attempt)

    prompt = f"""
    You are the ECO-EDGE Planner Agent. Create an orchestration plan based on the intent and environment.
    
    INTENT: {json.dumps(parsed_intent)}
    ENVIRONMENT: {json.dumps(env_snapshot)}
    ATTEMPT: {attempt}
    
    On attempt > 1, be more conservative with energy saving to ensure latency constraints are met.
    
    Return ONLY a valid JSON object with this structure:
    {{
        "action": {{
            "type": "composite" or "sleep_scheduling" or "workload_migration" or "traffic_steering",
            "target_nodes": ["edge-001", "edge-002"],
            "description": "Detailed description of action",
            "sub_actions": [] (if composite)
        }},
        "energy_target_pct": 25,
        "latency_limit_ms": 50,
        "attempt": {attempt},
        "rationale": "Chain-of-thought explanation for this plan"
    }}
    """
    
    try:
        response = slm.invoke(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return generate_plan_rules(parsed_intent, env_snapshot, attempt)


def planner_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Planner Agent."""
    parsed_intent = state.get("parsed_intent", {})
    env_snapshot = state.get("env_snapshot", {})
    attempt = state.get("plan_attempt", 1)

    plan = generate_plan(parsed_intent, env_snapshot, attempt)
    
    using_slm = state.get("using_slm", False) or (get_slm() is not None)

    return {
        **state,
        "proposed_plan": plan,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "PlannerAgent",
                "mode": "SLM" if using_slm else "Rules",
                "action": "generate_plan",
                "attempt": attempt,
                "result": f"Proposed: {plan['action']['type']} — {plan['rationale']}",
            }
        ],
    }
