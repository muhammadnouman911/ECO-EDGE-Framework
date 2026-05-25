"""
ECO-EDGE: Intent Agent
=======================
Receives a natural-language orchestration intent and converts it into
a structured task specification with goals, constraints, and priorities.
"""

from __future__ import annotations

import re
import json
import sys
import os
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestration.slm_config import get_slm


def parse_intent_rules(intent_text: str) -> Dict[str, Any]:
    """Fallback rule-based parser for intent."""
    result: Dict[str, Any] = {
        "raw_intent": intent_text,
        "goals": [],
        "constraints": [],
        "priority": "balanced",
        "sub_tasks": [],
    }

    text = intent_text.lower()

    # --- Extract energy reduction goals ---
    energy_match = re.search(r"(?:reduce|cut|lower|decrease)\s+(?:energy|power)\s+(?:consumption\s+)?(?:by\s+)?(\d+)\s*%", text)
    if energy_match:
        result["goals"].append({
            "metric": "energy_reduction",
            "target_pct": int(energy_match.group(1)),
        })
        result["priority"] = "energy"
        result["sub_tasks"].extend(["identify_idle_nodes", "sleep_scheduling", "workload_consolidation"])

    # --- Extract latency constraints ---
    latency_match = re.search(r"latency\s+(?:under|below|less\s+than|<=?)\s+(\d+)\s*(?:ms|milliseconds)?", text)
    if latency_match:
        result["constraints"].append({
            "metric": "latency",
            "max_ms": int(latency_match.group(1)),
        })

    # --- Extract throughput goals ---
    throughput_match = re.search(r"(?:increase|improve|boost)\s+(?:throughput|bandwidth)\s+(?:by\s+)?(\d+)\s*%", text)
    if throughput_match:
        result["goals"].append({
            "metric": "throughput_increase",
            "target_pct": int(throughput_match.group(1)),
        })
        result["sub_tasks"].append("traffic_steering")

    # --- Extract load balancing intents ---
    if any(kw in text for kw in ["balance", "distribute", "spread", "equalize"]):
        result["goals"].append({"metric": "load_balancing", "target_pct": None})
        result["sub_tasks"].append("workload_migration")

    # --- Extract QoS maintenance intents ---
    if any(kw in text for kw in ["maintain", "ensure", "guarantee", "preserve"]):
        qos_match = re.search(r"(?:maintain|ensure|guarantee)\s+(?:qos|quality)", text)
        if qos_match:
            result["constraints"].append({"metric": "qos", "min_level": "acceptable"})

    # --- Deduplicate sub-tasks ---
    result["sub_tasks"] = list(dict.fromkeys(result["sub_tasks"]))

    # --- Fallback ---
    if not result["goals"]:
        result["goals"].append({"metric": "general_optimization", "target_pct": 10})
        result["sub_tasks"] = ["workload_migration", "traffic_steering"]

    return result


def parse_intent(intent_text: str) -> Dict[str, Any]:
    """
    Parse a natural-language intent into a structured orchestration task.
    Tries SLM first, falls back to rules.
    """
    slm = get_slm()
    if not slm:
        return parse_intent_rules(intent_text)

    prompt = f"""
    You are the ECO-EDGE Intent Agent. Analyze the following user intent for edge computing orchestration.
    Extract goals, constraints, priority, and suggested sub-tasks.
    
    INTENT: "{intent_text}"
    
    Return ONLY a valid JSON object with this structure:
    {{
        "raw_intent": "{intent_text}",
        "goals": [{{ "metric": "energy_reduction", "target_pct": 25 }}],
        "constraints": [{{ "metric": "latency", "max_ms": 50 }}],
        "priority": "energy" or "latency" or "balanced",
        "sub_tasks": ["sleep_scheduling", "workload_migration", "traffic_steering"]
    }}
    
    If the intent is unclear, provide reasonable defaults based on edge computing optimization best practices.
    """
    
    try:
        response = slm.invoke(prompt)
        # Clean response string to extract only JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return parse_intent_rules(intent_text)


def intent_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Intent Agent."""
    intent_text = state.get("intent_text", "")
    parsed = parse_intent(intent_text)
    
    using_slm = state.get("using_slm", False) or (get_slm() is not None)

    return {
        **state,
        "parsed_intent": parsed,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "IntentAgent",
                "action": "parse_intent",
                "mode": "SLM" if using_slm else "Rules",
                "result": f"Extracted {len(parsed['goals'])} goals, "
                          f"{len(parsed['constraints'])} constraints, "
                          f"{len(parsed['sub_tasks'])} sub-tasks",
            }
        ],
    }
