"""
ECO-EDGE: Critic Agent
========================
Evaluates the simulated outcome against system constraints.
Approves or rejects actions using threshold-based criteria.

Part of the ActSimSecCrit pipeline (the final "Crit" phase).
"""

import json
import re
import sys
import os
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestration.slm_config import get_slm


# Default acceptance thresholds
DEFAULT_THRESHOLDS = {
    "min_success_rate": 0.85,
    "min_energy_reduction_pct": 5.0,
    "max_latency_ms": 100.0,  # overridden by intent constraints
}


def evaluate_plan_rules(
    simulation_result: Dict[str, Any],
    security_result: Dict[str, Any],
    plan: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Fallback rule-based evaluator."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # Override defaults with intent constraints
    latency_limit = plan.get("latency_limit_ms")
    if latency_limit is not None:
        thresholds["max_latency_ms"] = float(latency_limit)

    energy_target = plan.get("energy_target_pct")
    if energy_target is not None:
        thresholds["min_energy_reduction_pct"] = float(energy_target) * 0.7  # Allow 70% of target

    evaluations: List[Dict[str, Any]] = []
    all_passed = True

    # 1. Security gate
    sec_passed = security_result.get("security_passed", False)
    evaluations.append({
        "criterion": "security_clearance",
        "passed": sec_passed,
        "detail": security_result.get("summary", "No security data"),
    })
    if not sec_passed:
        all_passed = False

    # 2. Latency check
    pred_latency = simulation_result.get("predicted_latency_ms", 999)
    lat_ok = pred_latency <= thresholds["max_latency_ms"]
    evaluations.append({
        "criterion": "latency",
        "passed": lat_ok,
        "predicted": pred_latency,
        "threshold": thresholds["max_latency_ms"],
        "detail": f"{pred_latency:.1f}ms {'<=' if lat_ok else '>'} {thresholds['max_latency_ms']:.1f}ms",
    })
    if not lat_ok:
        all_passed = False

    # 3. Energy reduction check
    pred_energy = simulation_result.get("predicted_energy_reduction_pct", 0)
    energy_ok = pred_energy >= thresholds["min_energy_reduction_pct"]
    evaluations.append({
        "criterion": "energy_reduction",
        "passed": energy_ok,
        "predicted": pred_energy,
        "threshold": thresholds["min_energy_reduction_pct"],
        "detail": f"{pred_energy:.1f}% {'>=' if energy_ok else '<'} {thresholds['min_energy_reduction_pct']:.1f}%",
    })
    if not energy_ok:
        all_passed = False

    # 4. Success rate check
    pred_success = simulation_result.get("predicted_success_rate", 0)
    success_ok = pred_success >= thresholds["min_success_rate"]
    evaluations.append({
        "criterion": "success_rate",
        "passed": success_ok,
        "predicted": pred_success,
        "threshold": thresholds["min_success_rate"],
        "detail": f"{pred_success:.2%} {'>=' if success_ok else '<'} {thresholds['min_success_rate']:.2%}",
    })
    if not success_ok:
        all_passed = False

    return {
        "approved": all_passed,
        "evaluations": evaluations,
        "thresholds_used": thresholds,
        "summary": "APPROVED ✓ — All criteria met" if all_passed else "REJECTED X — Criteria not met",
        "failed_criteria": [e["criterion"] for e in evaluations if not e["passed"]],
    }


def evaluate_plan(
    simulation_result: Dict[str, Any],
    security_result: Dict[str, Any],
    plan: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate the proposed plan using SLM or rules.
    """
    slm = get_slm()
    if not slm:
        return evaluate_plan_rules(simulation_result, security_result, plan, thresholds)

    prompt = f"""
    You are the ECO-EDGE Critic Agent. Evaluate the orchestration plan performance.
    
    PLAN: {json.dumps(plan)}
    SIMULATION: {json.dumps(simulation_result)}
    SECURITY: {json.dumps(security_result)}
    THRESHOLDS: {json.dumps(thresholds or DEFAULT_THRESHOLDS)}
    
    Return ONLY a valid JSON object with this structure:
    {{
        "approved": true or false,
        "evaluations": [
            {{ "criterion": "latency", "passed": true, "detail": "Reasoning" }},
            {{ "criterion": "security", "passed": true, "detail": "Reasoning" }}
        ],
        "summary": "Final verdict summary string",
        "failed_criteria": ["list", "of", "failed"]
    }}
    """
    
    try:
        response = slm.invoke(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return evaluate_plan_rules(simulation_result, security_result, plan, thresholds)


def critic_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Critic Agent."""
    simulation_result = state.get("simulation_result", {})
    security_result = state.get("security_result", {})
    plan = state.get("proposed_plan", {})

    critic_result = evaluate_plan(simulation_result, security_result, plan)
    
    using_slm = state.get("using_slm", False) or (get_slm() is not None)

    return {
        **state,
        "critic_result": critic_result,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "CriticAgent",
                "mode": "SLM" if using_slm else "Rules",
                "action": "evaluate_plan",
                "result": critic_result["summary"],
            }
        ],
    }
