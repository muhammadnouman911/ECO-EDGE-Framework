"""
ECO-EDGE: Security Agent (Security-Critic)
============================================
Performs three verification checks on every proposed action:
  1. Credential / Role Authorization
  2. Prompt Integrity Check (anomalous resource requests)
  3. Behavioral Anomaly Detection (deviation scoring)

Part of the ActSimSecCrit pipeline (the "Sec" phase).
"""

from __future__ import annotations

from typing import Any, Dict, List


# Maximum fraction of total resources a single action may request
MAX_RESOURCE_REQUEST_RATIO = 0.6

# Anomaly threshold (z-score)
ANOMALY_THRESHOLD = 3.0

# Allowed action types per role
ROLE_PERMISSIONS = {
    "tier1_agent": ["sleep_scheduling", "traffic_steering"],
    "tier2_agent": ["workload_migration", "sleep_scheduling", "traffic_steering", "composite"],
    "tier3_agent": ["workload_migration", "sleep_scheduling", "traffic_steering", "composite", "policy_update"],
}


def check_role_authorization(action: Dict[str, Any], agent_role: str = "tier2_agent") -> Dict[str, Any]:
    """Check if the requesting agent has permission for this action type."""
    action_type = action.get("type", "unknown")
    allowed = ROLE_PERMISSIONS.get(agent_role, [])

    # For composite actions, check all sub-action types
    if action_type == "composite":
        sub_types = [sa.get("type", "unknown") for sa in action.get("sub_actions", [])]
        unauthorized = [t for t in sub_types if t not in allowed]
        return {
            "check": "role_authorization",
            "passed": len(unauthorized) == 0,
            "detail": f"Unauthorized sub-actions: {unauthorized}" if unauthorized else "All sub-actions authorized",
        }

    return {
        "check": "role_authorization",
        "passed": action_type in allowed,
        "detail": f"Action '{action_type}' {'is' if action_type in allowed else 'is NOT'} authorized for {agent_role}",
    }


def check_prompt_integrity(action: Dict[str, Any], total_nodes: int = 27) -> Dict[str, Any]:
    """Check for anomalous resource requests that could indicate prompt injection."""
    target_nodes = action.get("target_nodes", [])
    request_ratio = len(target_nodes) / max(total_nodes, 1)

    issues: List[str] = []

    if request_ratio > MAX_RESOURCE_REQUEST_RATIO:
        issues.append(
            f"Requesting {len(target_nodes)}/{total_nodes} nodes "
            f"({request_ratio:.0%}) exceeds safety threshold ({MAX_RESOURCE_REQUEST_RATIO:.0%})"
        )

    # Check for suspiciously uniform targets
    if len(target_nodes) > 5 and len(set(target_nodes)) < len(target_nodes) * 0.5:
        issues.append("Duplicate target nodes detected — possible injection pattern")

    return {
        "check": "prompt_integrity",
        "passed": len(issues) == 0,
        "detail": "; ".join(issues) if issues else "No anomalous patterns detected",
    }


def check_behavioral_anomaly(
    action: Dict[str, Any],
    historical_avg_targets: float = 5.0,
    historical_std_targets: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute behavioral deviation score:
      δ = ||a_proposed − ā_history|| / σ_history

    Flags actions with δ > threshold.
    """
    n_targets = len(action.get("target_nodes", []))
    if historical_std_targets < 0.01:
        historical_std_targets = 1.0

    deviation = abs(n_targets - historical_avg_targets) / historical_std_targets

    return {
        "check": "behavioral_anomaly",
        "passed": deviation <= ANOMALY_THRESHOLD,
        "deviation_score": round(deviation, 3),
        "detail": (
            f"Deviation score δ={deviation:.2f} "
            f"({'WITHIN' if deviation <= ANOMALY_THRESHOLD else 'EXCEEDS'} "
            f"threshold {ANOMALY_THRESHOLD})"
        ),
    }


def run_security_checks(
    action: Dict[str, Any],
    agent_role: str = "tier2_agent",
    total_nodes: int = 27,
) -> Dict[str, Any]:
    """Run all three security checks and return aggregate result."""
    checks = [
        check_role_authorization(action, agent_role),
        check_prompt_integrity(action, total_nodes),
        check_behavioral_anomaly(action),
    ]

    all_passed = all(c["passed"] for c in checks)
    failed = [c for c in checks if not c["passed"]]

    return {
        "security_passed": all_passed,
        "checks": checks,
        "failed_checks": [c["check"] for c in failed],
        "summary": "All security checks passed" if all_passed else f"FAILED: {', '.join(c['check'] for c in failed)}",
    }


def security_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Security Agent."""
    plan = state.get("proposed_plan", {})
    action = plan.get("action", {})
    env_snapshot = state.get("env_snapshot", {})
    total_nodes = env_snapshot.get("total_nodes", 27)

    security_result = run_security_checks(action, agent_role="tier2_agent", total_nodes=total_nodes)

    return {
        **state,
        "security_result": security_result,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "SecurityAgent",
                "action": "run_security_checks",
                "result": security_result["summary"],
            }
        ],
    }
