"""
ECO-EDGE: Execution Agent
===========================
Applies a validated (critic-approved) orchestration action to the
live edge environment and logs the outcome.
"""

from __future__ import annotations

from typing import Any, Dict

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.edge_environment import EdgeEnvironment


def execute_plan(plan: Dict[str, Any], environment: EdgeEnvironment) -> Dict[str, Any]:
    """
    Apply the approved action to the live environment.

    Returns execution summary with changes made and new system snapshot.
    """
    action = plan.get("action", {})
    result = environment.apply_action(action)

    return {
        "execution_status": "success",
        "action_type": result["action_type"],
        "changes_made": result["changes"],
        "new_snapshot": result["new_snapshot"],
        "num_changes": len(result["changes"]),
    }


def execution_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Execution Agent."""
    critic_result = state.get("critic_result", {})
    plan = state.get("proposed_plan", {})
    environment = state.get("environment")

    if not critic_result.get("approved", False):
        return {
            **state,
            "execution_result": {
                "execution_status": "skipped",
                "reason": "Plan was not approved by Critic",
            },
            "agent_trace": state.get("agent_trace", []) + [
                {
                    "agent": "ExecutionAgent",
                    "action": "execute_plan",
                    "result": "SKIPPED — Plan not approved",
                }
            ],
        }

    if environment is None:
        environment = EdgeEnvironment()

    exec_result = execute_plan(plan, environment)

    return {
        **state,
        "execution_result": exec_result,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "ExecutionAgent",
                "action": "execute_plan",
                "result": f"Applied {exec_result['action_type']} — {exec_result['num_changes']} changes made",
            }
        ],
    }
