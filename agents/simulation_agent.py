"""
ECO-EDGE: Simulation Agent
============================
Simulates the proposed action in the digital twin (EdgeEnvironment)
and reports predicted KPIs without modifying the live environment.
"""

from __future__ import annotations

from typing import Any, Dict

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.edge_environment import EdgeEnvironment


def simulate_plan(plan: Dict[str, Any], environment: EdgeEnvironment) -> Dict[str, Any]:
    """
    Run the proposed action through the digital twin simulator.

    Returns predicted KPIs: latency, energy reduction, success rate, load, warnings.
    """
    action = plan.get("action", {})
    sim_result = environment.simulate_action(action)

    return {
        "predicted_latency_ms": sim_result["predicted_latency_ms"],
        "predicted_energy_reduction_pct": sim_result["predicted_energy_reduction_pct"],
        "predicted_success_rate": sim_result["predicted_success_rate"],
        "predicted_load_after": sim_result["predicted_load_after"],
        "current_power_w": sim_result["current_power_w"],
        "warnings": sim_result.get("warnings", []),
        "simulation_passed": len(sim_result.get("warnings", [])) == 0,
    }


def simulation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the Simulation Agent."""
    plan = state.get("proposed_plan", {})
    environment = state.get("environment")

    if environment is None:
        # Create a default environment if none provided
        environment = EdgeEnvironment()

    sim_result = simulate_plan(plan, environment)

    return {
        **state,
        "simulation_result": sim_result,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "SimulationAgent",
                "action": "simulate_plan",
                "result": (
                    f"Predicted: latency={sim_result['predicted_latency_ms']:.1f}ms, "
                    f"energy_reduction={sim_result['predicted_energy_reduction_pct']:.1f}%, "
                    f"success_rate={sim_result['predicted_success_rate']:.2%}"
                ),
            }
        ],
    }
