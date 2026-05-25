"""
ECO-EDGE: LangGraph ActSimSecCrit Workflow
============================================
Builds a LangGraph StateGraph that wires all 6 agents into the
full ActSimSecCrit pipeline:

  Intent → Planner → Simulate → Security → Critic → Execute
                ↑                              |
                └──── re-plan (max 3) ─────────┘

Usage:
    from graph.eco_edge_graph import run_pipeline
    result = run_pipeline("Reduce energy consumption by 25% ...")
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, TypedDict, List, Optional

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, END

from agents.intent_agent import intent_agent_node
from agents.planner_agent import planner_agent_node
from agents.simulation_agent import simulation_agent_node
from agents.security_agent import security_agent_node
from agents.critic_agent import critic_agent_node
from agents.execution_agent import execution_agent_node
from simulation.edge_environment import EdgeEnvironment


# ──────────────────────────────────────────────
# Shared State
# ──────────────────────────────────────────────

class EcoEdgeState(TypedDict, total=False):
    """Typed state dictionary shared across all agent nodes."""
    # Input
    intent_text: str

    # Environment
    environment: Any  # EdgeEnvironment instance (not serializable)
    env_snapshot: Dict[str, Any]

    # Pipeline data
    parsed_intent: Dict[str, Any]
    proposed_plan: Dict[str, Any]
    simulation_result: Dict[str, Any]
    security_result: Dict[str, Any]
    critic_result: Dict[str, Any]
    execution_result: Dict[str, Any]

    # Control flow
    plan_attempt: int
    max_attempts: int
    agent_trace: List[Dict[str, Any]]


# ──────────────────────────────────────────────
# Re-planning Router
# ──────────────────────────────────────────────

def should_replan(state: EcoEdgeState) -> str:
    """
    Conditional edge after the Critic node.
    Routes to 'replan' (back to Planner) or 'execute' based on approval.
    """
    critic_result = state.get("critic_result", {})
    attempt = state.get("plan_attempt", 1)
    max_attempts = state.get("max_attempts", 3)

    if critic_result.get("approved", False):
        return "execute"
    elif attempt < max_attempts:
        return "replan"
    else:
        return "execute"  # execute anyway after max retries (will be skipped by exec agent)


def replan_node(state: EcoEdgeState) -> EcoEdgeState:
    """Increment the attempt counter before looping back to the Planner."""
    new_attempt = state.get("plan_attempt", 1) + 1
    return {
        **state,
        "plan_attempt": new_attempt,
        "agent_trace": state.get("agent_trace", []) + [
            {
                "agent": "Router",
                "action": "replan",
                "result": f"Critic rejected plan. Re-planning (attempt {new_attempt})",
            }
        ],
    }


# ──────────────────────────────────────────────
# Build the Graph
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the ECO-EDGE LangGraph workflow.

    Graph structure:
      intent → planner → simulate → security → critic
                  ↑                               |
                  └─── replan ←────── (rejected) ──┘
                                       |
                                  (approved) → execute → END
    """
    graph = StateGraph(EcoEdgeState)

    # Add nodes
    graph.add_node("intent", intent_agent_node)
    graph.add_node("planner", planner_agent_node)
    graph.add_node("simulate", simulation_agent_node)
    graph.add_node("security", security_agent_node)
    graph.add_node("critic", critic_agent_node)
    graph.add_node("replan", replan_node)
    graph.add_node("execute", execution_agent_node)

    # Add edges (linear pipeline)
    graph.set_entry_point("intent")
    graph.add_edge("intent", "planner")
    graph.add_edge("planner", "simulate")
    graph.add_edge("simulate", "security")
    graph.add_edge("security", "critic")

    # Conditional edge after critic
    graph.add_conditional_edges(
        "critic",
        should_replan,
        {
            "execute": "execute",
            "replan": "replan",
        },
    )

    # Re-plan loops back to planner
    graph.add_edge("replan", "planner")

    # Execute leads to END
    graph.add_edge("execute", END)

    return graph


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_pipeline(
    intent_text: str,
    num_tier1: int = 20,
    num_tier2: int = 5,
    num_tier3: int = 2,
    max_attempts: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the full ECO-EDGE ActSimSecCrit pipeline.

    Args:
        intent_text: Natural language orchestration intent.
        num_tier1: Number of Tier-1 edge nodes.
        num_tier2: Number of Tier-2 MEC servers.
        num_tier3: Number of Tier-3 cloud nodes.
        max_attempts: Maximum re-planning attempts.
        seed: Random seed for reproducibility.

    Returns:
        Final state dictionary with all agent outputs and traces.
    """
    # Initialize environment
    env = EdgeEnvironment(num_tier1=num_tier1, num_tier2=num_tier2, num_tier3=num_tier3, seed=seed)

    # Build and compile graph
    graph = build_graph()
    compiled = graph.compile()

    # Initial state
    initial_state: EcoEdgeState = {
        "intent_text": intent_text,
        "environment": env,
        "env_snapshot": env.get_snapshot(),
        "plan_attempt": 1,
        "max_attempts": max_attempts,
        "agent_trace": [],
    }

    # Run the graph
    final_state = compiled.invoke(initial_state)
    return final_state
