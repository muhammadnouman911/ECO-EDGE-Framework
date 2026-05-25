"""
ECO-EDGE: Experiment Runner
==============================
CLI entry point that demonstrates the full ActSimSecCrit pipeline
with a sample orchestration scenario.

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --intent "Your custom intent here"
    python experiments/run_experiment.py --nodes 50 --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.eco_edge_graph import run_pipeline


# ──────────────────────────────────────────────
# Pretty Printing
# ──────────────────────────────────────────────

SEPARATOR = "=" * 72
THIN_SEP = "-" * 72


def print_header():
    print()
    print(SEPARATOR)
    print("  ECO-EDGE: Decentralized Agentic Orchestration Framework")
    print("  ActSimSecCrit Pipeline — Research Prototype")
    print(SEPARATOR)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEPARATOR)
    print()


def print_section(title: str):
    print()
    print(THIN_SEP)
    print(f"  > {title}")
    print(THIN_SEP)


def print_agent_trace(trace: list):
    print_section("AGENT EXECUTION TRACE")
    for i, entry in enumerate(trace, 1):
        agent = entry.get("agent", "Unknown")
        action = entry.get("action", "")
        result = entry.get("result", "")
        attempt = entry.get("attempt", "")
        attempt_str = f" (attempt {attempt})" if attempt else ""
        print(f"  [{i}] {agent}{attempt_str}")
        print(f"      Action: {action}")
        print(f"      Result: {result}")
        print()


def print_environment(snapshot: dict):
    print_section("ENVIRONMENT SNAPSHOT")
    print(f"  Total Nodes:       {snapshot.get('total_nodes', '?')}")
    print(f"  Active Nodes:      {snapshot.get('active_nodes', '?')}")
    print(f"  Sleeping Nodes:    {snapshot.get('sleeping_nodes', '?')}")
    print(f"  Total Power:       {snapshot.get('total_power_w', '?')} W")
    print(f"  Avg Load:          {snapshot.get('avg_load', '?')}")
    print(f"  Avg Battery SOC:   {snapshot.get('avg_battery_soc', '?')}")
    tiers = snapshot.get("nodes_by_tier", {})
    print(f"  Nodes by Tier:     T1={tiers.get(1, '?')} | T2={tiers.get(2, '?')} | T3={tiers.get(3, '?')}")


def print_parsed_intent(parsed: dict):
    print_section("PARSED INTENT")
    print(f"  Raw: {parsed.get('raw_intent', '?')}")
    print(f"  Priority: {parsed.get('priority', '?')}")
    print(f"  Goals:")
    for g in parsed.get("goals", []):
        print(f"    • {g.get('metric', '?')}: {g.get('target_pct', '?')}%")
    print(f"  Constraints:")
    for c in parsed.get("constraints", []):
        detail = f"max {c.get('max_ms', '?')}ms" if "max_ms" in c else str(c)
        print(f"    • {c.get('metric', '?')}: {detail}")
    print(f"  Sub-tasks: {', '.join(parsed.get('sub_tasks', []))}")


def print_plan(plan: dict):
    print_section(f"PROPOSED PLAN (Attempt {plan.get('attempt', '?')})")
    action = plan.get("action", {})
    print(f"  Type:      {action.get('type', '?')}")
    print(f"  Rationale: {plan.get('rationale', '?')}")
    if action.get("type") == "composite":
        for i, sub in enumerate(action.get("sub_actions", []), 1):
            print(f"    Sub-action {i}: {sub.get('type', '?')} — {sub.get('description', '')}")
    targets = action.get("target_nodes", [])
    print(f"  Target Nodes ({len(targets)}): {', '.join(targets[:8])}{'...' if len(targets) > 8 else ''}")


def print_simulation(sim: dict):
    print_section("SIMULATION RESULTS")
    print(f"  Predicted Latency:          {sim.get('predicted_latency_ms', '?')} ms")
    print(f"  Predicted Energy Reduction: {sim.get('predicted_energy_reduction_pct', '?')}%")
    print(f"  Predicted Success Rate:     {sim.get('predicted_success_rate', '?')}")
    print(f"  Predicted Load After:       {sim.get('predicted_load_after', '?')}")
    warnings = sim.get("warnings", [])
    if warnings:
        print(f"  ! Warnings: {'; '.join(warnings)}")


def print_security(sec: dict):
    print_section("SECURITY CHECKS")
    for check in sec.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['check']}: {check.get('detail', '')}")
    print(f"  Summary: {sec.get('summary', '?')}")


def print_critic(critic: dict):
    print_section("CRITIC EVALUATION")
    for ev in critic.get("evaluations", []):
        status = "PASS" if ev["passed"] else "FAIL"
        print(f"  [{status}] {ev['criterion']}: {ev.get('detail', '')}")
    print()
    verdict = critic.get("summary", "?")
    print(f"  ==> VERDICT: {verdict}")


def print_execution(exec_result: dict):
    print_section("EXECUTION RESULT")
    status = exec_result.get("execution_status", "?")
    print(f"  Status: {status}")
    if status == "success":
        changes = exec_result.get("changes_made", [])
        print(f"  Changes Applied ({len(changes)}):")
        for c in changes[:10]:
            print(f"    • {c}")
        if len(changes) > 10:
            print(f"    ... and {len(changes) - 10} more")
        new_snap = exec_result.get("new_snapshot", {})
        if new_snap:
            print(f"  New Power: {new_snap.get('total_power_w', '?')} W")
            print(f"  New Avg Load: {new_snap.get('avg_load', '?')}")
    elif status == "skipped":
        print(f"  Reason: {exec_result.get('reason', '?')}")


def print_final_summary(state: dict):
    print()
    print(SEPARATOR)
    print("  FINAL SUMMARY")
    print(SEPARATOR)
    exec_result = state.get("execution_result", {})
    critic = state.get("critic_result", {})
    attempts = state.get("plan_attempt", 1)

    if exec_result.get("execution_status") == "success":
        print(f"  [COMPLETED] Orchestration plan EXECUTED SUCCESSFULLY")
        print(f"     Total planning attempts: {attempts}")
        sim = state.get("simulation_result", {})
        print(f"     Predicted Energy Reduction: {sim.get('predicted_energy_reduction_pct', '?')}%")
        print(f"     Predicted Latency: {sim.get('predicted_latency_ms', '?')} ms")
    else:
        print(f"  [FAILED] Orchestration plan was NOT executed")
        print(f"     Attempts exhausted: {attempts}")
        print(f"     Critic verdict: {critic.get('summary', '?')}")

    print()
    print(SEPARATOR)
    print("  Pipeline complete.")
    print(SEPARATOR)
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ECO-EDGE: Run the ActSimSecCrit orchestration pipeline"
    )
    parser.add_argument(
        "--intent",
        type=str,
        default="Reduce energy consumption by 25% in the edge network while keeping latency under 50 ms.",
        help="Natural language orchestration intent",
    )
    parser.add_argument("--nodes", type=int, default=20, help="Number of Tier-1 edge nodes")
    parser.add_argument("--mec", type=int, default=5, help="Number of Tier-2 MEC servers")
    parser.add_argument("--cloud", type=int, default=2, help="Number of Tier-3 cloud nodes")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max re-planning attempts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--slm", action="store_true", help="Enable Real SLM reasoning via Ollama")
    parser.add_argument("--model", type=str, default="phi3:mini", help="Ollama model name")
    parser.add_argument("--json", action="store_true", help="Output raw JSON state instead of formatted")

    args = parser.parse_args()

    # Set environment variables for SLM if flagged
    if args.slm:
        os.environ["USE_SLM"] = "true"
        os.environ["OLLAMA_MODEL"] = args.model

    if not args.json:
        print_header()
        print(f"  Intent: \"{args.intent}\"")
        print(f"  Network: {args.nodes} edge + {args.mec} MEC + {args.cloud} cloud nodes")
        if args.slm:
            print(f"  SLM Mode: ENABLED (Model: {args.model})")
        else:
            print(f"  SLM Mode: DISABLED (Rule-based fallback)")
        print()

    # Run the pipeline
    final_state = run_pipeline(
        intent_text=args.intent,
        num_tier1=args.nodes,
        num_tier2=args.mec,
        num_tier3=args.cloud,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    if args.json:
        # JSON output (exclude non-serializable environment object)
        output = {k: v for k, v in final_state.items() if k != "environment"}
        print(json.dumps(output, indent=2, default=str))
    else:
        # Formatted output
        print_environment(final_state.get("env_snapshot", {}))
        print_parsed_intent(final_state.get("parsed_intent", {}))
        print_plan(final_state.get("proposed_plan", {}))
        print_simulation(final_state.get("simulation_result", {}))
        print_security(final_state.get("security_result", {}))
        print_critic(final_state.get("critic_result", {}))
        print_execution(final_state.get("execution_result", {}))
        print_agent_trace(final_state.get("agent_trace", []))
        print_final_summary(final_state)


if __name__ == "__main__":
    main()
