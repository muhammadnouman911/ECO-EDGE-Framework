"""
ECO-EDGE: Edge Network Simulation Environment
===============================================
Lightweight digital-twin simulator for a heterogeneous 3-tier MEC network.
Models edge nodes with CPU, GPU, memory, bandwidth, power, and battery state.
Used by the Simulation Agent to predict KPIs before actions are applied.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class EdgeNode:
    """Represents a single edge/MEC/cloud node."""

    node_id: str
    tier: int  # 1 = edge device, 2 = MEC server, 3 = cloud
    cpu_gflops: float
    gpu_tflops: float
    memory_gb: float
    bandwidth_mbps: float
    max_power_w: float
    battery_soc: float = 1.0  # state-of-charge [0, 1]; 1.0 for powered nodes
    is_battery_powered: bool = False
    current_load: float = 0.0  # [0, 1]
    status: str = "active"  # active | sleeping | migrating
    slm_tier: str = "full"  # full | mid | min (Elastic Intelligence)

    @property
    def available_compute(self) -> float:
        return (self.cpu_gflops + self.gpu_tflops * 1000) * (1.0 - self.current_load)

    @property
    def power_consumption(self) -> float:
        """Dynamic power ∝ load³ (cubic DVFS model)."""
        return self.max_power_w * (0.1 + 0.9 * self.current_load ** 3)


class EdgeEnvironment:
    """
    Simulates a heterogeneous 3-tier edge-cloud network.

    Provides:
      - simulate_action(): predict KPIs for a proposed action (non-destructive)
      - apply_action(): actually mutate node states (destructive)
      - get_snapshot(): return current network state summary
    """

    def __init__(self, num_tier1: int = 20, num_tier2: int = 5, num_tier3: int = 2, seed: int = 42):
        random.seed(seed)
        self.nodes: Dict[str, EdgeNode] = {}
        self._build_network(num_tier1, num_tier2, num_tier3)

    def _build_network(self, n1: int, n2: int, n3: int) -> None:
        """Create heterogeneous nodes across 3 tiers using realistic hardware profiles."""
        node_id = 0

        # Hardware Profiles for accurate energy modeling (as requested by meta-review)
        hw_profiles_tier1 = [
            {"name": "Raspberry Pi 5", "cpu": 4.5, "gpu": 0.0, "mem": 8.0, "power": 12.0},
            {"name": "Jetson Nano", "cpu": 2.0, "gpu": 0.47, "mem": 4.0, "power": 10.0},
            {"name": "Jetson Orin Nano", "cpu": 6.0, "gpu": 20.0, "mem": 8.0, "power": 15.0} # 15W mode
        ]
        
        # Tier 1: Edge devices (resource-constrained, some battery-powered)
        for i in range(n1):
            is_batt = i < int(n1 * 0.6)  # 60% are battery-powered
            profile = random.choice(hw_profiles_tier1)
            self.nodes[f"edge-{node_id:03d}"] = EdgeNode(
                node_id=f"edge-{node_id:03d}",
                tier=1,
                cpu_gflops=profile["cpu"],
                gpu_tflops=profile["gpu"],
                memory_gb=profile["mem"],
                bandwidth_mbps=random.uniform(10, 100),
                max_power_w=profile["power"],
                battery_soc=random.uniform(0.2, 1.0) if is_batt else 1.0,
                is_battery_powered=is_batt,
                current_load=random.uniform(0.1, 0.7),
            )
            node_id += 1

        # Tier 2: MEC servers (e.g., localized 5G base stations with edge servers)
        hw_profiles_tier2 = [
            {"name": "Dell PowerEdge XR11", "cpu": 40.0, "gpu": 130.0, "mem": 128.0, "power": 350.0}, # A30 GPU
            {"name": "Supermicro IoT Gateway", "cpu": 25.0, "gpu": 20.0, "mem": 64.0, "power": 150.0}
        ]
        for _ in range(n2):
            profile = random.choice(hw_profiles_tier2)
            self.nodes[f"mec-{node_id:03d}"] = EdgeNode(
                node_id=f"mec-{node_id:03d}",
                tier=2,
                cpu_gflops=profile["cpu"],
                gpu_tflops=profile["gpu"],
                memory_gb=profile["mem"],
                bandwidth_mbps=random.uniform(100, 1000),
                max_power_w=profile["power"],
                current_load=random.uniform(0.2, 0.5),
            )
            node_id += 1

        # Tier 3: Cloud nodes (high resources, e.g., Regional Data Center)
        for _ in range(n3):
            self.nodes[f"cloud-{node_id:03d}"] = EdgeNode(
                node_id=f"cloud-{node_id:03d}",
                tier=3,
                cpu_gflops=random.uniform(200, 500),
                gpu_tflops=random.uniform(150, 312),  # Equivalent to A100/H100 tier
                memory_gb=random.uniform(256, 1024),
                bandwidth_mbps=random.uniform(1000, 10000),
                max_power_w=random.uniform(500, 2000),
                current_load=random.uniform(0.1, 0.3),
            )
            node_id += 1

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a summary of the current network state."""
        active = [n for n in self.nodes.values() if n.status == "active"]
        sleeping = [n for n in self.nodes.values() if n.status == "sleeping"]
        total_power = sum(n.power_consumption for n in active)
        avg_load = sum(n.current_load for n in active) / max(len(active), 1)
        avg_battery = [n.battery_soc for n in self.nodes.values() if n.is_battery_powered]

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active),
            "sleeping_nodes": len(sleeping),
            "total_power_w": round(total_power, 2),
            "avg_load": round(avg_load, 3),
            "avg_battery_soc": round(sum(avg_battery) / max(len(avg_battery), 1), 3),
            "nodes_by_tier": {
                1: len([n for n in self.nodes.values() if n.tier == 1]),
                2: len([n for n in self.nodes.values() if n.tier == 2]),
                3: len([n for n in self.nodes.values() if n.tier == 3]),
            },
        }

    def get_tier_nodes(self, tier: int) -> List[EdgeNode]:
        return [n for n in self.nodes.values() if n.tier == tier]

    # ------------------------------------------------------------------
    # Simulation (non-destructive prediction)
    # ------------------------------------------------------------------

    def simulate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict KPIs for a proposed action WITHOUT mutating state.

        Returns dict with: predicted_latency_ms, predicted_energy_reduction_pct,
        predicted_success_rate, predicted_load_after, warnings.
        """
        action_type = action.get("type", "unknown")
        target_nodes = action.get("target_nodes", [])
        warnings: List[str] = []

        # Base KPI predictions (heuristic model)
        if action_type == "sleep_scheduling":
            n_sleep = len(target_nodes)
            n_active = len([n for n in self.nodes.values() if n.status == "active"])
            sleep_ratio = n_sleep / max(n_active, 1)

            energy_reduction = sleep_ratio * 100 * random.uniform(0.85, 1.15)
            latency_increase = sleep_ratio * 20 * random.uniform(0.8, 1.2)  # ms penalty
            base_latency = 25.0  # ms baseline
            predicted_latency = base_latency + latency_increase
            success_rate = max(0.75, 1.0 - sleep_ratio * 0.3)

            if sleep_ratio > 0.5:
                warnings.append("Sleeping >50% of nodes risks capacity shortfall")

        elif action_type == "workload_migration":
            energy_reduction = random.uniform(10, 25)
            predicted_latency = random.uniform(20, 55)
            success_rate = random.uniform(0.85, 0.98)

        elif action_type == "traffic_steering":
            energy_reduction = random.uniform(5, 20)
            predicted_latency = random.uniform(15, 40)
            success_rate = random.uniform(0.90, 0.99)

        elif action_type == "composite":
            # Composite: combine sub-actions
            sub_actions = action.get("sub_actions", [])
            total_energy = 0
            max_latency = 0
            min_success = 1.0
            for sub in sub_actions:
                result = self.simulate_action(sub)
                total_energy += result["predicted_energy_reduction_pct"]
                max_latency = max(max_latency, result["predicted_latency_ms"])
                min_success = min(min_success, result["predicted_success_rate"])
                warnings.extend(result.get("warnings", []))
            energy_reduction = min(total_energy, 60)  # cap at 60%
            predicted_latency = max_latency
            success_rate = min_success

        else:
            energy_reduction = random.uniform(2, 10)
            predicted_latency = random.uniform(30, 80)
            success_rate = random.uniform(0.70, 0.90)
            warnings.append(f"Unknown action type: {action_type}")

        snapshot = self.get_snapshot()

        return {
            "predicted_latency_ms": round(predicted_latency, 2),
            "predicted_energy_reduction_pct": round(energy_reduction, 2),
            "predicted_success_rate": round(success_rate, 4),
            "predicted_load_after": round(snapshot["avg_load"] * (1 - energy_reduction / 200), 3),
            "current_power_w": snapshot["total_power_w"],
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Execution (destructive state mutation)
    # ------------------------------------------------------------------

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a validated action to the environment.
        Returns a summary of changes made.
        """
        action_type = action.get("type", "unknown")
        target_nodes = action.get("target_nodes", [])
        changes: List[str] = []

        if action_type == "sleep_scheduling":
            for nid in target_nodes:
                if nid in self.nodes:
                    self.nodes[nid].status = "sleeping"
                    self.nodes[nid].current_load = 0.0
                    changes.append(f"{nid} -> sleeping")

        elif action_type == "workload_migration":
            src = action.get("source_node")
            dst = action.get("destination_node")
            if src in self.nodes and dst in self.nodes:
                load_transfer = self.nodes[src].current_load * 0.5
                self.nodes[src].current_load -= load_transfer
                self.nodes[dst].current_load = min(1.0, self.nodes[dst].current_load + load_transfer)
                changes.append(f"Migrated {load_transfer:.2f} load from {src} -> {dst}")

        elif action_type == "traffic_steering":
            for nid in target_nodes:
                if nid in self.nodes:
                    self.nodes[nid].current_load *= 0.7
                    changes.append(f"{nid} load reduced to {self.nodes[nid].current_load:.2f}")

        elif action_type == "composite":
            for sub in action.get("sub_actions", []):
                sub_result = self.apply_action(sub)
                changes.extend(sub_result["changes"])

        # Update battery drain for active battery-powered nodes
        for node in self.nodes.values():
            if node.is_battery_powered and node.status == "active":
                drain = (node.power_consumption / node.max_power_w) * 0.01
                node.battery_soc = max(0, node.battery_soc - drain)

        return {
            "action_type": action_type,
            "changes": changes,
            "new_snapshot": self.get_snapshot(),
        }
