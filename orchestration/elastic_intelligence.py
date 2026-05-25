"""
ECO-EDGE: Elastic Intelligence — Battery-Aware Dynamic Model Scaling
=====================================================================
Implements the MDP-based policy from the paper that selects SLM complexity
based on the node's battery state-of-charge (SOC).

Policy π*(B_i(t)) ≈
  Full (2.7B)  if B > 0.7
  Mid  (1.1B)  if 0.3 ≤ B ≤ 0.7
  Min  (0.5B)  if B < 0.3
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelTier:
    """Describes a single SLM tier."""
    name: str
    param_billions: float
    quantization: str
    inference_energy_j: float  # Joules per inference
    accuracy_proxy: float  # reasoning quality [0, 1]


# Available SLM tiers (from the paper's Elastic Intelligence formulation)
FULL_TIER = ModelTier("full", 2.7, "INT8", 0.84, 0.95)
MID_TIER = ModelTier("mid", 1.1, "INT4", 0.31, 0.82)
MIN_TIER = ModelTier("min", 0.5, "INT4", 0.15, 0.65)

ALL_TIERS = [FULL_TIER, MID_TIER, MIN_TIER]


def select_model_tier(battery_soc: float) -> ModelTier:
    """
    MDP optimal policy: select model tier based on battery SOC.

    Args:
        battery_soc: Battery state-of-charge in [0, 1].

    Returns:
        The ModelTier to use for the current inference.
    """
    if battery_soc > 0.7:
        return FULL_TIER
    elif battery_soc >= 0.3:
        return MID_TIER
    else:
        return MIN_TIER


def compute_reward(
    model: ModelTier,
    battery_soc: float,
    lambda1: float = 0.6,
    lambda2: float = 0.4,
    epsilon: float = 0.01,
) -> float:
    """
    MDP reward function (Eq. from paper):
      R(s, a) = λ₁ · μ_acc(a) − λ₂ · E_think(a) / (B_i(t) + ε)

    Higher reward = better trade-off between accuracy and energy sustainability.
    """
    accuracy_reward = lambda1 * model.accuracy_proxy
    energy_penalty = lambda2 * model.inference_energy_j / (battery_soc + epsilon)
    return round(accuracy_reward - energy_penalty, 4)


def compute_battery_transition(
    battery_soc: float,
    model: ModelTier,
    e_base: float = 0.05,
    e_cap: float = 50.0,  # Wh total capacity
    harvesting: float = 0.0,
) -> float:
    """
    Battery state transition (Eq. from paper):
      B(t+1) = max(0, min(1, B(t) - (E_think + E_base) / E_cap + H(t)))
    """
    drain = (model.inference_energy_j + e_base) / e_cap
    new_soc = battery_soc - drain + harvesting
    return round(max(0.0, min(1.0, new_soc)), 4)


def get_elastic_status(battery_soc: float) -> dict:
    """
    Return a full status report for a node's Elastic Intelligence state.
    """
    tier = select_model_tier(battery_soc)
    reward = compute_reward(tier, battery_soc)
    next_soc = compute_battery_transition(battery_soc, tier)

    return {
        "battery_soc": round(battery_soc, 4),
        "selected_tier": tier.name,
        "model_params_b": tier.param_billions,
        "quantization": tier.quantization,
        "inference_energy_j": tier.inference_energy_j,
        "accuracy_proxy": tier.accuracy_proxy,
        "mdp_reward": reward,
        "predicted_next_soc": next_soc,
    }
