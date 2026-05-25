"""
PPO baseline for edge task offloading (Stable-Baselines3).

Trains outside the Java DES loop; exported policy weights can be validated against
EdgeCloudSim PPO_FIT (Java clipped policy-gradient orchestrator).

Hyperparameters (paper defaults):
  learning_rate=3e-4, clip_range=0.2, gae_lambda=0.95, n_steps=2048,
  batch_size=64, gamma=0.99, policy=MLP [128, 128], total_timesteps=1e6

Reward: r = w_s * success - w_tau * (tau/tau_max) - w_e * energy
         (w_s, w_tau, w_e) = (1.0, 0.4, 0.3)
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None  # type: ignore

# Gymnasium env wrapper would connect to EdgeCloudSim logs / replay buffer.
# For reproducibility without a live JVM bridge, use the Java PPO_FIT orchestrator.


def default_config() -> dict:
    return {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "net_arch": [128, 128],
        "total_timesteps": 1_000_000,
        "reward_weights": {"w_s": 1.0, "w_tau": 0.4, "w_e": 0.3},
        "state": [
            "vm_utilizations_normalized",
            "task_size",
            "deadline",
            "task_type",
            "tier_load",
        ],
        "action": "discrete_vm_index",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ppo_config.json")
    ap.add_argument("--train", action="store_true", help="Run SB3 training (requires custom Gym env)")
    args = ap.parse_args()

    cfg = default_config()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {args.out}")

    if args.train:
        if PPO is None:
            raise SystemExit("Install: pip install stable-baselines3 gymnasium")
        raise SystemExit(
            "Connect EdgeCloudSimGymEnv to your log-driven simulator, then call "
            "PPO('MlpPolicy', env, learning_rate=3e-4, clip_range=0.2, ...).learn(1_000_000)"
        )


if __name__ == "__main__":
    main()
