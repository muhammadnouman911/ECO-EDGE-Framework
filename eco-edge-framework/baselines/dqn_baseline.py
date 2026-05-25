"""
Tabular DQN-style baseline for edge task placement (software-only simulation).
Used by experiments/run_dqn_benchmark.py and plot_eta_comparison.py.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SimTask:
    size: float
    deadline: float
    task_type: int = 0


class DQNBaseline:
    """Simplified Q-learning scheduler for edge node assignment."""

    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        self.state_size = state_size
        self.action_size = action_size
        self.memory: deque = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.q_table: Dict[Tuple, float] = {}
        self._rng = random.Random(seed)

    def get_state(
        self, node_loads: Sequence[float], task_size: float, deadline: float
    ) -> Tuple:
        loads = tuple(np.round(list(node_loads)[: self.state_size], 1))
        return loads + (round(task_size, 1), round(deadline, 0))

    def choose_action(self, state: Tuple, available_nodes: List[int]) -> int:
        if not available_nodes:
            raise ValueError("No available nodes for assignment")
        if self._rng.random() < self.epsilon:
            return self._rng.choice(available_nodes)
        q_vals = {n: self.q_table.get((state, n), 0.0) for n in available_nodes}
        return max(q_vals, key=q_vals.get)

    def learn(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool,
    ) -> None:
        old_q = self.q_table.get((state, action), 0.0)
        if done:
            target = reward
        else:
            next_vals = [
                self.q_table.get((next_state, a), 0.0)
                for a in range(self.action_size)
            ]
            target = reward + self.gamma * max(next_vals, default=0.0)
        self.q_table[(state, action)] = old_q + self.lr * (target - old_q)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def compute_reward(latency: float, energy: float, deadline: float) -> float:
        deadline_penalty = -10.0 if latency > deadline else 0.0
        energy_cost = -energy * 0.1
        return deadline_penalty + energy_cost


_LOAD_STEP = 0.03


def next_fit_assign(node_loads: List[float], task: SimTask) -> Tuple[int, float, float]:
    """Heuristic next-fit: first node with capacity."""
    for i, load in enumerate(node_loads):
        if load + task.size * _LOAD_STEP <= 0.92:
            latency = (load + 0.08) * 80.0 + task.size * 2.0
            energy = task.size * 0.06
            node_loads[i] = min(1.0, load + task.size * _LOAD_STEP)
            return i, latency, energy
    return -1, 400.0, task.size * 0.15


def assign_task_to_node(
    node_loads: List[float], node_index: int, task: SimTask
) -> Tuple[float, float]:
    load = node_loads[node_index]
    latency = (load + 0.08) * 80.0 + task.size * 2.0
    energy = task.size * 0.06
    node_loads[node_index] = min(1.0, load + task.size * _LOAD_STEP)
    return latency, energy
