from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioSnapshot:
    scenario_id: int
    config: dict
    env_width: int
    env_height: int
    num_agents: int
    distance_map: np.ndarray        # shape: (num_agents, H, W, 4)
    rail_grid: np.ndarray           # shape: (H, W)
    rail_transitions: dict          # (row, col, direction) -> tuple[int, ...]
    agent_targets: list[tuple[int, int]]
    agent_initial_positions: list[tuple[int, int]]
    timesteps: list[dict]
