from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import composite

from flatland_sim.snapshot import ScenarioSnapshot


@composite
def snapshot_strategy(draw) -> ScenarioSnapshot:
    scenario_id = draw(st.integers(min_value=0, max_value=10_000))
    num_agents = draw(st.integers(min_value=1, max_value=5))
    config_key = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Ll",))))
    config_val = draw(st.integers(min_value=0, max_value=100))
    config = {config_key: config_val}

    num_timesteps = draw(st.integers(min_value=0, max_value=10))

    timesteps = []
    for step in range(num_timesteps):
        agents = []
        for agent_id in range(num_agents):
            label = draw(st.integers(min_value=0, max_value=4))
            agents.append({
                "id": agent_id,
                "position": None,
                "direction": 0,
                "status": "active",
                "action_planned": 0,
                "next_position": None,
                "transition_label": label,
            })
        timesteps.append({"step": step, "agents": agents})

    return ScenarioSnapshot(
        scenario_id=scenario_id,
        config=config,
        env_width=10,
        env_height=10,
        num_agents=num_agents,
        distance_map=np.zeros((num_agents, 10, 10, 4)),
        rail_grid=np.zeros((10, 10), dtype=int),
        rail_transitions={},
        agent_targets=[(0, 0)] * num_agents,
        agent_initial_positions=[(0, 0)] * num_agents,
        timesteps=timesteps,
    )
