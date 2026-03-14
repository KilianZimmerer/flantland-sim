import dill
import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

from flatland_sim.snapshot import ScenarioSnapshot


def make_snapshot(seed=42):
    params = {
        "grid_width": 30,
        "grid_height": 30,
        "num_trains": 2,
        "num_cities": 2,
        "max_rails_between_cities": 1,
        "max_rail_pairs_in_city": 1,
    }
    from flatland_sim.generator import ScenarioGenerator
    from flatland_sim.runner import SimulationRunner

    env, _ = ScenarioGenerator(params).build()
    rail_grid = env.rail.grid.astype(int)
    distance_map = env.distance_map.get()
    rail_transitions = {}
    for row in range(env.height):
        for col in range(env.width):
            if env.rail.grid[row, col] != 0:
                for direction in range(4):
                    rail_transitions[(row, col, direction)] = env.rail.get_transitions(
                        ((row, col), direction)
                    )
    agent_targets = [tuple(a.target) for a in env.agents]
    agent_initial_positions = [tuple(a.initial_position) for a in env.agents]
    timesteps = SimulationRunner(env, max_steps=50).run()
    return ScenarioSnapshot(
        scenario_id=seed,
        config=params,
        env_width=env.width,
        env_height=env.height,
        num_agents=env.get_num_agents(),
        distance_map=distance_map,
        rail_grid=rail_grid,
        rail_transitions=rail_transitions,
        agent_targets=agent_targets,
        agent_initial_positions=agent_initial_positions,
        timesteps=timesteps,
    )


# Feature: flatland-sim, Property 10: ScenarioSnapshot array shapes and list lengths match env dimensions
@settings(max_examples=20)
@given(st.integers(min_value=0, max_value=50))
def test_snapshot_field_shapes(seed):
    snapshot = make_snapshot(seed)

    assert snapshot.distance_map.shape == (
        snapshot.num_agents,
        snapshot.env_height,
        snapshot.env_width,
        4,
    )
    assert snapshot.rail_grid.shape == (snapshot.env_height, snapshot.env_width)
    assert len(snapshot.agent_targets) == snapshot.num_agents
    assert len(snapshot.agent_initial_positions) == snapshot.num_agents

    for key in snapshot.rail_transitions:
        assert len(key) == 3, f"Key {key} is not a 3-tuple"
        row, col, direction = key
        assert 0 <= direction <= 3, f"direction {direction} out of range in key {key}"


# Feature: flatland-sim, Property 11: dill round-trip preserves all ScenarioSnapshot fields
@settings(max_examples=20)
@given(st.integers(min_value=0, max_value=50))
def test_snapshot_dill_roundtrip(seed):
    snapshot = make_snapshot(seed)

    serialized = dill.dumps(snapshot)
    restored = dill.loads(serialized)

    assert restored.scenario_id == snapshot.scenario_id
    assert restored.config == snapshot.config
    assert restored.env_width == snapshot.env_width
    assert restored.env_height == snapshot.env_height
    assert restored.num_agents == snapshot.num_agents
    assert np.array_equal(restored.distance_map, snapshot.distance_map)
    assert np.array_equal(restored.rail_grid, snapshot.rail_grid)
    assert restored.rail_transitions == snapshot.rail_transitions
    assert restored.agent_targets == snapshot.agent_targets
    assert restored.agent_initial_positions == snapshot.agent_initial_positions
    assert restored.timesteps == snapshot.timesteps
