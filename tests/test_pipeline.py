from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import dill
import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flatland_sim.pipeline import Pipeline, generate_scenarios
from flatland_sim.snapshot import ScenarioSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(output_dir: str, seed: int = 42, num_scenarios: int = 3) -> dict:
    return {
        "seed": seed,
        "num_scenarios": num_scenarios,
        "max_steps": 50,
        "simulation_dir": str(output_dir),
        "randomization": {
            "num_trains": {"min": 2, "max": 2},
            "grid_width": {"min": 30, "max": 30},
            "grid_height": {"min": 30, "max": 30},
            "num_cities": {"min": 2, "max": 2},
            "max_rails_between_cities": {"min": 1, "max": 1},
            "max_rail_pairs_in_city": {"min": 1, "max": 1},
        },
    }


def make_fake_snapshot(scenario_id: int) -> ScenarioSnapshot:
    """Return a minimal, dill-serializable ScenarioSnapshot."""
    return ScenarioSnapshot(
        scenario_id=scenario_id,
        config={},
        env_width=30,
        env_height=30,
        num_agents=2,
        distance_map=np.zeros((2, 30, 30, 4)),
        rail_grid=np.zeros((30, 30), dtype=int),
        rail_transitions={(0, 0, 0): (0, 0, 0, 0)},
        agent_targets=[(0, 0), (1, 1)],
        agent_initial_positions=[(2, 2), (3, 3)],
        timesteps=[],
    )


def _make_mock_env() -> MagicMock:
    """Return a mock env whose extracted fields are all real (dill-safe) objects."""
    mock_env = MagicMock()
    mock_env.width = 30
    mock_env.height = 30
    mock_env.get_num_agents.return_value = 2

    agent0 = MagicMock()
    agent0.target = (0, 0)
    agent0.initial_position = (1, 1)
    agent1 = MagicMock()
    agent1.target = (2, 2)
    agent1.initial_position = (3, 3)
    mock_env.agents = [agent0, agent1]

    # rail.grid: astype returns a real numpy array; __ne__ returns False so the
    # rail_transitions loop body is never entered (grid is all zeros → no rail cells)
    real_grid = np.zeros((30, 30), dtype=np.uint16)
    mock_rail = MagicMock()
    mock_rail.grid = real_grid          # real ndarray, not a MagicMock
    mock_env.rail = mock_rail

    mock_env.distance_map.get.return_value = np.zeros((2, 30, 30, 4))
    return mock_env


# ---------------------------------------------------------------------------
# Property 1: Config loading equivalence
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 1: Config dict vs YAML path produce equivalent snapshot counts
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=99))
def test_p1_config_equivalence(seed, tmp_path):
    # Use unique sub-dirs per seed to avoid output-path collisions across examples
    dir_dict = tmp_path / f"dict_{seed}"
    dir_yaml = tmp_path / f"yaml_{seed}"
    dir_dict.mkdir(parents=True, exist_ok=True)
    dir_yaml.mkdir(parents=True, exist_ok=True)

    config_dict = make_config(str(dir_dict), seed=seed, num_scenarios=2)
    config_yaml_dict = make_config(str(dir_yaml), seed=seed, num_scenarios=2)

    yaml_path = str(tmp_path / f"config_{seed}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config_yaml_dict, f)

    snapshots_dict = generate_scenarios(config_dict)
    snapshots_yaml = generate_scenarios(yaml_path)

    assert len(snapshots_dict) == len(snapshots_yaml)
    assert [s.scenario_id for s in snapshots_dict] == [s.scenario_id for s in snapshots_yaml]


# ---------------------------------------------------------------------------
# Property 12: Pipeline attempt count
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 12: Pipeline makes exactly num_scenarios attempts
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(num_scenarios=st.integers(min_value=1, max_value=5))
def test_p12_attempt_count(num_scenarios, tmp_path):
    config = make_config(str(tmp_path), num_scenarios=num_scenarios)

    call_count = {"n": 0}

    def counting_build():
        call_count["n"] += 1
        raise RuntimeError("always fail")

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen:
        MockGen.return_value.build.side_effect = counting_build
        Pipeline(config).run()

    assert call_count["n"] == num_scenarios


# ---------------------------------------------------------------------------
# Property 13: Pipeline skips failures and continues
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 13: Pipeline returns only successful snapshots when some fail
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total=st.integers(min_value=1, max_value=4),
    num_failures=st.integers(min_value=0, max_value=3),
)
def test_p13_skip_failures(total, num_failures, tmp_path):
    num_failures = min(num_failures, total)
    config = make_config(str(tmp_path), num_scenarios=total)

    call_count = {"n": 0}

    def build_side_effect():
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < num_failures:
            raise RuntimeError(f"Simulated failure {idx}")
        return _make_mock_env(), {}

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen, \
         patch("flatland_sim.pipeline.SimulationRunner") as MockRunner:
        MockGen.return_value.build.side_effect = build_side_effect
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = []
        mock_runner_instance._frames = []

        result = Pipeline(config).run()

    assert len(result) == total - num_failures


# ---------------------------------------------------------------------------
# Property 14: Serialization round-trip
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 14: dill.load on output file equals Pipeline.run() return value
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=99))
def test_p14_serialization_round_trip(seed, tmp_path):
    config = make_config(str(tmp_path / f"run_{seed}"), seed=seed, num_scenarios=2)
    (tmp_path / f"run_{seed}").mkdir(parents=True, exist_ok=True)

    def build_side_effect():
        return _make_mock_env(), {}

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen, \
         patch("flatland_sim.pipeline.SimulationRunner") as MockRunner:
        MockGen.return_value.build.side_effect = build_side_effect
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = []
        mock_runner_instance._frames = []

        returned = Pipeline(config).run()

    pkl_path = Path(config["simulation_dir"]) / "scenarios.pkl"
    with open(pkl_path, "rb") as f:
        loaded = dill.load(f)

    assert len(loaded) == len(returned)
    assert [s.scenario_id for s in loaded] == [s.scenario_id for s in returned]


# ---------------------------------------------------------------------------
# Unit tests: edge cases
# ---------------------------------------------------------------------------

def test_empty_list_on_all_failures(tmp_path):
    """Pipeline returns [] and still writes the output file when all scenarios fail."""
    config = make_config(str(tmp_path), num_scenarios=3)

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen:
        MockGen.return_value.build.side_effect = RuntimeError("always fail")
        result = Pipeline(config).run()

    pkl_path = Path(config["simulation_dir"]) / "scenarios.pkl"
    assert result == []
    assert os.path.exists(pkl_path)
    with open(pkl_path, "rb") as f:
        loaded = dill.load(f)
    assert loaded == []


def test_parent_dir_creation(tmp_path):
    """Pipeline creates deeply nested parent directories for simulation_dir."""
    nested_sim_dir = str(tmp_path / "a" / "b" / "c")
    config = make_config(nested_sim_dir, num_scenarios=1)

    def build_side_effect():
        return _make_mock_env(), {}

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen, \
         patch("flatland_sim.pipeline.SimulationRunner") as MockRunner:
        MockGen.return_value.build.side_effect = build_side_effect
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = []
        mock_runner_instance._frames = []

        Pipeline(config).run()

    assert os.path.exists(Path(nested_sim_dir) / "scenarios.pkl")


# ---------------------------------------------------------------------------
# Task 1.2: test_pipeline_output_paths
# ---------------------------------------------------------------------------

def test_pipeline_output_paths(tmp_path):
    """scenarios.pkl lands at {simulation_dir}/scenarios.pkl and GIFs at {simulation_dir}/previews/."""
    sim_dir = str(tmp_path / "sim_output")
    config = make_config(sim_dir, num_scenarios=1)

    def build_side_effect():
        return _make_mock_env(), {}

    with patch("flatland_sim.pipeline.ScenarioGenerator") as MockGen, \
         patch("flatland_sim.pipeline.SimulationRunner") as MockRunner:
        MockGen.return_value.build.side_effect = build_side_effect
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = []
        mock_runner_instance._frames = []

        pipeline = Pipeline(config)

    assert pipeline.pkl_path == Path(sim_dir) / "scenarios.pkl"
    assert pipeline.previews_dir == Path(sim_dir) / "previews"


# ---------------------------------------------------------------------------
# Task 1.1: Property 1 — Output path derivation
# Feature: scenarios-analysis, Property 1: Output path derivation
# ---------------------------------------------------------------------------

@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(sim_dir=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="/_-")))
def test_p1_output_path_derivation(sim_dir, tmp_path):
    """All paths written by Pipeline are sub-paths of simulation_dir."""
    config = make_config(sim_dir, num_scenarios=1)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = config
    pipeline.num_scenarios = config["num_scenarios"]
    pipeline.max_steps = config["max_steps"]
    pipeline.previews_dir = Path(sim_dir) / "previews"
    pipeline.pkl_path = Path(sim_dir) / "scenarios.pkl"
    pipeline.preview = False
    pipeline.sampler = MagicMock()

    sim_path = Path(sim_dir)
    assert pipeline.pkl_path.parts[:len(sim_path.parts)] == sim_path.parts
    assert pipeline.previews_dir.parts[:len(sim_path.parts)] == sim_path.parts
    assert "output_path" not in config
    assert "preview_output_dir" not in config
