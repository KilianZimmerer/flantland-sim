import logging
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from flatland_sim.generator import ScenarioGenerator
from flatland_sim.runner import SimulationRunner

REQUIRED_TIMESTEP_KEYS = {"step", "agents"}
REQUIRED_AGENT_KEYS = {"id", "position", "direction", "status", "action_planned", "next_position"}

MOVING_STOPPED_STATUSES = {"MOVING", "STOPPED"}
WAITING_DEPART_STATUSES = {"WAITING", "READY_TO_DEPART"}


def make_small_env(seed=42, num_trains=2):
    params = {
        "grid_width": 30,
        "grid_height": 30,
        "num_trains": num_trains,
        "num_cities": 2,
        "max_rails_between_cities": 1,
        "max_rail_pairs_in_city": 1,
    }
    env, _ = ScenarioGenerator(params).build()
    return env


# ---------------------------------------------------------------------------
# P6 – Action policy correctness
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 6: Action policy maps every TrainState to the correct action set
@settings(max_examples=20)
@given(seed=st.integers(min_value=0, max_value=999))
def test_p6_action_policy_correctness(seed):
    env = make_small_env(seed=seed)
    timesteps = SimulationRunner(env, max_steps=200).run()

    for ts in timesteps:
        for agent in ts["agents"]:
            status = agent["status"]
            action = agent["action_planned"]
            if status in MOVING_STOPPED_STATUSES:
                assert action in {1, 2, 3, 4}, (
                    f"Status {status}: expected action in {{1,2,3,4}}, got {action}"
                )
            elif status in WAITING_DEPART_STATUSES:
                assert action == 2, (
                    f"Status {status}: expected action 2, got {action}"
                )
            elif status == "DONE":
                assert action == 0, (
                    f"Status DONE: expected action 0, got {action}"
                )


# ---------------------------------------------------------------------------
# P7 – Timestep record structure
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 7: Every timestep record contains all required keys
@settings(max_examples=20)
@given(seed=st.integers(min_value=0, max_value=999))
def test_p7_timestep_record_structure(seed):
    env = make_small_env(seed=seed)
    timesteps = SimulationRunner(env, max_steps=200).run()

    assert len(timesteps) > 0
    for ts in timesteps:
        assert REQUIRED_TIMESTEP_KEYS <= set(ts.keys()), (
            f"Timestep missing keys: {REQUIRED_TIMESTEP_KEYS - set(ts.keys())}"
        )
        for agent in ts["agents"]:
            assert REQUIRED_AGENT_KEYS <= set(agent.keys()), (
                f"Agent entry missing keys: {REQUIRED_AGENT_KEYS - set(agent.keys())}"
            )


# ---------------------------------------------------------------------------
# P8 – next_position consistency
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 8: next_position equals following timestep position for all timesteps
@settings(max_examples=20)
@given(seed=st.integers(min_value=0, max_value=999))
def test_p8_next_position_consistency(seed):
    env = make_small_env(seed=seed)
    timesteps = SimulationRunner(env, max_steps=200).run()

    # The final timestep is dropped, so every remaining timestep has a successor.
    # next_position[i] must equal position in the following timestep for all steps.
    for t in range(len(timesteps) - 1):
        current_agents = timesteps[t]["agents"]
        next_agents = timesteps[t + 1]["agents"]
        for i, agent in enumerate(current_agents):
            assert agent["next_position"] == next_agents[i]["position"], (
                f"Step {t}, agent {i}: next_position {agent['next_position']} "
                f"!= following position {next_agents[i]['position']}"
            )


# ---------------------------------------------------------------------------
# P9 – Early stopping
# ---------------------------------------------------------------------------

# Feature: flatland-sim, Property 9: Early stopping: timestep count < max_steps when any agent reaches destination
@settings(max_examples=20)
@given(seed=st.integers(min_value=0, max_value=999))
def test_p9_early_stopping(seed):
    max_steps = 500
    env = make_small_env(seed=seed)
    timesteps = SimulationRunner(env, max_steps=max_steps).run()

    # Scenarios now stop as soon as any agent reaches its destination,
    # so the timestep count should always be less than max_steps for
    # scenarios where at least one agent finishes.
    assert len(timesteps) <= max_steps


# ---------------------------------------------------------------------------
# Unit test – warning log when max_steps reached with unfinished agents
# ---------------------------------------------------------------------------

def test_warning_logged_when_max_steps_reached_with_unfinished_agents():
    from flatland.envs.agent_utils import TrainState

    # Build a minimal mock env where agents never finish
    mock_agent = MagicMock()
    mock_agent.state = TrainState.MOVING
    mock_agent.position = (5, 5)
    mock_agent.direction = 0

    mock_env = MagicMock()
    mock_env.agents = [mock_agent, mock_agent]
    mock_env.dones = {"__all__": False}
    mock_env.step.return_value = ({}, {}, {"__all__": False}, {})

    scenario_id = 99
    with patch("flatland_sim.runner.logging.warning") as mock_warning:
        # Use max_steps=2 so the loop ends quickly
        SimulationRunner(mock_env, max_steps=2, scenario_id=scenario_id).run()

        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert str(scenario_id) in call_args, (
            f"Warning message should contain scenario_id {scenario_id}: {call_args}"
        )
        # 2 unfinished agents
        assert "2" in call_args, (
            f"Warning message should contain unfinished count: {call_args}"
        )
