from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatland_sim.generator import ScenarioGenerator


# Feature: flatland-sim, Property 5: ScenarioGenerator produces env matching params
@settings(max_examples=100)
@given(
    grid_width=st.integers(min_value=30, max_value=40),
    grid_height=st.integers(min_value=30, max_value=40),
    num_trains=st.integers(min_value=1, max_value=3),
    num_cities=st.integers(min_value=2, max_value=3),
    max_rails_between_cities=st.integers(min_value=1, max_value=2),
    max_rail_pairs_in_city=st.integers(min_value=1, max_value=2),
)
def test_p5_env_dimensions_match_params(
    grid_width,
    grid_height,
    num_trains,
    num_cities,
    max_rails_between_cities,
    max_rail_pairs_in_city,
):
    params = {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "num_trains": num_trains,
        "num_cities": num_cities,
        "max_rails_between_cities": max_rails_between_cities,
        "max_rail_pairs_in_city": max_rail_pairs_in_city,
    }
    env, _ = ScenarioGenerator(params).build()
    assert env.width == params["grid_width"]
    assert env.height == params["grid_height"]
    assert env.get_num_agents() == params["num_trains"]


def test_build_propagates_reset_exception():
    params = {
        "grid_width": 15,
        "grid_height": 15,
        "num_trains": 1,
        "num_cities": 2,
        "max_rails_between_cities": 1,
        "max_rail_pairs_in_city": 1,
    }
    with patch("flatland_sim.generator.RailEnv") as MockRailEnv:
        mock_env = MockRailEnv.return_value
        mock_env.reset.side_effect = RuntimeError("reset failed")
        with pytest.raises(RuntimeError, match="reset failed"):
            ScenarioGenerator(params).build()
