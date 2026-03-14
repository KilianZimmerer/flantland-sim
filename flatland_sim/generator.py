from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import SparseRailGen


class ScenarioGenerator:
    def __init__(self, params: dict):
        self.params = params

    def build(self) -> tuple[RailEnv, dict]:
        p = self.params
        env = RailEnv(
            width=p["grid_width"],
            height=p["grid_height"],
            number_of_agents=p["num_trains"],
            rail_generator=SparseRailGen(
                max_num_cities=p["num_cities"],
                max_rails_between_cities=p["max_rails_between_cities"],
                max_rail_pairs_in_city=p["max_rail_pairs_in_city"],
            ),
        )
        obs, _ = env.reset()
        return env, obs
