import logging
import yaml
import dill
from pathlib import Path

from flatland_sim.sampler import RandomConfigSampler
from flatland_sim.generator import ScenarioGenerator
from flatland_sim.runner import SimulationRunner
from flatland_sim.snapshot import ScenarioSnapshot


_MAX_RETRIES = 10


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.num_scenarios = config["num_scenarios"]
        self.max_steps = config["max_steps"]
        simulation_dir = config["simulation_dir"]
        self.pkl_path = Path(simulation_dir) / "scenarios.pkl"
        self.sampler = RandomConfigSampler(config)

    def run(self) -> list[ScenarioSnapshot]:
        snapshots = []
        skip_count = 0

        for i in range(self.num_scenarios):
            snapshot = None
            for attempt in range(_MAX_RETRIES):
                try:
                    params = self.sampler.sample()
                    env, obs = ScenarioGenerator(params).build()

                    # Extract static fields from env after reset
                    rail_grid = env.rail.grid.astype(int)
                    distance_map = env.distance_map.get()

                    rail_transitions = {}
                    for row in range(env.height):
                        for col in range(env.width):
                            if env.rail.grid[row, col] != 0:
                                for direction in range(4):
                                    transitions = env.rail.get_transitions(((row, col), direction))
                                    rail_transitions[(row, col, direction)] = transitions

                    agent_targets = [tuple(agent.target) for agent in env.agents]
                    agent_initial_positions = [tuple(agent.initial_position) for agent in env.agents]

                    runner = SimulationRunner(
                        env, self.max_steps, scenario_id=i,
                        action_weights=self.config.get("action_weights"),
                    )
                    timesteps = runner.run()

                    if not timesteps:
                        logging.info(
                            f"Scenario {i}, attempt {attempt + 1}: no valid common-presence window, retrying"
                        )
                        continue

                    snapshot = ScenarioSnapshot(
                        scenario_id=i,
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
                    break

                except Exception as e:
                    logging.warning(f"Scenario {i}, attempt {attempt + 1} failed: {e}")
                    continue

            if snapshot is not None:
                print(f"Scenario {len(snapshots) + 1}/{self.num_scenarios} complete")
                snapshots.append(snapshot)
            else:
                logging.warning(f"Scenario {i}: skipped after {_MAX_RETRIES} attempts")
                skip_count += 1

        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pkl_path, "wb") as f:
            dill.dump(snapshots, f)

        print(f"Saved {len(snapshots)} scenarios ({skip_count} skipped) to {self.pkl_path}")
        return snapshots


def generate_scenarios(config: dict | str) -> list[ScenarioSnapshot]:
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return Pipeline(config).run()
