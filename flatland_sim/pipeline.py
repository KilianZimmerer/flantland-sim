import logging
import yaml
import dill
from pathlib import Path

from flatland.utils.rendertools import RenderTool
from flatland_sim.sampler import RandomConfigSampler
from flatland_sim.generator import ScenarioGenerator
from flatland_sim.runner import SimulationRunner
from flatland_sim.snapshot import ScenarioSnapshot


class Pipeline:
    def __init__(self, config: dict, preview: bool = False):
        self.config = config
        self.num_scenarios = config["num_scenarios"]
        self.max_steps = config["max_steps"]
        self.output_path = config["output_path"]
        self.preview_output_dir = config["preview_output_dir"]
        self.sampler = RandomConfigSampler(config)
        self.preview = preview

    def run(self) -> list[ScenarioSnapshot]:
        snapshots = []
        skip_count = 0

        for i in range(self.num_scenarios):
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

                renderer = RenderTool(env, gl="PIL") if self.preview else None
                runner = SimulationRunner(env, self.max_steps, scenario_id=i, renderer=renderer)
                timesteps = runner.run()

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
                print(f"Scenario {len(snapshots) + 1}/{self.num_scenarios} complete")
                snapshots.append(snapshot)
                preview_frames = runner._frames if self.preview else None

            except Exception as e:
                logging.warning(f"Scenario {i} failed: {e}")
                skip_count += 1
                continue

            # Save preview outside try/except so a render failure doesn't skip the snapshot
            if self.preview and preview_frames is not None:
                try:
                    self._save_preview(preview_frames, i)
                except Exception as e:
                    logging.warning(f"Scenario {i} preview failed: {e}")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            dill.dump(snapshots, f)

        print(f"Saved {len(snapshots)} scenarios ({skip_count} skipped) to {self.output_path}")
        return snapshots

    def _save_preview(self, frames: list, scenario_id: int) -> None:
        Path(self.preview_output_dir).mkdir(parents=True, exist_ok=True)
        gif_path = Path(self.preview_output_dir) / f"scenario_{scenario_id}.gif"
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
        print(f"Saved preview to {self.preview_output_dir}/scenario_{scenario_id}.gif")


def generate_scenarios(config: dict | str, preview: bool = False) -> list[ScenarioSnapshot]:
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return Pipeline(config, preview).run()
