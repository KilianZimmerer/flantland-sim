import random
import logging

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState
from PIL import Image


class SimulationRunner:
    def __init__(self, env: RailEnv, max_steps: int, scenario_id: int = 0, renderer=None):
        self.env = env
        self.max_steps = max_steps
        self.scenario_id = scenario_id
        self.renderer = renderer
        self._frames = []

    def _get_action(self, state: TrainState) -> int:
        if state in (TrainState.MOVING, TrainState.STOPPED):
            return random.randint(1, 4)
        elif state in (TrainState.WAITING, TrainState.READY_TO_DEPART):
            return 2
        else:  # TrainState.DONE and any other state
            return 0

    def run(self) -> list[dict]:
        timesteps = []

        for t in range(self.max_steps):
            # Guard: don't step a finished episode
            if self.env.dones.get("__all__", False):
                break

            agents = self.env.agents
            actions = {i: self._get_action(agent.state) for i, agent in enumerate(agents)}

            # Record current timestep (next_position starts as None)
            record = {
                "step": t,
                "agents": [
                    {
                        "id": i,
                        "position": agent.position,
                        "direction": agent.direction,
                        "status": agent.state.name,
                        "action_taken": actions[i],
                        "next_position": None,
                    }
                    for i, agent in enumerate(agents)
                ],
            }
            timesteps.append(record)

            # Step the environment
            self.env.step(actions)

            # Collect frame if renderer is provided
            if self.renderer is not None:
                frame = self.renderer.render_env(show=False, return_image=True)
                # render_env returns a numpy array; convert to PIL Image for GIF saving
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                self._frames.append(frame)

            # Backfill next_position of the just-recorded timestep
            for agent_record in record["agents"]:
                i = agent_record["id"]
                agent_record["next_position"] = self.env.agents[i].position

            # Early stopping: all agents done
            if all(agent.state == TrainState.DONE for agent in self.env.agents):
                break

        # Check for unfinished agents after the loop
        unfinished = sum(1 for agent in self.env.agents if agent.state != TrainState.DONE)
        if unfinished > 0:
            logging.warning(
                f"Scenario {self.scenario_id}: max_steps reached with {unfinished} unfinished agents"
            )

        # Final timestep: next_position must be None (already set by default, but
        # backfill above overwrites it — so we reset the last timestep's next_positions)
        if timesteps:
            for agent_record in timesteps[-1]["agents"]:
                agent_record["next_position"] = None

        return timesteps
