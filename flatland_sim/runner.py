import random
import logging

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState
from PIL import Image

# transition_label constants
WAITING = 0
INTENTIONAL_STOP = 1
FREE_FORWARD = 2
FREE_TURN = 3
BLOCKED = 4
END = 5

_DEADLOCK_WINDOW = 5


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
        else:  # TrainState.DONE
            return 0

    @staticmethod
    def _transition_label(action: int, prev_pos, next_pos, cur_status: str, next_status: str, is_terminal: bool) -> int:
        """Compute transition_label for one agent based on the physical outcome."""
        # Already DONE in a previous step — agent is inactive, don't count again
        if cur_status == "DONE":
            return WAITING
        # Transitioning into DONE this step (arrival)
        if is_terminal or next_status == "DONE":
            return END
        if prev_pos is None:
            return WAITING
        if action == 4 and prev_pos == next_pos:
            return INTENTIONAL_STOP
        if action == 2 and prev_pos != next_pos:
            return FREE_FORWARD
        if action in (1, 3) and prev_pos != next_pos:
            return FREE_TURN
        if action in (1, 2, 3) and prev_pos == next_pos:
            return BLOCKED
        # fallback (e.g. action==0 / DONE)
        return WAITING

    def run(self) -> list[dict]:
        timesteps = []
        # Tracks how many consecutive steps each active agent hasn't moved.
        # Keyed by agent index; only agents not in DONE state are considered.
        no_move_streak: list[int] = [0] * len(self.env.agents)
        deadlock = False

        for t in range(self.max_steps):
            # Guard: don't step a finished episode
            if self.env.dones.get("__all__", False):
                break

            agents = self.env.agents
            actions = {i: self._get_action(agent.state) for i, agent in enumerate(agents)}

            # Snapshot positions before stepping
            pre_positions = [agent.position for agent in agents]

            # Record current timestep (next_position and transition_label filled after step)
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
                        "transition_label": None,
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
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                self._frames.append(frame)

            # Backfill next_position
            for agent_record in record["agents"]:
                i = agent_record["id"]
                agent_record["next_position"] = self.env.agents[i].position

            # Update no-move streaks (only for non-DONE agents)
            active_count = 0
            for i, agent in enumerate(self.env.agents):
                if agent.state == TrainState.DONE:
                    no_move_streak[i] = 0
                    continue
                active_count += 1
                if agent.position == pre_positions[i] and pre_positions[i] is not None:
                    no_move_streak[i] += 1
                else:
                    no_move_streak[i] = 0

            # Check for global deadlock: all active agents stuck for _DEADLOCK_WINDOW steps
            if active_count > 0 and all(
                no_move_streak[i] >= _DEADLOCK_WINDOW
                for i, agent in enumerate(self.env.agents)
                if agent.state != TrainState.DONE
            ):
                deadlock = True
                logging.warning(
                    f"Scenario {self.scenario_id}: global deadlock detected at step {t}, stopping early"
                )
                break

            # Early stopping: all agents done
            if all(agent.state == TrainState.DONE for agent in self.env.agents):
                break

        # Determine terminal condition for transition_label backfill
        unfinished = sum(1 for agent in self.env.agents if agent.state != TrainState.DONE)
        is_max_steps = len(timesteps) >= self.max_steps and not deadlock

        if unfinished > 0 and is_max_steps:
            logging.warning(
                f"Scenario {self.scenario_id}: max_steps reached with {unfinished} unfinished agents"
            )

        # Backfill transition_label for all but the final timestep
        for idx in range(len(timesteps) - 1):
            cur_agents = timesteps[idx]["agents"]
            nxt_agents = timesteps[idx + 1]["agents"]
            for j, agent_record in enumerate(cur_agents):
                agent_record["transition_label"] = self._transition_label(
                    action=agent_record["action_taken"],
                    prev_pos=agent_record["position"],
                    next_pos=nxt_agents[j]["position"],
                    cur_status=agent_record["status"],
                    next_status=nxt_agents[j]["status"],
                    is_terminal=False,
                )

        # Drop leading timesteps where no agent has appeared on the grid yet
        while timesteps and all(a["position"] is None for a in timesteps[0]["agents"]):
            timesteps.pop(0)

        # Drop the final timestep — next_position and transition_label are unknowable
        # without taking another step, so every label in the returned list is ground truth.
        if timesteps:
            timesteps.pop()

        return timesteps
