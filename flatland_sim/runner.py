import random
import logging

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState

# transition_label constants
INTENTIONAL_STOP = 0
FREE_FORWARD = 1
FREE_LEFT = 2
FREE_RIGHT = 3
BLOCKED = 4

_DEADLOCK_WINDOW = 5


_ACTION_NAME_TO_ID = {"left": 1, "forward": 2, "right": 3, "stop": 4}


class SimulationRunner:
    def __init__(self, env: RailEnv, max_steps: int, scenario_id: int = 0,
                 action_weights: dict[str, float] | None = None):
        self.env = env
        self.max_steps = max_steps
        self.scenario_id = scenario_id
        # Map action *id* → weight.  None means uniform random.
        self._action_weights: dict[int, float] | None = None
        if action_weights:
            self._action_weights = {
                _ACTION_NAME_TO_ID[name]: w
                for name, w in action_weights.items()
                if name in _ACTION_NAME_TO_ID
            }

    def _get_valid_actions(self, agent_handle: int) -> list[int]:
        """Return the list of Flatland actions (1-4) that lead to a valid
        transition from the agent's current cell and orientation.

        Actions: 1=left, 2=forward, 3=right, 4=stop.
        """
        agent = self.env.agents[agent_handle]
        if agent.position is None:
            return [2]  # not on grid yet – only forward/depart makes sense

        row, col = agent.position
        direction = agent.direction
        transition_map = self.env.rail.get_full_transitions(row, col)

        # Flatland direction encoding: 0=N, 1=E, 2=S, 3=W
        # Action mapping relative to current direction:
        #   left  (1) → (direction - 1) % 4
        #   fwd   (2) → direction
        #   right (3) → (direction + 1) % 4
        action_to_new_dir = {
            1: (direction - 1) % 4,
            2: direction,
            3: (direction + 1) % 4,
        }

        valid = []
        for action, new_dir in action_to_new_dir.items():
            # transition_map is a 16-bit int; bits [new_dir] for each source dir
            if (transition_map >> ((3 - direction) * 4 + (3 - new_dir))) & 1:
                valid.append(action)

        # Stop (4) is always a legal action for a moving/stopped agent
        valid.append(4)
        return valid

    def _get_action(self, agent_handle: int, state: TrainState) -> int:
        if state in (TrainState.MOVING, TrainState.STOPPED):
            valid = self._get_valid_actions(agent_handle)
            if self._action_weights is not None:
                weights = [self._action_weights.get(a, 1.0) for a in valid]
                return random.choices(valid, weights=weights, k=1)[0]
            return random.choice(valid)
        elif state in (TrainState.WAITING, TrainState.READY_TO_DEPART):
            return 2
        else:  # TrainState.DONE
            return 0

    @staticmethod
    def _transition_label(action: int, prev_pos, next_pos) -> int:
        """Compute transition_label for one agent based on the physical outcome."""
        if action == 4 and prev_pos == next_pos:
            return INTENTIONAL_STOP
        if action == 2 and prev_pos != next_pos:
            return FREE_FORWARD
        if action == 1 and prev_pos != next_pos:
            return FREE_LEFT
        if action == 3 and prev_pos != next_pos:
            return FREE_RIGHT
        if action in (1, 2, 3) and prev_pos == next_pos:
            return BLOCKED
        # fallback
        return INTENTIONAL_STOP

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
            actions = {i: self._get_action(i, agent.state) for i, agent in enumerate(agents)}

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
                        "action_planned": actions[i],
                        "next_position": None,
                        "transition_label": None,
                    }
                    for i, agent in enumerate(agents)
                ],
            }
            timesteps.append(record)

            # Log timestep info
            statuses = [agent.state.name for agent in agents]
            status_summary = ", ".join(
                f"A{i}:{s}" for i, s in enumerate(statuses)
            )
            logging.info(
                f"Scenario {self.scenario_id}, step {t}: actions={actions}, statuses=[{status_summary}]"
            )

            # Step the environment
            self.env.step(actions)

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
                logging.info(
                    f"Scenario {self.scenario_id}: global deadlock detected at step {t}, stopping early"
                )
                break

            # Early stopping: any agent reached destination
            if any(agent.state == TrainState.DONE for agent in self.env.agents):
                logging.info(
                    f"Scenario {self.scenario_id}: agent reached destination at step {t}, stopping"
                )
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
                    action=agent_record["action_planned"],
                    prev_pos=agent_record["position"],
                    next_pos=nxt_agents[j]["position"],
                )

        # Drop the final timestep — next_position and transition_label are unknowable
        # without taking another step, so every label in the returned list is ground truth.
        if timesteps:
            timesteps.pop()

        # Trim to the time range where ALL trains are simultaneously present
        # on the grid (all departed, none yet DONE).
        timesteps = self._trim_to_common_presence(timesteps)

        return timesteps

    @staticmethod
    def _trim_to_common_presence(timesteps: list[dict]) -> list[dict]:
        """Return the sub-sequence of *timesteps* where every agent is on the
        grid and none has reached DONE status yet.

        The first valid step is the one where the last agent departs (position
        becomes non-None).  The last valid step is the one just before the
        first agent reaches DONE.  If no such window exists the full list is
        returned unchanged so downstream code still has data to work with.
        """
        if not timesteps:
            return timesteps

        num_agents = len(timesteps[0]["agents"])

        # Find the first step where ALL agents have a non-None position
        start = None
        for idx, ts in enumerate(timesteps):
            if all(a["position"] is not None for a in ts["agents"]):
                start = idx
                break

        # Find the first step where ANY agent has status DONE
        end = len(timesteps)
        for idx, ts in enumerate(timesteps):
            if any(a["status"] == "DONE" for a in ts["agents"]):
                end = idx
                break

        if start is not None and start < end:
            return timesteps[start:end]

        # No valid common-presence window — return as-is
        return timesteps
