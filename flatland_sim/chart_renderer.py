import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from flatland_sim.snapshot import ScenarioSnapshot


class ChartRenderer:
    def render(self, snap: ScenarioSnapshot, metrics: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Bar chart of transition-label counts
        labels = ["waiting", "intentional_stop", "free_forward", "free_turn", "blocked", "end"]
        counts = [
            metrics["waiting_count"],
            metrics["intentional_stop_count"],
            metrics["free_forward_count"],
            metrics["free_turn_count"],
            metrics["blocked_count"],
            metrics["end_count"],
        ]
        ax1.bar(labels, counts)
        ax1.set_title("Transition Label Counts")
        ax1.set_xlabel("Label")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=30)

        # Subplot 2: Line chart of per-timestep blocked-agent count
        blocked_per_step = [
            sum(1 for agent in step["agents"] if agent["transition_label"] == 4)
            for step in snap.timesteps
        ]
        ax2.plot(blocked_per_step)
        ax2.set_title("Blocked Agents per Timestep")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Blocked Count")

        # Figure title with scenario metadata
        fig.suptitle(
            f"Scenario {metrics['scenario_id']} | "
            f"Agents: {snap.num_agents} | "
            f"Completion: {metrics['completion_rate']:.2%} | "
            f"Deadlock: {metrics['deadlock_detected']}"
        )

        fig.tight_layout()
        fig.savefig(path)
        plt.close("all")
