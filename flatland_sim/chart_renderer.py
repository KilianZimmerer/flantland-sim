import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from flatland_sim.snapshot import ScenarioSnapshot


class ChartRenderer:
    def render(self, snap: ScenarioSnapshot, metrics: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pre-compute per-timestep label counts
        label_names = ["waiting", "intentional_stop", "free_forward", "free_turn", "blocked", "end", "done"]
        per_step_counts = {name: [] for name in label_names}
        for step in snap.timesteps:
            step_counts = {name: 0 for name in label_names}
            for agent in step["agents"]:
                lbl = agent["transition_label"]
                step_counts[label_names[lbl]] += 1
            for name in label_names:
                per_step_counts[name].append(step_counts[name])

        # Subplot 1: Bar chart of transition-label counts
        counts = [metrics[f"{name}_count"] for name in label_names]
        ax1.bar(label_names, counts)
        ax1.set_title("Transition Label Counts")
        ax1.set_xlabel("Label")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=30)

        # Subplot 2: Stacked area of all transition labels per timestep
        steps = list(range(len(snap.timesteps)))
        ax2.stackplot(
            steps,
            per_step_counts["free_forward"],
            per_step_counts["free_turn"],
            per_step_counts["waiting"],
            per_step_counts["intentional_stop"],
            per_step_counts["blocked"],
            per_step_counts["end"],
            per_step_counts["done"],
            labels=["free_forward", "free_turn", "waiting", "intentional_stop", "blocked", "end", "done"],
            alpha=0.8,
        )
        ax2.set_title("Agent State Composition")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Agent Count")
        ax2.legend(loc="upper right", fontsize="small")

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
