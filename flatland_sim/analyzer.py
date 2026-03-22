from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from flatland_sim.scenario_store import ScenarioStore
from flatland_sim.snapshot import ScenarioSnapshot


@dataclass
class ScenarioMetrics:
    scenario_id: int
    completion_rate: float        # [0.0, 1.0]
    total_steps: int
    deadlock_detected: bool
    waiting_count: int
    intentional_stop_count: int
    free_forward_count: int
    free_turn_count: int
    blocked_count: int
    end_count: int
    done_count: int
    avg_blocked_ratio: float      # blocked_count / (total_steps * num_agents) or 0.0


@dataclass
class AggregateMetrics:
    mean_completion_rate: float
    deadlock_count: int
    mean_total_steps: float
    mean_avg_blocked_ratio: float


@dataclass
class AnalysisReport:
    per_scenario: list[dict]      # ScenarioMetrics serialised to dicts
    aggregate: dict               # AggregateMetrics serialised to dict


class Analyzer:
    def __init__(self, store: ScenarioStore, max_steps: int) -> None:
        self._store = store
        self._max_steps = max_steps

    def analyse(self) -> AnalysisReport:
        metrics = [self._analyse_scenario(snap) for snap in self._store.snapshots]
        aggregate = self._aggregate(metrics)
        return AnalysisReport(
            per_scenario=[dataclasses.asdict(m) for m in metrics],
            aggregate=dataclasses.asdict(aggregate),
        )

    def _analyse_scenario(self, snap: ScenarioSnapshot) -> ScenarioMetrics:
        total_steps = len(snap.timesteps)
        num_agents = snap.num_agents

        # Per-label counts
        waiting_count = 0
        intentional_stop_count = 0
        free_forward_count = 0
        free_turn_count = 0
        blocked_count = 0
        end_count = 0
        done_count = 0

        # Track which agents have seen at least one END label
        agents_with_end: set[int] = set()

        for timestep in snap.timesteps:
            for agent in timestep["agents"]:
                label = agent["transition_label"]
                if label == 0:
                    waiting_count += 1
                elif label == 1:
                    intentional_stop_count += 1
                elif label == 2:
                    free_forward_count += 1
                elif label == 3:
                    free_turn_count += 1
                elif label == 4:
                    blocked_count += 1
                elif label == 5:
                    end_count += 1
                    agents_with_end.add(agent["id"])
                elif label == 6:
                    done_count += 1

        completion_rate = len(agents_with_end) / num_agents if num_agents > 0 else 0.0
        deadlock_detected = total_steps < self._max_steps and completion_rate < 1.0

        denom = total_steps * num_agents
        avg_blocked_ratio = blocked_count / denom if denom > 0 else 0.0

        return ScenarioMetrics(
            scenario_id=snap.scenario_id,
            completion_rate=completion_rate,
            total_steps=total_steps,
            deadlock_detected=deadlock_detected,
            waiting_count=waiting_count,
            intentional_stop_count=intentional_stop_count,
            free_forward_count=free_forward_count,
            free_turn_count=free_turn_count,
            blocked_count=blocked_count,
            end_count=end_count,
            done_count=done_count,
            avg_blocked_ratio=avg_blocked_ratio,
        )

    def _aggregate(self, metrics: list[ScenarioMetrics]) -> AggregateMetrics:
        if not metrics:
            return AggregateMetrics(
                mean_completion_rate=0.0,
                deadlock_count=0,
                mean_total_steps=0.0,
                mean_avg_blocked_ratio=0.0,
            )
        n = len(metrics)
        mean_completion_rate = sum(m.completion_rate for m in metrics) / n
        deadlock_count = sum(1 for m in metrics if m.deadlock_detected)
        mean_total_steps = sum(m.total_steps for m in metrics) / n
        mean_avg_blocked_ratio = sum(m.avg_blocked_ratio for m in metrics) / n
        return AggregateMetrics(
            mean_completion_rate=mean_completion_rate,
            deadlock_count=deadlock_count,
            mean_total_steps=mean_total_steps,
            mean_avg_blocked_ratio=mean_avg_blocked_ratio,
        )
