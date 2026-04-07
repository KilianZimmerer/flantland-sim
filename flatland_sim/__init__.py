from pathlib import Path

from flatland_sim.pipeline import generate_scenarios
from flatland_sim.scenario_store import ScenarioStore
from flatland_sim.snapshot import ScenarioSnapshot
from flatland_sim import schema


def load_scenarios(path: str | Path) -> list[ScenarioSnapshot]:
    """Load a ``scenarios.pkl`` file and return its snapshots.

    Thin wrapper around :meth:`ScenarioStore.load` for convenience.
    """
    return ScenarioStore.load(path).snapshots


__all__ = ["generate_scenarios", "load_scenarios", "ScenarioSnapshot", "schema"]
