from __future__ import annotations

from pathlib import Path
from typing import Callable

import dill

from flatland_sim.snapshot import ScenarioSnapshot


class ScenarioStore:
    def __init__(self, snapshots: list[ScenarioSnapshot]) -> None:
        self._snapshots = list(snapshots)

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "rb") as f:
            data = dill.load(f)
        if not isinstance(data, list) or not all(
            isinstance(s, ScenarioSnapshot) for s in data
        ):
            raise ValueError(
                f"Expected list[ScenarioSnapshot], got {type(data).__name__}"
            )
        return cls(data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            dill.dump(self._snapshots, f)

    @property
    def snapshots(self) -> list[ScenarioSnapshot]:
        return list(self._snapshots)

    @property
    def ids(self) -> list[int]:
        return sorted(s.scenario_id for s in self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def filter(self, predicate: Callable[[ScenarioSnapshot], bool]) -> "ScenarioStore":
        return ScenarioStore([s for s in self._snapshots if predicate(s)])

    def filter_by(self, **kwargs) -> "ScenarioStore":
        def predicate(s: ScenarioSnapshot) -> bool:
            return all(s.config.get(k) == v for k, v in kwargs.items())

        return self.filter(predicate)

    def get(self, scenario_id: int) -> ScenarioSnapshot:
        for s in self._snapshots:
            if s.scenario_id == scenario_id:
                return s
        raise KeyError(scenario_id)
