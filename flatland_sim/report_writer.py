import json
from pathlib import Path

from flatland_sim.analyzer import AnalysisReport


class ReportWriter:
    def write(self, report: AnalysisReport, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"per_scenario": report.per_scenario, "aggregate": report.aggregate}, f, indent=2)
