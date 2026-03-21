import argparse
import sys
import dataclasses
from pathlib import Path
import yaml

from flatland_sim.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run flatland-sim scenario generation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--preview", action="store_true", help="Enable preview mode.")
    parser.add_argument("--analyze", action="store_true", help="Run analysis after generation.")
    parser.add_argument("--analyze-only", action="store_true", dest="analyze_only",
                        help="Run analysis only against existing scenarios.pkl.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    simulation_dir = Path(config["simulation_dir"])
    max_steps = config["max_steps"]

    if not args.analyze_only:
        Pipeline(config, args.preview).run()

    if args.analyze or args.analyze_only:
        from flatland_sim.scenario_store import ScenarioStore
        from flatland_sim.analyzer import Analyzer
        from flatland_sim.report_writer import ReportWriter
        from flatland_sim.chart_renderer import ChartRenderer

        pkl_path = simulation_dir / "scenarios.pkl"
        if not pkl_path.exists():
            print(f"Error: {pkl_path} does not exist. Run without --analyze-only first.")
            sys.exit(1)

        store = ScenarioStore.load(pkl_path)
        report = Analyzer(store, max_steps).analyse()
        ReportWriter().write(report, simulation_dir / "analysis_report.json")
        for snap in store.snapshots:
            metrics = next(m for m in report.per_scenario if m["scenario_id"] == snap.scenario_id)
            ChartRenderer().render(
                snap, metrics,
                simulation_dir / "analysis" / f"scenario_{snap.scenario_id}.png"
            )


if __name__ == "__main__":
    main()
