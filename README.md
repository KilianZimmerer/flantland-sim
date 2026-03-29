# flatland-sim

A standalone Python package for generating and analysing serialized [Flatland](https://flatland.aicrowd.com/) railway simulation datasets. It produces a `scenarios.pkl` file containing randomized simulation snapshots, usable for ML training, research, or analysis. An optional analysis pipeline computes per-scenario metrics, writes a JSON report, and renders per-scenario PNG charts.

## Installation

```bash
uv sync
```

## Usage

### CLI

Generate scenarios:

```bash
uv run python run.py --config config.yaml
```

Generate scenarios and run analysis:

```bash
uv run python run.py --config config.yaml --analyze
```

Run analysis only against an existing `scenarios.pkl`:

```bash
uv run python run.py --config config.yaml --analyze-only
```

With preview GIFs:

```bash
uv run python run.py --config config.yaml --preview
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to config YAML |
| `--preview` | off | Save a GIF per scenario under `{simulation_dir}/previews/` |
| `--analyze` | off | Run analysis pipeline after generation |
| `--analyze-only` | off | Run analysis only (skip generation) |

### Scenario Navigator (GUI)

Launch the interactive GUI to browse and replay scenarios from an existing `.pkl` file:

```bash
uv run python -m flatland_sim.navigator output/scenarios.pkl
```

The navigator renders the rail grid on a canvas and lets you step through timesteps, play/pause the simulation, and inspect per-agent state including transition labels.

### Python API

```python
from flatland_sim.pipeline import Pipeline
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

snapshots = Pipeline(config).run()
```

Returns a `list[ScenarioSnapshot]`.

To run analysis programmatically:

```python
from flatland_sim.scenario_store import ScenarioStore
from flatland_sim.analyzer import Analyzer
from flatland_sim.report_writer import ReportWriter
from flatland_sim.chart_renderer import ChartRenderer
import dataclasses
from pathlib import Path

store = ScenarioStore.load("output/scenarios.pkl")
report = Analyzer(store, max_steps=500).analyse()

ReportWriter().write(report, "output/analysis_report.json")

for snap in store.snapshots:
    metrics = next(m for m in report.per_scenario if m["scenario_id"] == snap.scenario_id)
    ChartRenderer().render(snap, metrics, f"output/analysis/scenario_{snap.scenario_id}.png")
```

## Configuration

```yaml
seed: 42
num_scenarios: 100
max_steps: 500
simulation_dir: "output"   # all outputs written here

randomization:
  num_trains:    { min: 2, max: 8 }
  grid_width:    { min: 20, max: 50 }
  grid_height:   { min: 20, max: 50 }
  num_cities:    { min: 2, max: 5 }
  max_rails_between_cities: { min: 1, max: 3 }
  max_rail_pairs_in_city:   { min: 1, max: 3 }
```

All outputs are written under `simulation_dir`:

| Path | Description |
|---|---|
| `{simulation_dir}/scenarios.pkl` | Serialized scenario snapshots |
| `{simulation_dir}/previews/scenario_{id}.gif` | Preview GIFs (with `--preview`) |
| `{simulation_dir}/analysis_report.json` | Aggregate + per-scenario metrics (with `--analyze`) |
| `{simulation_dir}/analysis/scenario_{id}.png` | Per-scenario charts (with `--analyze`) |

## ScenarioSnapshot

Each snapshot is a dataclass with:

| Field | Type | Description |
|---|---|---|
| `scenario_id` | `int` | Index of the scenario |
| `config` | `dict` | Sampled params used to generate it |
| `env_width` / `env_height` | `int` | Grid dimensions |
| `num_agents` | `int` | Number of trains |
| `rail_grid` | `ndarray (H, W)` | Transition bitmask per cell |
| `distance_map` | `ndarray (N, H, W, 4)` | Shortest-path distances per agent |
| `rail_transitions` | `dict` | `(row, col, dir)` → valid outgoing directions |
| `agent_targets` | `list[tuple]` | Target positions |
| `agent_initial_positions` | `list[tuple]` | Initial positions |
| `timesteps` | `list[dict]` | Per-step agent states including `transition_label` |

Each agent entry in `timesteps` includes a `transition_label` (int):

| Value | Label | Meaning |
|---|---|---|
| 0 | WAITING | Agent not yet departed |
| 1 | INTENTIONAL_STOP | Agent chose to stop |
| 2 | FREE_FORWARD | Moved forward unobstructed |
| 3 | FREE_TURN | Turned unobstructed |
| 4 | BLOCKED | Tried to move but was blocked |
| 5 | END | Agent reached its target |

Load the output file directly with:

```python
import dill

with open("output/scenarios.pkl", "rb") as f:
    snapshots = dill.load(f)
```

## Analysis Output

`analysis_report.json` structure:

```json
{
  "per_scenario": [
    {
      "scenario_id": 0,
      "completion_rate": 0.75,
      "total_steps": 312,
      "deadlock_detected": false,
      "waiting_count": 120,
      "intentional_stop_count": 30,
      "free_forward_count": 800,
      "free_turn_count": 50,
      "blocked_count": 200,
      "end_count": 48,
      "avg_blocked_ratio": 0.08
    }
  ],
  "aggregate": {
    "mean_completion_rate": 0.75,
    "deadlock_count": 1,
    "mean_total_steps": 312.0,
    "mean_avg_blocked_ratio": 0.08
  }
}
```

## Project Structure

```
flatland-sim/
├── pyproject.toml
├── config.yaml
├── run.py
├── flatland_sim/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── sampler.py
│   ├── generator.py
│   ├── runner.py
│   ├── snapshot.py
│   ├── scenario_store.py
│   ├── analyzer.py
│   ├── report_writer.py
│   └── chart_renderer.py
└── tests/
    ├── strategies.py
    ├── test_sampler.py
    ├── test_generator.py
    ├── test_runner.py
    ├── test_snapshot.py
    ├── test_pipeline.py
    ├── test_scenario_store.py
    ├── test_analyzer.py
    ├── test_report_writer.py
    ├── test_chart_renderer.py
    └── test_cli.py
```

## Tests

```bash
uv run pytest
```
