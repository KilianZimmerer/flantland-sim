# flatland-sim

A standalone Python package for generating serialized [Flatland](https://flatland.aicrowd.com/) railway simulation datasets. It produces a `scenarios.pkl` file containing randomized simulation snapshots, usable for ML training, research, or analysis.

## Installation

```bash
uv sync
```

## Usage

### CLI

```bash
uv run python run.py --config config.yaml
```

With preview GIFs:

```bash
uv run python run.py --config config.yaml --preview
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to config YAML |
| `--preview` | off | Save a GIF per scenario to `preview_output_dir` |

### Python API

```python
from flatland_sim import generate_scenarios

# From a config dict
snapshots = generate_scenarios(config={"seed": 42, ...})

# From a YAML path
snapshots = generate_scenarios(config="config.yaml")
```

Returns a `list[ScenarioSnapshot]`.

## Configuration

```yaml
seed: 42
num_scenarios: 100
max_steps: 500
output_path: "output/scenarios.pkl"
preview_output_dir: "output/previews"

randomization:
  num_trains:    { min: 2, max: 8 }
  grid_width:    { min: 20, max: 50 }
  grid_height:   { min: 20, max: 50 }
  num_cities:    { min: 2, max: 5 }
  max_rails_between_cities: { min: 1, max: 3 }
  max_rail_pairs_in_city:   { min: 1, max: 3 }
```

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
| `timesteps` | `list[dict]` | Per-step agent states |

Load the output file with:

```python
import dill

with open("output/scenarios.pkl", "rb") as f:
    snapshots = dill.load(f)
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
│   └── snapshot.py
└── tests/
    ├── test_sampler.py
    ├── test_generator.py
    ├── test_runner.py
    ├── test_snapshot.py
    ├── test_pipeline.py
    └── test_cli.py
```

## Tests

```bash
uv run pytest
```
