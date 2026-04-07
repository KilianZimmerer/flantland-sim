# flatland-sim

A standalone Python package for generating serialized [Flatland](https://flatland.aicrowd.com/) railway simulation datasets. It produces a `scenarios.pkl` file containing randomized simulation snapshots, usable for ML training, research, or analysis.

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

### Scenario Navigator (GUI)

Launch the interactive GUI to browse and replay scenarios from an existing `.pkl` file:

```bash
uv run python -m flatland_sim.navigator output/scenarios.pkl
```

The navigator renders the rail grid on a canvas and lets you step through timesteps, play/pause the simulation, and inspect per-agent state including action→transition labels.

### Python API

```python
from flatland_sim import generate_scenarios

snapshots = generate_scenarios("config.yaml")
```

Returns a `list[ScenarioSnapshot]`.

### Use as a dependency

Install `flatland-sim` in another project:

```bash
# from a git remote
uv add "flatland-sim @ git+https://github.com/youruser/flatland-sim.git"

# or from a local checkout
uv add --editable ../flatland-sim
```

Then generate scenarios programmatically:

```python
from flatland_sim import generate_scenarios, load_scenarios

# generate
snapshots = generate_scenarios("path/to/config.yaml")

# load an existing file
snapshots = load_scenarios("output/tst/scenarios.pkl")
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

Each agent entry in `timesteps` includes:

| Field | Description |
|---|---|
| `id` | Agent index |
| `position` | `(row, col)` before the step |
| `direction` | Facing direction (0=N, 1=E, 2=S, 3=W) |
| `status` | Flatland TrainState name |
| `action_planned` | Action to be executed (0=noop, 1=left, 2=forward, 3=right, 4=stop) |
| `next_position` | `(row, col)` after the step |
| `transition_label` | Outcome of the action |

Transition label values:

| Value | Label | Meaning |
|---|---|---|
| 0 | INTENTIONAL_STOP | Agent chose to stop |
| 1 | FREE_FORWARD | Moved forward unobstructed |
| 2 | FREE_LEFT | Turned left unobstructed |
| 3 | FREE_RIGHT | Turned right unobstructed |
| 4 | BLOCKED | Tried to move but was blocked |

Load the output file directly with:

```python
from flatland_sim import load_scenarios

snapshots = load_scenarios("output/scenarios.pkl")
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
│   ├── schema.py
│   ├── snapshot.py
│   ├── scenario_store.py
│   └── navigator.py
└── tests/
    ├── strategies.py
    ├── test_sampler.py
    ├── test_generator.py
    ├── test_runner.py
    ├── test_snapshot.py
    ├── test_pipeline.py
    ├── test_scenario_store.py
    ├── test_navigator.py
    └── test_cli.py
```

## Tests

```bash
uv run pytest
```
