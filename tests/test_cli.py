"""Unit tests for run.py CLI argument parsing and main() behaviour."""
import importlib
from pathlib import Path
from unittest.mock import patch

import yaml


def _make_config(tmp_path: Path) -> Path:
    """Write a minimal config.yaml into tmp_path and return its path."""
    cfg = {
        "seed": 42,
        "num_scenarios": 1,
        "max_steps": 50,
        "simulation_dir": str(tmp_path),
        "randomization": {
            "num_trains": {"min": 2, "max": 4},
            "grid_width": {"min": 20, "max": 30},
            "grid_height": {"min": 20, "max": 30},
            "num_cities": {"min": 2, "max": 3},
            "max_rails_between_cities": {"min": 1, "max": 2},
            "max_rail_pairs_in_city": {"min": 1, "max": 2},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


def _reload_run():
    import run
    importlib.reload(run)
    return run


def test_main_calls_pipeline_run(tmp_path):
    """main() calls Pipeline.run() when invoked."""
    config_path = _make_config(tmp_path)

    with patch("sys.argv", ["run.py", "--config", str(config_path)]):
        with patch("flatland_sim.pipeline.Pipeline.run") as mock_run:
            run = _reload_run()
            run.main()
            mock_run.assert_called_once()
