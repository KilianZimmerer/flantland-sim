"""Unit tests for run.py CLI argument parsing and main() behaviour."""
import importlib
import sys
import tempfile
import dill
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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


def _make_pkl(tmp_path: Path) -> Path:
    """Create an empty scenarios.pkl in tmp_path."""
    pkl_path = tmp_path / "scenarios.pkl"
    with open(pkl_path, "wb") as f:
        dill.dump([], f)
    return pkl_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_run():
    import run
    importlib.reload(run)
    return run


# ---------------------------------------------------------------------------
# Basic Pipeline invocation
# ---------------------------------------------------------------------------

def test_main_calls_pipeline_run(tmp_path):
    """main() calls Pipeline.run() when no --analyze-only flag is given."""
    config_path = _make_config(tmp_path)

    with patch("sys.argv", ["run.py", "--config", str(config_path)]):
        with patch("flatland_sim.pipeline.Pipeline.run") as mock_run:
            run = _reload_run()
            run.main()
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# --analyze flag
# ---------------------------------------------------------------------------

def test_cli_analyze_flag(tmp_path):
    """--analyze causes Pipeline.run, Analyzer.analyse, ReportWriter.write to be called."""
    config_path = _make_config(tmp_path)
    _make_pkl(tmp_path)

    fake_store = MagicMock()
    fake_store.snapshots = []  # empty — no ChartRenderer calls needed

    fake_report = MagicMock()
    fake_report.per_scenario = []

    with patch("sys.argv", ["run.py", "--config", str(config_path), "--analyze"]):
        with patch("flatland_sim.pipeline.Pipeline.run") as mock_pipeline_run, \
             patch("flatland_sim.scenario_store.ScenarioStore.load", return_value=fake_store) as mock_load, \
             patch("flatland_sim.analyzer.Analyzer.analyse", return_value=fake_report) as mock_analyse, \
             patch("flatland_sim.report_writer.ReportWriter.write") as mock_write, \
             patch("flatland_sim.chart_renderer.ChartRenderer.render") as mock_render:

            run = _reload_run()
            run.main()

            mock_pipeline_run.assert_called_once()
            mock_load.assert_called_once()
            mock_analyse.assert_called_once()
            mock_write.assert_called_once()
            mock_render.assert_not_called()  # no snapshots


# ---------------------------------------------------------------------------
# --analyze-only flag
# ---------------------------------------------------------------------------

def test_cli_analyze_only_flag(tmp_path):
    """--analyze-only skips Pipeline.run but still runs analysis."""
    config_path = _make_config(tmp_path)
    _make_pkl(tmp_path)

    fake_store = MagicMock()
    fake_store.snapshots = []

    fake_report = MagicMock()
    fake_report.per_scenario = []

    with patch("sys.argv", ["run.py", "--config", str(config_path), "--analyze-only"]):
        with patch("flatland_sim.pipeline.Pipeline.run") as mock_pipeline_run, \
             patch("flatland_sim.scenario_store.ScenarioStore.load", return_value=fake_store), \
             patch("flatland_sim.analyzer.Analyzer.analyse", return_value=fake_report), \
             patch("flatland_sim.report_writer.ReportWriter.write"), \
             patch("flatland_sim.chart_renderer.ChartRenderer.render"):

            run = _reload_run()
            run.main()

            mock_pipeline_run.assert_not_called()


# ---------------------------------------------------------------------------
# --analyze-only with missing pkl
# ---------------------------------------------------------------------------

def test_cli_analyze_only_missing_pkl(tmp_path):
    """--analyze-only exits non-zero when scenarios.pkl does not exist."""
    config_path = _make_config(tmp_path)
    # deliberately do NOT create scenarios.pkl

    with patch("sys.argv", ["run.py", "--config", str(config_path), "--analyze-only"]):
        with patch("flatland_sim.pipeline.Pipeline.run"):
            run = _reload_run()
            with pytest.raises(SystemExit) as exc_info:
                run.main()
            assert exc_info.value.code != 0
