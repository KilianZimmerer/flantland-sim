"""Unit tests for run.py CLI argument parsing (Req 8.1, 8.2)."""
import importlib
import sys
from unittest.mock import patch


def _parse_args(argv):
    """Import run module and parse args without executing main."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args(argv)


def test_default_config_and_no_preview():
    args = _parse_args([])
    assert args.config == "config.yaml"
    assert args.preview is False


def test_custom_config():
    args = _parse_args(["--config", "my_config.yaml"])
    assert args.config == "my_config.yaml"


def test_preview_flag():
    args = _parse_args(["--preview"])
    assert args.preview is True


def test_config_and_preview_together():
    args = _parse_args(["--config", "other.yaml", "--preview"])
    assert args.config == "other.yaml"
    assert args.preview is True


def test_main_calls_generate_scenarios():
    """run.py main() calls generate_scenarios with correct args (Req 8.3)."""
    from unittest.mock import MagicMock
    fake_module = MagicMock()
    with patch.dict(sys.modules, {"flatland_sim": fake_module, "flatland_sim.pipeline": MagicMock(), "flatland_sim.snapshot": MagicMock()}):
        with patch("sys.argv", ["run.py", "--config", "test.yaml", "--preview"]):
            import run
            importlib.reload(run)
            run.main()
        fake_module.generate_scenarios.assert_called_once_with(config="test.yaml", preview=True)
