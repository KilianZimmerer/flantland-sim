import dataclasses
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hypothesis import given, settings, HealthCheck

from flatland_sim.analyzer import Analyzer
from flatland_sim.chart_renderer import ChartRenderer
from flatland_sim.scenario_store import ScenarioStore
from tests.strategies import snapshot_strategy


# Feature: scenarios-analysis, Property 16: ChartRenderer produces PNG file
@given(snap=snapshot_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_chart_renderer_produces_png(snap, tmp_path):
    """Validates: Requirements 6.1, 6.5"""
    metrics = dataclasses.asdict(
        Analyzer(ScenarioStore([snap]), max_steps=1000)._analyse_scenario(snap)
    )
    output_path = tmp_path / "nested" / "subdir" / f"scenario_{snap.scenario_id}.png"

    ChartRenderer().render(snap, metrics, output_path)

    assert output_path.exists()


# Feature: scenarios-analysis, Property 17: ChartRenderer closes all figures
@given(snap=snapshot_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_chart_renderer_closes_figures(snap, tmp_path):
    """Validates: Requirements 6.6"""
    metrics = dataclasses.asdict(
        Analyzer(ScenarioStore([snap]), max_steps=1000)._analyse_scenario(snap)
    )
    output_path = tmp_path / f"scenario_{snap.scenario_id}.png"

    before = len(plt.get_fignums())
    ChartRenderer().render(snap, metrics, output_path)
    after = len(plt.get_fignums())

    assert after == before
