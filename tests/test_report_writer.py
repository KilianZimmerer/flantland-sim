"""Tests for ReportWriter — Property 15 and unit tests."""
from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flatland_sim.analyzer import Analyzer, AnalysisReport
from flatland_sim.report_writer import ReportWriter
from flatland_sim.scenario_store import ScenarioStore
from tests.strategies import snapshot_strategy


# ---------------------------------------------------------------------------
# Sub-task 5.1 — Property 15: AnalysisReport JSON round-trip
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=5))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_report_json_round_trip(snapshots, tmp_path):
    """**Validates: Requirements 1.5**"""
    report = Analyzer(ScenarioStore(snapshots), max_steps=100).analyse()
    writer = ReportWriter()
    out = tmp_path / "report.json"
    writer.write(report, out)

    with out.open() as f:
        data = json.load(f)

    assert data["per_scenario"] == report.per_scenario
    assert data["aggregate"] == report.aggregate


# ---------------------------------------------------------------------------
# Sub-task 5.2 — Unit tests: dir auto-creation and indentation
# ---------------------------------------------------------------------------

def test_report_writer_creates_dirs(tmp_path):
    report = Analyzer(ScenarioStore([]), max_steps=10).analyse()
    nested = tmp_path / "a" / "b" / "report.json"
    ReportWriter().write(report, nested)
    assert nested.exists()


def test_report_writer_indentation(tmp_path):
    report = Analyzer(ScenarioStore([]), max_steps=10).analyse()
    out = tmp_path / "report.json"
    ReportWriter().write(report, out)
    content = out.read_text()
    assert "  " in content
