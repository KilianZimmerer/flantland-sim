from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from flatland_sim.analyzer import Analyzer
from flatland_sim.scenario_store import ScenarioStore
from tests.strategies import snapshot_strategy


# Feature: scenarios-analysis, Property 10: completion_rate formula
@given(snap=snapshot_strategy())
@settings(max_examples=100)
def test_analyzer_completion_rate(snap):
    """Validates: Requirements 3.1"""
    analyzer = Analyzer(ScenarioStore([snap]), max_steps=1000)
    metrics = analyzer._analyse_scenario(snap)

    num_agents = snap.num_agents
    agents_with_end = set()
    for timestep in snap.timesteps:
        for agent in timestep["agents"]:
            if agent["transition_label"] == 6:
                agents_with_end.add(agent["id"])

    expected = len(agents_with_end) / num_agents if num_agents > 0 else 0.0
    assert metrics.completion_rate == expected
    assert 0.0 <= metrics.completion_rate <= 1.0


# Feature: scenarios-analysis, Property 11: Transition label counts sum invariant
@given(snap=snapshot_strategy())
@settings(max_examples=100)
def test_analyzer_label_counts_sum(snap):
    """Validates: Requirements 3.2, 3.4"""
    analyzer = Analyzer(ScenarioStore([snap]), max_steps=1000)
    metrics = analyzer._analyse_scenario(snap)

    total = (
        metrics.waiting_count
        + metrics.intentional_stop_count
        + metrics.free_forward_count
        + metrics.free_left_count
        + metrics.free_right_count
        + metrics.blocked_count
        + metrics.end_count
        + metrics.done_count
    )
    assert total == metrics.total_steps * snap.num_agents


# Feature: scenarios-analysis, Property 12: deadlock_detected formula
@given(snap=snapshot_strategy(), max_steps=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_analyzer_deadlock_formula(snap, max_steps):
    """Validates: Requirements 3.3"""
    analyzer = Analyzer(ScenarioStore([snap]), max_steps=max_steps)
    metrics = analyzer._analyse_scenario(snap)

    expected = metrics.total_steps < max_steps and metrics.completion_rate < 1.0
    assert metrics.deadlock_detected == expected


# Feature: scenarios-analysis, Property 13: avg_blocked_ratio formula
@given(snap=snapshot_strategy())
@settings(max_examples=100)
def test_analyzer_blocked_ratio(snap):
    """Validates: Requirements 3.5"""
    analyzer = Analyzer(ScenarioStore([snap]), max_steps=1000)
    metrics = analyzer._analyse_scenario(snap)

    denom = metrics.total_steps * snap.num_agents
    if denom > 0:
        assert metrics.avg_blocked_ratio == metrics.blocked_count / denom
    else:
        assert metrics.avg_blocked_ratio == 0.0


# Feature: scenarios-analysis, Property 14: Aggregate metrics consistency
@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=10))
@settings(max_examples=100)
def test_analyzer_aggregate_consistency(snapshots):
    """Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5"""
    store = ScenarioStore(snapshots)
    analyzer = Analyzer(store, max_steps=1000)
    report = analyzer.analyse()

    if not snapshots:
        assert report.per_scenario == []
        agg = report.aggregate
        assert agg["mean_completion_rate"] == 0.0
        assert agg["deadlock_count"] == 0
        assert agg["mean_total_steps"] == 0.0
        assert agg["mean_avg_blocked_ratio"] == 0.0
    else:
        per = report.per_scenario
        agg = report.aggregate
        n = len(per)

        assert agg["mean_completion_rate"] == sum(m["completion_rate"] for m in per) / n
        assert agg["deadlock_count"] == sum(1 for m in per if m["deadlock_detected"])
        assert agg["mean_total_steps"] == sum(m["total_steps"] for m in per) / n
        assert agg["mean_avg_blocked_ratio"] == sum(m["avg_blocked_ratio"] for m in per) / n


def test_analyzer_empty_store():
    """Validates: Requirements 4.5"""
    store = ScenarioStore([])
    analyzer = Analyzer(store, max_steps=500)
    report = analyzer.analyse()

    assert report.per_scenario == []
    assert report.aggregate["mean_completion_rate"] == 0.0
    assert report.aggregate["deadlock_count"] == 0
    assert report.aggregate["mean_total_steps"] == 0.0
    assert report.aggregate["mean_avg_blocked_ratio"] == 0.0
