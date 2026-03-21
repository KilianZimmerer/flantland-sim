from __future__ import annotations

import tempfile
from pathlib import Path

import dill
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flatland_sim.scenario_store import ScenarioStore
from tests.strategies import snapshot_strategy


# Feature: scenarios-analysis, Property 3: ScenarioStore load/save round-trip
@given(snapshots=st.lists(snapshot_strategy(), min_size=1, max_size=10))
@settings(max_examples=100, deadline=None)
def test_store_round_trip(snapshots):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        tmp_path = Path(tmp)
        pkl = tmp_path / "scenarios.pkl"
        with open(pkl, "wb") as f:
            dill.dump(snapshots, f)

        store = ScenarioStore.load(pkl)
        assert len(store) == len(snapshots)
        assert store.ids == sorted(s.scenario_id for s in snapshots)

        pkl2 = tmp_path / "scenarios2.pkl"
        store.save(pkl2)
        store2 = ScenarioStore.load(pkl2)
        assert len(store2) == len(store)
        assert store2.ids == store.ids


# Feature: scenarios-analysis, Property 4: Missing path raises FileNotFoundError
@given(path_str=st.text(min_size=1, alphabet=st.characters(blacklist_characters="/\\\x00")))
@settings(max_examples=100)
def test_store_missing_path(path_str):
    # Build a path guaranteed not to exist
    nonexistent = Path("/nonexistent_hypothesis_dir") / path_str
    with pytest.raises(FileNotFoundError) as exc_info:
        ScenarioStore.load(nonexistent)
    assert str(nonexistent) in str(exc_info.value)


# Feature: scenarios-analysis, Property 5: ScenarioStore len invariant
@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=20))
@settings(max_examples=100)
def test_store_len(snapshots):
    store = ScenarioStore(snapshots)
    assert len(store) == len(snapshots)


# Feature: scenarios-analysis, Property 6: filter correctness
@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=10))
@settings(max_examples=100)
def test_store_filter(snapshots):
    store = ScenarioStore(snapshots)

    # Every result satisfies the predicate
    predicate = lambda s: s.num_agents >= 1
    filtered = store.filter(predicate)
    for s in filtered.snapshots:
        assert predicate(s)

    # Empty result when predicate is always False
    empty = store.filter(lambda s: False)
    assert len(empty) == 0


# Feature: scenarios-analysis, Property 7: filter_by correctness
@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=10))
@settings(max_examples=100)
def test_store_filter_by(snapshots):
    store = ScenarioStore(snapshots)
    if not snapshots:
        return

    # Pick a config key/value from the first snapshot
    first = snapshots[0]
    if not first.config:
        return
    k, v = next(iter(first.config.items()))

    result = store.filter_by(**{k: v})
    for s in result.snapshots:
        assert s.config[k] == v


# Feature: scenarios-analysis, Property 8: get correctness
@given(snapshots=st.lists(snapshot_strategy(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_store_get(snapshots):
    store = ScenarioStore(snapshots)

    # get returns correct snapshot for present ids
    for snap in snapshots:
        result = store.get(snap.scenario_id)
        assert result.scenario_id == snap.scenario_id

    # get raises KeyError for absent ids
    absent_id = max(s.scenario_id for s in snapshots) + 1
    with pytest.raises(KeyError):
        store.get(absent_id)


# Feature: scenarios-analysis, Property 9: ids sorted invariant
@given(snapshots=st.lists(snapshot_strategy(), min_size=0, max_size=20))
@settings(max_examples=100)
def test_store_ids_sorted(snapshots):
    store = ScenarioStore(snapshots)
    ids = store.ids
    # ids must be sorted
    assert ids == sorted(ids)
    # ids must contain exactly the scenario_id of every snapshot (including duplicates)
    assert ids == sorted(s.scenario_id for s in snapshots)


def test_store_invalid_file(tmp_path):
    bad_file = tmp_path / "bad.pkl"
    with open(bad_file, "wb") as f:
        dill.dump({"not": "a list"}, f)

    with pytest.raises(ValueError):
        ScenarioStore.load(bad_file)
