from hypothesis import given, settings
from hypothesis import strategies as st

from flatland_sim.sampler import RandomConfigSampler

REQUIRED_KEYS = {
    "num_trains",
    "grid_width",
    "grid_height",
    "num_cities",
    "max_rails_between_cities",
    "max_rail_pairs_in_city",
}


def make_config(seed: int, ranges: dict) -> dict:
    """Build a valid config dict for RandomConfigSampler."""
    return {
        "seed": seed,
        "randomization": {
            key: {"min": lo, "max": hi}
            for key, (lo, hi) in ranges.items()
        },
    }


def fixed_ranges(min_val: int = 1, max_val: int = 10) -> dict:
    """Return a fixed set of randomization ranges covering all required keys."""
    return {key: (min_val, max_val) for key in REQUIRED_KEYS}


# ---------------------------------------------------------------------------
# P2 – Sampler reproducibility
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    seed=st.integers(),
    n=st.integers(min_value=1, max_value=20),
)
def test_sampler_reproducibility(seed: int, n: int):
    # Feature: flatland-sim, Property 2: Sampler reproducibility
    config = make_config(seed, fixed_ranges())

    sampler_a = RandomConfigSampler(config)
    sampler_b = RandomConfigSampler(config)

    results_a = [sampler_a.sample() for _ in range(n)]
    results_b = [sampler_b.sample() for _ in range(n)]

    assert results_a == results_b


# ---------------------------------------------------------------------------
# P3 – Sampled values within declared range
# ---------------------------------------------------------------------------

range_strategy = st.integers(min_value=0, max_value=100)


@settings(max_examples=100)
@given(
    seed=st.integers(),
    mins=st.lists(range_strategy, min_size=len(REQUIRED_KEYS), max_size=len(REQUIRED_KEYS)),
    widths=st.lists(st.integers(min_value=0, max_value=50), min_size=len(REQUIRED_KEYS), max_size=len(REQUIRED_KEYS)),
    n=st.integers(min_value=1, max_value=10),
)
def test_sampled_values_within_range(seed: int, mins: list, widths: list, n: int):
    # Feature: flatland-sim, Property 3: Sampled values within declared range
    keys = sorted(REQUIRED_KEYS)
    ranges = {key: (lo, lo + w) for key, lo, w in zip(keys, mins, widths)}
    config = make_config(seed, ranges)

    sampler = RandomConfigSampler(config)
    for _ in range(n):
        result = sampler.sample()
        for key in keys:
            lo, hi = ranges[key]
            assert lo <= result[key] <= hi, (
                f"Key '{key}': value {result[key]} not in [{lo}, {hi}]"
            )


# ---------------------------------------------------------------------------
# P4 – Sampled dict contains all required keys
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    seed=st.integers(),
    n=st.integers(min_value=1, max_value=10),
)
def test_sampled_dict_has_required_keys(seed: int, n: int):
    # Feature: flatland-sim, Property 4: Sampled dict contains all required keys
    config = make_config(seed, fixed_ranges())

    sampler = RandomConfigSampler(config)
    for _ in range(n):
        result = sampler.sample()
        assert set(result.keys()) == REQUIRED_KEYS, (
            f"Expected keys {REQUIRED_KEYS}, got {set(result.keys())}"
        )
