"""Formal schema definitions for flatland-sim output data.

This module is the single source of truth for the structure of
``scenarios.pkl`` files produced by :class:`flatland_sim.pipeline.Pipeline`.
Downstream consumers (e.g. a JEPA training repo) can depend on these
definitions without importing the full flatland-sim package — just copy
or vendor this file.

Versioned via :data:`SCHEMA_VERSION`.  Bump the *major* component when a
breaking change is made to field names, types, or semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import TypedDict

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Action & transition enums
# ---------------------------------------------------------------------------


class Action(IntEnum):
    """Flatland action codes used in ``action_planned``."""
    NOOP = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    STOP = 4


class TransitionLabel(IntEnum):
    """Outcome labels assigned to each agent-step after the environment
    has been stepped.  See :meth:`SimulationRunner._transition_label`."""
    INTENTIONAL_STOP = 0
    FREE_FORWARD = 1
    FREE_LEFT = 2
    FREE_RIGHT = 3
    BLOCKED = 4


class AgentStatus:
    """String constants for the ``status`` field in agent records.
    These mirror ``flatland.envs.agent_utils.TrainState`` names."""
    WAITING = "WAITING"
    READY_TO_DEPART = "READY_TO_DEPART"
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    DONE = "DONE"


# ---------------------------------------------------------------------------
# Typed dicts for timestep / agent records
# ---------------------------------------------------------------------------


class AgentRecord(TypedDict):
    """Schema for a single agent entry inside a timestep."""
    id: int
    """Agent index (0-based)."""
    position: tuple[int, int] | None
    """(row, col) on the grid, or None if not yet departed."""
    direction: int
    """Facing direction: 0=N, 1=E, 2=S, 3=W."""
    status: str
    """One of :class:`AgentStatus` constants."""
    action_planned: int
    """Action chosen this step — see :class:`Action`."""
    next_position: tuple[int, int] | None
    """Position after the environment step."""
    transition_label: int | None
    """Outcome of the action — see :class:`TransitionLabel`."""


class TimestepRecord(TypedDict):
    """Schema for one simulation timestep."""
    step: int
    """Zero-based timestep index."""
    agents: list[AgentRecord]
    """Per-agent records for this timestep."""


# ---------------------------------------------------------------------------
# Scenario snapshot field descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Describes one field of :class:`ScenarioSnapshot`."""
    name: str
    dtype: str
    shape: str
    description: str


SCENARIO_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("scenario_id", "int", "scalar", "Unique scenario index."),
    FieldSpec("config", "dict", "scalar",
              "Sampled generation parameters (num_trains, grid_width, …)."),
    FieldSpec("env_width", "int", "scalar", "Rail grid width in cells."),
    FieldSpec("env_height", "int", "scalar", "Rail grid height in cells."),
    FieldSpec("num_agents", "int", "scalar", "Number of trains in the scenario."),
    FieldSpec("distance_map", "np.ndarray[int]", "(num_agents, H, W, 4)",
              "Shortest-path distance from every cell/direction to each agent's target."),
    FieldSpec("rail_grid", "np.ndarray[int]", "(H, W)",
              "Encoded rail topology — non-zero cells contain track."),
    FieldSpec("rail_transitions", "dict[(row,col,dir), tuple[int,...]]", "variable",
              "Per-cell, per-direction transition bitvectors."),
    FieldSpec("agent_targets", "list[tuple[int,int]]", "(num_agents,)",
              "Target (row, col) for each agent."),
    FieldSpec("agent_initial_positions", "list[tuple[int,int]]", "(num_agents,)",
              "Starting (row, col) for each agent."),
    FieldSpec("timesteps", "list[TimestepRecord]", "(T,)",
              "Ordered simulation timesteps trimmed to the common-presence window."),
)
