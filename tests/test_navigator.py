"""Tests for PlaybackEngine (Task 1) and transition label formatter (Task 2)."""

import pytest

from flatland_sim.navigator import PlaybackEngine, TRANSITION_LABELS, format_transition_label


# ── __init__ ──────────────────────────────────────────────────────────────

class TestPlaybackEngineInit:
    def test_defaults(self):
        engine = PlaybackEngine(total_steps=10)
        assert engine.current_index == 0
        assert engine.total_steps == 10
        assert engine.is_playing is False
        assert engine.speed_ms == 200

    def test_speed_clamped_low(self):
        engine = PlaybackEngine(total_steps=5, speed_ms=10)
        assert engine.speed_ms == 50

    def test_speed_clamped_high(self):
        engine = PlaybackEngine(total_steps=5, speed_ms=5000)
        assert engine.speed_ms == 2000

    def test_speed_within_range(self):
        engine = PlaybackEngine(total_steps=5, speed_ms=500)
        assert engine.speed_ms == 500

    def test_zero_total_steps(self):
        engine = PlaybackEngine(total_steps=0)
        assert engine.total_steps == 0
        assert engine.current_index == 0


# ── set_speed ─────────────────────────────────────────────────────────────

class TestSetSpeed:
    def test_clamp_low(self):
        engine = PlaybackEngine(total_steps=5)
        engine.set_speed(0)
        assert engine.speed_ms == 50

    def test_clamp_high(self):
        engine = PlaybackEngine(total_steps=5)
        engine.set_speed(9999)
        assert engine.speed_ms == 2000

    def test_within_range(self):
        engine = PlaybackEngine(total_steps=5)
        engine.set_speed(1000)
        assert engine.speed_ms == 1000


# ── play / pause ──────────────────────────────────────────────────────────

class TestPlayPause:
    def test_play(self):
        engine = PlaybackEngine(total_steps=5)
        engine.play()
        assert engine.is_playing is True

    def test_pause(self):
        engine = PlaybackEngine(total_steps=5)
        engine.play()
        engine.pause()
        assert engine.is_playing is False

    def test_pause_preserves_index(self):
        engine = PlaybackEngine(total_steps=5)
        engine.play()
        engine._current_index = 3
        engine.pause()
        assert engine.current_index == 3
        assert engine.is_playing is False


# ── step_forward ──────────────────────────────────────────────────────────

class TestStepForward:
    def test_advances(self):
        engine = PlaybackEngine(total_steps=5)
        assert engine.step_forward() is True
        assert engine.current_index == 1

    def test_at_end(self):
        engine = PlaybackEngine(total_steps=5)
        engine._current_index = 4
        assert engine.step_forward() is False
        assert engine.current_index == 4

    def test_pauses_if_playing(self):
        engine = PlaybackEngine(total_steps=5)
        engine.play()
        engine.step_forward()
        assert engine.is_playing is False

    def test_zero_steps(self):
        engine = PlaybackEngine(total_steps=0)
        assert engine.step_forward() is False
        assert engine.current_index == 0

    def test_single_step(self):
        engine = PlaybackEngine(total_steps=1)
        assert engine.step_forward() is False
        assert engine.current_index == 0


# ── step_backward ─────────────────────────────────────────────────────────

class TestStepBackward:
    def test_decreases(self):
        engine = PlaybackEngine(total_steps=5)
        engine._current_index = 3
        assert engine.step_backward() is True
        assert engine.current_index == 2

    def test_at_start(self):
        engine = PlaybackEngine(total_steps=5)
        assert engine.step_backward() is False
        assert engine.current_index == 0

    def test_pauses_if_playing(self):
        engine = PlaybackEngine(total_steps=5)
        engine._current_index = 2
        engine.play()
        engine.step_backward()
        assert engine.is_playing is False

    def test_zero_steps(self):
        engine = PlaybackEngine(total_steps=0)
        assert engine.step_backward() is False


# ── jump_to ───────────────────────────────────────────────────────────────

class TestJumpTo:
    def test_jump_within_range(self):
        engine = PlaybackEngine(total_steps=10)
        engine.jump_to(5)
        assert engine.current_index == 5

    def test_clamp_high(self):
        engine = PlaybackEngine(total_steps=10)
        engine.jump_to(100)
        assert engine.current_index == 9

    def test_clamp_low(self):
        engine = PlaybackEngine(total_steps=10)
        engine.jump_to(-5)
        assert engine.current_index == 0

    def test_pauses_if_playing(self):
        engine = PlaybackEngine(total_steps=10)
        engine.play()
        engine.jump_to(5)
        assert engine.is_playing is False

    def test_zero_steps(self):
        engine = PlaybackEngine(total_steps=0)
        engine.jump_to(5)
        assert engine.current_index == 0


# ── tick ──────────────────────────────────────────────────────────────────

class TestTick:
    def test_not_playing(self):
        engine = PlaybackEngine(total_steps=5)
        assert engine.tick() is False
        assert engine.current_index == 0

    def test_advances_when_playing(self):
        engine = PlaybackEngine(total_steps=5)
        engine.play()
        assert engine.tick() is True
        assert engine.current_index == 1
        assert engine.is_playing is True

    def test_auto_pauses_at_end(self):
        engine = PlaybackEngine(total_steps=5)
        engine._current_index = 4
        engine.play()
        assert engine.tick() is False
        assert engine.current_index == 4
        assert engine.is_playing is False

    def test_zero_steps_playing(self):
        engine = PlaybackEngine(total_steps=0)
        engine._playing = True
        assert engine.tick() is False
        assert engine.is_playing is False

    def test_advances_to_last_then_pauses(self):
        engine = PlaybackEngine(total_steps=3)
        engine._current_index = 1
        engine.play()
        # tick from index 1 -> 2 (last), should advance
        assert engine.tick() is True
        assert engine.current_index == 2
        assert engine.is_playing is True
        # tick at index 2 (end), should auto-pause
        assert engine.tick() is False
        assert engine.current_index == 2
        assert engine.is_playing is False


# ── reset ─────────────────────────────────────────────────────────────────

class TestReset:
    def test_resets_state(self):
        engine = PlaybackEngine(total_steps=10)
        engine._current_index = 7
        engine.play()
        engine.reset(20)
        assert engine.current_index == 0
        assert engine.is_playing is False
        assert engine.total_steps == 20

    def test_reset_to_zero(self):
        engine = PlaybackEngine(total_steps=10)
        engine._current_index = 5
        engine.reset(0)
        assert engine.current_index == 0
        assert engine.total_steps == 0


# ── TRANSITION_LABELS & format_transition_label ───────────────────────────

class TestTransitionLabels:
    def test_all_known_labels(self):
        expected = {
            0: "WAITING",
            1: "INTENTIONAL_STOP",
            2: "FREE_FORWARD",
            3: "FREE_LEFT",
            4: "FREE_RIGHT",
            5: "BLOCKED",
            6: "END",
            7: "DONE",
        }
        assert TRANSITION_LABELS == expected

    def test_format_known_labels(self):
        for label_int, name in TRANSITION_LABELS.items():
            assert format_transition_label(label_int) == name

    def test_format_unknown_positive(self):
        assert format_transition_label(99) == "UNKNOWN(99)"

    def test_format_unknown_negative(self):
        assert format_transition_label(-1) == "UNKNOWN(-1)"

    def test_format_unknown_eight(self):
        assert format_transition_label(8) == "UNKNOWN(8)"


# ── CLI main() ────────────────────────────────────────────────────────────

import sys
import dill
from pathlib import Path
from unittest.mock import patch, MagicMock

from flatland_sim.navigator import main, NavigatorApp
from flatland_sim.snapshot import ScenarioSnapshot


class TestMainCLI:
    def test_no_args_exits_with_usage(self, capsys):
        """No path argument → exit 1 with usage on stderr."""
        with patch("sys.argv", ["navigator"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Usage:" in captured.err

    def test_nonexistent_path_exits_with_error(self, capsys, tmp_path):
        """Non-existent path → exit 1 with error on stderr."""
        bad_path = tmp_path / "does_not_exist.pkl"
        with patch("sys.argv", ["navigator", str(bad_path)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "file not found" in captured.err

    def test_invalid_pickle_exits_with_error(self, capsys, tmp_path):
        """Invalid pickle data → ValueError caught, exit 1 with error on stderr."""
        pkl = tmp_path / "bad.pkl"
        with open(pkl, "wb") as f:
            dill.dump("not a list of snapshots", f)
        with patch("sys.argv", ["navigator", str(pkl)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_valid_path_creates_app_and_runs(self, tmp_path):
        """Valid store file → NavigatorApp created and run() called."""
        pkl = tmp_path / "store.pkl"
        with open(pkl, "wb") as f:
            dill.dump([], f)  # empty list of snapshots is valid

        with patch("sys.argv", ["navigator", str(pkl)]), \
             patch("flatland_sim.navigator.NavigatorApp") as MockApp:
            mock_instance = MagicMock()
            MockApp.return_value = mock_instance
            main()
            MockApp.assert_called_once()
            mock_instance.run.assert_called_once()


# ── NavigatorApp (Task 4) ─────────────────────────────────────────────────

import numpy as np
from flatland_sim.scenario_store import ScenarioStore


def _make_snapshot(scenario_id: int, num_timesteps: int = 3) -> ScenarioSnapshot:
    """Helper to create a minimal ScenarioSnapshot for testing."""
    return ScenarioSnapshot(
        scenario_id=scenario_id,
        config={"n_agents": 1},
        env_width=5,
        env_height=5,
        num_agents=1,
        distance_map=np.zeros((1, 5, 5, 4)),
        rail_grid=np.zeros((5, 5), dtype=np.uint16),
        rail_transitions={},
        agent_targets=[(4, 4)],
        agent_initial_positions=[(0, 0)],
        timesteps=[
            {"step": i, "agents": [{"id": 0, "position": (0, i), "direction": 0,
                                     "status": "ACTIVE", "action_taken": 2,
                                     "next_position": (0, i + 1), "transition_label": 2}]}
            for i in range(num_timesteps)
        ],
    )


class TestNavigatorAppLogic:
    """Test NavigatorApp internal logic by mocking tkinter."""

    def _build_app(self, snapshots: list[ScenarioSnapshot]) -> NavigatorApp:
        """Create a NavigatorApp with mocked tkinter widgets."""
        store = ScenarioStore(snapshots)

        # Patch tkinter at the module level so the import inside __init__ gets mocks
        mock_tk = MagicMock()
        mock_ttk = MagicMock()

        # Make StringVar return a mock that tracks get/set
        string_var_instances = []

        def make_string_var(*args, **kwargs):
            sv = MagicMock()
            sv._value = kwargs.get("value", "")
            sv.get.side_effect = lambda: sv._value
            sv.set.side_effect = lambda v: setattr(sv, "_value", v)
            string_var_instances.append(sv)
            return sv

        mock_tk.StringVar = make_string_var

        # Make Tk() return a mock root
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root

        # Make Canvas return a mock
        mock_canvas = MagicMock()
        mock_tk.Canvas.return_value = mock_canvas

        with patch.dict("sys.modules", {"tkinter": mock_tk, "tkinter.ttk": mock_ttk}):
            # Need to reimport to pick up the mocked tkinter
            import importlib
            import flatland_sim.navigator as nav_module
            importlib.reload(nav_module)
            app = nav_module.NavigatorApp(store)

        # Restore the real module after construction
        import importlib
        import flatland_sim.navigator as nav_module
        importlib.reload(nav_module)

        # after_idle doesn't fire in mocked tkinter, so clear the guard manually
        app._updating_slider = False
        return app

    def test_init_with_scenarios_selects_first(self):
        """With scenarios, the first should be auto-selected."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=3)
        app = self._build_app([snap1, snap2])

        assert app._snapshot is not None
        assert app._snapshot.scenario_id == 10
        assert app._engine.total_steps == 5
        assert app._engine.current_index == 0

    def test_init_empty_store(self):
        """With no scenarios, snapshot should be None and engine has 0 steps."""
        app = self._build_app([])

        assert app._snapshot is None
        assert app._engine.total_steps == 0

    def test_on_scenario_selected_loads_snapshot(self):
        """Selecting a scenario loads its snapshot and resets the engine."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=8)
        app = self._build_app([snap1, snap2])

        # Initially scenario 10 is selected
        assert app._snapshot.scenario_id == 10
        assert app._engine.total_steps == 5

        # Select scenario 20
        app._on_scenario_selected(20)
        assert app._snapshot.scenario_id == 20
        assert app._engine.total_steps == 8
        assert app._engine.current_index == 0
        assert app._engine.is_playing is False

    def test_on_scenario_selected_resets_engine(self):
        """Switching scenarios resets the engine index to 0 and pauses."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=3)
        app = self._build_app([snap1, snap2])

        # Advance engine on first scenario
        app._engine.play()
        app._engine.tick()
        app._engine.tick()
        assert app._engine.current_index == 2

        # Switch scenario
        app._on_scenario_selected(20)
        assert app._engine.current_index == 0
        assert app._engine.is_playing is False
        assert app._engine.total_steps == 3

    def test_on_combo_selected_delegates(self):
        """_on_combo_selected parses the string value and delegates."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=3)
        app = self._build_app([snap1, snap2])

        # Simulate combobox selecting "20"
        app._scenario_var._value = "20"
        app._on_combo_selected()
        assert app._snapshot.scenario_id == 20


# ── Canvas rendering (Task 5) ────────────────────────────────────────────


def _make_snapshot_with_rail(
    scenario_id: int = 1,
    width: int = 5,
    height: int = 5,
    num_agents: int = 2,
    num_timesteps: int = 3,
) -> ScenarioSnapshot:
    """Create a snapshot with non-zero rail cells and multiple agents."""
    grid = np.zeros((height, width), dtype=np.uint16)
    # Put rail on the first row
    for c in range(width):
        grid[0, c] = 1
    return ScenarioSnapshot(
        scenario_id=scenario_id,
        config={"n_agents": num_agents},
        env_width=width,
        env_height=height,
        num_agents=num_agents,
        distance_map=np.zeros((num_agents, height, width, 4)),
        rail_grid=grid,
        rail_transitions={},
        agent_targets=[(height - 1, width - 1), (height - 1, 0)][:num_agents],
        agent_initial_positions=[(0, 0), (0, 1)][:num_agents],
        timesteps=[
            {
                "step": i,
                "agents": [
                    {
                        "id": a,
                        "position": (0, min(a + i, width - 1)),
                        "direction": 0,
                        "status": "ACTIVE",
                        "action_taken": 2,
                        "next_position": (0, min(a + i + 1, width - 1)),
                        "transition_label": 2,
                    }
                    for a in range(num_agents)
                ],
            }
            for i in range(num_timesteps)
        ],
    )


class TestRenderGrid:
    """Tests for _render_grid canvas rendering (Task 5)."""

    def _build_app(self, snapshots: list[ScenarioSnapshot]) -> "NavigatorApp":
        """Create a NavigatorApp with mocked tkinter, reusing the pattern from TestNavigatorAppLogic."""
        store = ScenarioStore(snapshots)
        mock_tk = MagicMock()
        mock_ttk = MagicMock()

        def make_string_var(*args, **kwargs):
            sv = MagicMock()
            sv._value = kwargs.get("value", "")
            sv.get.side_effect = lambda: sv._value
            sv.set.side_effect = lambda v: setattr(sv, "_value", v)
            return sv

        mock_tk.StringVar = make_string_var
        mock_tk.Tk.return_value = MagicMock()
        mock_canvas = MagicMock()
        mock_tk.Canvas.return_value = mock_canvas

        with patch.dict("sys.modules", {"tkinter": mock_tk, "tkinter.ttk": mock_ttk}):
            import importlib
            import flatland_sim.navigator as nav_module
            importlib.reload(nav_module)
            app = nav_module.NavigatorApp(store)

        import importlib
        import flatland_sim.navigator as nav_module
        importlib.reload(nav_module)
        app._updating_slider = False
        return app

    def test_render_clears_canvas(self):
        """_render_grid should call canvas.delete('all') for static layer."""
        snap = _make_snapshot_with_rail()
        app = self._build_app([snap])
        # _render_grid (called via _on_scenario_selected) does delete("all") for static
        # then delete("agent") for dynamic layer
        delete_calls = [str(c) for c in app._canvas.delete.call_args_list]
        assert any("all" in c for c in delete_calls)

    def test_render_draws_rectangles_for_grid(self):
        """Should draw create_rectangle for each cell in the grid."""
        snap = _make_snapshot_with_rail(width=3, height=2, num_agents=1)
        app = self._build_app([snap])
        rect_calls = [c for c in app._canvas.create_rectangle.call_args_list]
        # 3 cols * 2 rows = 6 rectangles
        assert len(rect_calls) == 6

    def test_render_draws_agents_as_ovals(self):
        """Should draw create_oval for each agent with a position."""
        snap = _make_snapshot_with_rail(num_agents=2)
        app = self._build_app([snap])
        oval_calls = app._canvas.create_oval.call_args_list
        assert len(oval_calls) == 2

    def test_render_no_snapshot(self):
        """With no snapshot, _render_grid should clear and return without drawing."""
        app = self._build_app([])
        app._canvas.reset_mock()
        app._render_grid()
        # Static layer clears with "all", dynamic layer clears with "agent"
        delete_calls = [str(c) for c in app._canvas.delete.call_args_list]
        assert any("all" in c for c in delete_calls)
        app._canvas.create_rectangle.assert_not_called()

    def test_render_agent_with_none_position(self):
        """Agents with position=None should not produce an oval."""
        snap = _make_snapshot_with_rail(num_agents=1, num_timesteps=1)
        snap.timesteps[0]["agents"][0]["position"] = None
        app = self._build_app([snap])
        # Re-render after modifying the snapshot
        app._canvas.reset_mock()
        app._render_grid()
        oval_calls = app._canvas.create_oval.call_args_list
        assert len(oval_calls) == 0

    def test_status_shows_agent_count(self):
        """_update_controls should include agent count in status."""
        snap = _make_snapshot_with_rail(num_agents=3)
        app = self._build_app([snap])
        status = app._status_var._value
        assert "Agents: 3" in status

    def test_status_shows_timestep(self):
        """_update_controls should include timestep info in status."""
        snap = _make_snapshot_with_rail(num_timesteps=5)
        app = self._build_app([snap])
        status = app._status_var._value
        assert "Step 1/5" in status

    def test_cell_size_scales_to_canvas(self):
        """Cell size should be min(canvas_w/cols, canvas_h/rows)."""
        # With fallback 600x400 and a 10x20 grid:
        # cell = min(600/10, 400/20) = min(60, 20) = 20
        snap = _make_snapshot_with_rail(width=10, height=20, num_agents=1)
        app = self._build_app([snap])
        # Check that rectangles are drawn with correct cell size
        rect_calls = app._canvas.create_rectangle.call_args_list
        # First cell at (0,0) should be (0, 0, 20, 20)
        first_call_args = rect_calls[0][0]
        assert first_call_args[0] == 0.0  # x0
        assert first_call_args[1] == 0.0  # y0
        assert first_call_args[2] == 20.0  # x1
        assert first_call_args[3] == 20.0  # y1


# ── Playback controls (Task 6) ───────────────────────────────────────────


class TestPlaybackControls:
    """Tests for playback control buttons, speed slider, and tick loop (Task 6)."""

    def _build_app(self, snapshots: list[ScenarioSnapshot]) -> "NavigatorApp":
        """Create a NavigatorApp with mocked tkinter widgets."""
        store = ScenarioStore(snapshots)
        mock_tk = MagicMock()
        mock_ttk = MagicMock()

        def make_string_var(*args, **kwargs):
            sv = MagicMock()
            sv._value = kwargs.get("value", "")
            sv.get.side_effect = lambda: sv._value
            sv.set.side_effect = lambda v: setattr(sv, "_value", v)
            return sv

        mock_tk.StringVar = make_string_var
        mock_tk.Tk.return_value = MagicMock()
        mock_tk.Canvas.return_value = MagicMock()
        # Make Scale return a mock with a set method
        mock_scale = MagicMock()
        mock_tk.Scale.return_value = mock_scale
        mock_tk.HORIZONTAL = "horizontal"

        with patch.dict("sys.modules", {"tkinter": mock_tk, "tkinter.ttk": mock_ttk}):
            import importlib
            import flatland_sim.navigator as nav_module
            importlib.reload(nav_module)
            app = nav_module.NavigatorApp(store)

        import importlib
        import flatland_sim.navigator as nav_module
        importlib.reload(nav_module)
        app._updating_slider = False
        return app

    def test_play_button_exists(self):
        """Play button should be created in the control frame."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_btn_play")

    def test_pause_button_exists(self):
        """Pause button should be created in the control frame."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_btn_pause")

    def test_step_fwd_button_exists(self):
        """Step Forward button should be created."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_btn_step_fwd")

    def test_step_back_button_exists(self):
        """Step Backward button should be created."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_btn_step_back")

    def test_speed_slider_exists(self):
        """Speed slider should be created."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_speed_slider")

    def test_on_play_sets_playing(self):
        """_on_play should set engine to playing state."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_play()
        assert app._engine.is_playing is True

    def test_on_pause_stops_playing(self):
        """_on_pause should stop playback."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.play()
        app._on_pause()
        assert app._engine.is_playing is False

    def test_on_step_fwd_advances_index(self):
        """_on_step_fwd should advance the engine index by 1."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert app._engine.current_index == 0
        app._on_step_fwd()
        assert app._engine.current_index == 1

    def test_on_step_back_decreases_index(self):
        """_on_step_back should decrease the engine index by 1."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.jump_to(3)
        app._on_step_back()
        assert app._engine.current_index == 2

    def test_on_step_fwd_pauses_playback(self):
        """_on_step_fwd should pause if playing."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.play()
        app._on_step_fwd()
        assert app._engine.is_playing is False

    def test_on_step_back_pauses_playback(self):
        """_on_step_back should pause if playing."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.jump_to(3)
        app._engine.play()
        app._on_step_back()
        assert app._engine.is_playing is False

    def test_on_speed_change_updates_engine(self):
        """_on_speed_change should update engine speed."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_speed_change("500")
        assert app._engine.speed_ms == 500

    def test_on_speed_change_handles_float_string(self):
        """_on_speed_change should handle float string values from Scale widget."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_speed_change("750.0")
        assert app._engine.speed_ms == 750

    def test_on_play_schedules_tick(self):
        """_on_play should schedule a tick via root.after."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_play()
        app._root.after.assert_called()

    def test_tick_advances_when_playing(self):
        """_tick should advance the engine when playing."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.play()
        app._tick()
        assert app._engine.current_index == 1

    def test_tick_schedules_next_when_still_playing(self):
        """_tick should schedule another tick if still playing."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.play()
        app._root.after.reset_mock()
        app._tick()
        # Should schedule next tick since engine is still playing
        app._root.after.assert_called()

    def test_tick_does_not_schedule_when_paused(self):
        """_tick should not schedule another tick if engine is paused."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        # Engine is paused by default
        app._root.after.reset_mock()
        app._tick()
        # Should not schedule since engine is not playing
        app._root.after.assert_not_called()

    def test_tick_auto_pauses_at_end(self):
        """_tick at the last step should auto-pause and not schedule next tick."""
        snap = _make_snapshot(10, num_timesteps=3)
        app = self._build_app([snap])
        app._engine.jump_to(2)  # last step
        app._engine.play()
        app._root.after.reset_mock()
        app._tick()
        assert app._engine.is_playing is False
        app._root.after.assert_not_called()


# ── Timeline Slider (Task 7) ─────────────────────────────────────────────


class TestTimelineSlider:
    """Tests for the timeline slider widget and its interactions."""

    def _build_app(self, snapshots: list[ScenarioSnapshot]) -> "NavigatorApp":
        """Create a NavigatorApp with mocked tkinter widgets."""
        store = ScenarioStore(snapshots)
        mock_tk = MagicMock()
        mock_ttk = MagicMock()

        def make_string_var(*args, **kwargs):
            sv = MagicMock()
            sv._value = kwargs.get("value", "")
            sv.get.side_effect = lambda: sv._value
            sv.set.side_effect = lambda v: setattr(sv, "_value", v)
            return sv

        mock_tk.StringVar = make_string_var
        mock_tk.Tk.return_value = MagicMock()
        mock_tk.Canvas.return_value = MagicMock()
        mock_tk.HORIZONTAL = "horizontal"

        # Track Scale instances to distinguish timeline vs speed slider
        scale_instances = []

        def make_scale(*args, **kwargs):
            s = MagicMock()
            s._from = kwargs.get("from_", 0)
            s._to = kwargs.get("to", 0)
            s._last_set = None

            def mock_set(val):
                s._last_set = val

            def mock_configure(**kw):
                if "to" in kw:
                    s._to = kw["to"]
                if "from_" in kw:
                    s._from = kw["from_"]

            s.set = mock_set
            s.configure = mock_configure
            scale_instances.append(s)
            return s

        mock_tk.Scale = make_scale

        with patch.dict("sys.modules", {"tkinter": mock_tk, "tkinter.ttk": mock_ttk}):
            import importlib
            import flatland_sim.navigator as nav_module
            importlib.reload(nav_module)
            app = nav_module.NavigatorApp(store)

        import importlib
        import flatland_sim.navigator as nav_module
        importlib.reload(nav_module)
        app._updating_slider = False
        return app

    def test_timeline_slider_exists(self):
        """Timeline slider should be created as self._timeline_slider."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert hasattr(app, "_timeline_slider")

    def test_timeline_slider_separate_from_speed(self):
        """Timeline slider and speed slider should be distinct objects."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        assert app._timeline_slider is not app._speed_slider

    def test_slider_range_set_on_scenario_select(self):
        """Selecting a scenario should set the slider range to 0..total_steps-1."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        # After init, scenario 10 is auto-selected with 5 timesteps
        assert app._timeline_slider._to == 4  # max(0, 5-1)

    def test_slider_range_updates_on_new_scenario(self):
        """Switching scenarios should update the slider range."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=10)
        app = self._build_app([snap1, snap2])
        assert app._timeline_slider._to == 4  # first scenario

        app._on_scenario_selected(20)
        assert app._timeline_slider._to == 9  # second scenario

    def test_slider_resets_to_zero_on_scenario_change(self):
        """Switching scenarios should reset the slider position to 0."""
        snap1 = _make_snapshot(10, num_timesteps=5)
        snap2 = _make_snapshot(20, num_timesteps=8)
        app = self._build_app([snap1, snap2])

        # Advance on first scenario
        app._engine.jump_to(3)
        app._update_controls()

        # Switch scenario
        app._on_scenario_selected(20)
        assert app._timeline_slider._last_set == 0

    def test_on_slider_change_jumps_engine(self):
        """_on_slider_change should call engine.jump_to with the slider value."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_slider_change("3")
        assert app._engine.current_index == 3

    def test_on_slider_change_handles_float_string(self):
        """_on_slider_change should handle float string values from Scale widget."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._on_slider_change("2.0")
        assert app._engine.current_index == 2

    def test_update_controls_syncs_slider_position(self):
        """_update_controls should set the slider to engine.current_index."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.jump_to(3)
        app._update_controls()
        assert app._timeline_slider._last_set == 3

    def test_updating_slider_flag_prevents_reentry(self):
        """When _updating_slider is True, _on_slider_change should be a no-op."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.jump_to(2)
        app._updating_slider = True
        app._on_slider_change("4")
        # Engine should NOT have jumped because the flag was set
        assert app._engine.current_index == 2

    def test_slider_change_pauses_playback(self):
        """Dragging the slider should pause playback (via engine.jump_to)."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.play()
        assert app._engine.is_playing is True
        app._on_slider_change("2")
        assert app._engine.is_playing is False


# ── Agent Information Panel (Task 8) ─────────────────────────────────────


class TestInfoPanel:
    """Tests for the message log panel (Task 8)."""

    def _build_app(self, snapshots: list[ScenarioSnapshot]) -> "NavigatorApp":
        """Create a NavigatorApp with mocked tkinter, including a trackable Text widget."""
        store = ScenarioStore(snapshots)
        mock_tk = MagicMock()
        mock_ttk = MagicMock()

        def make_string_var(*args, **kwargs):
            sv = MagicMock()
            sv._value = kwargs.get("value", "")
            sv.get.side_effect = lambda: sv._value
            sv.set.side_effect = lambda v: setattr(sv, "_value", v)
            return sv

        mock_tk.StringVar = make_string_var
        mock_tk.Tk.return_value = MagicMock()
        mock_tk.Canvas.return_value = MagicMock()
        mock_tk.HORIZONTAL = "horizontal"

        # Constants that tkinter.Text uses
        mock_tk.DISABLED = "disabled"
        mock_tk.NORMAL = "normal"
        mock_tk.NONE = "none"
        mock_tk.END = "end"
        mock_tk.W = "w"
        mock_tk.TOP = "top"
        mock_tk.BOTTOM = "bottom"
        mock_tk.LEFT = "left"
        mock_tk.RIGHT = "right"
        mock_tk.X = "x"
        mock_tk.Y = "y"
        mock_tk.BOTH = "both"

        # Trackable Text widget mock
        text_widget = MagicMock()
        text_widget._content = ""
        text_widget._state = "disabled"

        def text_configure(**kw):
            if "state" in kw:
                text_widget._state = kw["state"]

        def text_delete(start, end):
            text_widget._content = ""

        def text_insert(pos, content):
            if pos == "end" and text_widget._content:
                text_widget._content += content
            else:
                text_widget._content += content

        def text_get(start, end):
            return text_widget._content

        text_widget.configure = text_configure
        text_widget.delete = text_delete
        text_widget.insert = text_insert
        text_widget.get = text_get
        text_widget.see = MagicMock()
        mock_tk.Text.return_value = text_widget

        with patch.dict("sys.modules", {"tkinter": mock_tk, "tkinter.ttk": mock_ttk}):
            import importlib
            import flatland_sim.navigator as nav_module
            importlib.reload(nav_module)
            app = nav_module.NavigatorApp(store)

        import importlib
        import flatland_sim.navigator as nav_module
        importlib.reload(nav_module)
        app._updating_slider = False
        return app

    def test_info_text_widget_exists(self):
        """Info panel should have a Text widget as self._info_text."""
        snap = _make_snapshot(10, num_timesteps=3)
        app = self._build_app([snap])
        assert hasattr(app, "_info_text")

    def test_log_message_appends_to_panel(self):
        """_log_message should append a formatted line to the text widget."""
        snap = _make_snapshot(10, num_timesteps=3)
        app = self._build_app([snap])
        app._log_message("INFO", "test message")
        content = app._info_text._content
        assert "INFO" in content
        assert "test message" in content

    def test_log_message_includes_step(self):
        """_log_message should include the current step number."""
        snap = _make_snapshot(10, num_timesteps=5)
        app = self._build_app([snap])
        app._engine.jump_to(3)
        app._log_message("WARNING", "something happened")
        content = app._info_text._content
        assert "[Step 3]" in content

    def test_update_info_panel_logs_blocked_agents(self):
        """_update_info_panel should log an INFO when agents are blocked."""
        snap = _make_snapshot_with_rail(num_agents=1, num_timesteps=1)
        snap.timesteps[0]["agents"][0]["transition_label"] = 5  # BLOCKED
        app = self._build_app([snap])
        app._update_info_panel()
        content = app._info_text._content
        assert "INFO" in content
        assert "Blocked" in content

    def test_update_info_panel_logs_ended_agents(self):
        """_update_info_panel should log INFO when agents reach target."""
        snap = _make_snapshot_with_rail(num_agents=1, num_timesteps=1)
        snap.timesteps[0]["agents"][0]["transition_label"] = 6  # END
        app = self._build_app([snap])
        app._update_info_panel()
        content = app._info_text._content
        assert "INFO" in content
        assert "reached target" in content

    def test_update_info_panel_logs_summary_for_normal_step(self):
        """_update_info_panel should log a timestep summary even for normal steps."""
        snap = _make_snapshot(10, num_timesteps=3)
        app = self._build_app([snap])
        app._update_info_panel()
        content = app._info_text._content
        assert "INFO" in content
        assert "Agents:" in content
        assert "FREE_FORWARD" in content

    def test_update_info_panel_no_crash_when_no_snapshot(self):
        """_update_info_panel with no snapshot should be a no-op."""
        app = self._build_app([])
        app._update_info_panel()
        assert app._info_text._content == ""

    def test_info_panel_is_read_only_after_log(self):
        """After _log_message, the Text widget should be set back to DISABLED state."""
        snap = _make_snapshot(10, num_timesteps=3)
        app = self._build_app([snap])
        app._log_message("ERROR", "something broke")
        assert app._info_text._state == "disabled"

    def test_render_agents_draws_text_labels(self):
        """_render_agents should draw create_text for each visible agent."""
        snap = _make_snapshot_with_rail(num_agents=2)
        app = self._build_app([snap])
        text_calls = app._canvas.create_text.call_args_list
        assert len(text_calls) == 2


# ══════════════════════════════════════════════════════════════════════════
# Property-Based Tests (Task 9)
# ══════════════════════════════════════════════════════════════════════════

import tempfile
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from flatland_sim.navigator import PlaybackEngine, format_transition_label, TRANSITION_LABELS
from flatland_sim.scenario_store import ScenarioStore
from flatland_sim.snapshot import ScenarioSnapshot


# ── Hypothesis strategy for PlaybackEngine in arbitrary valid states ──────

@st.composite
def playback_engine_strategy(draw):
    """Generate a PlaybackEngine in an arbitrary valid state."""
    total_steps = draw(st.integers(min_value=1, max_value=1000))
    current_index = draw(st.integers(min_value=0, max_value=total_steps - 1))
    is_playing = draw(st.booleans())
    speed_ms = draw(st.integers(min_value=50, max_value=2000))
    engine = PlaybackEngine(total_steps)
    engine._current_index = current_index
    engine._playing = is_playing
    engine._speed_ms = speed_ms
    return engine


# ── Property 1: Store loading preserves scenario IDs ─────────────────────
# Feature: scenario-time-navigator, Property 1: Store loading preserves scenario IDs
# **Validates: Requirements 1.1**

@given(
    scenario_ids=st.lists(
        st.integers(min_value=0, max_value=100_000),
        min_size=1,
        max_size=20,
        unique=True,
    )
)
@settings(max_examples=100)
def test_prop_store_loading_preserves_scenario_ids(scenario_ids):
    """Loading a store built from generated snapshots preserves scenario IDs."""
    snapshots = [
        ScenarioSnapshot(
            scenario_id=sid,
            config={"n_agents": 1},
            env_width=3,
            env_height=3,
            num_agents=1,
            distance_map=np.zeros((1, 3, 3, 4)),
            rail_grid=np.zeros((3, 3), dtype=np.uint16),
            rail_transitions={},
            agent_targets=[(2, 2)],
            agent_initial_positions=[(0, 0)],
            timesteps=[{"step": 0, "agents": []}],
        )
        for sid in scenario_ids
    ]
    store = ScenarioStore(snapshots)
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "store.pkl"
        store.save(pkl_path)
        reloaded = ScenarioStore.load(pkl_path)
    assert reloaded.ids == sorted(scenario_ids)


# ── Property 3: reset(n) sets current_index == 0, is_playing == False, total_steps == n
# Feature: scenario-time-navigator, Property 3: reset sets index to zero and pauses
# **Validates: Requirements 2.1, 2.3**

@given(engine=playback_engine_strategy(), new_total=st.integers(min_value=0, max_value=1000))
@settings(max_examples=100)
def test_prop_reset_sets_index_zero_and_pauses(engine, new_total):
    engine.reset(new_total)
    assert engine.current_index == 0
    assert engine.is_playing is False
    assert engine.total_steps == new_total


# ── Property 4: tick() advances by 1 when playing and not at end; auto-pauses at end
# Feature: scenario-time-navigator, Property 4: Tick advances or auto-pauses
# **Validates: Requirements 4.1, 4.3**

@given(engine=playback_engine_strategy())
@settings(max_examples=100)
def test_prop_tick_advances_or_auto_pauses(engine):
    # Force playing state to test tick behavior
    engine._playing = True
    old_index = engine.current_index
    total = engine.total_steps

    changed = engine.tick()

    if old_index < total - 1:
        assert changed is True
        assert engine.current_index == old_index + 1
        assert engine.is_playing is True
    else:
        # At end — auto-pause, index unchanged
        assert changed is False
        assert engine.current_index == old_index
        assert engine.is_playing is False


# ── Property 5: pause() sets is_playing = False and preserves current_index
# Feature: scenario-time-navigator, Property 5: Pause preserves index
# **Validates: Requirements 4.2**

@given(engine=playback_engine_strategy())
@settings(max_examples=100)
def test_prop_pause_preserves_index(engine):
    old_index = engine.current_index
    engine.pause()
    assert engine.is_playing is False
    assert engine.current_index == old_index


# ── Property 6: set_speed(ms) clamps to [50, 2000]
# Feature: scenario-time-navigator, Property 6: Speed clamping
# **Validates: Requirements 4.5**

@given(ms=st.integers(min_value=-10_000, max_value=100_000))
@settings(max_examples=100)
def test_prop_set_speed_clamps(ms):
    engine = PlaybackEngine(total_steps=10)
    engine.set_speed(ms)
    assert 50 <= engine.speed_ms <= 2000
    if ms < 50:
        assert engine.speed_ms == 50
    elif ms > 2000:
        assert engine.speed_ms == 2000
    else:
        assert engine.speed_ms == ms


# ── Property 7: step_forward() advances by 1 when not at end; no-op at end
# Feature: scenario-time-navigator, Property 7: Step forward advances by one
# **Validates: Requirements 5.1, 5.3**

@given(engine=playback_engine_strategy())
@settings(max_examples=100)
def test_prop_step_forward(engine):
    old_index = engine.current_index
    total = engine.total_steps
    result = engine.step_forward()

    if old_index < total - 1:
        assert result is True
        assert engine.current_index == old_index + 1
    else:
        assert result is False
        assert engine.current_index == old_index


# ── Property 8: step_backward() decreases by 1 when not at start; no-op at start
# Feature: scenario-time-navigator, Property 8: Step backward decreases by one
# **Validates: Requirements 5.2, 5.4**

@given(engine=playback_engine_strategy())
@settings(max_examples=100)
def test_prop_step_backward(engine):
    old_index = engine.current_index
    result = engine.step_backward()

    if old_index > 0:
        assert result is True
        assert engine.current_index == old_index - 1
    else:
        assert result is False
        assert engine.current_index == 0


# ── Property 9: step_forward(), step_backward(), jump_to() all pause playback
# Feature: scenario-time-navigator, Property 9: Navigation actions pause playback
# **Validates: Requirements 5.5, 6.4**

@given(engine=playback_engine_strategy(), jump_target=st.integers(min_value=-100, max_value=1100))
@settings(max_examples=100)
def test_prop_navigation_pauses_playback(engine, jump_target):
    # Test step_forward pauses
    engine._playing = True
    engine.step_forward()
    assert engine.is_playing is False

    # Test step_backward pauses
    engine._playing = True
    engine.step_backward()
    assert engine.is_playing is False

    # Test jump_to pauses
    engine._playing = True
    engine.jump_to(jump_target)
    assert engine.is_playing is False


# ── Property 10: jump_to(n) sets index to clamp(n, 0, total_steps - 1)
# Feature: scenario-time-navigator, Property 10: Jump-to sets index
# **Validates: Requirements 6.2**

@given(engine=playback_engine_strategy(), n=st.integers(min_value=-1000, max_value=2000))
@settings(max_examples=100)
def test_prop_jump_to_clamps(engine, n):
    total = engine.total_steps
    engine.jump_to(n)
    expected = max(0, min(total - 1, n))
    assert engine.current_index == expected


# ── Property 11: format_transition_label returns correct name for 0–7, fallback for others
# Feature: scenario-time-navigator, Property 11: Transition label formatting round-trip
# **Validates: Requirements 7.3**

@given(label=st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=100)
def test_prop_format_transition_label(label):
    result = format_transition_label(label)
    if 0 <= label <= 7:
        expected_names = {
            0: "WAITING",
            1: "INTENTIONAL_STOP",
            2: "FREE_FORWARD",
            3: "FREE_LEFT",
            4: "FREE_RIGHT",
            5: "BLOCKED",
            6: "END",
            7: "DONE",
        }
        assert result == expected_names[label]
    else:
        assert result == f"UNKNOWN({label})"


# ══════════════════════════════════════════════════════════════════════════
# Task 10: CLI Property Test + Edge Case Classes
# ══════════════════════════════════════════════════════════════════════════


# ── Property 2: Invalid path produces error ──────────────────────────────
# Feature: scenario-time-navigator, Property 2: Invalid path produces error
# **Validates: Requirements 1.2, 8.3**

@st.composite
def nonexistent_path_strategy(draw):
    """Generate random file paths that are very unlikely to exist."""
    segments = draw(st.lists(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=4,
    ))
    filename = draw(st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
        min_size=1,
        max_size=12,
    ))
    ext = draw(st.sampled_from([".pkl", ".pickle", ".dat", ".bin", ""]))
    return "/tmp/_nonexistent_test_dir_/" + "/".join(segments) + "/" + filename + ext


@given(bad_path=nonexistent_path_strategy())
@settings(max_examples=100)
def test_prop_invalid_path_produces_error(bad_path):
    """main() with a non-existent path exits non-zero with error on stderr."""
    import os
    import io
    # Ensure the path truly doesn't exist
    assume(not os.path.exists(bad_path))

    stderr_capture = io.StringIO()
    with patch("sys.argv", ["navigator", bad_path]), \
         patch("sys.stderr", stderr_capture):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    assert len(stderr_capture.getvalue()) > 0  # some error message was printed


# ── TestPlaybackEngineZeroSteps ──────────────────────────────────────────


class TestPlaybackEngineZeroSteps:
    """Comprehensive edge-case tests for PlaybackEngine with total_steps=0."""

    def test_step_forward_is_noop(self):
        """step_forward returns False and index stays 0."""
        engine = PlaybackEngine(total_steps=0)
        assert engine.step_forward() is False
        assert engine.current_index == 0

    def test_step_backward_is_noop(self):
        """step_backward returns False and index stays 0."""
        engine = PlaybackEngine(total_steps=0)
        assert engine.step_backward() is False
        assert engine.current_index == 0

    def test_jump_to_is_noop(self):
        """jump_to any value keeps index at 0."""
        engine = PlaybackEngine(total_steps=0)
        engine.jump_to(5)
        assert engine.current_index == 0
        engine.jump_to(-1)
        assert engine.current_index == 0
        engine.jump_to(0)
        assert engine.current_index == 0

    def test_tick_when_playing_auto_pauses(self):
        """tick when playing returns False and auto-pauses."""
        engine = PlaybackEngine(total_steps=0)
        engine._playing = True
        assert engine.tick() is False
        assert engine.is_playing is False
        assert engine.current_index == 0

    def test_play_pause_work_but_tick_does_nothing(self):
        """play/pause toggle works, but tick still does nothing useful."""
        engine = PlaybackEngine(total_steps=0)
        engine.play()
        assert engine.is_playing is True
        # tick immediately auto-pauses
        assert engine.tick() is False
        assert engine.is_playing is False
        # pause is idempotent
        engine.pause()
        assert engine.is_playing is False

    def test_reset_works_correctly(self):
        """reset to a new total_steps works from zero-step state."""
        engine = PlaybackEngine(total_steps=0)
        engine.reset(10)
        assert engine.current_index == 0
        assert engine.total_steps == 10
        assert engine.is_playing is False
        # And reset back to 0
        engine.reset(0)
        assert engine.total_steps == 0
        assert engine.current_index == 0


# ── TestPlaybackEngineSingleStep ─────────────────────────────────────────


class TestPlaybackEngineSingleStep:
    """Comprehensive edge-case tests for PlaybackEngine with total_steps=1."""

    def test_step_forward_at_index_0_is_noop(self):
        """With total_steps=1, index 0 is already at end (total_steps-1==0), so step_forward is no-op."""
        engine = PlaybackEngine(total_steps=1)
        assert engine.step_forward() is False
        assert engine.current_index == 0

    def test_step_backward_at_index_0_is_noop(self):
        """At index 0, step_backward is no-op."""
        engine = PlaybackEngine(total_steps=1)
        assert engine.step_backward() is False
        assert engine.current_index == 0

    def test_jump_to_0_works(self):
        """jump_to(0) sets index to 0."""
        engine = PlaybackEngine(total_steps=1)
        engine.jump_to(0)
        assert engine.current_index == 0

    def test_jump_to_1_clamps_to_0(self):
        """jump_to(1) clamps to total_steps-1 == 0."""
        engine = PlaybackEngine(total_steps=1)
        engine.jump_to(1)
        assert engine.current_index == 0

    def test_jump_to_negative_clamps_to_0(self):
        """jump_to(-5) clamps to 0."""
        engine = PlaybackEngine(total_steps=1)
        engine.jump_to(-5)
        assert engine.current_index == 0

    def test_tick_when_playing_at_index_0_auto_pauses(self):
        """With total_steps=1, index 0 is the end, so tick auto-pauses."""
        engine = PlaybackEngine(total_steps=1)
        engine.play()
        assert engine.tick() is False
        assert engine.is_playing is False
        assert engine.current_index == 0
