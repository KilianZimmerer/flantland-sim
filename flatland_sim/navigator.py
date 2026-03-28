"""Scenario Time Navigator — playback engine and GUI for Flatland simulations."""

from __future__ import annotations

import sys
from pathlib import Path

from flatland_sim.scenario_store import ScenarioStore
from flatland_sim.snapshot import ScenarioSnapshot

TRANSITION_LABELS: dict[int, str] = {
    0: "WAITING",
    1: "INTENTIONAL_STOP",
    2: "FREE_FORWARD",
    3: "FREE_LEFT",
    4: "FREE_RIGHT",
    5: "BLOCKED",
    6: "END",
    7: "DONE",
}


def format_transition_label(label: int) -> str:
    """Return human-readable name for a transition label integer."""
    return TRANSITION_LABELS.get(label, f"UNKNOWN({label})")


class PlaybackEngine:
    """Manages playback state (current index, play/pause, speed) independently of GUI."""

    def __init__(self, total_steps: int, speed_ms: int = 200) -> None:
        self._current_index: int = 0
        self._total_steps: int = total_steps
        self._playing: bool = False
        self._speed_ms: int = self._clamp_speed(speed_ms)

    # -- helpers --

    @staticmethod
    def _clamp_speed(ms: int) -> int:
        return max(50, min(2000, ms))

    # -- read-only properties --

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def speed_ms(self) -> int:
        return self._speed_ms

    # -- mutators --

    def set_speed(self, ms: int) -> None:
        """Clamp *ms* to [50, 2000] and update."""
        self._speed_ms = self._clamp_speed(ms)

    def play(self) -> None:
        """Set playing state to True."""
        self._playing = True

    def pause(self) -> None:
        """Set playing state to False."""
        self._playing = False

    def step_forward(self) -> bool:
        """Advance index by 1. Returns True if index changed. Pauses if playing."""
        self._playing = False
        if self._total_steps == 0:
            return False
        if self._current_index < self._total_steps - 1:
            self._current_index += 1
            return True
        return False

    def step_backward(self) -> bool:
        """Decrease index by 1. Returns True if index changed. Pauses if playing."""
        self._playing = False
        if self._current_index > 0:
            self._current_index -= 1
            return True
        return False

    def jump_to(self, index: int) -> None:
        """Set index to clamped value. Pauses if playing."""
        self._playing = False
        if self._total_steps == 0:
            self._current_index = 0
            return
        self._current_index = max(0, min(self._total_steps - 1, index))

    def tick(self) -> bool:
        """Called by the timer. If playing, advance by 1. Auto-pauses at end. Returns True if index changed."""
        if not self._playing:
            return False
        if self._total_steps == 0:
            self._playing = False
            return False
        if self._current_index < self._total_steps - 1:
            self._current_index += 1
            return True
        # At end — auto-pause, index unchanged
        self._playing = False
        return False

    def reset(self, total_steps: int) -> None:
        """Reset for a new scenario. Sets index to 0, pauses."""
        self._current_index = 0
        self._playing = False
        self._total_steps = total_steps


class NavigatorApp:
    """Tkinter GUI application for scenario time navigation."""

    def __init__(self, store: ScenarioStore) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._store = store
        self._snapshot: ScenarioSnapshot | None = None
        self._engine = PlaybackEngine(total_steps=0)
        self._cell_size: float = 0.0

        # -- root window --
        self._root = tk.Tk()
        self._root.title("Flatland Scenario Navigator")
        self._root.geometry("900x650")

        # -- top bar: scenario selector --
        top_frame = ttk.Frame(self._root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Scenario:").pack(side=tk.LEFT, padx=(0, 5))

        self._scenario_var = tk.StringVar()
        self._scenario_combo = ttk.Combobox(
            top_frame,
            textvariable=self._scenario_var,
            state="readonly",
            width=20,
        )
        self._scenario_combo.pack(side=tk.LEFT)
        self._scenario_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)

        # -- status label (right side of top bar) --
        self._status_var = tk.StringVar(value="")
        ttk.Label(top_frame, textvariable=self._status_var).pack(side=tk.RIGHT)

        # -- canvas --
        self._canvas = tk.Canvas(self._root, bg="white")
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # -- timeline slider (between canvas and control bar) --
        self._updating_slider = False
        timeline_frame = ttk.Frame(self._root)
        timeline_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(timeline_frame, text="Timeline:").pack(side=tk.LEFT, padx=(0, 5))
        self._timeline_slider = tk.Scale(
            timeline_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_slider_change,
        )
        self._timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- control bar frame --
        self._control_frame = ttk.Frame(self._root)
        self._control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self._btn_step_back = ttk.Button(
            self._control_frame, text="⏮ Step Back", command=self._on_step_back,
        )
        self._btn_step_back.pack(side=tk.LEFT, padx=2)

        self._btn_play = ttk.Button(
            self._control_frame, text="▶ Play", command=self._on_play,
        )
        self._btn_play.pack(side=tk.LEFT, padx=2)

        self._btn_pause = ttk.Button(
            self._control_frame, text="⏸ Pause", command=self._on_pause,
        )
        self._btn_pause.pack(side=tk.LEFT, padx=2)

        self._btn_step_fwd = ttk.Button(
            self._control_frame, text="Step Fwd ⏭", command=self._on_step_fwd,
        )
        self._btn_step_fwd.pack(side=tk.LEFT, padx=2)

        ttk.Label(self._control_frame, text="Speed (ms):").pack(side=tk.LEFT, padx=(10, 2))
        self._speed_slider = tk.Scale(
            self._control_frame,
            from_=50,
            to=2000,
            orient=tk.HORIZONTAL,
            command=self._on_speed_change,
        )
        self._speed_slider.set(self._engine.speed_ms)
        self._speed_slider.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self._tick_id: str | None = None

        # -- info panel frame --
        self._info_frame = ttk.Frame(self._root)
        self._info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        ttk.Label(self._info_frame, text="Agent Info:").pack(anchor=tk.W)
        self._info_text = tk.Text(self._info_frame, height=6, state=tk.DISABLED, wrap=tk.NONE)
        self._info_text.pack(fill=tk.X, expand=True)

        # -- populate scenario list --
        scenario_ids = store.ids
        if scenario_ids:
            self._scenario_combo["values"] = [str(sid) for sid in scenario_ids]
            self._scenario_combo.current(0)
            self._on_scenario_selected(scenario_ids[0])
        else:
            self._scenario_combo["values"] = []
            self._status_var.set("No scenarios loaded")

    # -- event handlers --

    def _on_combo_selected(self, _event: object = None) -> None:
        """Handle combobox selection event."""
        value = self._scenario_var.get()
        if value:
            self._on_scenario_selected(int(value))

    def _on_canvas_resize(self, _event: object = None) -> None:
        """Redraw the grid when the canvas is resized."""
        if self._snapshot is not None:
            self._render_static()
            self._render_agents()

    def _on_scenario_selected(self, scenario_id: int) -> None:
        """Load the selected scenario snapshot, reset the engine, and trigger a redraw."""
        self._snapshot = self._store.get(scenario_id)
        total_steps = len(self._snapshot.timesteps)
        self._engine.reset(total_steps)
        self._timeline_slider.configure(to=max(0, total_steps - 1))
        self._timeline_slider.set(0)
        self._cell_size = 0.0  # will be computed in _render_static
        self._render_static()
        self._render_agents()
        self._update_controls()

    # -- playback control handlers --

    def _on_play(self) -> None:
        """Start playback and begin the tick loop."""
        self._engine.play()
        self._render_agents()
        self._update_controls()
        self._update_info_panel()
        self._start_tick()

    def _on_pause(self) -> None:
        """Pause playback and refresh display."""
        self._engine.pause()
        self._render_agents()
        self._update_controls()
        self._update_info_panel()

    def _on_step_fwd(self) -> None:
        """Step forward one timestep and refresh display."""
        self._engine.step_forward()
        self._render_agents()
        self._update_controls()
        self._update_info_panel()

    def _on_step_back(self) -> None:
        """Step backward one timestep and refresh display."""
        self._engine.step_backward()
        self._render_agents()
        self._update_controls()
        self._update_info_panel()

    def _on_slider_change(self, value: str) -> None:
        """Handle timeline slider change — jump engine to the selected timestep."""
        if self._updating_slider:
            return
        self._engine.jump_to(int(float(value)))
        self._render_agents()
        self._update_controls()
        self._update_info_panel()

    def _on_speed_change(self, value: str) -> None:
        """Handle speed slider change."""
        self._engine.set_speed(int(float(value)))

    def _start_tick(self) -> None:
        """Schedule the first tick callback."""
        if self._tick_id is not None:
            self._root.after_cancel(self._tick_id)
        self._tick_id = self._root.after(self._engine.speed_ms, self._tick)

    def _tick(self) -> None:
        """Auto-advance one step during playback, then schedule the next tick."""
        self._tick_id = None
        changed = self._engine.tick()
        if changed:
            self._render_agents()
            self._update_controls()
            self._update_info_panel()
        if self._engine.is_playing:
            self._tick_id = self._root.after(self._engine.speed_ms, self._tick)

    # -- rendering (Tasks 5-8) --

    # Agent colors for up to 10 agents; cycles if more.
    _AGENT_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ]

    # Direction offsets for drawing: index 0=N, 1=E, 2=S, 3=W
    # In canvas coords: N is up (-y), E is right (+x), S is down (+y), W is left (-x)
    _DIR_OFFSETS = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}

    def _compute_cell_size(self) -> float:
        """Compute cell size to fit the canvas, caching the result."""
        if self._snapshot is None:
            return 0.0
        self._canvas.update_idletasks()
        try:
            canvas_w = int(self._canvas.winfo_width())
            canvas_h = int(self._canvas.winfo_height())
        except (TypeError, ValueError):
            canvas_w = 0
            canvas_h = 0
        if canvas_w <= 1:
            canvas_w = 600
        if canvas_h <= 1:
            canvas_h = 400
        cols = self._snapshot.env_width
        rows = self._snapshot.env_height
        if cols == 0 or rows == 0:
            return 0.0
        return min(canvas_w / cols, canvas_h / rows)

    def _render_static(self) -> None:
        """Draw the rail grid, transition indicators, and targets (called once per scenario)."""
        self._canvas.delete("all")
        if self._snapshot is None:
            return

        snap = self._snapshot
        cell = self._compute_cell_size()
        self._cell_size = cell
        if cell == 0.0:
            return

        cols = snap.env_width
        rows = snap.env_height

        # -- draw rail grid --
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell
                y0 = r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                val = snap.rail_grid[r, c]
                fill = "#d0d0d0" if val != 0 else "#ffffff"
                self._canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#e0e0e0", tags="grid")

        # -- draw transition indicators --
        # Collect all possible exit directions per cell across all entry directions.
        # rail_transitions: (row, col, entry_dir) -> (n, e, s, w) where non-zero = can exit
        exit_dirs: dict[tuple[int, int], set[int]] = {}
        for (r, c, _entry_dir), transitions in snap.rail_transitions.items():
            for exit_dir, val in enumerate(transitions):
                if val != 0:
                    exit_dirs.setdefault((r, c), set()).add(exit_dir)

        for (r, c), dirs in exit_dirs.items():
            cx = c * cell + cell / 2
            cy = r * cell + cell / 2
            for d in dirs:
                dx, dy = self._DIR_OFFSETS.get(d, (0, 0))
                ex = cx + dx * cell
                ey = cy + dy * cell
                self._canvas.create_line(
                    cx, cy, ex, ey,
                    fill="#7090b0", width=max(1, cell * 0.08),
                    tags="transition",
                )

    def _render_agents(self) -> None:
        """Redraw only the agent markers for the current timestep (no flicker)."""
        self._canvas.delete("agent")
        if self._snapshot is None:
            return

        snap = self._snapshot
        cell = self._cell_size
        if cell == 0.0:
            return
        idx = self._engine.current_index

        if idx < len(snap.timesteps):
            timestep = snap.timesteps[idx]
            for agent in timestep.get("agents", []):
                pos = agent.get("position")
                if pos is None:
                    continue
                ar, ac = pos
                aid = agent.get("id", 0)
                color = self._AGENT_COLORS[aid % len(self._AGENT_COLORS)]
                cx = ac * cell + cell / 2
                cy = ar * cell + cell / 2
                radius = cell * 0.3
                self._canvas.create_oval(
                    cx - radius, cy - radius, cx + radius, cy + radius,
                    fill=color, outline="black", width=1, tags="agent",
                )

    def _render_grid(self) -> None:
        """Full redraw — static grid + agents. Used by tests and initial render."""
        self._render_static()
        self._render_agents()

    def _update_controls(self) -> None:
        """Sync control widgets with engine state."""
        if self._snapshot is None:
            self._status_var.set("No scenarios loaded")
            return
        idx = self._engine.current_index
        total = self._engine.total_steps
        agents = self._snapshot.num_agents
        self._status_var.set(f"Step {idx + 1}/{total} | Agents: {agents}")
        # Guard against slider command callback pausing the engine.
        # Keep the flag set until the next idle cycle so any deferred
        # Scale command callbacks are also suppressed.
        self._updating_slider = True
        self._timeline_slider.set(idx)
        self._root.after_idle(self._clear_updating_slider)

    def _clear_updating_slider(self) -> None:
        """Reset the slider guard after the event loop has processed pending events."""
        self._updating_slider = False

    def _update_info_panel(self) -> None:
        """Update the agent information panel with current timestep agent details."""
        import tkinter as tk

        self._info_text.configure(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)

        if self._snapshot is None:
            self._info_text.configure(state=tk.DISABLED)
            return

        idx = self._engine.current_index
        if idx >= len(self._snapshot.timesteps):
            self._info_text.configure(state=tk.DISABLED)
            return

        timestep = self._snapshot.timesteps[idx]
        lines: list[str] = []
        for agent in timestep.get("agents", []):
            aid = agent.get("id", "?")
            status = agent.get("status", "?")
            pos = agent.get("position")
            direction = agent.get("direction", "?")
            transition = format_transition_label(agent.get("transition_label", -1))
            pos_str = f"({pos[0]}, {pos[1]})" if pos is not None else "None"
            lines.append(
                f"Agent {aid} | status={status} | pos={pos_str} | dir={direction} | transition={transition}"
            )

        self._info_text.insert("1.0", "\n".join(lines))
        self._info_text.configure(state=tk.DISABLED)

    # -- public API --

    def run(self) -> None:  # pragma: no cover
        """Start the tkinter mainloop."""
        self._root.mainloop()


def main() -> None:
    """CLI entry point: load a ScenarioStore and launch the navigator GUI."""
    if len(sys.argv) < 2:
        print("Usage: python -m flatland_sim.navigator <path_to_store.pkl>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        store = ScenarioStore.load(path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    app = NavigatorApp(store)
    app.run()


if __name__ == "__main__":
    main()
