#!/usr/bin/env python3
"""
Parametric coil tab (TX / RX).
"""

import os, sys, math, threading, queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
if _modules not in sys.path:
    sys.path.insert(0, _modules)

import parametric_coil as pc

try:
    import fasthenry_runner as runner
    import zc_parser
    _RUNNER_OK = True
except Exception:
    runner = None
    zc_parser = None
    _RUNNER_OK = False

# --- Visual constants -----------------------------------------------------
# User convention: layer 1 topmost = red, 2 green, 3 orange, 4 blue.
LAYER_COLORS = ["#e63434", "#18d935", "#ce6c33", "#2080d0"]

ROW_BG_EMPTY    = "#f0f0f0"
ROW_BG_FILLED   = "#dfeeff"
ROW_BG_SELECTED = "#ffd89a"

TOPOLOGY_CHOICES = [
    ("All parallel",   "parallel"),
    ("All series",     "series"),
    ("1p2 -|- 3p4",    "parallel_pairs_ser"),
]

_STATUS_COLORS = {
    "nothing":  ("Nothing loaded in Sim",        "#c01010"),
    "current":  ("Current Coil loaded in Sim",   "#1a8020"),
    "outdated": ("Outdated Coil loaded in Sim",  "#c06010"),
}

# For the default (non-true-scale) 3D view, exaggerate z gaps so thin
# PCB stacks are visible alongside coil ODs of tens of mm.
Z_EXAGGERATION = 12.0


class ParametricCoilTab(ttk.Frame):
    DEBOUNCE_MS = 150

    DEFAULT_OD_MM         = 52.0
    DEFAULT_TRACE_W_MM    = 0.4
    DEFAULT_SPACING_MM    = 0.16
    DEFAULT_TURNS         = 11
    DEFAULT_RESOLUTION_MM = 0.6
    DEFAULT_OUTER_GAP     = 0.2
    DEFAULT_INNER_GAP     = 1.3
    DEFAULT_COPPER_OZ     = [1.0, 0.5, 0.5, 1.0]

    QUICKSIM_RESOLUTION_MM = 1.2
    QUICKSIM_TIMEOUT_SEC   = 120   # wall-clock limit for quickSim

    def __init__(self, parent, app, role, coil_index, temp_dir,
                 on_next_tab=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.role = role
        self.coil_index = coil_index
        self.temp_dir = temp_dir
        self._on_next_tab = on_next_tab

        self._layer_data = None           # display-native (for visualizer)
        self._layer_data_emitted = None   # what was last sent to sim
        self._last_params = None
        self._selected_slot_idx = None
        self._debounce_id = None
        self._enabled = True
        self._my_temp_files = set()

        self._link_cu = tk.BooleanVar(value=True)
        self._true_scale = tk.BooleanVar(value=False)
        self._port_inside = tk.BooleanVar(value=False)

        # QuickSim state.
        self._quicksim_value = None    # float µH or None
        self._quicksim_stale = False
        self._quicksim_thread = None
        self._quicksim_queue = queue.Queue()

        self._build()
        self.after(100, self._post_build_init)

    # ---- post-init: restore from savestate, then first refresh -----------
    def _post_build_init(self):
        self._suspend_savestate = True
        try:
            self._restore_from_savestate()
        finally:
            self._suspend_savestate = False
        self._schedule_refresh()
        self.after(self.DEBOUNCE_MS + 50, self._poll_quicksim)

    # ---- UI build --------------------------------------------------------
    def _build(self):
        main = ttk.Frame(self); main.pack(fill="both", expand=True,
                                           padx=6, pady=6)
        ctrl = ttk.Frame(main); ctrl.pack(side="left", fill="y",
                                           padx=(0, 8))
        self._build_controls(ctrl)
        right = ttk.Frame(main); right.pack(side="left", fill="both",
                                             expand=True)
        self._build_viewer(right)

    def _build_controls(self, parent):
        # Topology picker.
        tf = ttk.LabelFrame(parent, text="Topology")
        tf.pack(fill="x", pady=(0, 6))
        self.topology_var = tk.StringVar(value="parallel")
        for label, val in TOPOLOGY_CHOICES:
            ttk.Radiobutton(tf, text=label, variable=self.topology_var,
                            value=val,
                            command=self._on_topology_change
                            ).pack(anchor="w", padx=6, pady=1)
        self._port_inside_cb = ttk.Checkbutton(
            tf, text="Port on Inside",
            variable=self._port_inside,
            command=self._on_port_inside_change,
            state="enabled")
        self._port_inside_cb.pack(anchor="w", padx=6, pady=(2, 2))

        # Spiral settings.
        sp = ttk.LabelFrame(parent, text="Spiral (applies to all active layers)")
        sp.pack(fill="x", pady=(0, 6))
        self.od_var    = self._entry(sp, "OD (mm):",            self.DEFAULT_OD_MM)
        self.w_var     = self._entry(sp, "Trace width (mm):",   self.DEFAULT_TRACE_W_MM)
        self.s_var     = self._entry(sp, "Trace spacing (mm):", self.DEFAULT_SPACING_MM)
        self.turns_var = self._entry(sp, "Turns/layer:",        self.DEFAULT_TURNS)
        self.res_var   = self._entry(sp, "Resolution (mm):",    self.DEFAULT_RESOLUTION_MM)

        # Stack-up gaps.
        zf = ttk.LabelFrame(parent, text="Stack-up gaps")
        zf.pack(fill="x", pady=(0, 6))
        self.outer_gap_var = self._entry(zf, "Outer L. gap (mm):", self.DEFAULT_OUTER_GAP)
        self.inner_gap_var = self._entry(zf, "Inner L. gap (mm):", self.DEFAULT_INNER_GAP)

        # Layer row table.
        lt = ttk.LabelFrame(parent, text="Layers")
        lt.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(lt, text="Link copper oz (outer pair + inner pair)",
                        variable=self._link_cu,
                        command=self._on_link_toggle
                        ).pack(anchor="w", padx=4, pady=(2, 4))

        self.slot_active_vars = []
        self.slot_oz_vars = []
        self.slot_row_frames = []
        for i in range(4):
            row = tk.Frame(lt, bg=ROW_BG_EMPTY, bd=1, relief="solid")
            row.pack(fill="x", padx=4, pady=1)

            av = tk.BooleanVar(value=(i == 0))
            ttk.Checkbutton(row, text=f"Layer {i+1}", variable=av,
                            command=self._on_active_toggle
                            ).pack(side="left", padx=4, pady=2)

            tk.Label(row, text="Cu oz:", bg=ROW_BG_EMPTY
                     ).pack(side="left", padx=(4, 0))
            ozv = tk.StringVar(value=f"{self.DEFAULT_COPPER_OZ[i]}")
            ttk.Entry(row, textvariable=ozv, width=6
                      ).pack(side="left", padx=2, pady=2)
            ozv.trace_add("write", lambda *_a, idx=i: self._on_oz_change(idx))

            self.slot_active_vars.append(av)
            self.slot_oz_vars.append(ozv)
            self.slot_row_frames.append(row)

            row.bind("<Button-1>", lambda e, idx=i: self._focus_slot(idx))
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.bind("<Button-1>",
                               lambda e, idx=i: self._focus_slot(idx))

        # View-mode buttons.
        vf = ttk.Frame(parent); vf.pack(fill="x", pady=(4, 2))
        ttk.Button(vf, text="View Stack (3D)",
                   command=self._view_stack).pack(fill="x", padx=4, pady=1)
        ttk.Checkbutton(vf, text="True Scale",
                        variable=self._true_scale,
                        command=self._on_true_scale_toggle
                        ).pack(anchor="w", padx=4, pady=(2, 2))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # QuickSim row.
        qs = ttk.Frame(parent); qs.pack(fill="x", pady=(0, 2))
        self.quicksim_btn = ttk.Button(qs, text="Quick Sim L",
                                        command=self._on_quick_sim,
                                        width=14)
        self.quicksim_btn.pack(side="left", padx=4)
        self.quicksim_label = tk.Label(qs, text="~ — µH", fg="gray")
        self.quicksim_label.pack(side="left", padx=4)

        # Send to Sim + status + Next Tab.
        ttk.Button(parent, text=f"Send to Simulation ({self.role})",
                   width=28,
                   command=self._on_send_to_sim).pack(fill="x", padx=4, pady=2)
        self.sim_status_label = tk.Label(parent, text="",
                                          fg=_STATUS_COLORS["nothing"][1],
                                          anchor="w")
        self.sim_status_label.pack(fill="x", padx=4, pady=(0, 2))
        self._set_sim_status("nothing")

        ttk.Button(parent, text="Next Tab →", width=28,
                   command=self._on_next_tab_click
                   ).pack(fill="x", padx=4, pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

        # Export / reset.
        ttk.Button(parent, text="Export INP…", width=28,
                   command=self.export_inp).pack(fill="x", padx=4, pady=1)
        ttk.Button(parent, text="Reset This Tab", width=28,
                   command=self.reset_this_tab).pack(fill="x", padx=4, pady=1)

    def _build_viewer(self, parent):
        self.fig = Figure(figsize=(6.5, 6.5), dpi=95)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        bar = ttk.Frame(parent); bar.pack(fill="x", pady=(4, 0))
        self.status_var = tk.StringVar(value="")
        self.info_var   = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status_var,
                  foreground="#a03020").pack(side="left")
        ttk.Label(bar, textvariable=self.info_var,
                  foreground="gray").pack(side="right")

    def _entry(self, parent, label, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        var = tk.StringVar(value=f"{default}")
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        var.trace_add("write", lambda *_a: self._schedule_refresh())
        return var

    # ---- enable/disable --------------------------------------------------
    def set_enabled(self, enabled):
        self._enabled = enabled
        state = "normal" if enabled else "disabled"
        self._walk_state(self, state)
        if not enabled:
            self.status_var.set(
                f"DXF loaded on paired {self.role} tab — "
                f"parametric input ignored")
            self.info_var.set("")
            self.fig.clf(); self.canvas.draw_idle()
        else:
            self.status_var.set("")
            self._schedule_refresh()

    def _walk_state(self, widget, state):
        try: widget.configure(state=state)
        except Exception: pass
        for child in widget.winfo_children():
            self._walk_state(child, state)

    # ---- user events -----------------------------------------------------
    def _on_topology_change(self):
        topo = self.topology_var.get()
        if topo == "parallel_pairs_ser":
            for av in self.slot_active_vars:
                av.set(True)
        if topo == "parallel":
            self._port_inside.set(False)
            self._port_inside_cb.configure(state="disabled")
        else:
            self._port_inside_cb.configure(state="normal")
        self._mark_quicksim_stale()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()
        self._save_state()

    def _on_port_inside_change(self):
        self._mark_quicksim_stale()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()
        self._save_state()

    def _on_active_toggle(self):
        self._mark_quicksim_stale()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()
        self._save_state()

    def _on_link_toggle(self):
        if self._link_cu.get():
            # Outer pair: slots 1 & 4 share value of slot 1.
            self.slot_oz_vars[3].set(self.slot_oz_vars[0].get())
            # Inner pair: slots 2 & 3 share value of slot 2.
            self.slot_oz_vars[2].set(self.slot_oz_vars[1].get())
        self._mark_quicksim_stale()
        self._schedule_refresh()
        self._save_state()

    def _on_oz_change(self, changed_idx):
        if self._link_cu.get():
            v = self.slot_oz_vars[changed_idx].get()
            # Outer pair: 0 ↔ 3. Inner pair: 1 ↔ 2.
            pair = {0: 3, 3: 0, 1: 2, 2: 1}.get(changed_idx)
            if pair is not None and self.slot_oz_vars[pair].get() != v:
                self.slot_oz_vars[pair].set(v)
        self._mark_quicksim_stale()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()

    def _on_true_scale_toggle(self):
        # View-only toggle; redraw but don't invalidate anything.
        if self._selected_slot_idx is None:
            self._draw_stack_3d()
        self._save_state()

    def _focus_slot(self, slot_idx):
        if not self._enabled or self._layer_data is None:
            return
        match = [ld for ld in self._layer_data
                 if (ld["slot"] - 1) == slot_idx]
        if not match:
            return
        self._selected_slot_idx = slot_idx
        self._paint_row_bgs()
        self._draw_single_layer_ribbon(match[0])

    def _view_stack(self):
        self._selected_slot_idx = None
        self._paint_row_bgs()
        self._draw_stack_3d()

    def _on_next_tab_click(self):
        if self._on_next_tab:
            self._on_next_tab()

    def _paint_row_bgs(self):
        for i, row in enumerate(self.slot_row_frames):
            active = self.slot_active_vars[i].get()
            if self._selected_slot_idx == i:
                bg = ROW_BG_SELECTED
            elif active:
                bg = ROW_BG_FILLED
            else:
                bg = ROW_BG_EMPTY
            row.configure(bg=bg)
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=bg)

    # ---- refresh / validation -------------------------------------------
    def _schedule_refresh(self, *_):
        if not self._enabled: return
        if self._debounce_id is not None:
            try: self.after_cancel(self._debounce_id)
            except tk.TclError: pass
        self._debounce_id = self.after(self.DEBOUNCE_MS, self._refresh_now)
        self._save_state()

    def _refresh_now(self):
        self._debounce_id = None
        if not self._enabled: return
        params, stackup, err = self._snapshot_inputs()
        if err is not None:
            self._show_fail(err); return
        ok, msg = pc.validate_spiral(params)
        if not ok:
            self._show_fail(msg); return
        topo = self.topology_var.get()
        n_active = sum(1 for s in stackup.slots if s.active)
        if topo == "parallel_pairs_ser" and n_active != 4:
            self._show_fail("This topology requires all 4 layers active.")
            return
        if n_active < 1:
            self._show_fail("Activate at least one layer."); return
        try:
            layers = pc.active_layer_data(params, stackup)
        except Exception as e:
            self._show_fail(f"Generation error: {e}"); return

        # Apply reverse flags so the *visualizer* shows true current flow
        # for series topologies (per user's preference).
        flags = pc.series_reverse_flags_for_topology(topo, len(layers))
        if self._port_inside.get() and topo != "parallel":
            flags = [not f for f in flags]
        layers = pc.reverse_nodes_for_series_flow(layers, flags)

        self._layer_data = layers
        self._last_params = params
        self.status_var.set("")
        self._update_info_line()
        self._paint_row_bgs()

        if self._selected_slot_idx is not None:
            match = [ld for ld in layers
                     if (ld["slot"] - 1) == self._selected_slot_idx]
            if match:
                self._draw_single_layer_ribbon(match[0])
            else:
                self._selected_slot_idx = None
                self._draw_stack_3d()
        else:
            self._draw_stack_3d()

    def _update_info_line(self):
        if not self._layer_data:
            self.info_var.set(""); return
        topo = self.topology_var.get()
        n_nodes = sum(len(ld["nodes"]) for ld in self._layer_data)
        t_per = self._last_params.turns
        n_act = len(self._layer_data)
        if   topo == "parallel":            eff_t, eff_m = t_per,      1
        elif topo == "series":              eff_t, eff_m = t_per*n_act, n_act
        elif topo == "parallel_pairs_ser":  eff_t, eff_m = t_per*2,    2
        else:                               eff_t, eff_m = t_per,      1
        single_len = self._single_layer_length_mm()
        self.info_var.set(
            f"{n_act} active | {topo} | eff. turns={eff_t:g} | "
            f"eff. length={single_len*eff_m:.1f} mm | {n_nodes} nodes")

    def _single_layer_length_mm(self):
        if not self._layer_data: return 0.0
        nodes = self._layer_data[0]["nodes"]
        total = 0.0
        for i in range(len(nodes) - 1):
            dx = nodes[i+1][0] - nodes[i][0]
            dy = nodes[i+1][1] - nodes[i][1]
            dz = nodes[i+1][2] - nodes[i][2]
            total += math.sqrt(dx*dx + dy*dy + dz*dz)
        return total

    def _show_fail(self, msg):
        self.status_var.set(msg); self.info_var.set("")
        self._layer_data = None
        self.fig.clf(); self.canvas.draw_idle()

    def _snapshot_inputs(self):
        try:
            params = pc.SpiralParams(
                od_mm=float(self.od_var.get()),
                trace_width_mm=float(self.w_var.get()),
                spacing_mm=float(self.s_var.get()),
                turns=float(self.turns_var.get()),
                resolution_mm=float(self.res_var.get()))
        except ValueError:
            return None, None, "Enter numeric spiral parameters."
        try:
            slots = [pc.LayerSlot(active=bool(self.slot_active_vars[i].get()),
                                  copper_oz=float(self.slot_oz_vars[i].get()))
                     for i in range(4)]
            stackup = pc.StackUp(slots=slots,
                                 outer_gap_mm=float(self.outer_gap_var.get()),
                                 inner_gap_mm=float(self.inner_gap_var.get()))
        except ValueError:
            return None, None, "Enter numeric stack-up parameters."
        return params, stackup, None

    # ---- drawing ---------------------------------------------------------
    def _build_ribbon_world(self, nodes, trace_w):
        """Outer/inner offset polygon along a centerline — 2D."""
        n = len(nodes)
        if n < 2: return []
        half = trace_w / 2.0
        outer, inner = [], []
        for i in range(n):
            if i == 0:
                dx = nodes[1][0]-nodes[0][0]; dy = nodes[1][1]-nodes[0][1]
            elif i == n-1:
                dx = nodes[i][0]-nodes[i-1][0]; dy = nodes[i][1]-nodes[i-1][1]
            else:
                dx = nodes[i+1][0]-nodes[i-1][0]; dy = nodes[i+1][1]-nodes[i-1][1]
            mag = math.hypot(dx, dy) or 1e-9
            px, py = -dy/mag, dx/mag
            x, y = nodes[i][0], nodes[i][1]
            outer.append((x + half*px, y + half*py))
            inner.append((x - half*px, y - half*py))
        return outer + inner[::-1]

    def _draw_single_layer_ribbon(self, ld):
        """Single-layer focus: true-width ribbon, correct color."""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        w = self._last_params.trace_width_mm
        poly = self._build_ribbon_world(ld["nodes"], w)
        color = LAYER_COLORS[(ld["slot"] - 1) % len(LAYER_COLORS)]
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color=color, alpha=0.85,
                    edgecolor="#202020", linewidth=0.5)
        # Start/end markers (reflect reversed direction if series).
        n = ld["nodes"]
        ax.plot(n[0][0],  n[0][1],  "go", markersize=8, label="Entry")
        ax.plot(n[-1][0], n[-1][1], "rs", markersize=8, label="Exit")
        ax.set_aspect("equal")
        ax.set_title(f"Layer {ld['slot']}  (z = {ld['z']:.3f} mm, "
                     f"h = {ld['h_mm']:.3f} mm)")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_stack_3d(self):
        """
        3D stack with user's mental-model z: layer 1 topmost.

        Code-z in parametric_coil.generate_nodes: slot 1 = 0, slot 4 > 0.
        We invert the display axis so z_display = -z_code; matplotlib
        then renders slot 1 at the top of the 3D plot.
        """
        if not self._layer_data: return
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection="3d")

        true_scale = self._true_scale.get()
        z_scale = 1.0 if true_scale else Z_EXAGGERATION

        for ld in self._layer_data:
            xs = [p[0] for p in ld["nodes"]]
            ys = [p[1] for p in ld["nodes"]]
            # Flip z so slot 1 (z_code=0) ends up at the top visually.
            zs = [-p[2] * z_scale for p in ld["nodes"]]
            color = LAYER_COLORS[(ld["slot"] - 1) % len(LAYER_COLORS)]
            ax.plot(xs, ys, zs, color=color, linewidth=1.1,
                    label=f"Layer {ld['slot']}")
            # Entry/exit dots on each layer (makes reversed directions visible).
            ax.scatter([xs[0]], [ys[0]], [zs[0]],
                       color="#00a000", s=20, edgecolors="black", linewidth=0.5)
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                       color="#c01010", s=20, edgecolors="black", linewidth=0.5)

        # Draw via connections as purple lines between layer endpoints.
        topo = self.topology_var.get()
        via_conns = pc.via_connections_for_topology(topo, len(self._layer_data))
        for ki, ei, kj, ej in via_conns:
            pa = self._layer_data[ki]["nodes"][ei]
            pb = self._layer_data[kj]["nodes"][ej]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    [-pa[2] * z_scale, -pb[2] * z_scale],
                    color="#9020c0", linewidth=1.8, zorder=5)

        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_zlabel("-Z (mm, layer 1 top)"
                      + ("" if true_scale else f"  [×{Z_EXAGGERATION:g}]"))
        title = f"{self.role} stack — {self.topology_var.get()}"
        if not true_scale:
            title += "  (z exaggerated)"
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        self._equalize_3d_axes(ax)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _equalize_3d_axes(ax):
        xl = ax.get_xlim3d(); yl = ax.get_ylim3d(); zl = ax.get_zlim3d()
        span = max(xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0])
        cx = 0.5*(xl[0]+xl[1]); cy = 0.5*(yl[0]+yl[1]); cz = 0.5*(zl[0]+zl[1])
        ax.set_xlim3d(cx-span/2, cx+span/2)
        ax.set_ylim3d(cy-span/2, cy+span/2)
        ax.set_zlim3d(cz-span/2, cz+span/2)

    # ---- public ---------------------------------------------------------
    def is_ready(self):
        return self._layer_data is not None

    # ---- Send to sim / Export / Reset -----------------------------------
    def _compose_inp(self, dest_path, fmin, fmax, resolution_override=None,
                     record_emitted=True):
        topo = self.topology_var.get()
        port_inside = self._port_inside.get()
        # Re-generate layers at possibly different resolution (for QuickSim).
        if resolution_override is not None:
            sp = pc.SpiralParams(
                od_mm=self._last_params.od_mm,
                trace_width_mm=self._last_params.trace_width_mm,
                spacing_mm=self._last_params.spacing_mm,
                turns=self._last_params.turns,
                resolution_mm=resolution_override)
            _, stackup, _ = self._snapshot_inputs()
            layers = pc.active_layer_data(sp, stackup)
        else:
            layers = [dict(ld, nodes=list(ld["nodes"]))
                      for ld in self._layer_data]
            # Undo the visualizer's reverse (including any port_inside flip)
            # so the writer can apply its own flags from scratch.
            if topo != "parallel":
                flags = pc.series_reverse_flags_for_topology(topo, len(layers))
                if port_inside:
                    flags = [not f for f in flags]
                layers = pc.reverse_nodes_for_series_flow(layers, flags)

        pc.write_topology_inp(topo, layers, dest_path,
                              w_mm=self._last_params.trace_width_mm,
                              port_inside=port_inside,
                              fmin=fmin, fmax=fmax)
        if record_emitted and resolution_override is None:
            self._layer_data_emitted = [dict(ld) for ld in layers]
        return topo, layers

    def _on_send_to_sim(self):
        if not self.is_ready():
            messagebox.showwarning("Send to Sim",
                                   "Valid coil required."); return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Send to Sim",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
        dest = os.path.join(self.temp_dir,
                            f"param_{self.role.lower()}.inp")
        try:
            topo, _ = self._compose_inp(dest, fmin, fmax)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Write failed: {e}"); return
        self._my_temp_files.add(dest)

        layer_params = [
            (self._last_params.trace_width_mm, ld["h_mm"], len(ld["nodes"]))
            for ld in self._layer_data]
        meta = {"role": self.role, "topology": topo,
                "layer_params": layer_params,
                "nodes_by_layer": [list(ld["nodes"]) for ld in self._layer_data]}
        self.app.sim_tab.register_coil(
            self.role, "Parametric Generator", dest, meta)
        self._set_sim_status("current")

        # Also push parameters to the Simulation NN tab.
        try:
            nn_tab = getattr(self.app, "sim_nn_tab", None)
            if nn_tab is not None:
                nn_params = {
                    "od":       self._last_params.od_mm,
                    "turns":    self._last_params.turns,
                    "w":        self._last_params.trace_width_mm,
                    "topology": topo,
                }
                nn_tab.receive_parametric_params(self.role, nn_params)
        except Exception:
            pass

    def export_inp(self):
        if not self.is_ready():
            messagebox.showwarning("Export", "Generate a valid coil first.")
            return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Export",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
        dest = filedialog.asksaveasfilename(
            defaultextension=".inp",
            initialfile=f"param_{self.role.lower()}.inp",
            filetypes=[("FastHenry INP", "*.inp")])
        if not dest: return
        try: self._compose_inp(dest, fmin, fmax, record_emitted=False)
        except Exception as e:
            messagebox.showerror("Export", f"Write failed: {e}"); return
        messagebox.showinfo("Export", f"Wrote:\n{dest}")

    def reset_this_tab(self):
        self._suspend_savestate = True
        try:
            self.od_var.set(f"{self.DEFAULT_OD_MM}")
            self.w_var.set(f"{self.DEFAULT_TRACE_W_MM}")
            self.s_var.set(f"{self.DEFAULT_SPACING_MM}")
            self.turns_var.set(f"{self.DEFAULT_TURNS}")
            self.res_var.set(f"{self.DEFAULT_RESOLUTION_MM}")
            self.outer_gap_var.set(f"{self.DEFAULT_OUTER_GAP}")
            self.inner_gap_var.set(f"{self.DEFAULT_INNER_GAP}")
            self.topology_var.set("parallel")
            self._port_inside.set(False)
            self._port_inside_cb.configure(state="disabled")
            self._link_cu.set(True)
            self._true_scale.set(False)
            for i in range(4):
                self.slot_active_vars[i].set(i == 0)
                self.slot_oz_vars[i].set(f"{self.DEFAULT_COPPER_OZ[i]}")
            self._layer_data = None
            self._layer_data_emitted = None
            self._last_params = None
            self._selected_slot_idx = None
            self._quicksim_value = None
            self._quicksim_stale = False
            self._refresh_quicksim_label()
            self.fig.clf(); self.canvas.draw_idle()
            for f in list(self._my_temp_files):
                try:
                    if os.path.exists(f): os.remove(f)
                except Exception: pass
            self._my_temp_files.clear()
            self.app.sim_tab.unregister_coil(self.role, "Parametric Generator")
            self._set_sim_status("nothing")
        finally:
            self._suspend_savestate = False
        self._save_state()
        self._paint_row_bgs()
        self._schedule_refresh()

    # ---- QuickSim -------------------------------------------------------
    def _mark_quicksim_stale(self):
        if self._quicksim_value is not None:
            self._quicksim_stale = True
            self._refresh_quicksim_label()

    def _refresh_quicksim_label(self):
        if self._quicksim_value is None:
            self.quicksim_label.configure(text="~ — µH", fg="gray")
        elif self._quicksim_stale:
            self.quicksim_label.configure(
                text=f"~ {self._quicksim_value:.3f} µH (stale)", fg="#c01010")
        else:
            self.quicksim_label.configure(
                text=f"~ {self._quicksim_value:.3f} µH", fg="#1a8020")

    def _on_quick_sim(self):
        if not _RUNNER_OK:
            messagebox.showerror("Quick Sim",
                                 "FastHenry runner unavailable.")
            return
        if not self.is_ready():
            messagebox.showwarning("Quick Sim",
                                   "Valid coil required."); return
        if self._quicksim_thread and self._quicksim_thread.is_alive():
            return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Quick Sim",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
        # Use a dedicated sub-directory so quickSim's Zc.mat never collides
        # with the sim-tab's Zc.mat (both would otherwise land in temp/).
        qs_dir = os.path.join(self.temp_dir, f"qs_{self.role.lower()}")
        os.makedirs(qs_dir, exist_ok=True)
        dest = os.path.join(qs_dir, f"quicksim_{self.role.lower()}.inp")
        try:
            self._compose_inp(dest, fmin, fmax,
                              resolution_override=self.QUICKSIM_RESOLUTION_MM,
                              record_emitted=False)
        except Exception as e:
            messagebox.showerror("Quick Sim", f"Write failed: {e}"); return
        self._my_temp_files.add(dest)

        self.quicksim_btn.configure(state="disabled", text="Running…")
        self._quicksim_thread = threading.Thread(
            target=self._quicksim_worker,
            args=(dest, fmin + 5000.0),  # pick mid-ish frequency
            daemon=True)
        self._quicksim_thread.start()

    def _quicksim_worker(self, inp_path, target_f):
        try:
            with runner.FastHenryRunner() as fh:
                ok = fh.run(inp_path,
                            timeout_sec=self.QUICKSIM_TIMEOUT_SEC,
                            progress_cb=None)
                if not ok:
                    self._quicksim_queue.put(("error", "Timed out")); return
                zc_path = os.path.join(os.path.dirname(inp_path), "Zc.mat")
                fh.export_zc_mat(zc_path)
                blocks = zc_parser.parse_zc_mat(zc_path)
                if not blocks:
                    self._quicksim_queue.put(("error", "Zc.mat empty")); return
                f_used, Zmat = zc_parser.matrix_at(blocks, target_f)
                L_h = Zmat[0][0].imag / (2.0 * math.pi * f_used)
                self._quicksim_queue.put(("done", L_h * 1e6))
        except Exception as e:
            self._quicksim_queue.put(("error",
                                      f"{type(e).__name__}: {e}"))

    def _poll_quicksim(self):
        try:
            while True:
                kind, payload = self._quicksim_queue.get_nowait()
                self.quicksim_btn.configure(state="normal", text="Quick Sim L")
                if kind == "done":
                    self._quicksim_value = payload
                    self._quicksim_stale = False
                elif kind == "error":
                    self._quicksim_value = None
                    self._quicksim_stale = False
                    messagebox.showerror("Quick Sim", str(payload))
                self._refresh_quicksim_label()
        except queue.Empty:
            pass
        finally:
            self.after(self.DEBOUNCE_MS, self._poll_quicksim)

    # ---- sim-loaded-status helpers --------------------------------------
    def _set_sim_status(self, key):
        text, color = _STATUS_COLORS[key]
        self.sim_status_label.configure(text=text, fg=color)
        self._sim_status_key = key

    def _mark_sim_outdated_if_loaded(self):
        if getattr(self, "_sim_status_key", "nothing") == "current":
            self._set_sim_status("outdated")

    def on_sim_slot_cleared(self):
        """Called by SimTab when our slot is cleared externally."""
        self._set_sim_status("nothing")

    # ---- savestate ------------------------------------------------------
    def _save_state(self):
        if getattr(self, "_suspend_savestate", False): return
        self.app.persist_parametric_tab(self.role, self.to_state())

    def to_state(self):
        return {
            "topology":    self.topology_var.get(),
            "port_inside": self._port_inside.get(),
            "od":          self.od_var.get(),
            "w":           self.w_var.get(),
            "s":           self.s_var.get(),
            "turns":       self.turns_var.get(),
            "res":         self.res_var.get(),
            "outer_gap":   self.outer_gap_var.get(),
            "inner_gap":   self.inner_gap_var.get(),
            "link_cu":     self._link_cu.get(),
            "true_scale":  self._true_scale.get(),
            "active":      [v.get() for v in self.slot_active_vars],
            "oz":          [v.get() for v in self.slot_oz_vars],
        }

    def _restore_from_savestate(self):
        state = self.app.load_parametric_tab_state(self.role)
        if not state: return
        try:
            topo = state.get("topology", "parallel")
            self.topology_var.set(topo)
            port_inside = bool(state.get("port_inside", False))
            self._port_inside.set(port_inside)
            if topo == "parallel":
                self._port_inside_cb.configure(state="disabled")
            else:
                self._port_inside_cb.configure(state="normal")
            self.od_var.set(state.get("od", f"{self.DEFAULT_OD_MM}"))
            self.w_var.set(state.get("w", f"{self.DEFAULT_TRACE_W_MM}"))
            self.s_var.set(state.get("s", f"{self.DEFAULT_SPACING_MM}"))
            self.turns_var.set(state.get("turns", f"{self.DEFAULT_TURNS}"))
            self.res_var.set(state.get("res", f"{self.DEFAULT_RESOLUTION_MM}"))
            self.outer_gap_var.set(state.get("outer_gap", f"{self.DEFAULT_OUTER_GAP}"))
            self.inner_gap_var.set(state.get("inner_gap", f"{self.DEFAULT_INNER_GAP}"))
            self._link_cu.set(bool(state.get("link_cu", True)))
            self._true_scale.set(bool(state.get("true_scale", False)))
            act = state.get("active", [True, False, False, False])
            oz  = state.get("oz", [str(v) for v in self.DEFAULT_COPPER_OZ])
            for i in range(4):
                self.slot_active_vars[i].set(bool(act[i]) if i < len(act) else False)
                if i < len(oz):
                    self.slot_oz_vars[i].set(str(oz[i]))
        except Exception:
            pass