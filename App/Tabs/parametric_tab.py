#!/usr/bin/env python3
"""
Parametric coil tab (TX / RX).

TX: layers 1+2 only, series topology, port on outside (all hardcoded).
    Optional independent L2 turns (auto-computed L2 trace width).
RX: all 4 layers in series, port on inside, copper [1,0.5,0.5,1] (hardcoded).
    Optional independent inner-layer turns (slots 2&3 share auto-computed
    endpoint-matched trace width; slots 1&4 use the outer turns/width).
"""

import os, sys, math, threading, queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

# Layer 1=red, 2=green, 3=orange, 4=blue.
LAYER_COLORS = ["#e63434", "#18d935", "#ce6c33", "#2080d0"]

ROW_BG_EMPTY    = "#f0f0f0"
ROW_BG_FILLED   = "#dfeeff"
ROW_BG_SELECTED = "#ffd89a"

_STATUS_COLORS = {
    "nothing":  ("Nothing loaded in Sim",        "#c01010"),
    "current":  ("Current Coil loaded in Sim",   "#1a8020"),
    "outdated": ("Outdated Coil loaded in Sim",  "#c06010"),
}

Z_EXAGGERATION = 12.0

# Hardcoded per role.
_TX_TOPOLOGY    = "series"
_TX_PORT_INSIDE = False
_TX_ACTIVE_SLOTS = [True, True, False, False]

_RX_TOPOLOGY    = "series"
_RX_PORT_INSIDE = True
_RX_ACTIVE_SLOTS = [True, True, True, True]
_RX_COPPER_OZ   = [1.0, 0.5, 0.5, 1.0]


class ParametricCoilTab(ttk.Frame):
    DEBOUNCE_MS = 150

    DEFAULT_OD_MM         = 52.0
    DEFAULT_TRACE_W_MM    = 0.4
    DEFAULT_SPACING_MM    = 0.16
    DEFAULT_TURNS         = 11
    DEFAULT_RESOLUTION_MM = 0.6
    DEFAULT_LAYER_GAP     = 0.2104  # TX: L1-L2 prepreg gap; RX: outer prepreg gap

    QUICKSIM_RESOLUTION_MM = 1.2
    QUICKSIM_TIMEOUT_SEC   = 120

    def __init__(self, parent, app, role, coil_index, temp_dir,
                 on_next_tab=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.role = role          # "TX" or "RX"
        self.coil_index = coil_index
        self.temp_dir = temp_dir
        self._on_next_tab = on_next_tab

        self._layer_data = None
        self._layer_data_emitted = None
        self._last_params = None
        self._selected_slot_idx = None
        self._debounce_id = None
        self._enabled = True
        self._my_temp_files = set()

        self._true_scale = tk.BooleanVar(value=False)
        # TX-only: independent layer 2 turns.
        self._tx_indep_l2 = tk.BooleanVar(value=False)
        self._tx_l2_turns_var = tk.StringVar(value="11")
        # RX-only: independent inner-layer turns (slots 2&3).
        self._rx_indep_inner = tk.BooleanVar(value=False)
        self._rx_inner_turns_var = tk.StringVar(value="11")

        self._quicksim_value = None
        self._quicksim_stale = False
        self._quicksim_thread = None
        self._quicksim_queue = queue.Queue()

        self._build()
        self.after(100, self._post_build_init)

    # ---- role helpers -------------------------------------------------------

    @property
    def _is_tx(self):
        return self.role == "TX"

    def _topology(self):
        return _TX_TOPOLOGY if self._is_tx else _RX_TOPOLOGY

    def _port_inside(self):
        return _TX_PORT_INSIDE if self._is_tx else _RX_PORT_INSIDE

    def _active_slots(self):
        return _TX_ACTIVE_SLOTS if self._is_tx else _RX_ACTIVE_SLOTS

    def _copper_oz_list(self):
        if self._is_tx:
            vals = [1.0, 0.5]
            for i, v in enumerate(self.tx_oz_vars):
                try: vals[i] = float(v.get())
                except Exception: pass
            return [vals[0], vals[1], 0.5, 1.0]   # slots 3,4 inactive for TX
        return _RX_COPPER_OZ

    # ---- post-init ----------------------------------------------------------

    def _post_build_init(self):
        self._suspend_savestate = True
        try:
            self._restore_from_savestate()
        finally:
            self._suspend_savestate = False
        self._schedule_refresh()
        self.after(self.DEBOUNCE_MS + 50, self._poll_quicksim)

    # ---- UI build -----------------------------------------------------------

    def _build(self):
        main = ttk.Frame(self); main.pack(fill="both", expand=True,
                                          padx=6, pady=6)
        ctrl = ttk.Frame(main); ctrl.pack(side="left", fill="y", padx=(0, 8))
        self._build_controls(ctrl)
        right = ttk.Frame(main); right.pack(side="left", fill="both", expand=True)
        self._build_viewer(right)

    def _build_controls(self, parent):
        # Spiral settings.
        sp = ttk.LabelFrame(parent, text="Spiral (applies to all active layers)")
        sp.pack(fill="x", pady=(0, 6))
        self.od_var    = self._entry(sp, "OD (mm):",            self.DEFAULT_OD_MM)
        self.w_var     = self._entry(sp, "Trace width (mm):",   self.DEFAULT_TRACE_W_MM)
        self.s_var     = self._entry(sp, "Trace spacing (mm):", self.DEFAULT_SPACING_MM)
        self.turns_var = self._entry(sp, "Turns/layer:",        self.DEFAULT_TURNS)
        self.res_var   = self._entry(sp, "Resolution (mm):",    self.DEFAULT_RESOLUTION_MM)

        # RX is fixed: series across all 4 layers, port inside.

        # Stack-up gap (role-specific label/count).
        zf = ttk.LabelFrame(parent, text="Stack-up gap")
        zf.pack(fill="x", pady=(0, 6))
        if self._is_tx:
            self.layer_gap_var = self._entry(zf, "Layer 1 - 2 gap (mm):", self.DEFAULT_LAYER_GAP)
        else:
            self.layer_gap_var = self._entry(zf, "Outer L. gap (mm):", self.DEFAULT_LAYER_GAP)
            self.inner_gap_var = self._entry(zf, "Inner L. gap (mm):", 0.6)

        # Layer info / copper weight rows.
        lf = ttk.LabelFrame(parent, text="Layers")
        lf.pack(fill="x", pady=(0, 6))
        if self._is_tx:
            ttk.Label(lf, text="Series  |  Port outside  (fixed)",
                      foreground="#606060").pack(anchor="w", padx=6, pady=(4, 2))
        else:
            ttk.Label(lf, text="All 4 layers  |  Port inside  (fixed)",
                      foreground="#606060").pack(anchor="w", padx=6, pady=2)
            ttk.Label(lf, text="Cu oz: 1.0 / 0.5 / 0.5 / 1.0  (fixed)",
                      foreground="#606060").pack(anchor="w", padx=6, pady=(0, 4))

        # TX: per-layer rows with Cu oz entry. RX: label-only rows.
        self.slot_row_frames = []
        self.tx_oz_vars = []   # only populated for TX
        active = self._active_slots()
        tx_oz_defaults = [1.0, 0.5]
        for i in range(2 if self._is_tx else 4):
            row = tk.Frame(lf, bg=ROW_BG_FILLED if active[i] else ROW_BG_EMPTY,
                           bd=1, relief="solid")
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=f"Layer {i+1}",
                     bg=ROW_BG_FILLED if active[i] else ROW_BG_EMPTY
                     ).pack(side="left", padx=4, pady=2)
            if self._is_tx:
                tk.Label(row, text="Cu oz:", bg=ROW_BG_FILLED
                         ).pack(side="left", padx=(6, 0))
                ozv = tk.StringVar(value=f"{tx_oz_defaults[i]}")
                ttk.Entry(row, textvariable=ozv, width=6
                          ).pack(side="left", padx=2, pady=2)
                ozv.trace_add("write", lambda *_a: self._schedule_refresh())
                self.tx_oz_vars.append(ozv)
            row.bind("<Button-1>", lambda e, idx=i: self._focus_slot(idx))
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.bind("<Button-1>",
                               lambda e, idx=i: self._focus_slot(idx))
            self.slot_row_frames.append(row)

        # TX-only: independent layer 2 turns.
        if self._is_tx:
            l2f = ttk.LabelFrame(parent, text="Layer 2 (independent)")
            l2f.pack(fill="x", pady=(0, 6))
            ttk.Checkbutton(l2f, text="Independent L2 turns",
                            variable=self._tx_indep_l2,
                            command=self._on_tx_indep_l2_toggle
                            ).pack(anchor="w", padx=6, pady=(4, 2))
            self._l2_controls_frame = ttk.Frame(l2f)
            self._l2_controls_frame.pack(fill="x")
            row = ttk.Frame(self._l2_controls_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text="L2 turns:", width=22, anchor="w").pack(side="left")
            ttk.Entry(row, textvariable=self._tx_l2_turns_var, width=10
                      ).pack(side="left")
            self._tx_l2_turns_var.trace_add("write", lambda *_: self._on_l2_turns_changed())
            self._tx_l2_w_label = ttk.Label(self._l2_controls_frame,
                                             text="L2 width: —",
                                             foreground="#404080")
            self._tx_l2_w_label.pack(anchor="w", padx=6, pady=(0, 4))
            self._update_l2_controls_state()

        # RX-only: independent inner-layer turns (slots 2 & 3).
        if not self._is_tx:
            inf = ttk.LabelFrame(parent, text="Inner layers (independent)")
            inf.pack(fill="x", pady=(0, 6))
            ttk.Checkbutton(inf, text="Independent inner turns (L2 & L3)",
                            variable=self._rx_indep_inner,
                            command=self._on_rx_indep_inner_toggle
                            ).pack(anchor="w", padx=6, pady=(4, 2))
            self._rx_inner_controls_frame = ttk.Frame(inf)
            self._rx_inner_controls_frame.pack(fill="x")
            row = ttk.Frame(self._rx_inner_controls_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text="Inner turns:", width=22, anchor="w").pack(side="left")
            ttk.Entry(row, textvariable=self._rx_inner_turns_var, width=10
                      ).pack(side="left")
            self._rx_inner_turns_var.trace_add(
                "write", lambda *_: self._on_rx_inner_turns_changed())
            self._rx_inner_w_label = ttk.Label(self._rx_inner_controls_frame,
                                                text="Inner width: —",
                                                foreground="#404080")
            self._rx_inner_w_label.pack(anchor="w", padx=6, pady=(0, 4))
            self._update_rx_inner_controls_state()

        # View mode.
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
                                       command=self._on_quick_sim, width=14)
        self.quicksim_btn.pack(side="left", padx=4)
        self.quicksim_label = tk.Label(qs, text="~ — µH", fg="gray")
        self.quicksim_label.pack(side="left", padx=4)

        # Send to Sim + status + Next Tab.
        ttk.Button(parent, text=f"Send to Simulation ({self.role})",
                   width=28, command=self._on_send_to_sim).pack(fill="x", padx=4, pady=2)
        self.sim_status_label = tk.Label(parent, text="",
                                          fg=_STATUS_COLORS["nothing"][1],
                                          anchor="w")
        self.sim_status_label.pack(fill="x", padx=4, pady=(0, 2))
        self._set_sim_status("nothing")

        ttk.Button(parent, text="Next Tab →", width=28,
                   command=self._on_next_tab_click
                   ).pack(fill="x", padx=4, pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=4)

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

    # ---- enable/disable -----------------------------------------------------

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

    # ---- user events --------------------------------------------------------

    def _on_l2_turns_changed(self):
        try:
            n2 = int(self._tx_l2_turns_var.get())
            n1 = int(float(self.turns_var.get()))
            if n2 > n1:
                self._tx_l2_turns_var.set(str(n1))
                return   # setting the var triggers this callback again
        except ValueError:
            pass
        self._schedule_refresh()

    def _on_tx_indep_l2_toggle(self):
        self._update_l2_controls_state()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()
        self._save_state()

    def _update_l2_controls_state(self):
        if not self._is_tx:
            return
        enabled = self._tx_indep_l2.get()
        state = "normal" if enabled else "disabled"
        self._walk_state(self._l2_controls_frame, state)
        if not enabled:
            self._tx_l2_w_label.configure(text="L2 width: —")

    def _on_rx_inner_turns_changed(self):
        try:
            n_inner = int(self._rx_inner_turns_var.get())
            n_outer = int(float(self.turns_var.get()))
            if n_inner > n_outer:
                self._rx_inner_turns_var.set(str(n_outer))
                return   # setting the var triggers this callback again
        except ValueError:
            pass
        self._schedule_refresh()

    def _on_rx_indep_inner_toggle(self):
        self._update_rx_inner_controls_state()
        self._mark_sim_outdated_if_loaded()
        self._schedule_refresh()
        self._save_state()

    def _update_rx_inner_controls_state(self):
        if self._is_tx:
            return
        enabled = self._rx_indep_inner.get()
        state = "normal" if enabled else "disabled"
        self._walk_state(self._rx_inner_controls_frame, state)
        if not enabled:
            self._rx_inner_w_label.configure(text="Inner width: —")

    def _on_true_scale_toggle(self):
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
        active = self._active_slots()
        for i, row in enumerate(self.slot_row_frames):
            if self._selected_slot_idx == i:
                bg = ROW_BG_SELECTED
            elif active[i]:
                bg = ROW_BG_FILLED
            else:
                bg = ROW_BG_EMPTY
            row.configure(bg=bg)
            for child in row.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg=bg)

    # ---- refresh / validation -----------------------------------------------

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

        # TX with independent L2 turns / RX with independent inner turns
        # both take separate code paths.
        if self._is_tx and self._tx_indep_l2.get():
            try:
                n2 = int(self._tx_l2_turns_var.get())
                n1 = int(params.turns)
                if n2 < 1 or n2 > n1:
                    raise ValueError(f"must be 1 – {n1}")
            except ValueError as e:
                self._show_fail(f"L2 turns: {e}"); return
            try:
                layers, w2 = pc.active_layer_data_tx_independent(
                    params, stackup, n2)
                self._tx_l2_w_label.configure(
                    text=f"L2 width: {w2:.4f} mm")
            except Exception as e:
                self._show_fail(f"Layer 2 error: {e}"); return
        elif (not self._is_tx) and self._rx_indep_inner.get():
            try:
                n_inner = int(self._rx_inner_turns_var.get())
                n_outer = int(params.turns)
                if n_inner < 1 or n_inner > n_outer:
                    raise ValueError(f"must be 1 – {n_outer}")
            except ValueError as e:
                self._show_fail(f"Inner turns: {e}"); return
            try:
                layers, w_inner = pc.active_layer_data_rx_independent(
                    params, stackup, n_inner)
                self._rx_inner_w_label.configure(
                    text=f"Inner width: {w_inner:.4f} mm")
            except Exception as e:
                self._show_fail(f"Inner layer error: {e}"); return
        else:
            if self._is_tx and hasattr(self, "_tx_l2_w_label"):
                self._tx_l2_w_label.configure(text="L2 width: —")
            if (not self._is_tx) and hasattr(self, "_rx_inner_w_label"):
                self._rx_inner_w_label.configure(text="Inner width: —")
            try:
                layers = pc.active_layer_data(params, stackup)
            except Exception as e:
                self._show_fail(f"Generation error: {e}"); return

        topo = self._topology()
        flags = pc.series_reverse_flags_for_topology(topo, len(layers))
        if self._port_inside() and topo != "parallel":
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
        topo = self._topology()
        n_nodes = sum(len(ld["nodes"]) for ld in self._layer_data)
        n_act = len(self._layer_data)
        if self._is_tx and self._tx_indep_l2.get() and n_act == 2:
            try:
                n2 = float(self._tx_l2_turns_var.get())
            except ValueError:
                n2 = 0.0
            eff_t = self._last_params.turns + n2
            self.info_var.set(
                f"{n_act} active | series | "
                f"L1={self._last_params.turns:g}T L2={n2:g}T | "
                f"eff. turns={eff_t:g} | {n_nodes} nodes")
        elif (not self._is_tx) and self._rx_indep_inner.get() and n_act == 4:
            try:
                n_inner = float(self._rx_inner_turns_var.get())
            except ValueError:
                n_inner = 0.0
            n_outer = self._last_params.turns
            eff_t = 2.0 * n_outer + 2.0 * n_inner
            self.info_var.set(
                f"{n_act} active | series | "
                f"outer={n_outer:g}T inner={n_inner:g}T | "
                f"eff. turns={eff_t:g} | {n_nodes} nodes")
        else:
            t_per = self._last_params.turns
            eff_m = 2 if topo == "parallel_pairs_ser" else n_act
            eff_t = t_per * eff_m
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
            outer_gap = float(self.layer_gap_var.get())
            if self._is_tx:
                inner_gap = 0.0
            else:
                inner_gap = float(self.inner_gap_var.get())

            oz_list = self._copper_oz_list()
            active = self._active_slots()
            slots = [pc.LayerSlot(active=active[i], copper_oz=oz_list[i])
                     for i in range(4)]
            stackup = pc.StackUp(slots=slots,
                                 outer_gap_mm=outer_gap,
                                 inner_gap_mm=inner_gap)
        except ValueError:
            return None, None, "Enter numeric stack-up parameters."
        return params, stackup, None

    # ---- drawing ------------------------------------------------------------

    def _build_ribbon_world(self, nodes, trace_w):
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
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        w = ld.get("w_mm", self._last_params.trace_width_mm)
        poly = self._build_ribbon_world(ld["nodes"], w)
        color = LAYER_COLORS[(ld["slot"] - 1) % len(LAYER_COLORS)]
        if poly:
            xs, ys = zip(*poly)
            ax.fill(xs, ys, color=color, alpha=0.85,
                    edgecolor="#202020", linewidth=0.5)
        n = ld["nodes"]
        ax.plot(n[0][0],  n[0][1],  "go", markersize=8, label="Entry")
        ax.plot(n[-1][0], n[-1][1], "rs", markersize=8, label="Exit")
        ax.set_aspect("equal")
        ax.set_title(f"Layer {ld['slot']}  (z = {ld['z']:.3f} mm, "
                     f"h = {ld['h_mm']:.3f} mm, w = {w:.4f} mm)")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_stack_3d(self):
        if not self._layer_data: return
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection="3d")

        true_scale = self._true_scale.get()
        z_scale = 1.0 if true_scale else Z_EXAGGERATION

        for ld in self._layer_data:
            xs = [p[0] for p in ld["nodes"]]
            ys = [p[1] for p in ld["nodes"]]
            zs = [-p[2] * z_scale for p in ld["nodes"]]
            color = LAYER_COLORS[(ld["slot"] - 1) % len(LAYER_COLORS)]
            ax.plot(xs, ys, zs, color=color, linewidth=1.1,
                    label=f"Layer {ld['slot']}")
            ax.scatter([xs[0]], [ys[0]], [zs[0]],
                       color="#00a000", s=20, edgecolors="black", linewidth=0.5)
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                       color="#c01010", s=20, edgecolors="black", linewidth=0.5)

        topo = self._topology()
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
        title = f"{self.role} stack — {self._topology()}"
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

    # ---- public -------------------------------------------------------------

    def is_ready(self):
        return self._layer_data is not None

    # ---- Send to sim / Export / Reset ---------------------------------------

    def _compose_inp(self, dest_path, fmin, fmax, resolution_override=None,
                     record_emitted=True):
        topo = self._topology()
        port_inside = self._port_inside()
        tx_indep = self._is_tx and self._tx_indep_l2.get()
        rx_indep = (not self._is_tx) and self._rx_indep_inner.get()

        if resolution_override is not None:
            _, stackup, _ = self._snapshot_inputs()
            sp = pc.SpiralParams(
                od_mm=self._last_params.od_mm,
                trace_width_mm=self._last_params.trace_width_mm,
                spacing_mm=self._last_params.spacing_mm,
                turns=self._last_params.turns,
                resolution_mm=resolution_override)
            if tx_indep:
                n2 = int(self._tx_l2_turns_var.get())
                layers, _ = pc.active_layer_data_tx_independent(sp, stackup, n2)
            elif rx_indep:
                n_inner = int(self._rx_inner_turns_var.get())
                layers, _ = pc.active_layer_data_rx_independent(sp, stackup, n_inner)
            else:
                layers = pc.active_layer_data(sp, stackup)
        else:
            layers = [dict(ld, nodes=list(ld["nodes"]))
                      for ld in self._layer_data]
            # Undo visualizer's reversal so writer can apply from scratch.
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
            messagebox.showwarning("Send to Sim", "Valid coil required."); return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Send to Sim",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
        dest = os.path.join(self.temp_dir, f"param_{self.role.lower()}.inp")
        try:
            topo, _ = self._compose_inp(dest, fmin, fmax)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Write failed: {e}"); return
        self._my_temp_files.add(dest)

        layer_params = [
            (ld.get("w_mm", self._last_params.trace_width_mm), ld["h_mm"], len(ld["nodes"]))
            for ld in self._layer_data]
        meta = {"role": self.role, "topology": topo,
                "layer_params": layer_params,
                "nodes_by_layer": [list(ld["nodes"]) for ld in self._layer_data]}
        self.app.sim_tab.register_coil(
            self.role, "Parametric Generator", dest, meta)
        self._set_sim_status("current")

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
            self.layer_gap_var.set(f"{self.DEFAULT_LAYER_GAP}")
            if self._is_tx:
                for i, v in enumerate(self.tx_oz_vars):
                    v.set("1.0" if i == 0 else "0.5")
                self._tx_indep_l2.set(False)
                self._tx_l2_turns_var.set("11")
                self._update_l2_controls_state()
            else:
                self.inner_gap_var.set("0.6")
                self._rx_indep_inner.set(False)
                self._rx_inner_turns_var.set("11")
                self._update_rx_inner_controls_state()
            self._true_scale.set(False)
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

    # ---- QuickSim -----------------------------------------------------------

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
            messagebox.showerror("Quick Sim", "FastHenry runner unavailable.")
            return
        if not self.is_ready():
            messagebox.showwarning("Quick Sim", "Valid coil required."); return
        if self._quicksim_thread and self._quicksim_thread.is_alive():
            return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Quick Sim",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
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
            args=(dest, fmin + 5000.0),
            daemon=True)
        self._quicksim_thread.start()

    def _quicksim_worker(self, inp_path, target_f):
        try:
            with runner.FastHenryRunner() as fh:
                ok = fh.run(inp_path, timeout_sec=self.QUICKSIM_TIMEOUT_SEC,
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
            self._quicksim_queue.put(("error", f"{type(e).__name__}: {e}"))

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

    # ---- sim-loaded-status helpers ------------------------------------------

    def _set_sim_status(self, key):
        text, color = _STATUS_COLORS[key]
        self.sim_status_label.configure(text=text, fg=color)
        self._sim_status_key = key

    def _mark_sim_outdated_if_loaded(self):
        if getattr(self, "_sim_status_key", "nothing") == "current":
            self._set_sim_status("outdated")

    def on_sim_slot_cleared(self):
        self._set_sim_status("nothing")

    # ---- savestate ----------------------------------------------------------

    def _save_state(self):
        if getattr(self, "_suspend_savestate", False): return
        self.app.persist_parametric_tab(self.role, self.to_state())

    def to_state(self):
        state = {
            "od":         self.od_var.get(),
            "w":          self.w_var.get(),
            "s":          self.s_var.get(),
            "turns":      self.turns_var.get(),
            "res":        self.res_var.get(),
            "layer_gap":  self.layer_gap_var.get(),
            "true_scale": self._true_scale.get(),
        }
        if self._is_tx:
            state["tx_oz"] = [v.get() for v in self.tx_oz_vars]
            state["tx_indep_l2"] = self._tx_indep_l2.get()
            state["tx_l2_turns"] = self._tx_l2_turns_var.get()
        else:
            state["inner_gap"] = self.inner_gap_var.get()
            state["rx_indep_inner"] = self._rx_indep_inner.get()
            state["rx_inner_turns"] = self._rx_inner_turns_var.get()
        return state

    def _restore_from_savestate(self):
        state = self.app.load_parametric_tab_state(self.role)
        if not state: return
        try:
            self.od_var.set(state.get("od", f"{self.DEFAULT_OD_MM}"))
            self.w_var.set(state.get("w", f"{self.DEFAULT_TRACE_W_MM}"))
            self.s_var.set(state.get("s", f"{self.DEFAULT_SPACING_MM}"))
            self.turns_var.set(state.get("turns", f"{self.DEFAULT_TURNS}"))
            self.res_var.set(state.get("res", f"{self.DEFAULT_RESOLUTION_MM}"))
            self.layer_gap_var.set(state.get("layer_gap", f"{self.DEFAULT_LAYER_GAP}"))
            self._true_scale.set(bool(state.get("true_scale", False)))
            if self._is_tx:
                saved_oz = state.get("tx_oz", ["1.0", "0.5"])
                defaults = ["1.0", "0.5"]
                for i, v in enumerate(self.tx_oz_vars):
                    v.set(saved_oz[i] if i < len(saved_oz) else defaults[i])
                self._tx_indep_l2.set(bool(state.get("tx_indep_l2", False)))
                self._tx_l2_turns_var.set(state.get("tx_l2_turns", "11"))
                self._update_l2_controls_state()
            else:
                self.inner_gap_var.set(state.get("inner_gap", "0.6"))
                self._rx_indep_inner.set(bool(state.get("rx_indep_inner", False)))
                self._rx_inner_turns_var.set(state.get("rx_inner_turns", "11"))
                self._update_rx_inner_controls_state()
        except Exception:
            pass
