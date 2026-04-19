#!/usr/bin/env python3
"""
Parametric coil GUI tab — ribbon preview + Send/Reset integration.
"""

import os
import sys
import math
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
if _modules not in sys.path:
    sys.path.insert(0, _modules)

import parametric_coil as pc


# ---------------------------------------------------------------------------
# Canvas preview with filled trace ribbons
# ---------------------------------------------------------------------------

def _build_ribbon_world(nodes_2d, trace_width_mm):
    """
    Given centerline (x,y) points, emit the closed polygon of the trace
    ribbon: outer side forward, inner side backward. Perpendiculars are
    computed from the CENTRAL tangent at each interior node and from the
    adjacent segment at each endpoint.
    """
    n = len(nodes_2d)
    if n < 2:
        return []
    half = trace_width_mm / 2.0
    outer, inner = [], []

    for i in range(n):
        if i == 0:
            dx = nodes_2d[1][0] - nodes_2d[0][0]
            dy = nodes_2d[1][1] - nodes_2d[0][1]
        elif i == n - 1:
            dx = nodes_2d[i][0] - nodes_2d[i-1][0]
            dy = nodes_2d[i][1] - nodes_2d[i-1][1]
        else:
            dx = nodes_2d[i+1][0] - nodes_2d[i-1][0]
            dy = nodes_2d[i+1][1] - nodes_2d[i-1][1]
        mag = math.hypot(dx, dy) or 1e-9
        # 90°-CCW perpendicular, unit-scaled
        px, py = -dy / mag, dx / mag
        x, y = nodes_2d[i]
        outer.append((x + half * px, y + half * py))
        inner.append((x - half * px, y - half * py))

    return outer + inner[::-1]


class SpiralPreview(tk.Canvas):
    """
    2D preview drawing each layer as a filled ribbon.

    Stacked layers use a per-layer screen-space shift (not z-scaled)
    because real PCB layer gaps (~0.4 mm) would be invisible alongside
    coil ODs of 10–50 mm.
    """
    LAYER_COLORS  = ["#2080d0", "#d06020", "#30a040", "#a040c0"]
    LAYER_OUTLINE = "#2a2a2a"
    ISO_PER_LAYER = 1.5     # world-unit shift per active-layer index
    MARGIN_PX     = 20

    def __init__(self, parent, **kw):
        super().__init__(parent, bg="white",
                         highlightthickness=1, highlightbackground="#b0b0b0",
                         **kw)
        self._layers = []
        self._trace_width_mm = 0.5
        self.bind("<Configure>", lambda e: self._redraw())

    def set_layers(self, layers, trace_width_mm):
        self._layers = layers
        self._trace_width_mm = trace_width_mm
        self._redraw()

    def clear(self):
        self._layers = []
        self.delete("all")

    def _redraw(self):
        self.delete("all")
        if not self._layers:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2 or h <= 2:
            return

        # Build ribbon polygons per layer in world coordinates.
        polys = []
        for k, ld in enumerate(self._layers):
            shift = self.ISO_PER_LAYER * k
            flat2d = [(p[0] + shift, p[1] + shift) for p in ld["nodes"]]
            polys.append(_build_ribbon_world(flat2d, self._trace_width_mm))

        # BBox across every ribbon vertex.
        all_pts = [p for poly in polys for p in poly]
        if not all_pts:
            return
        xs, ys = zip(*all_pts)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        span_x = max(1e-6, xmax - xmin)
        span_y = max(1e-6, ymax - ymin)
        scale = min((w - 2*self.MARGIN_PX) / span_x,
                    (h - 2*self.MARGIN_PX) / span_y)
        cx_w, cy_w = (xmin + xmax)/2.0, (ymin + ymax)/2.0
        cx_s, cy_s = w/2.0, h/2.0

        def to_screen(p):
            return (cx_s + (p[0] - cx_w) * scale,
                    cy_s - (p[1] - cy_w) * scale)     # y flip

        # Draw bottom-up so upper layers visually sit on top.
        for k, poly in enumerate(polys):
            flat = []
            for p in poly:
                sx, sy = to_screen(p)
                flat.extend([sx, sy])
            color = self.LAYER_COLORS[k % len(self.LAYER_COLORS)]
            self.create_polygon(*flat,
                                fill=color, outline=self.LAYER_OUTLINE,
                                width=0.5)

        # Start/end markers on bottom-layer centerline.
        bottom = self._layers[0]["nodes"]
        if bottom:
            sx, sy = to_screen((bottom[0][0], bottom[0][1]))
            self.create_oval(sx-5, sy-5, sx+5, sy+5,
                             fill="#00a000", outline="white", width=1.5)
            ex, ey = to_screen((bottom[-1][0], bottom[-1][1]))
            self.create_oval(ex-5, ey-5, ex+5, ey+5,
                             fill="#c01010", outline="white", width=1.5)


# ---------------------------------------------------------------------------
# The tab itself
# ---------------------------------------------------------------------------

class ParametricCoilTab(ttk.Frame):
    DEBOUNCE_MS = 150

    DEFAULT_OD_MM           = 30.0
    DEFAULT_TRACE_W_MM      = 0.5
    DEFAULT_SPACING_MM      = 0.16
    DEFAULT_TURNS           = 8
    DEFAULT_RESOLUTION_MM   = 0.5
    DEFAULT_OUTER_SPACING   = 0.4
    DEFAULT_INNER_SPACING   = 0.6
    DEFAULT_COPPER_OZ       = [1.0, 0.5, 0.5, 1.0]

    def __init__(self, parent, app, coil_index=1, temp_dir=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.coil_index = coil_index
        self.temp_dir = temp_dir

        self._layer_data = None
        self._last_params = None
        self._last_stackup = None
        self._debounce_id = None
        self._enabled = True
        self._my_temp_files = set()    # files this tab wrote — cleared on reset

        self._build()
        self.after(200, self._schedule_refresh)

    # -------- UI construction --------
    def _build(self):
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=6, pady=6)

        ctrl = ttk.Frame(main)
        ctrl.pack(side="left", fill="y", padx=(0, 8))
        self._build_controls(ctrl)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)
        self.preview = SpiralPreview(right, width=520, height=520)
        self.preview.pack(fill="both", expand=True)

        bar = ttk.Frame(right); bar.pack(fill="x", pady=(4, 0))
        self.status_var = tk.StringVar(value="")
        self.count_var  = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status_var,
                  foreground="#a03020").pack(side="left")
        ttk.Label(bar, textvariable=self.count_var,
                  foreground="gray").pack(side="right")

    def _build_controls(self, parent):
        spiral = ttk.LabelFrame(parent, text="Spiral")
        spiral.pack(fill="x", pady=(0, 6))
        self.od_var    = self._entry(spiral, "OD (mm):",            self.DEFAULT_OD_MM)
        self.w_var     = self._entry(spiral, "Trace width (mm):",   self.DEFAULT_TRACE_W_MM)
        self.s_var     = self._entry(spiral, "Trace spacing (mm):", self.DEFAULT_SPACING_MM)
        self.turns_var = self._entry(spiral, "Turns:",              self.DEFAULT_TURNS)
        self.res_var   = self._entry(spiral, "Resolution (mm):",    self.DEFAULT_RESOLUTION_MM)

        stack = ttk.LabelFrame(parent, text="Stack-up (up to 4 parallel layers)")
        stack.pack(fill="x", pady=(0, 6))

        self.outer_gap_var = self._entry(stack, "OuterLayerSpacing (mm):",
                                         self.DEFAULT_OUTER_SPACING)
        self.inner_gap_var = self._entry(stack, "InnerLayerSpacing (mm):",
                                         self.DEFAULT_INNER_SPACING)

        self.slot_active_vars = []
        self.slot_oz_vars = []
        for i in range(4):
            row = ttk.Frame(stack); row.pack(fill="x", padx=4, pady=1)
            av = tk.BooleanVar(value=(i == 0))
            ttk.Checkbutton(row, text=f"Slot {i+1}", variable=av,
                            command=self._schedule_refresh,
                            width=8).pack(side="left")
            ttk.Label(row, text="Cu oz:").pack(side="left")
            ozv = tk.StringVar(value=f"{self.DEFAULT_COPPER_OZ[i]}")
            ttk.Entry(row, textvariable=ozv, width=6).pack(side="left", padx=2)
            ozv.trace_add("write", lambda *_a: self._schedule_refresh())
            self.slot_active_vars.append(av)
            self.slot_oz_vars.append(ozv)

        act = ttk.LabelFrame(parent, text="Actions")
        act.pack(fill="x", pady=(6, 0))
        ttk.Button(act, text="Send to Simulation", width=22,
                   command=self._on_send_to_sim).pack(fill="x", padx=4, pady=2)
        ttk.Button(act, text="Export INP…", width=22,
                   command=self.export_inp).pack(fill="x", padx=4, pady=2)
        ttk.Button(act, text="Reset This Tab", width=22,
                   command=self.reset_this_tab).pack(fill="x", padx=4, pady=2)

        self.send_status_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.send_status_var,
                  foreground="#206020",
                  wraplength=230).pack(fill="x", padx=4, pady=(4, 0))

    def _entry(self, parent, label, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        var = tk.StringVar(value=f"{default}")
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        var.trace_add("write", lambda *_a: self._schedule_refresh())
        return var

    # -------- Enable/disable (greying by paired DXF tab) --------
    def set_enabled(self, enabled):
        self._enabled = enabled
        state = "normal" if enabled else "disabled"
        self._walk_state(self, state)
        if not enabled:
            self.preview.clear()
            self.status_var.set("DXF loaded on paired tab — "
                                "parametric input ignored")
            self.count_var.set("")
            self.send_status_var.set("")
        else:
            self.status_var.set("")
            self._schedule_refresh()

    def _walk_state(self, widget, state):
        try:
            widget.configure(state=state)
        except (tk.TclError, Exception):
            pass
        for child in widget.winfo_children():
            self._walk_state(child, state)

    # -------- Debounced refresh --------
    def _schedule_refresh(self, *_):
        if not self._enabled:
            return
        if self._debounce_id is not None:
            try:
                self.after_cancel(self._debounce_id)
            except tk.TclError:
                pass
        self._debounce_id = self.after(self.DEBOUNCE_MS, self._refresh_now)

    def _refresh_now(self):
        self._debounce_id = None
        if not self._enabled:
            return
        params, stackup, err = self._snapshot_inputs()
        if err is not None:
            self._show_fail(err); return
        ok, msg = pc.validate_spiral(params)
        if not ok:
            self._show_fail(msg); return
        ok_s, msg_s = pc.validate_stackup(stackup, min_active=1, max_active=4)
        if not ok_s:
            self._show_fail(msg_s); return

        try:
            layers = pc.active_layer_data(params, stackup)
        except Exception as e:
            self._show_fail(f"Generation error: {e}"); return

        self._layer_data   = layers
        self._last_params  = params
        self._last_stackup = stackup
        self.status_var.set("")
        n_nodes = sum(len(ld["nodes"]) for ld in layers)
        self.count_var.set(
            f"{len(layers)} layer(s), {n_nodes} nodes, "
            f"inner R = {params.r_inner_centerline:.2f} mm")
        self.preview.set_layers(layers, params.trace_width_mm)

    def _show_fail(self, msg):
        self.status_var.set(msg); self.count_var.set("")
        self.preview.clear(); self._layer_data = None

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
            stackup = pc.StackUp(
                slots=slots,
                outer_gap_mm=float(self.outer_gap_var.get()),
                inner_gap_mm=float(self.inner_gap_var.get()))
        except ValueError:
            return None, None, "Enter numeric stack-up parameters."
        return params, stackup, None

    # -------- Public API --------
    def is_ready(self):
        return self._layer_data is not None

    # -------- Send / Reset --------
    def _on_send_to_sim(self):
        if not self.is_ready():
            messagebox.showwarning("Send to Sim",
                                   "Valid coil required before sending.")
            return
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Send to Sim",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0

        dest = os.path.join(self.temp_dir,
                            f"param_coil{self.coil_index}.inp")
        try:
            if len(self._layer_data) == 1:
                ld = self._layer_data[0]
                pc.write_single_layer_inp(
                    ld["nodes"], dest,
                    w_mm=self._last_params.trace_width_mm,
                    h_mm=ld["h_mm"],
                    fmin=fmin, fmax=fmax)
                topology = "single"
            else:
                pc.write_parallel_multilayer_inp(
                    self._layer_data, dest,
                    w_mm=self._last_params.trace_width_mm,
                    fmin=fmin, fmax=fmax)
                topology = "parallel"
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Write failed: {e}")
            return
        self._my_temp_files.add(dest)

        layer_params = [
            (self._last_params.trace_width_mm, ld["h_mm"], len(ld["nodes"]))
            for ld in self._layer_data]

        meta = {
            "topology":       topology,
            "layer_params":   layer_params,
            "nodes_by_layer": [list(ld["nodes"]) for ld in self._layer_data],
        }
        self.app.sim_tab.register_coil(
            self.coil_index, "Parametric Generator", dest, meta)
        self.send_status_var.set(
            f"Sent as Coil {self.coil_index}.")

    def reset_this_tab(self):
        self.od_var.set(f"{self.DEFAULT_OD_MM}")
        self.w_var.set(f"{self.DEFAULT_TRACE_W_MM}")
        self.s_var.set(f"{self.DEFAULT_SPACING_MM}")
        self.turns_var.set(f"{self.DEFAULT_TURNS}")
        self.res_var.set(f"{self.DEFAULT_RESOLUTION_MM}")
        self.outer_gap_var.set(f"{self.DEFAULT_OUTER_SPACING}")
        self.inner_gap_var.set(f"{self.DEFAULT_INNER_SPACING}")
        for i in range(4):
            self.slot_active_vars[i].set(i == 0)
            self.slot_oz_vars[i].set(f"{self.DEFAULT_COPPER_OZ[i]}")

        self._layer_data = None
        self._last_params = None
        self._last_stackup = None
        self.preview.clear()
        self.send_status_var.set("")

        # Remove only the files this tab authored.
        for f in list(self._my_temp_files):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self._my_temp_files.clear()

        # Vacate our slot on the sim tab only if WE were the source.
        self.app.sim_tab.unregister_coil(
            self.coil_index, "Parametric Generator")
        self._schedule_refresh()

    # -------- Export-only (manual file write) --------
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
            initialfile=f"parametric_coil{self.coil_index}.inp",
            filetypes=[("FastHenry INP", "*.inp")])
        if not dest:
            return
        try:
            if len(self._layer_data) == 1:
                ld = self._layer_data[0]
                pc.write_single_layer_inp(
                    ld["nodes"], dest,
                    w_mm=self._last_params.trace_width_mm,
                    h_mm=ld["h_mm"],
                    fmin=fmin, fmax=fmax)
            else:
                pc.write_parallel_multilayer_inp(
                    self._layer_data, dest,
                    w_mm=self._last_params.trace_width_mm,
                    fmin=fmin, fmax=fmax)
        except Exception as e:
            messagebox.showerror("Export", f"Write failed: {e}")
            return
        messagebox.showinfo("Export", f"Wrote:\n{dest}")