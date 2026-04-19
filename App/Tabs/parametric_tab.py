#!/usr/bin/env python3
"""
Parametric coil GUI tab — ribbon preview + Send/Reset integration.

Updates in this revision:
- Persistent 3-state sim-status line (red "Nothing" / green "Current" /
  orange "Outdated") replacing the transient green "Sent" text.
- "Next Tab" button advances the notebook through the app.
- Stackup preview now uses SLOT index for iso-shift and color, so a
  missing slot shows up as a visible gap. Slot 1 is the top of the
  visual stack (largest shift, red); slot 4 is the bottom (zero shift,
  blue).
- "LayerLinked Copper weight" checkbox: when on, slot 1 ↔ slot 4 and
  slot 2 ↔ slot 3 copper-oz values mirror each other.
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Make Modules/ and this Tabs/ dir importable when loaded standalone.
_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
for _p in (_modules, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parametric_coil as pc


MAX_SLOTS = 4


# ---------------------------------------------------------------------------
# Canvas preview with filled trace ribbons
# ---------------------------------------------------------------------------

def _build_ribbon_world(nodes_2d, trace_width_mm):
    """
    Given centerline (x,y) points, emit the closed polygon of the trace
    ribbon: outer side forward, inner side backward.
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
        px, py = -dy / mag, dx / mag        # 90° CCW unit normal
        x, y = nodes_2d[i]
        outer.append((x + half * px, y + half * py))
        inner.append((x - half * px, y - half * py))
    return outer + inner[::-1]


class SpiralPreview(tk.Canvas):
    """
    2D preview: each layer = filled ribbon, iso-offset by SLOT index so
    disabled slots leave visible gaps in the stack.

    Visual convention (matches the tab's colour legend):
        Slot 1 = visual top (largest iso shift)  — red
        Slot 2                                   — green
        Slot 3                                   — orange
        Slot 4 = visual bottom (zero iso shift)  — blue

    NOTE: this is a display-only convention. Physical z in the .inp still
    follows parametric_coil.layer_z_positions (slot 1 at z=0, slot 4
    highest) — FastHenry sees the stackup that way regardless of how
    we draw it.
    """
    SLOT_COLORS = {
        1: "#d04040",   # red
        2: "#30a040",   # green
        3: "#e68020",   # orange
        4: "#2080d0",   # blue
    }
    LAYER_OUTLINE = "#2a2a2a"
    ISO_PER_LAYER = 1.5     # world-unit iso shift per slot index
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

    @staticmethod
    def _slot_shift(slot):
        """Slot 1 -> (MAX_SLOTS-1) * ISO; slot 4 -> 0."""
        return (MAX_SLOTS - slot) * SpiralPreview.ISO_PER_LAYER

    def _redraw(self):
        self.delete("all")
        if not self._layers:
            return
        w = self.winfo_width(); h = self.winfo_height()
        if w <= 2 or h <= 2:
            return

        # Build polys in draw order (bottom of visual stack first = slot 4).
        polys_with_meta = []
        for ld in sorted(self._layers, key=lambda ld: -ld["slot"]):
            slot = ld["slot"]
            shift = self._slot_shift(slot)
            flat2d = [(p[0] + shift, p[1] + shift) for p in ld["nodes"]]
            poly = _build_ribbon_world(flat2d, self._trace_width_mm)
            polys_with_meta.append((poly, self.SLOT_COLORS.get(slot, "#888"),
                                    slot))

        all_pts = [p for (poly, _, _) in polys_with_meta for p in poly]
        if not all_pts:
            return
        xs, ys = zip(*all_pts)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        span_x = max(1e-6, xmax - xmin)
        span_y = max(1e-6, ymax - ymin)
        scale = min((w - 2 * self.MARGIN_PX) / span_x,
                    (h - 2 * self.MARGIN_PX) / span_y)
        cx_w, cy_w = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        cx_s, cy_s = w / 2.0, h / 2.0

        def to_screen(p):
            return (cx_s + (p[0] - cx_w) * scale,
                    cy_s - (p[1] - cy_w) * scale)

        # Draw ribbons bottom-to-top of the visual stack.
        for poly, color, _slot in polys_with_meta:
            flat = []
            for p in poly:
                sx, sy = to_screen(p)
                flat.extend([sx, sy])
            self.create_polygon(*flat, fill=color,
                                outline=self.LAYER_OUTLINE, width=0.5)

        # Markers go on the lowest-slot active layer (= electrical port
        # layer; same layer FastHenry wires .external to). Drawn LAST so
        # they sit on top of every ribbon.
        terminal_layer = min(self._layers, key=lambda ld: ld["slot"])
        term_shift = self._slot_shift(terminal_layer["slot"])
        if terminal_layer["nodes"]:
            s = terminal_layer["nodes"][0]
            e = terminal_layer["nodes"][-1]
            sx, sy = to_screen((s[0] + term_shift, s[1] + term_shift))
            ex, ey = to_screen((e[0] + term_shift, e[1] + term_shift))
            self.create_oval(sx-5, sy-5, sx+5, sy+5,
                             fill="#00a000", outline="white", width=1.5)
            self.create_oval(ex-5, ey-5, ex+5, ey+5,
                             fill="#c01010", outline="white", width=1.5)


# ---------------------------------------------------------------------------
# The tab itself
# ---------------------------------------------------------------------------

class ParametricCoilTab(ttk.Frame):
    DEBOUNCE_MS = 150

    DEFAULT_OD_MM           = 52.0
    DEFAULT_TRACE_W_MM      = 0.5
    DEFAULT_SPACING_MM      = 0.16
    DEFAULT_TURNS           = 8
    DEFAULT_RESOLUTION_MM   = 0.5
    DEFAULT_OUTER_SPACING   = 0.4
    DEFAULT_INNER_SPACING   = 0.6
    DEFAULT_COPPER_OZ       = [1.0, 0.5, 0.5, 1.0]
    DEFAULT_LINK_COPPER     = True

    # Pair map for linked-copper mode: outer (slot 1 <-> slot 4),
    # inner (slot 2 <-> slot 3). Keys/values are 0-based slot indices.
    _LINK_PAIRS = {0: 3, 3: 0, 1: 2, 2: 1}

    _STATUS_COLORS = {
        "nothing":  ("Nothing loaded in Sim",         "#c01010"),
        "current":  ("Current Coil loaded in Sim",    "#1a8020"),
        "outdated": ("Outdated Coil loaded in Sim",   "#c06010"),
    }

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
        self._my_temp_files = set()

        # Snapshot of inputs captured the last time we successfully sent
        # this coil to Sim. None = nothing currently registered.
        self._sent_snapshot = None

        # Recursion guard for LayerLinked copper-oz mirroring.
        self._linking_in_progress = False

        self._build()
        self.after(200, self._schedule_refresh)
        self._update_sim_status()

    # -------- UI construction --------
    def _build(self):
        main = ttk.Frame(self); main.pack(fill="both", expand=True,
                                          padx=6, pady=6)

        ctrl = ttk.Frame(main); ctrl.pack(side="left", fill="y",
                                          padx=(0, 8))
        self._build_controls(ctrl)

        right = ttk.Frame(main); right.pack(side="left", fill="both",
                                             expand=True)
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

        # Linked-copper checkbox.
        self.link_copper_var = tk.BooleanVar(value=self.DEFAULT_LINK_COPPER)
        ttk.Checkbutton(stack, text="LayerLinked Copper weight",
                        variable=self.link_copper_var,
                        command=self._on_link_toggle).pack(
                            anchor="w", padx=6, pady=(2, 2))

        # Slot rows. Note the copper-oz trace dispatches to _on_oz_change
        # so linked-mode can mirror the paired slot.
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
            ozv.trace_add("write", lambda *_a, idx=i: self._on_oz_change(idx))
            self.slot_active_vars.append(av)
            self.slot_oz_vars.append(ozv)

        # Actions. "Next Tab" sits directly under Send to Simulation.
        act = ttk.LabelFrame(parent, text="Actions")
        act.pack(fill="x", pady=(6, 0))
        ttk.Button(act, text="Send to Simulation", width=22,
                   command=self._on_send_to_sim).pack(fill="x", padx=4, pady=2)
        ttk.Button(act, text="Next Tab", width=22,
                   command=self._on_next_tab).pack(fill="x", padx=4, pady=2)
        ttk.Button(act, text="Export INP…", width=22,
                   command=self.export_inp).pack(fill="x", padx=4, pady=2)
        ttk.Button(act, text="Reset This Tab", width=22,
                   command=self.reset_this_tab).pack(fill="x", padx=4, pady=2)

        # Persistent sim-status line (red/green/orange).
        self.sim_status_var = tk.StringVar(value="")
        self.sim_status_lbl = ttk.Label(parent, textvariable=self.sim_status_var,
                                         wraplength=230)
        self.sim_status_lbl.pack(fill="x", padx=4, pady=(4, 0))

    def _entry(self, parent, label, default):
        row = ttk.Frame(parent); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")
        var = tk.StringVar(value=f"{default}")
        ttk.Entry(row, textvariable=var, width=10).pack(side="left")
        var.trace_add("write", lambda *_a: self._schedule_refresh())
        return var

    # -------- LayerLinked copper handling --------
    def _on_oz_change(self, idx):
        """
        Copper-oz entry changed. Always triggers preview refresh; if
        link-mode is on, mirror the value to the paired slot (with a
        recursion guard so the mirror-set doesn't loop back here).
        """
        self._schedule_refresh()
        if not self.link_copper_var.get() or self._linking_in_progress:
            return
        target = self._LINK_PAIRS.get(idx)
        if target is None:
            return
        self._linking_in_progress = True
        try:
            self.slot_oz_vars[target].set(self.slot_oz_vars[idx].get())
        finally:
            self._linking_in_progress = False

    def _on_link_toggle(self):
        """
        Checkbox toggled. When turning ON, snap the passive side of each
        pair to match the driving side (outer pair: slot 4 <- slot 1,
        inner pair: slot 3 <- slot 2). Turning OFF is purely latent.
        """
        if self.link_copper_var.get():
            self._linking_in_progress = True
            try:
                self.slot_oz_vars[3].set(self.slot_oz_vars[0].get())
                self.slot_oz_vars[2].set(self.slot_oz_vars[1].get())
            finally:
                self._linking_in_progress = False
        self._schedule_refresh()

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
        else:
            self.status_var.set("")
            self._schedule_refresh()
        self._update_sim_status()

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
            try: self.after_cancel(self._debounce_id)
            except tk.TclError: pass
        # Sim-status is cheap (dict compare) — update synchronously so
        # the user sees red/orange flip the instant they edit a field.
        self._update_sim_status()
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

    # -------- Sim-status (persistent 3-state line) --------
    def _current_input_tuple(self):
        """
        Hashable snapshot of every user-visible input that affects the
        generated .inp. Compared to `_sent_snapshot` to decide the
        status-line state (current vs outdated).
        """
        return (
            self.od_var.get(),
            self.w_var.get(),
            self.s_var.get(),
            self.turns_var.get(),
            self.res_var.get(),
            self.outer_gap_var.get(),
            self.inner_gap_var.get(),
            tuple((bool(self.slot_active_vars[i].get()),
                   self.slot_oz_vars[i].get()) for i in range(4)),
        )

    def _update_sim_status(self):
        if self._sent_snapshot is None:
            key = "nothing"
        else:
            key = ("current"
                   if self._current_input_tuple() == self._sent_snapshot
                   else "outdated")
        text, fg = self._STATUS_COLORS[key]
        self.sim_status_var.set(text)
        try:
            self.sim_status_lbl.configure(foreground=fg)
        except tk.TclError:
            pass

    # -------- Send / Next / Reset --------
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
            self.coil_index, "Parametric Generator", dest, meta,
            on_unregister=self.clear_send_status)

        # Snapshot NOW so subsequent input edits flip status to "outdated".
        self._sent_snapshot = self._current_input_tuple()
        self._update_sim_status()

    def _on_next_tab(self):
        self.app.go_to_next_tab()

    def clear_send_status(self):
        """Invoked by SimTab when our coil slot is unregistered or reset."""
        self._sent_snapshot = None
        self._update_sim_status()

    def reset_this_tab(self):
        # Clear snapshot FIRST so input-var .set() calls don't flash
        # the status through an "outdated" state before settling on "nothing".
        self._sent_snapshot = None

        self.od_var.set(f"{self.DEFAULT_OD_MM}")
        self.w_var.set(f"{self.DEFAULT_TRACE_W_MM}")
        self.s_var.set(f"{self.DEFAULT_SPACING_MM}")
        self.turns_var.set(f"{self.DEFAULT_TURNS}")
        self.res_var.set(f"{self.DEFAULT_RESOLUTION_MM}")
        self.outer_gap_var.set(f"{self.DEFAULT_OUTER_SPACING}")
        self.inner_gap_var.set(f"{self.DEFAULT_INNER_SPACING}")
        self.link_copper_var.set(self.DEFAULT_LINK_COPPER)
        for i in range(4):
            self.slot_active_vars[i].set(i == 0)
            self.slot_oz_vars[i].set(f"{self.DEFAULT_COPPER_OZ[i]}")

        self._layer_data = None
        self._last_params = None
        self._last_stackup = None
        self.preview.clear()

        for f in list(self._my_temp_files):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self._my_temp_files.clear()

        self.app.sim_tab.unregister_coil(
            self.coil_index, "Parametric Generator")
        self._update_sim_status()
        self._schedule_refresh()

    # -------- Export-only --------
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

    # -------- save/load --------
    def save_state(self):
        return {
            "od_mm":         self.od_var.get(),
            "trace_w_mm":    self.w_var.get(),
            "spacing_mm":    self.s_var.get(),
            "turns":         self.turns_var.get(),
            "resolution_mm": self.res_var.get(),
            "outer_gap_mm":  self.outer_gap_var.get(),
            "inner_gap_mm":  self.inner_gap_var.get(),
            "link_copper":   bool(self.link_copper_var.get()),
            "slots": [
                {"active": bool(self.slot_active_vars[i].get()),
                 "copper_oz": self.slot_oz_vars[i].get()}
                for i in range(4)
            ],
        }

    def load_state(self, d):
        if not d:
            return
        def _set(v, k):
            if k in d: v.set(str(d[k]))
        _set(self.od_var, "od_mm")
        _set(self.w_var, "trace_w_mm")
        _set(self.s_var, "spacing_mm")
        _set(self.turns_var, "turns")
        _set(self.res_var, "resolution_mm")
        _set(self.outer_gap_var, "outer_gap_mm")
        _set(self.inner_gap_var, "inner_gap_mm")
        if "link_copper" in d:
            self.link_copper_var.set(bool(d["link_copper"]))
        # Guard so loaded slot values don't immediately mirror each other
        # before we've finished setting them all.
        self._linking_in_progress = True
        try:
            for i, slot in enumerate(d.get("slots", [])[:4]):
                self.slot_active_vars[i].set(bool(slot.get("active", i == 0)))
                self.slot_oz_vars[i].set(str(slot.get(
                    "copper_oz", self.DEFAULT_COPPER_OZ[i])))
        finally:
            self._linking_in_progress = False
        self._update_sim_status()
        self._schedule_refresh()