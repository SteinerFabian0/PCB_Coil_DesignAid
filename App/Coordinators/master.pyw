#!/usr/bin/env python3
"""
Coil2Inductor — master GUI.

Orchestrates:
    DXF load  ->  DXF->INP convert  ->  per-layer preview
              ->  via match + merge  ->  combined overlay preview
              ->  FastHenry2 solve   ->  L / R_ac / R_dc / Q / f0 readout

Window is a single top-level, laid out top-to-bottom:
    [Layer 1 panel] [Layer 2 panel]
    [Global settings + Convert & Merge]
    [Combined overlay]
    [Simulation controls]
    [Results]
"""

import os
import sys
import time
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Import wiring: master.py lives in App/Coordinators/, modules next door.
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
if _MODULES_DIR not in sys.path:
    sys.path.insert(0, _MODULES_DIR)

import dxf_to_inp_converter as d2i
import inp_visualizer as viz
import coil_merger as merger
import coil_analysis as analysis

# fasthenry_runner is Windows-only in practice (COM). We tolerate missing
# pywin32 so the GUI can at least render on dev machines without the solver.
try:
    import fasthenry_runner as runner
    _RUNNER_IMPORT_OK = True
except Exception as _e:
    runner = None
    _RUNNER_IMPORT_OK = False

# temp/ sits at the project root (sibling of App/).
PROJECT_ROOT = os.path.dirname(_APP_ROOT)
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


# 1 oz/ft^2 copper ~= 35 um thickness; the converter expects mm.
OZ_TO_MM = 0.035


def oz_to_mm(oz):
    return oz * OZ_TO_MM


# =========================================================================
# LayerPanel — one per DXF slot
# =========================================================================

class LayerPanel(ttk.LabelFrame):
    """
    Holds one layer's DXF picker, width/weight inputs, an export button,
    and an embedded matplotlib preview of the converted .inp.

    Conversion itself is driven from the parent app (so both layers can
    share the same fmin/fmax), but each panel caches its own inp_path
    and node list for downstream steps (merge, path-length).
    """

    def __init__(self, parent, title, **kw):
        super().__init__(parent, text=title, **kw)

        # State set by browse() / convert().
        self.dxf_path = None
        self.inp_path = None
        self.nodes = None   # list of (x,y,z)
        self._canvas = None

        # -- Row: file picker --
        r0 = ttk.Frame(self)
        r0.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(r0, text="DXF:").pack(side="left")
        self.path_var = tk.StringVar(value="(none)")
        ttk.Label(r0, textvariable=self.path_var, foreground="gray",
                  width=30, anchor="w").pack(side="left", padx=4,
                                             fill="x", expand=True)
        ttk.Button(r0, text="Browse…", command=self._browse).pack(side="right")
        ttk.Button(r0, text="Clear", command=self._clear).pack(side="right",
                                                               padx=(0, 4))

        # -- Row: copper width + weight --
        r1 = ttk.Frame(self)
        r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="Width (mm):").pack(side="left")
        self.w_var = tk.StringVar(value="0.52")
        ttk.Entry(r1, textvariable=self.w_var, width=8).pack(side="left",
                                                              padx=4)
        ttk.Label(r1, text="Weight (oz):").pack(side="left", padx=(12, 0))
        self.oz_var = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.oz_var, width=6).pack(side="left",
                                                               padx=4)

        # -- Row: export + status --
        r2 = ttk.Frame(self)
        r2.pack(fill="x", padx=6, pady=2)
        ttk.Button(r2, text="Export INP next to DXF",
                   command=self._export_inp).pack(side="left")
        self.status_var = tk.StringVar(value="No DXF loaded")
        ttk.Label(r2, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

        # -- Preview plot area --
        self.plot_frame = ttk.Frame(self, height=260)
        self.plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    # ---- UI handlers ----

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select DXF",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")])
        if not p:
            return
        # Soft unit check — converter assumes mm.
        try:
            if not d2i.check_dxf_is_mm(p):
                if not messagebox.askyesno(
                        "Units",
                        "DXF $INSUNITS is not 'millimeters'. "
                        "Proceed anyway (values will be treated as mm)?"):
                    return
        except Exception:
            # Header read failed — not fatal. Trust the user.
            pass
        self.dxf_path = p
        self.path_var.set(os.path.basename(p))
        self.status_var.set("DXF loaded (not converted yet)")

    def _clear(self):
        self.dxf_path = None
        self.inp_path = None
        self.nodes = None
        self.path_var.set("(none)")
        self.status_var.set("No DXF loaded")
        for c in self.plot_frame.winfo_children():
            c.destroy()

    def _export_inp(self):
        if self.inp_path is None or not self.dxf_path:
            messagebox.showwarning("Export", "Run Convert & Merge first.")
            return
        base = os.path.splitext(os.path.basename(self.dxf_path))[0]
        dest = os.path.join(os.path.dirname(self.dxf_path), base + ".inp")
        shutil.copyfile(self.inp_path, dest)
        messagebox.showinfo("Export", f"Wrote:\n{dest}")

    # ---- Used by parent ----

    def get_params(self):
        """Parse the entry fields. Raises ValueError on bad input."""
        w = float(self.w_var.get())
        h = oz_to_mm(float(self.oz_var.get()))
        if w <= 0 or h <= 0:
            raise ValueError("width and oz must be positive")
        return w, h

    def convert(self, temp_name, fmin, fmax):
        """DXF -> INP. Caches nodes and inp_path. Returns True on success."""
        if not self.dxf_path:
            return False
        w, h = self.get_params()
        self.inp_path = os.path.join(TEMP_DIR, temp_name)
        self.nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, self.inp_path,
            w=w, h=h, fmin=fmin, fmax=fmax)
        if len(self.nodes) < 2:
            raise RuntimeError(
                f"Only {len(self.nodes)} node(s) after chaining — "
                "DXF may have disjoint segments.")
        self.status_var.set(f"Converted ({len(self.nodes)} nodes)")
        return True

    def render_preview(self, title, highlight=None):
        """Drop a matplotlib figure for the current inp_path into the panel."""
        if self.inp_path is None:
            return
        for c in self.plot_frame.winfo_children():
            c.destroy()
        fig, _, _ = viz.build_figure(
            self.inp_path, highlight=highlight, title=title,
            view="xy", figsize=(4.5, 4.0), dpi=90)
        self._canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.draw()


# =========================================================================
# CoilApp — the window
# =========================================================================

class CoilApp(tk.Tk):

    POLL_MS = 200   # how often the main thread drains the sim queue

    def __init__(self):
        super().__init__()
        self.title("Coil2Inductor")
        self.geometry("1320x960")

        # Simulation state
        self.combined_inp = None
        self.combined_nodes = None        # list of (x,y,z), for path length
        self.combined_highlight = None    # dict for visualizer
        self.last_result = None           # {"frequency","L_henry","R_ohm"}
        self._sim_thread = None
        self._sim_queue = queue.Queue()
        self._sim_t0 = None

        self._build_ui()
        self.after(self.POLL_MS, self._poll_sim_queue)

    # ---- UI construction ----

    def _build_ui(self):
        # Row 1: two layer panels side by side
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 4))
        self.layer1 = LayerPanel(top, "Layer 1")
        self.layer1.pack(side="left", fill="both", expand=True, padx=4)
        self.layer2 = LayerPanel(top, "Layer 2 (optional)")
        self.layer2.pack(side="left", fill="both", expand=True, padx=4)

        # Row 2: global settings
        gf = ttk.LabelFrame(self, text="Global settings")
        gf.pack(fill="x", padx=8, pady=4)

        gr = ttk.Frame(gf)
        gr.pack(fill="x", padx=6, pady=6)
        ttk.Label(gr, text="Layer spacing (mm):").pack(side="left")
        self.spacing_var = tk.StringVar(value="1.4")
        ttk.Entry(gr, textvariable=self.spacing_var,
                  width=8).pack(side="left", padx=4)
        ttk.Label(gr, text="Target f (Hz):").pack(side="left", padx=(12, 0))
        self.freq_var = tk.StringVar(value="130000")
        ttk.Entry(gr, textvariable=self.freq_var,
                  width=10).pack(side="left", padx=4)
        ttk.Label(gr, text="Max iterations:").pack(side="left", padx=(12, 0))
        self.maxiter_var = tk.StringVar(value="180")
        ttk.Entry(gr, textvariable=self.maxiter_var,
                  width=6).pack(side="left", padx=4)
        ttk.Label(gr, text="Tol (blank=default):").pack(side="left",
                                                        padx=(12, 0))
        self.tol_var = tk.StringVar(value="")
        ttk.Entry(gr, textvariable=self.tol_var,
                  width=8).pack(side="left", padx=4)

        gr2 = ttk.Frame(gf)
        gr2.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(gr2, text="Convert & Merge",
                   command=self._on_convert_merge).pack(side="left")
        ttk.Button(gr2, text="Export combined INP",
                   command=self._export_combined).pack(side="left", padx=6)
        self.global_status = tk.StringVar(value="Idle")
        ttk.Label(gr2, textvariable=self.global_status,
                  foreground="gray").pack(side="right")

        # Row 3: combined overlay
        cf = ttk.LabelFrame(self, text="Combined (both layers)")
        cf.pack(fill="both", expand=True, padx=8, pady=4)
        self.combined_plot_frame = ttk.Frame(cf, height=280)
        self.combined_plot_frame.pack(fill="both", expand=True,
                                      padx=6, pady=6)
        self._combined_canvas = None

        # Row 4: simulation controls
        sf = ttk.LabelFrame(self, text="Simulation")
        sf.pack(fill="x", padx=8, pady=4)
        sr = ttk.Frame(sf)
        sr.pack(fill="x", padx=6, pady=6)
        self.start_btn = ttk.Button(sr, text="Start simulation",
                                     command=self._on_start_sim)
        self.start_btn.pack(side="left")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00")
        ttk.Label(sr, textvariable=self.elapsed_var).pack(side="left",
                                                          padx=12)
        self.sim_status = tk.StringVar(value="Not started")
        ttk.Label(sr, textvariable=self.sim_status,
                  foreground="gray").pack(side="right")

        # Row 5: results
        rf = ttk.LabelFrame(self, text="Results")
        rf.pack(fill="x", padx=8, pady=(4, 8))

        grid = ttk.Frame(rf)
        grid.pack(fill="x", padx=6, pady=6)

        self.res_L = tk.StringVar(value="—")
        self.res_Rac = tk.StringVar(value="—")
        self.res_Rdc = tk.StringVar(value="—")
        self.res_len = tk.StringVar(value="—")
        self.res_Q = tk.StringVar(value="—")
        self.res_f0 = tk.StringVar(value="—")

        def put(row, col, label, var):
            ttk.Label(grid, text=label,
                      font=("", 9, "bold")).grid(row=row, column=col,
                                                 sticky="w", padx=4, pady=2)
            ttk.Label(grid, textvariable=var).grid(row=row, column=col + 1,
                                                   sticky="w", padx=4, pady=2)

        put(0, 0, "Inductance:", self.res_L)
        put(0, 2, "AC resistance (solver):", self.res_Rac)
        put(1, 0, "DC resistance (30 °C):", self.res_Rdc)
        put(1, 2, "Total path length:", self.res_len)
        put(2, 0, "Q at target f:", self.res_Q)
        put(2, 2, "f0 (LC resonance):", self.res_f0)

        cr = ttk.Frame(rf)
        cr.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(cr, text="Capacitance (nF):").pack(side="left")
        self.cap_var = tk.StringVar(value="")
        ttk.Entry(cr, textvariable=self.cap_var,
                  width=10).pack(side="left", padx=4)
        self.cap_var.trace_add("write", lambda *_: self._update_resonance())
        ttk.Button(cr, text="Export Zc.mat",
                   command=self._export_zcmat).pack(side="right")

    # =====================================================================
    # Convert & Merge pipeline
    # =====================================================================

    def _on_convert_merge(self):
        # Parse global inputs first — fail fast on bad numbers.
        try:
            fmin = float(self.freq_var.get())
            if fmin <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error",
                                 "Target frequency must be a positive number.")
            return
        fmax = fmin + 15000.0

        two_layer = (self.layer2.dxf_path is not None)

        # Layer spacing is mandatory in two-layer mode.
        spacing = 0.0
        if two_layer:
            try:
                spacing = float(self.spacing_var.get())
            except ValueError:
                messagebox.showerror("Error",
                                     "Layer spacing must be numeric.")
                return
            if spacing <= 0:
                messagebox.showerror(
                    "Error",
                    "Layer spacing must be > 0 when two DXFs are loaded.")
                return

        if not self.layer1.dxf_path:
            messagebox.showerror("Error", "Load Layer 1's DXF first.")
            return

        self.global_status.set("Converting…")
        self.update_idletasks()

        # --- Convert each layer ---
        try:
            self.layer1.convert("layer1.inp", fmin=fmin, fmax=fmax)
            self.layer1.render_preview(
                title="Layer 1",
                highlight={"start": "N0",
                           "end": f"N{len(self.layer1.nodes) - 1}"})

            if two_layer:
                self.layer2.convert("layer2.inp", fmin=fmin, fmax=fmax)
                self.layer2.render_preview(
                    title="Layer 2",
                    highlight={"start": "N0",
                               "end": f"N{len(self.layer2.nodes) - 1}"})
        except Exception as e:
            messagebox.showerror("Conversion error", str(e))
            self.global_status.set("Error")
            return

        # --- Single-layer mode: layer1.inp is the final artifact ---
        if not two_layer:
            self.combined_inp = self.layer1.inp_path
            self.combined_nodes = list(self.layer1.nodes)
            self.combined_highlight = {
                "start": "N0",
                "end": f"N{len(self.layer1.nodes) - 1}",
            }
            self._render_combined(view="xy", title="Single layer")
            self.global_status.set("Single-layer ready.")
            return

        # --- Two-layer merge ---
        match = merger.find_via_match(self.layer1.nodes, self.layer2.nodes)
        if match is None:
            messagebox.showerror(
                "Via match failed",
                "No coincident endpoints within "
                f"{merger.DEFAULT_VIA_MATCH_TOL_MM} mm. "
                "Make sure both DXFs share a point where the via should go.")
            self.global_status.set("Via match failed.")
            return

        _, _, via_dist = match
        a_oriented, b_oriented = merger.orient_layers(
            self.layer1.nodes, self.layer2.nodes, match)

        try:
            w1, h1 = self.layer1.get_params()
            w2, h2 = self.layer2.get_params()
        except ValueError as e:
            messagebox.showerror("Error", f"Bad copper params: {e}")
            self.global_status.set("Error")
            return

        self.combined_inp = os.path.join(TEMP_DIR, "combined.inp")

        # NOTE: this call depends on the per-layer w2/h2 patch to
        # coil_merger.write_combined_inp. If you haven't applied it yet,
        # drop `w2=w2, h2=h2` here — the GUI will still run but layer-2
        # copper dimensions will be ignored.
        try:
            indices = merger.write_combined_inp(
                a_oriented, b_oriented, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1, w2=w2, h2=h2,
                fmin=fmin, fmax=fmax)
        except TypeError:
            # Fallback for unpatched merger signature.
            indices = merger.write_combined_inp(
                a_oriented, b_oriented, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1,
                fmin=fmin, fmax=fmax)

        # Cache the combined node list (read from disk so it reflects the
        # snap / z-offset the writer applied).
        nd, _ = viz.parse_inp(self.combined_inp)
        self.combined_nodes = [nd[n] for n in viz.sorted_node_names(nd)]

        self.combined_highlight = {
            "start": f"N{indices['start_idx']}",
            "via":   f"N{indices['via_idx']}",
            "end":   f"N{indices['end_idx']}",
        }
        self._render_combined(view="iso", title="Combined (layer overlay)")
        self.global_status.set(
            f"Merged. Via gap before snap: {via_dist * 1000:.1f} µm.")

    def _render_combined(self, view, title):
        for c in self.combined_plot_frame.winfo_children():
            c.destroy()
        fig, _, _ = viz.build_figure(
            self.combined_inp,
            highlight=self.combined_highlight,
            title=title, view=view,
            figsize=(9, 4.5), dpi=90)
        self._combined_canvas = FigureCanvasTkAgg(
            fig, master=self.combined_plot_frame)
        self._combined_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._combined_canvas.draw()

    def _export_combined(self):
        if self.combined_inp is None:
            messagebox.showwarning("Export", "Run Convert & Merge first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".inp",
            filetypes=[("FastHenry INP", "*.inp")],
            initialfile="combined.inp")
        if p:
            shutil.copyfile(self.combined_inp, p)

    # =====================================================================
    # Simulation (threaded)
    # =====================================================================

    def _on_start_sim(self):
        if self.combined_inp is None:
            messagebox.showwarning("Simulation", "Convert & Merge first.")
            return
        if not _RUNNER_IMPORT_OK:
            messagebox.showerror(
                "Simulation",
                "FastHenry runner unavailable (pywin32 + FastHenry2 "
                "required, Windows-only).")
            return
        if self._sim_thread is not None and self._sim_thread.is_alive():
            return

        try:
            max_iter = int(self.maxiter_var.get())
            target_f = float(self.freq_var.get())
        except ValueError:
            messagebox.showerror("Error", "Iterations / frequency invalid.")
            return

        tol = None
        if self.tol_var.get().strip():
            try:
                tol = float(self.tol_var.get())
            except ValueError:
                messagebox.showerror("Error", "Tolerance must be numeric.")
                return

        self.start_btn.config(state="disabled")
        self.sim_status.set("Running…")
        self.elapsed_var.set("Elapsed: 00:00")
        self._sim_t0 = time.time()

        self._sim_thread = threading.Thread(
            target=self._sim_worker,
            args=(self.combined_inp, max_iter, tol, target_f),
            daemon=True)
        self._sim_thread.start()

    def _sim_worker(self, inp_path, max_iter, tol, target_f):
        """Background thread. Posts ('tick', s) / ('done', dict) / ('error', str)."""
        try:
            with runner.FastHenryRunner() as fh:
                def _tick(elapsed):
                    self._sim_queue.put(("tick", elapsed))
                ok = fh.run(inp_path, max_iter=max_iter, tol=tol,
                            progress_cb=_tick)
                if not ok:
                    self._sim_queue.put(("error", "Simulation timed out."))
                    return
                result = fh.single_port_result(target_f)
                # Copy Zc.mat into temp/ for later export; non-fatal if it fails.
                try:
                    fh.export_zc_mat(os.path.join(TEMP_DIR, "Zc.mat"))
                except Exception:
                    pass
                self._sim_queue.put(("done", result))
        except Exception as e:
            self._sim_queue.put(("error", f"{type(e).__name__}: {e}"))

    def _poll_sim_queue(self):
        try:
            while True:
                kind, payload = self._sim_queue.get_nowait()
                if kind == "tick":
                    self._update_elapsed(payload)
                elif kind == "done":
                    self._on_sim_done(payload)
                elif kind == "error":
                    messagebox.showerror("Simulation", payload)
                    self.sim_status.set("Error")
                    self.start_btn.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._poll_sim_queue)

    def _update_elapsed(self, seconds):
        m, s = divmod(int(seconds), 60)
        self.elapsed_var.set(f"Elapsed: {m:02d}:{s:02d}")

    # =====================================================================
    # Results + derived metrics
    # =====================================================================

    def _on_sim_done(self, result):
        self.last_result = result
        L_h = result["L_henry"]
        R_ac = result["R_ohm"]
        f = result["frequency"]

        # Path length from the combined node list we cached at merge time.
        length_mm = analysis.path_length_mm(self.combined_nodes or [])

        # DC R: use layer 1's cross-section as the representative one.
        # If layers differ, this is approximate — fine for a rough sanity
        # check, since we're comparing against the solver's AC R anyway.
        try:
            w1, h1 = self.layer1.get_params()
        except ValueError:
            w1, h1 = 0.52, 0.035
        R_dc = analysis.dc_resistance_ohm(length_mm, w1, h1, temp_c=30.0)

        Q = analysis.q_factor(L_h, R_ac, f)

        self.res_L.set(f"{L_h * 1e6:.4f} µH")
        self.res_Rac.set(f"{R_ac * 1000:.2f} mΩ  (@ {f/1000:.2f} kHz)")
        self.res_Rdc.set(f"{R_dc * 1000:.2f} mΩ")
        self.res_len.set(f"{length_mm:.2f} mm")
        self.res_Q.set(f"{Q:.2f}")
        self._update_resonance()

        self.sim_status.set("Done.")
        self.start_btn.config(state="normal")

    def _update_resonance(self):
        """Recompute f0 live as the user types into the capacitance field."""
        if self.last_result is None:
            self.res_f0.set("—")
            return
        txt = self.cap_var.get().strip()
        if not txt:
            self.res_f0.set("—")
            return
        try:
            c_nf = float(txt)
        except ValueError:
            self.res_f0.set("—")
            return
        if c_nf <= 0:
            self.res_f0.set("—")
            return
        f0 = analysis.series_resonant_freq_hz(
            self.last_result["L_henry"], c_nf * 1e-9)
        self.res_f0.set(f"{f0 / 1000:.3f} kHz")

    def _export_zcmat(self):
        src = os.path.join(TEMP_DIR, "Zc.mat")
        if not os.path.exists(src):
            messagebox.showwarning("Export", "Run simulation first.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mat",
            initialfile="Zc.mat",
            filetypes=[("FastHenry Zc.mat", "*.mat"),
                       ("All files", "*.*")])
        if dest:
            shutil.copyfile(src, dest)


# =========================================================================

def main():
    app = CoilApp()
    app.mainloop()


if __name__ == "__main__":
    main()