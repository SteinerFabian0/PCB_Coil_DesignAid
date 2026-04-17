#!/usr/bin/env python3
"""
Coil2Inductor — master GUI.

Tabbed layout:
    Tab "Coil 1"     : layer panels (up to 4 DXFs), convert & merge, previews
    Tab "Simulation"  : solver settings, run, results, exports
    Tab "Coil 2"      : placeholder for second coil (coupled sim)
    Tab "Automation"   : placeholder for parametric sweeps
"""

import os
import sys
import glob
import time
import math
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------------------------------------------------------------------
# Import wiring
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
if _MODULES_DIR not in sys.path:
    sys.path.insert(0, _MODULES_DIR)

import dxf_to_inp_converter as d2i
import inp_visualizer as viz
import coil_merger as merger
import coil_analysis as analysis
import zc_parser

try:
    import fasthenry_runner as runner
    _RUNNER_OK = True
except Exception:
    runner = None
    _RUNNER_OK = False

PROJECT_ROOT = os.path.dirname(_APP_ROOT)
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OZ_TO_MM = 0.035
MAX_LAYERS = 2          # bump to 4 once N-layer merger is ready


# ---------------------------------------------------------------------------
# Startup: wipe temp/
# ---------------------------------------------------------------------------
def _clean_temp():
    if os.path.isdir(TEMP_DIR):
        for entry in glob.glob(os.path.join(TEMP_DIR, "*")):
            try:
                if os.path.isfile(entry):
                    os.remove(entry)
                elif os.path.isdir(entry):
                    shutil.rmtree(entry)
            except Exception:
                pass
    os.makedirs(TEMP_DIR, exist_ok=True)


_clean_temp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def oz_to_mm(oz):
    return oz * OZ_TO_MM


def _figure_to_photo(fig, png_path, max_width=None):
    """Rasterize a matplotlib Figure to PNG -> ImageTk.PhotoImage."""
    import matplotlib.pyplot as plt
    FigureCanvasAgg(fig)
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    img = Image.open(png_path)
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# =========================================================================
#  LayerPanel — single DXF layer slot
# =========================================================================
class LayerPanel(ttk.LabelFrame):
    """One DXF layer: browse, copper params, instant preview."""

    def __init__(self, parent, tag, **kw):
        super().__init__(parent, text=tag, **kw)
        self.tag = tag          # "Layer 1", "Layer 2", …
        self.dxf_path = None
        self.inp_path = None
        self.nodes = None       # list of (x,y,z)
        self._photo = None

        # -- file picker --
        r0 = ttk.Frame(self); r0.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(r0, text="DXF:").pack(side="left")
        self.path_var = tk.StringVar(value="(none)")
        ttk.Label(r0, textvariable=self.path_var, foreground="gray",
                  width=22, anchor="w").pack(side="left", padx=4,
                                             fill="x", expand=True)
        ttk.Button(r0, text="Browse…", command=self._browse).pack(side="right")
        ttk.Button(r0, text="Clear", command=self._clear).pack(
            side="right", padx=(0, 4))

        # -- copper params --
        r1 = ttk.Frame(self); r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="Width mm:").pack(side="left")
        self.w_var = tk.StringVar(value="0.52")
        ttk.Entry(r1, textvariable=self.w_var, width=7).pack(side="left",
                                                              padx=4)
        ttk.Label(r1, text="Cu oz:").pack(side="left", padx=(8, 0))
        self.oz_var = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.oz_var, width=5).pack(side="left",
                                                               padx=4)

        # -- status + export --
        r2 = ttk.Frame(self); r2.pack(fill="x", padx=6, pady=2)
        ttk.Button(r2, text="Export INP",
                   command=self._export_inp).pack(side="left")
        self.status_var = tk.StringVar(value="—")
        ttk.Label(r2, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

        # -- preview --
        self.plot_frame = ttk.Frame(self, height=200)
        self.plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    # --- internal ---
    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select DXF",
            filetypes=[("DXF files", "*.dxf"), ("All", "*.*")])
        if not p:
            return
        try:
            if not d2i.check_dxf_is_mm(p):
                if not messagebox.askyesno(
                        "Units", "DXF units are not mm. Proceed anyway?"):
                    return
        except Exception:
            pass
        self.dxf_path = p
        self.path_var.set(os.path.basename(p))
        self.status_var.set("Loaded")
        try:
            self._quick_preview()
        except Exception as e:
            self.status_var.set(f"Preview error: {e}")

    def _quick_preview(self):
        w, h = self.get_params()
        tag_safe = self.tag.lower().replace(" ", "")
        tmp_inp = os.path.join(TEMP_DIR, f"{tag_safe}_quick.inp")
        nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, tmp_inp, w=w, h=h,
            fmin=130000, fmax=145000)
        if len(nodes) < 2:
            self.status_var.set("Too few nodes"); return
        self.nodes = nodes
        self.inp_path = tmp_inp
        self.status_var.set(f"{len(nodes)} nodes")
        self.render_preview(
            title=self.tag,
            png_name=f"{tag_safe}_preview.png",
            highlight={"start": "N0", "end": f"N{len(nodes)-1}"})

    def _clear(self):
        self.dxf_path = self.inp_path = self.nodes = self._photo = None
        self.path_var.set("(none)"); self.status_var.set("—")
        for c in self.plot_frame.winfo_children():
            c.destroy()

    def _export_inp(self):
        if not self.inp_path or not self.dxf_path:
            messagebox.showwarning("Export", "Convert first."); return
        base = os.path.splitext(os.path.basename(self.dxf_path))[0]
        dest = os.path.join(os.path.dirname(self.dxf_path), base + ".inp")
        shutil.copyfile(self.inp_path, dest)
        messagebox.showinfo("Export", f"Wrote:\n{dest}")

    # --- public API for parent ---
    def get_params(self):
        w = float(self.w_var.get())
        h = oz_to_mm(float(self.oz_var.get()))
        if w <= 0 or h <= 0:
            raise ValueError("width and oz must be positive")
        return w, h

    def convert(self, temp_name, fmin, fmax, nwinc=3, nhinc=1):
        if not self.dxf_path:
            return False
        w, h = self.get_params()
        self.inp_path = os.path.join(TEMP_DIR, temp_name)
        self.nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, self.inp_path, w=w, h=h,
            nwinc=nwinc, nhinc=nhinc,
            fmin=fmin, fmax=fmax)
        if len(self.nodes) < 2:
            raise RuntimeError(
                f"Only {len(self.nodes)} node(s) — check DXF segments")
        self.status_var.set(f"Converted ({len(self.nodes)} nodes)")
        return True

    def render_preview(self, title, png_name, highlight=None):
        if self.inp_path is None:
            return
        for c in self.plot_frame.winfo_children():
            c.destroy()
        fig, _, _ = viz.build_figure(
            self.inp_path, highlight=highlight, title=title,
            view="xy", figsize=(3.2, 2.8), dpi=90)
        png_path = os.path.join(TEMP_DIR, png_name)
        self._photo = _figure_to_photo(fig, png_path, max_width=360)
        ttk.Label(self.plot_frame, image=self._photo).pack(
            fill="both", expand=True)


# =========================================================================
#  CoilTab — "Coil 1" (or "Coil 2") notebook page
# =========================================================================
class CoilTab(ttk.Frame):
    """
    Holds up to MAX_LAYERS LayerPanels, global coil settings, convert
    & merge pipeline, and the combined overlay preview.

    Exposes these for SimTab to read:
        .combined_inp       path to merged .inp (or single-layer .inp)
        .combined_nodes     ordered list of (x,y,z)
        .layer_params       list of (w_mm, h_mm, node_count) per layer
        .via_index          index into combined_nodes where the via sits
    """

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app

        # Shared state consumed by SimTab
        self.combined_inp = None
        self.combined_nodes = None
        self.combined_highlight = None
        self.layer_params = []      # [(w, h, n_nodes), …]
        self.via_index = None
        self._combined_photo = None

        self._build()

    def _build(self):
        # --- Top row: layer panels + combined preview ---
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=4, pady=4)

        # Layer panels (left side)
        self.layers_frame = ttk.Frame(top)
        self.layers_frame.pack(side="left", fill="both", expand=True)

        self.panels = []
        for i in range(MAX_LAYERS):
            opt = "" if i == 0 else " (optional)"
            p = LayerPanel(self.layers_frame, f"Layer {i+1}{opt}")
            p.pack(side="left", fill="both", expand=True, padx=2)
            self.panels.append(p)

        # Combined preview (right side)
        cf = ttk.LabelFrame(top, text="Combined")
        cf.pack(side="left", fill="both", expand=True, padx=2)
        self.combined_frame = ttk.Frame(cf, height=200)
        self.combined_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Bottom row: settings + buttons ---
        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=8, pady=4)

        ttk.Label(bot, text="Layer spacing mm:").pack(side="left")
        self.spacing_var = tk.StringVar(value="1.4")
        ttk.Entry(bot, textvariable=self.spacing_var,
                  width=6).pack(side="left", padx=4)

        ttk.Label(bot, text="nwinc:").pack(side="left", padx=(12, 0))
        self.nwinc_var = tk.StringVar(value="3")
        ttk.Entry(bot, textvariable=self.nwinc_var,
                  width=4).pack(side="left", padx=4)

        ttk.Button(bot, text="Convert & Merge",
                   command=self.convert_and_merge).pack(side="left", padx=12)
        ttk.Button(bot, text="Export combined INP",
                   command=self._export_combined).pack(side="left")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bot, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

    # -----------------------------------------------------------------
    # Convert & Merge
    # -----------------------------------------------------------------
    def convert_and_merge(self):
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Error", "Set a valid target frequency "
                                 "on the Simulation tab first.")
            return
        fmax = fmin + 15000.0

        try:
            nwinc = max(1, int(self.nwinc_var.get()))
        except ValueError:
            nwinc = 3

        # Which layers have DXFs loaded?
        active = [p for p in self.panels if p.dxf_path is not None]
        if not active:
            messagebox.showerror("Error", "Load at least one DXF."); return

        two_layer = len(active) >= 2
        spacing = 0.0
        if two_layer:
            try:
                spacing = float(self.spacing_var.get())
            except ValueError:
                messagebox.showerror("Error", "Layer spacing must be numeric.")
                return
            if spacing <= 0:
                messagebox.showerror("Error", "Spacing must be > 0 for "
                                     "two layers.")
                return

        self.status_var.set("Converting…")
        self.update_idletasks()

        # --- Convert each layer ---
        try:
            for i, p in enumerate(active):
                p.convert(f"layer{i+1}.inp", fmin=fmin, fmax=fmax,
                          nwinc=nwinc)
                p.render_preview(
                    title=f"Layer {i+1}",
                    png_name=f"layer{i+1}_preview.png",
                    highlight={"start": "N0",
                               "end": f"N{len(p.nodes)-1}"})
        except Exception as e:
            messagebox.showerror("Conversion error", str(e))
            self.status_var.set("Error"); return

        # --- Single-layer shortcut ---
        if not two_layer:
            p0 = active[0]
            self.combined_inp = p0.inp_path
            self.combined_nodes = list(p0.nodes)
            self.via_index = None
            w, h = p0.get_params()
            self.layer_params = [(w, h, len(p0.nodes))]
            self.combined_highlight = {
                "start": "N0", "end": f"N{len(p0.nodes)-1}"}
            self._render_combined("xy", "Single layer")
            self.status_var.set("Single-layer ready.")
            return

        # --- Two-layer merge ---
        p0, p1 = active[0], active[1]
        match = merger.find_via_match(p0.nodes, p1.nodes)
        if match is None:
            messagebox.showerror(
                "Via match",
                f"No matching endpoints within "
                f"{merger.DEFAULT_VIA_MATCH_TOL_MM} mm.")
            self.status_var.set("Via match failed."); return

        _, _, via_dist = match
        a_ori, b_ori = merger.orient_layers(p0.nodes, p1.nodes, match)

        try:
            w1, h1 = p0.get_params()
            w2, h2 = p1.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error"); return

        self.combined_inp = os.path.join(TEMP_DIR, "combined.inp")

        try:
            indices = merger.write_combined_inp(
                a_ori, b_ori, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1, w2=w2, h2=h2,
                nwinc=nwinc,
                fmin=fmin, fmax=fmax)
        except TypeError:
            # Fallback if merger doesn't have w2/h2 yet
            indices = merger.write_combined_inp(
                a_ori, b_ori, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1, nwinc=nwinc,
                fmin=fmin, fmax=fmax)

        nd, _ = viz.parse_inp(self.combined_inp)
        self.combined_nodes = [nd[n] for n in viz.sorted_node_names(nd)]
        self.via_index = indices["via_idx"]
        n1 = indices["layer1_count"]
        n2 = indices["layer2_count"]
        self.layer_params = [(w1, h1, n1), (w2, h2, n2)]

        self.combined_highlight = {
            "start": f"N{indices['start_idx']}",
            "via":   f"N{indices['via_idx']}",
            "end":   f"N{indices['end_idx']}",
        }
        self._render_combined("iso", "Combined (layer overlay)")
        self.status_var.set(
            f"Merged. Via gap: {via_dist*1000:.1f} µm.")

    def _render_combined(self, view, title):
        for c in self.combined_frame.winfo_children():
            c.destroy()
        fig, _, _ = viz.build_figure(
            self.combined_inp,
            highlight=self.combined_highlight,
            title=title, view=view,
            figsize=(3.2, 2.8), dpi=90)
        png = os.path.join(TEMP_DIR, "combined_preview.png")
        self._combined_photo = _figure_to_photo(fig, png, max_width=360)
        ttk.Label(self.combined_frame,
                  image=self._combined_photo).pack(fill="both", expand=True)

    def _export_combined(self):
        if not self.combined_inp:
            messagebox.showwarning("Export", "Convert & Merge first."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".inp",
            filetypes=[("FastHenry INP", "*.inp")],
            initialfile="combined.inp")
        if p:
            shutil.copyfile(self.combined_inp, p)


# =========================================================================
#  SimTab — simulation settings, run, results
# =========================================================================
class SimTab(ttk.Frame):
    """
    Solver settings, run control, and results display.
    Reads coil data from app.coil1_tab when starting a simulation.
    """

    POLL_MS = 200

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.last_result = None
        self._sim_thread = None
        self._sim_queue = queue.Queue()
        self._build()
        self.after(self.POLL_MS, self._poll)

    def _build(self):
        # --- Settings row ---
        sf = ttk.LabelFrame(self, text="Solver settings")
        sf.pack(fill="x", padx=8, pady=(8, 4))
        sr = ttk.Frame(sf); sr.pack(fill="x", padx=6, pady=6)

        ttk.Label(sr, text="Target f (Hz):").pack(side="left")
        self.freq_var = tk.StringVar(value="130000")
        ttk.Entry(sr, textvariable=self.freq_var,
                  width=10).pack(side="left", padx=4)

        ttk.Label(sr, text="Max iter (blank=default):").pack(
            side="left", padx=(12, 0))
        self.maxiter_var = tk.StringVar(value="")
        ttk.Entry(sr, textvariable=self.maxiter_var,
                  width=6).pack(side="left", padx=4)

        ttk.Label(sr, text="Tol (blank=default):").pack(
            side="left", padx=(12, 0))
        self.tol_var = tk.StringVar(value="")
        ttk.Entry(sr, textvariable=self.tol_var,
                  width=8).pack(side="left", padx=4)

        # --- Run row ---
        rf = ttk.Frame(self); rf.pack(fill="x", padx=8, pady=4)
        self.start_btn = ttk.Button(rf, text="Start simulation",
                                     command=self._on_start)
        self.start_btn.pack(side="left")
        self.elapsed_var = tk.StringVar(value="Elapsed: —")
        ttk.Label(rf, textvariable=self.elapsed_var).pack(side="left",
                                                          padx=12)
        self.sim_status = tk.StringVar(value="Not started")
        ttk.Label(rf, textvariable=self.sim_status,
                  foreground="gray").pack(side="right")

        # --- Results ---
        res = ttk.LabelFrame(self, text="Results")
        res.pack(fill="x", padx=8, pady=4)

        grid = ttk.Frame(res); grid.pack(fill="x", padx=6, pady=6)
        self.res_L    = tk.StringVar(value="—")
        self.res_Rac  = tk.StringVar(value="—")
        self.res_Rdc  = tk.StringVar(value="—")
        self.res_len  = tk.StringVar(value="—")
        self.res_Q    = tk.StringVar(value="—")
        self.res_f0   = tk.StringVar(value="—")
        self.res_ratio = tk.StringVar(value="—")

        def put(r, c, label, var):
            ttk.Label(grid, text=label, font=("", 9, "bold")).grid(
                row=r, column=c, sticky="w", padx=4, pady=2)
            ttk.Label(grid, textvariable=var).grid(
                row=r, column=c+1, sticky="w", padx=4, pady=2)

        put(0, 0, "Inductance:",             self.res_L)
        put(0, 2, "AC resistance (solver):", self.res_Rac)
        put(1, 0, "DC resistance (30 °C):",  self.res_Rdc)
        put(1, 2, "Total path length:",      self.res_len)
        put(2, 0, "Q at target f:",          self.res_Q)
        put(2, 2, "µH / mΩ ratio:",         self.res_ratio)
        put(3, 0, "f₀ (LC resonance):",     self.res_f0)

        # Capacitance input for resonance calc
        cr = ttk.Frame(res); cr.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(cr, text="Capacitance (nF):").pack(side="left")
        self.cap_var = tk.StringVar(value="")
        ttk.Entry(cr, textvariable=self.cap_var,
                  width=10).pack(side="left", padx=4)
        self.cap_var.trace_add("write", lambda *_: self._update_f0())

        # Exports
        ef = ttk.Frame(self); ef.pack(fill="x", padx=8, pady=4)
        ttk.Button(ef, text="Export Zc.mat",
                   command=self._export_zcmat).pack(side="left")

    # --- public ---
    def get_target_freq(self):
        """Return target frequency as float, or None on bad input."""
        try:
            f = float(self.freq_var.get())
            return f if f > 0 else None
        except ValueError:
            return None

    # --- simulation ---
    def _on_start(self):
        coil = self.app.coil1_tab
        if coil.combined_inp is None:
            messagebox.showwarning("Simulation",
                                   "Convert & Merge on the Coil 1 tab first.")
            return
        if not _RUNNER_OK:
            messagebox.showerror("Simulation",
                                 "FastHenry runner unavailable "
                                 "(pywin32 + FastHenry2 required, Windows).")
            return
        if self._sim_thread and self._sim_thread.is_alive():
            return

        target_f = self.get_target_freq()
        if target_f is None:
            messagebox.showerror("Error", "Invalid target frequency."); return

        max_iter = None
        if self.maxiter_var.get().strip():
            try:
                max_iter = int(self.maxiter_var.get())
            except ValueError:
                messagebox.showerror("Error", "Max iter must be integer.")
                return

        tol = None
        if self.tol_var.get().strip():
            try:
                tol = float(self.tol_var.get())
            except ValueError:
                messagebox.showerror("Error", "Tol must be numeric."); return

        self.start_btn.config(state="disabled")
        self.sim_status.set("Running…")
        self.elapsed_var.set("Elapsed: 00:00")

        self._sim_thread = threading.Thread(
            target=self._worker,
            args=(coil.combined_inp, max_iter, tol, target_f),
            daemon=True)
        self._sim_thread.start()

    def _worker(self, inp_path, max_iter, tol, target_f):
        """Background thread. Runs solver, parses Zc.mat for results."""
        try:
            with runner.FastHenryRunner() as fh:
                def _tick(elapsed):
                    self._sim_queue.put(("tick", elapsed))

                ok = fh.run(inp_path, max_iter=max_iter, tol=tol,
                            progress_cb=_tick)
                if not ok:
                    self._sim_queue.put(("error", "Timed out.")); return

                # Pull results from Zc.mat — reliable, avoids COM quirks.
                zc_path = os.path.join(TEMP_DIR, "Zc.mat")
                fh.export_zc_mat(zc_path)

                blocks = zc_parser.parse_zc_mat(zc_path)
                if not blocks:
                    self._sim_queue.put(("error",
                                         "Zc.mat is empty or unparseable."))
                    return

                z = zc_parser.impedance_at(blocks, target_f)
                freq_used = min(
                    blocks,
                    key=lambda b: abs(b["frequency"] - target_f)
                )["frequency"]

                result = {
                    "frequency": freq_used,
                    "L_henry":   zc_parser.inductance_from_z(z, freq_used),
                    "R_ohm":     zc_parser.resistance_from_z(z),
                }
                self._sim_queue.put(("done", result))

        except Exception as e:
            self._sim_queue.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self._sim_queue.get_nowait()
                if kind == "tick":
                    m, s = divmod(int(payload), 60)
                    self.elapsed_var.set(f"Elapsed: {m:02d}:{s:02d}")
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    messagebox.showerror("Simulation", payload)
                    self.sim_status.set("Error")
                    self.start_btn.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._poll)

    # --- results ---
    def _on_done(self, result):
        self.last_result = result
        L_h  = result["L_henry"]
        R_ac = result["R_ohm"]
        f    = result["frequency"]

        coil = self.app.coil1_tab
        nodes = coil.combined_nodes or []
        length_mm = analysis.path_length_mm(nodes)

        # Per-layer DC resistance
        R_dc = self._compute_dcr(coil, nodes)

        Q = analysis.q_factor(L_h, R_ac, f)
        ratio = analysis.quality_ratio_uh_per_mohm(L_h, R_dc)

        self.res_L.set(f"{L_h * 1e6:.4f} µH")
        self.res_Rac.set(f"{R_ac * 1000:.2f} mΩ  (@ {f/1000:.1f} kHz)")
        self.res_Rdc.set(f"{R_dc * 1000:.2f} mΩ")
        self.res_len.set(f"{length_mm:.1f} mm")
        self.res_Q.set(f"{Q:.2f}")
        self.res_ratio.set(f"{ratio:.3f}")
        self._update_f0()

        self.sim_status.set("Done.")
        self.start_btn.config(state="normal")

    @staticmethod
    def _compute_dcr(coil, nodes):
        """
        Sum DC resistance per layer using each layer's own copper
        cross-section. Falls back to a flat estimate if layer_params
        isn't populated.
        """
        if not coil.layer_params or not nodes:
            # Fallback: single layer, use first panel's params
            try:
                w, h = coil.panels[0].get_params()
            except (ValueError, IndexError):
                w, h = 0.52, 0.035
            return analysis.dc_resistance_ohm(
                analysis.path_length_mm(nodes), w, h, temp_c=30.0)

        total_r = 0.0
        offset = 0
        for w, h, n in coil.layer_params:
            seg = nodes[offset:offset + n]
            seg_len = analysis.path_length_mm(seg)
            total_r += analysis.dc_resistance_ohm(seg_len, w, h, temp_c=30.0)
            offset += n
        return total_r

    def _update_f0(self):
        if self.last_result is None:
            self.res_f0.set("—"); return
        txt = self.cap_var.get().strip()
        if not txt:
            self.res_f0.set("—"); return
        try:
            c_nf = float(txt)
        except ValueError:
            self.res_f0.set("—"); return
        if c_nf <= 0:
            self.res_f0.set("—"); return
        f0 = analysis.series_resonant_freq_hz(
            self.last_result["L_henry"], c_nf * 1e-9)
        self.res_f0.set(f"{f0 / 1000:.3f} kHz")

    def _export_zcmat(self):
        src = os.path.join(TEMP_DIR, "Zc.mat")
        if not os.path.exists(src):
            messagebox.showwarning("Export", "Run simulation first."); return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mat", initialfile="Zc.mat",
            filetypes=[("Zc.mat", "*.mat"), ("All", "*.*")])
        if dest:
            shutil.copyfile(src, dest)


# =========================================================================
#  Placeholder tabs
# =========================================================================
class PlaceholderTab(ttk.Frame):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=text, foreground="gray",
                  font=("", 12)).pack(expand=True)


# =========================================================================
#  CoilApp — main window
# =========================================================================
class CoilApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Coil2Inductor")
        self.geometry("1400x720")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Build sim_tab first so coil_tab can reference it for target freq
        self.sim_tab = SimTab(nb, app=self)
        self.coil1_tab = CoilTab(nb, app=self)
        self.coil2_tab = PlaceholderTab(
            nb, "Coil 2 — coming soon\n\n"
            "Second coil for coupled / mutual-inductance simulation.")
        self.auto_tab = PlaceholderTab(
            nb, "Automation — coming soon\n\n"
            "Parametric sweeps: vary turns, layers, spacing, trace width.\n"
            "Batch-run FastHenry and collect L / R / Q data.")

        nb.add(self.coil1_tab, text="  Coil 1  ")
        nb.add(self.sim_tab,   text="  Simulation  ")
        nb.add(self.coil2_tab, text="  Coil 2  ")
        nb.add(self.auto_tab,  text="  Automation  ")


# =========================================================================
def main():
    app = CoilApp()
    app.mainloop()


if __name__ == "__main__":
    main()