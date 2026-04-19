#!/usr/bin/env python3
"""
Coil2Inductor — master GUI.

Tab order:
    DXF Coil 1, DXF Coil 2, Parametric Coil 1, Parametric Coil 2,
    Simulation, Automation.

Each coil tab has a "Send to Simulation" control. The Simulation tab
shows which source is currently feeding each coil slot and runs either
1-port (single coil) or 2-port (both coils, via port_combiner).
"""

import os
import sys
import glob
import time
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
for _p in (_MODULES_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dxf_to_inp_converter as d2i
import inp_visualizer as viz
import coil_merger as merger
import coil_analysis as analysis
import zc_parser
import port_combiner

try:
    import fasthenry_runner as runner
    _RUNNER_OK = True
except Exception:
    runner = None
    _RUNNER_OK = False

from parametric_tab import ParametricCoilTab


PROJECT_ROOT = os.path.dirname(_APP_ROOT)
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OZ_TO_MM = 0.035
MAX_LAYERS = 2


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
    import matplotlib.pyplot as plt
    FigureCanvasAgg(fig)
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    img = Image.open(png_path)
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# =========================================================================
#  LayerPanel
# =========================================================================
class LayerPanel(ttk.LabelFrame):
    """One DXF layer slot."""

    def __init__(self, parent, tag, coil_index, layer_index,
                 on_state_change=None, on_send_single=None, **kw):
        super().__init__(parent, text=tag, **kw)
        self.tag = tag
        self.coil_index = coil_index
        self.layer_index = layer_index          # 1-based
        self.dxf_path = None
        self.inp_path = None
        self.nodes = None
        self._photo = None
        self._on_state_change = on_state_change
        self._on_send_single  = on_send_single

        # File picker row
        r0 = ttk.Frame(self); r0.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(r0, text="DXF:").pack(side="left")
        self.path_var = tk.StringVar(value="(none)")
        ttk.Label(r0, textvariable=self.path_var, foreground="gray",
                  width=18, anchor="w").pack(side="left", padx=4,
                                              fill="x", expand=True)
        ttk.Button(r0, text="Browse…",
                   command=self._browse).pack(side="right")
        ttk.Button(r0, text="Clear",
                   command=self._clear).pack(side="right", padx=(0, 4))

        # Copper params
        r1 = ttk.Frame(self); r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="Width mm:").pack(side="left")
        self.w_var = tk.StringVar(value="0.52")
        ttk.Entry(r1, textvariable=self.w_var, width=7).pack(side="left", padx=4)
        ttk.Label(r1, text="Cu oz:").pack(side="left", padx=(8, 0))
        self.oz_var = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.oz_var, width=5).pack(side="left", padx=4)

        # Export + per-layer send row
        r2 = ttk.Frame(self); r2.pack(fill="x", padx=6, pady=2)
        ttk.Button(r2, text="Export INP", width=12,
                   command=self._export_inp).pack(side="left")
        ttk.Button(r2, text="Send to Simulation", width=20,
                   command=self._send_to_sim).pack(side="left", padx=(6, 0))

        # Status row
        r3 = ttk.Frame(self); r3.pack(fill="x", padx=6, pady=2)
        self.status_var = tk.StringVar(value="—")
        ttk.Label(r3, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

        # Preview
        self.plot_frame = ttk.Frame(self, height=200)
        self.plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    # -- DXF loading / clearing --
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
        if self._on_state_change:
            self._on_state_change()
        try:
            self._quick_preview()
        except Exception as e:
            self.status_var.set(f"Preview error: {e}")

    def _quick_preview(self):
        w, h = self.get_params()
        tag_safe = self._tag_safe()
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
        if self._on_state_change:
            self._on_state_change()

    # -- Tag helpers --
    def _tag_safe(self):
        return f"coil{self.coil_index}_layer{self.layer_index}"

    # -- Public API for parent coil tab --
    def get_params(self):
        w = float(self.w_var.get())
        h = oz_to_mm(float(self.oz_var.get()))
        if w <= 0 or h <= 0:
            raise ValueError("width and oz must be positive")
        return w, h

    def convert(self, fmin, fmax, nwinc=3, nhinc=1):
        if not self.dxf_path:
            return False
        w, h = self.get_params()
        tag_safe = self._tag_safe()
        self.inp_path = os.path.join(TEMP_DIR, f"{tag_safe}.inp")
        self.nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, self.inp_path, w=w, h=h,
            nwinc=nwinc, nhinc=nhinc, fmin=fmin, fmax=fmax)
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

    def _export_inp(self):
        if not self.inp_path or not self.dxf_path:
            messagebox.showwarning("Export", "Convert first."); return
        base = os.path.splitext(os.path.basename(self.dxf_path))[0]
        dest = os.path.join(os.path.dirname(self.dxf_path), base + ".inp")
        shutil.copyfile(self.inp_path, dest)
        messagebox.showinfo("Export", f"Wrote:\n{dest}")

    def _send_to_sim(self):
        if self.inp_path is None or self.nodes is None:
            messagebox.showwarning("Send to Sim",
                                   "Load a DXF first."); return
        if self._on_send_single is None:
            return
        try:
            w, h = self.get_params()
        except ValueError as e:
            messagebox.showerror("Send to Sim", str(e)); return
        self._on_send_single(self, w, h)


# =========================================================================
#  CoilTab — a full DXF-based coil (1 or 2 layers, optionally merged)
# =========================================================================
class CoilTab(ttk.Frame):
    """One DXF-based coil. Holds up to MAX_LAYERS LayerPanels + merge."""

    def __init__(self, parent, app, coil_index, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.coil_index = coil_index

        self.combined_inp = None
        self.combined_nodes = None
        self.combined_highlight = None
        self.layer_params = []
        self.via_index = None
        self._combined_photo = None
        self._build()

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="both", expand=True,
                                         padx=4, pady=4)

        self.layers_frame = ttk.Frame(top)
        self.layers_frame.pack(side="left", fill="both", expand=True)
        self.panels = []
        for i in range(MAX_LAYERS):
            opt = "" if i == 0 else " (optional)"
            p = LayerPanel(
                self.layers_frame,
                tag=f"Layer {i+1}{opt}",
                coil_index=self.coil_index, layer_index=i+1,
                on_state_change=self._on_layer_dxf_change,
                on_send_single=self._send_layer_to_sim)
            p.pack(side="left", fill="both", expand=True, padx=2)
            self.panels.append(p)

        cf = ttk.LabelFrame(top, text="Combined")
        cf.pack(side="left", fill="both", expand=True, padx=2)
        self.combined_frame = ttk.Frame(cf, height=200)
        self.combined_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Settings + actions row
        bot = ttk.Frame(self); bot.pack(fill="x", padx=8, pady=4)

        ttk.Label(bot, text="Layer spacing mm:").pack(side="left")
        self.spacing_var = tk.StringVar(value="1.4")
        ttk.Entry(bot, textvariable=self.spacing_var,
                  width=6).pack(side="left", padx=4)
        ttk.Label(bot, text="nwinc:").pack(side="left", padx=(12, 0))
        self.nwinc_var = tk.StringVar(value="3")
        ttk.Entry(bot, textvariable=self.nwinc_var,
                  width=4).pack(side="left", padx=4)

        ttk.Button(bot, text="Convert & Merge", width=18,
                   command=self.convert_and_merge).pack(side="left", padx=12)
        ttk.Button(bot, text="Send Combined to Simulation", width=26,
                   command=self._send_combined_to_sim).pack(side="left")
        ttk.Button(bot, text="Export combined INP", width=20,
                   command=self._export_combined).pack(side="left", padx=(8, 0))
        ttk.Button(bot, text="Reset This Tab", width=16,
                   command=self.reset_this_tab).pack(side="right")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bot, textvariable=self.status_var,
                  foreground="gray").pack(side="right", padx=8)

    # -- Parametric-grey wiring --
    def any_dxf_loaded(self):
        return any(p.dxf_path for p in self.panels)

    def _on_layer_dxf_change(self):
        self.app.on_coil_dxf_state_changed(self)

    # -- Convert & Merge --
    def convert_and_merge(self):
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Error", "Set a valid target frequency "
                                 "on the Simulation tab first."); return
        fmax = fmin + 15000.0

        try:
            nwinc = max(1, int(self.nwinc_var.get()))
        except ValueError:
            nwinc = 3

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
                messagebox.showerror("Error",
                                     "Spacing must be > 0 for two layers.")
                return

        self.status_var.set("Converting…"); self.update_idletasks()

        try:
            for p in active:
                p.convert(fmin=fmin, fmax=fmax, nwinc=nwinc)
                p.render_preview(
                    title=p.tag,
                    png_name=f"{p._tag_safe()}_preview.png",
                    highlight={"start": "N0",
                               "end": f"N{len(p.nodes)-1}"})
        except Exception as e:
            messagebox.showerror("Conversion error", str(e))
            self.status_var.set("Error"); return

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

        self.combined_inp = os.path.join(
            TEMP_DIR, f"coil{self.coil_index}_combined.inp")
        try:
            indices = merger.write_combined_inp(
                a_ori, b_ori, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1, w2=w2, h2=h2,
                nwinc=nwinc, fmin=fmin, fmax=fmax)
        except TypeError:
            indices = merger.write_combined_inp(
                a_ori, b_ori, self.combined_inp,
                layer_spacing_mm=spacing,
                w=w1, h=h1, nwinc=nwinc, fmin=fmin, fmax=fmax)

        nd, _ = viz.parse_inp(self.combined_inp)
        self.combined_nodes = [nd[n] for n in viz.sorted_node_names(nd)]
        self.via_index = indices["via_idx"]
        self.layer_params = [(w1, h1, indices["layer1_count"]),
                             (w2, h2, indices["layer2_count"])]
        self.combined_highlight = {
            "start": f"N{indices['start_idx']}",
            "via":   f"N{indices['via_idx']}",
            "end":   f"N{indices['end_idx']}",
        }
        self._render_combined("iso", "Combined (layer overlay)")
        self.status_var.set(f"Merged. Via gap: {via_dist*1000:.1f} µm.")

    def _render_combined(self, view, title):
        for c in self.combined_frame.winfo_children():
            c.destroy()
        fig, _, _ = viz.build_figure(
            self.combined_inp, highlight=self.combined_highlight,
            title=title, view=view, figsize=(3.2, 2.8), dpi=90)
        png = os.path.join(TEMP_DIR,
                           f"coil{self.coil_index}_combined_preview.png")
        self._combined_photo = _figure_to_photo(fig, png, max_width=360)
        ttk.Label(self.combined_frame,
                  image=self._combined_photo).pack(fill="both", expand=True)

    # -- Export / Send / Reset --
    def _export_combined(self):
        if not self.combined_inp:
            messagebox.showwarning("Export", "Convert & Merge first."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".inp",
            filetypes=[("FastHenry INP", "*.inp")],
            initialfile=f"coil{self.coil_index}_combined.inp")
        if p:
            shutil.copyfile(self.combined_inp, p)

    def _send_combined_to_sim(self):
        if self.combined_inp is None:
            messagebox.showwarning("Send to Sim",
                                   "Convert & Merge first."); return
        topology = "series" if len(self.layer_params) > 1 else "single"
        meta = {
            "topology":     topology,
            "layer_params": self.layer_params,
            "nodes":        list(self.combined_nodes or []),
        }
        self.app.sim_tab.register_coil(
            self.coil_index, "DXF Import", self.combined_inp, meta)
        self.status_var.set(f"Sent as Coil {self.coil_index}.")

    def _send_layer_to_sim(self, panel, w, h):
        """Called by a LayerPanel's Send button — registers just that layer."""
        inp_dest = os.path.join(
            TEMP_DIR, f"coil{self.coil_index}_layer{panel.layer_index}_sent.inp")
        try:
            shutil.copyfile(panel.inp_path, inp_dest)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Copy failed: {e}")
            return
        meta = {
            "topology":     "single",
            "layer_params": [(w, h, len(panel.nodes))],
            "nodes":        list(panel.nodes),
        }
        self.app.sim_tab.register_coil(
            self.coil_index, "DXF Import", inp_dest, meta)
        self.status_var.set(
            f"Sent Layer {panel.layer_index} as Coil {self.coil_index}.")

    def reset_this_tab(self):
        # Clear panels (individually, respecting their own state-change hook).
        for p in self.panels:
            p._clear()

        # Remove this coil tab's temp artifacts. We only touch files whose
        # name is prefixed with coil{N}_, so the sibling coil tab's files
        # are safe.
        prefix = f"coil{self.coil_index}_"
        for f in glob.glob(os.path.join(TEMP_DIR, prefix + "*")):
            _safe_remove(f)

        self.combined_inp = None
        self.combined_nodes = None
        self.combined_highlight = None
        self.layer_params = []
        self.via_index = None
        self.spacing_var.set("1.4")
        self.nwinc_var.set("3")
        for c in self.combined_frame.winfo_children():
            c.destroy()
        self.status_var.set("Reset.")

        # Vacate sim-tab slot only if WE were the source.
        self.app.sim_tab.unregister_coil(self.coil_index, "DXF Import")


# =========================================================================
#  SimTab — solver + results (1-port or 2-port)
# =========================================================================
class SimTab(ttk.Frame):
    POLL_MS = 200
    DEFAULT_FREQ = "130000"

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.last_result = None
        self._sim_thread = None
        self._sim_queue = queue.Queue()

        # Registered coil sources
        self._sources   = [None, None]    # "DXF Import" / "Parametric Generator" / None
        self._inp_paths = [None, None]
        self._metadata  = [None, None]

        self._build()
        self.after(self.POLL_MS, self._poll)
        self._update_source_labels()

    # -- build --
    def _build(self):
        # Source panel: what's currently wired into each coil slot.
        src = ttk.LabelFrame(self, text="Coil sources")
        src.pack(fill="x", padx=8, pady=(8, 4))
        self.coil1_src_var = tk.StringVar(value="Coil 1: None")
        self.coil2_src_var = tk.StringVar(value="Coil 2: None")
        ttk.Label(src, textvariable=self.coil1_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Label(src, textvariable=self.coil2_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(0, 4))

        # Solver settings
        sf = ttk.LabelFrame(self, text="Solver settings")
        sf.pack(fill="x", padx=8, pady=4)
        sr = ttk.Frame(sf); sr.pack(fill="x", padx=6, pady=6)
        ttk.Label(sr, text="Target f (Hz):").pack(side="left")
        self.freq_var = tk.StringVar(value=self.DEFAULT_FREQ)
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

        # Run + reset row
        rf = ttk.Frame(self); rf.pack(fill="x", padx=8, pady=4)
        self.start_btn = ttk.Button(rf, text="Start simulation", width=20,
                                     command=self._on_start)
        self.start_btn.pack(side="left")
        self.elapsed_var = tk.StringVar(value="Elapsed: —")
        ttk.Label(rf, textvariable=self.elapsed_var).pack(
            side="left", padx=12)
        ttk.Button(rf, text="Reset This Tab", width=16,
                   command=self.reset_this_tab).pack(side="right")
        self.sim_status = tk.StringVar(value="Not started")
        ttk.Label(rf, textvariable=self.sim_status,
                  foreground="gray").pack(side="right", padx=8)

        # Results: per-coil sections + coupling section.
        self.coil1_frame = self._make_coil_result_frame(1)
        self.coil2_frame = self._make_coil_result_frame(2)

        coup = ttk.LabelFrame(self, text="Coupling (2-port only)")
        coup.pack(fill="x", padx=8, pady=4)
        cg = ttk.Frame(coup); cg.pack(fill="x", padx=6, pady=6)
        self.res_M = tk.StringVar(value="—")
        self.res_k = tk.StringVar(value="—")
        self.res_Zmat = tk.StringVar(value="—")
        self._put(cg, 0, 0, "Mutual inductance M:", self.res_M)
        self._put(cg, 0, 2, "Coupling k:",          self.res_k)
        self._put(cg, 1, 0, "Z-matrix (Ω):",        self.res_Zmat)
        self.coupling_frame = coup

        # Legacy fields (single coil)
        extra = ttk.LabelFrame(self, text="Series LC resonance (Coil 1 only)")
        extra.pack(fill="x", padx=8, pady=4)
        eg = ttk.Frame(extra); eg.pack(fill="x", padx=6, pady=6)
        self.res_f0 = tk.StringVar(value="—")
        ttk.Label(eg, text="Capacitance (nF):").pack(side="left")
        self.cap_var = tk.StringVar(value="")
        ttk.Entry(eg, textvariable=self.cap_var,
                  width=10).pack(side="left", padx=4)
        ttk.Label(eg, text="  f₀:").pack(side="left", padx=(12, 0))
        ttk.Label(eg, textvariable=self.res_f0).pack(side="left", padx=4)
        self.cap_var.trace_add("write", lambda *_: self._update_f0())

    def _make_coil_result_frame(self, idx):
        frame = ttk.LabelFrame(self, text=f"Coil {idx} results")
        frame.pack(fill="x", padx=8, pady=4)
        grid = ttk.Frame(frame); grid.pack(fill="x", padx=6, pady=6)
        vars_ = {
            "L":      tk.StringVar(value="—"),
            "Rac":    tk.StringVar(value="—"),
            "Rdc":    tk.StringVar(value="—"),
            "length": tk.StringVar(value="—"),
            "Q":      tk.StringVar(value="—"),
            "ratio":  tk.StringVar(value="—"),
        }
        self._put(grid, 0, 0, "Inductance:",             vars_["L"])
        self._put(grid, 0, 2, "AC resistance (solver):", vars_["Rac"])
        self._put(grid, 1, 0, "DC resistance (30 °C):",  vars_["Rdc"])
        self._put(grid, 1, 2, "Total path length:",      vars_["length"])
        self._put(grid, 2, 0, "Q at target f:",          vars_["Q"])
        self._put(grid, 2, 2, "µH / mΩ ratio:",          vars_["ratio"])
        if idx == 1:
            self.res1 = vars_
        else:
            self.res2 = vars_
        return frame

    @staticmethod
    def _put(parent, r, c, label, var):
        ttk.Label(parent, text=label, font=("", 9, "bold")).grid(
            row=r, column=c, sticky="w", padx=4, pady=2)
        ttk.Label(parent, textvariable=var).grid(
            row=r, column=c+1, sticky="w", padx=4, pady=2)

    # -- coil source registration --
    def register_coil(self, coil_index, source_name, inp_path, metadata):
        idx = coil_index - 1
        self._sources[idx]   = source_name
        self._inp_paths[idx] = inp_path
        self._metadata[idx]  = metadata
        self._update_source_labels()

    def unregister_coil(self, coil_index, source_name):
        idx = coil_index - 1
        if self._sources[idx] == source_name:
            self._sources[idx]   = None
            self._inp_paths[idx] = None
            self._metadata[idx]  = None
            self._update_source_labels()

    def _update_source_labels(self):
        s1 = self._sources[0] or "None"
        s2 = self._sources[1] or "None"
        self.coil1_src_var.set(f"Coil 1: {s1}")
        self.coil2_src_var.set(f"Coil 2: {s2}")

    # -- target freq accessor --
    def get_target_freq(self):
        try:
            f = float(self.freq_var.get())
            return f if f > 0 else None
        except ValueError:
            return None

    # -- start/run --
    def _on_start(self):
        if not _RUNNER_OK:
            messagebox.showerror(
                "Simulation",
                "FastHenry runner unavailable (pywin32 + FastHenry2 "
                "required, Windows-only)."); return
        if self._sim_thread and self._sim_thread.is_alive():
            return

        target_f = self.get_target_freq()
        if target_f is None:
            messagebox.showerror("Error", "Invalid target frequency."); return

        n_sources = sum(1 for s in self._sources if s is not None)
        if n_sources == 0:
            messagebox.showwarning(
                "Simulation",
                "No coils registered. Use 'Send to Simulation' on a coil "
                "tab first."); return

        max_iter = self._parse_opt_int(self.maxiter_var.get(), "Max iter")
        if max_iter is False:
            return
        tol = self._parse_opt_float(self.tol_var.get(), "Tol")
        if tol is False:
            return

        # Build the .inp we're about to run.
        try:
            run_inp = self._prepare_run_inp()
        except Exception as e:
            messagebox.showerror("Simulation", f"Prepare failed: {e}")
            return

        self.start_btn.config(state="disabled")
        self.sim_status.set("Running…")
        self.elapsed_var.set("Elapsed: 00:00")
        self._sim_thread = threading.Thread(
            target=self._worker,
            args=(run_inp, max_iter, tol, target_f),
            daemon=True)
        self._sim_thread.start()

    def _parse_opt_int(self, s, label):
        if not s.strip():
            return None
        try:
            return int(s)
        except ValueError:
            messagebox.showerror("Error", f"{label} must be integer.")
            return False

    def _parse_opt_float(self, s, label):
        if not s.strip():
            return None
        try:
            return float(s)
        except ValueError:
            messagebox.showerror("Error", f"{label} must be numeric.")
            return False

    def _prepare_run_inp(self):
        """
        Single-port: copy the registered .inp to temp/run.inp and return.
        Two-port: combine via port_combiner and return the merged path.
        """
        run_path = os.path.join(TEMP_DIR, "run.inp")
        if self._sources[0] and self._sources[1]:
            port_combiner.combine_two_port(
                self._inp_paths[0], self._inp_paths[1], run_path)
        elif self._sources[0]:
            shutil.copyfile(self._inp_paths[0], run_path)
        elif self._sources[1]:
            shutil.copyfile(self._inp_paths[1], run_path)
        else:
            raise RuntimeError("No coil source registered")
        return run_path

    def _worker(self, inp_path, max_iter, tol, target_f):
        try:
            with runner.FastHenryRunner() as fh:
                def _tick(elapsed):
                    self._sim_queue.put(("tick", elapsed))
                ok = fh.run(inp_path, max_iter=max_iter, tol=tol,
                            progress_cb=_tick)
                if not ok:
                    self._sim_queue.put(("error", "Timed out.")); return

                zc_path = os.path.join(TEMP_DIR, "Zc.mat")
                fh.export_zc_mat(zc_path)

                blocks = zc_parser.parse_zc_mat(zc_path)
                if not blocks:
                    self._sim_queue.put(
                        ("error", "Zc.mat is empty or unparseable."))
                    return

                f_used, Zmat = zc_parser.matrix_at(blocks, target_f)
                n_ports = zc_parser.port_count(blocks)
                self._sim_queue.put(("done", {
                    "frequency": f_used,
                    "Zmat":      Zmat,
                    "n_ports":   n_ports,
                }))
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

    # -- results --
    def _on_done(self, result):
        self.last_result = result
        Zmat = result["Zmat"]
        f    = result["frequency"]
        n    = result["n_ports"]

        # Map matrix port index to our coil slot. If only coil 2 was
        # registered (coil 1 empty), port 0 is still coil 2.
        port_to_coil = [i for i in (0, 1) if self._sources[i] is not None]

        # Per-coil scalars.
        for p_idx in range(n):
            coil_slot = port_to_coil[p_idx] if p_idx < len(port_to_coil) else p_idx
            z = Zmat[p_idx][p_idx]
            self._fill_coil_result(coil_slot, z, f)

        # Clear the un-registered coil's panel.
        for slot in (0, 1):
            if slot not in port_to_coil[:n]:
                self._clear_coil_result(slot)

        # Coupling (2-port only)
        if n == 2:
            z11 = Zmat[0][0]; z22 = Zmat[1][1]; z12 = Zmat[0][1]
            L1 = z11.imag / (2 * 3.141592653589793 * f)
            L2 = z22.imag / (2 * 3.141592653589793 * f)
            M  = z12.imag / (2 * 3.141592653589793 * f)
            self.res_M.set(f"{M * 1e6:+.4f} µH")
            if L1 > 0 and L2 > 0:
                k = M / ((L1 * L2) ** 0.5)
                self.res_k.set(f"{k:+.4f}")
            else:
                self.res_k.set("—")
            self.res_Zmat.set(
                f"[{Zmat[0][0].real:.3f}+{Zmat[0][0].imag:.3f}j, "
                f"{Zmat[0][1].real:.3f}+{Zmat[0][1].imag:.3f}j; "
                f"{Zmat[1][0].real:.3f}+{Zmat[1][0].imag:.3f}j, "
                f"{Zmat[1][1].real:.3f}+{Zmat[1][1].imag:.3f}j]")
        else:
            self.res_M.set("—"); self.res_k.set("—"); self.res_Zmat.set("—")

        self._update_f0()
        self.sim_status.set(f"Done ({n}-port).")
        self.start_btn.config(state="normal")

    def _fill_coil_result(self, coil_slot, z_self, f):
        """Populate one coil-result panel from the diagonal Z value."""
        vars_ = self.res1 if coil_slot == 0 else self.res2
        meta  = self._metadata[coil_slot]

        import math as _m
        L_h  = z_self.imag / (2 * _m.pi * f)
        R_ac = z_self.real
        vars_["L"].set(f"{L_h * 1e6:.4f} µH")
        vars_["Rac"].set(f"{R_ac * 1000:.2f} mΩ  (@ {f/1000:.1f} kHz)")
        Q = analysis.q_factor(L_h, R_ac, f)
        vars_["Q"].set(f"{Q:.2f}")

        R_dc = self._compute_coil_dcr(meta) if meta else 0.0
        vars_["Rdc"].set(f"{R_dc * 1000:.2f} mΩ" if R_dc > 0 else "—")

        length_mm = self._compute_coil_length(meta) if meta else 0.0
        vars_["length"].set(f"{length_mm:.1f} mm" if length_mm > 0 else "—")

        ratio = analysis.quality_ratio_uh_per_mohm(L_h, R_dc)
        vars_["ratio"].set(f"{ratio:.3f}" if R_dc > 0 else "—")

    def _clear_coil_result(self, coil_slot):
        vars_ = self.res1 if coil_slot == 0 else self.res2
        for v in vars_.values():
            v.set("—")

    @staticmethod
    def _compute_coil_dcr(meta):
        """
        Topology-aware DC resistance.
        - series / single: sum of per-segment R
        - parallel: 1 / sum(1/R_k)  across independent layers
        """
        topology = meta.get("topology", "single")
        layer_params = meta.get("layer_params", [])
        if not layer_params:
            return 0.0

        if topology == "parallel":
            nodes_by_layer = meta.get("nodes_by_layer", [])
            resistances = []
            for (w, h, _n), lnodes in zip(layer_params, nodes_by_layer):
                seg_len = analysis.path_length_mm(lnodes)
                r = analysis.dc_resistance_ohm(seg_len, w, h, temp_c=30.0)
                if r > 0:
                    resistances.append(r)
            if not resistances:
                return 0.0
            return 1.0 / sum(1.0 / r for r in resistances)

        # series / single: flat node list with per-layer counts
        nodes = meta.get("nodes", [])
        if not nodes:
            return 0.0
        total = 0.0
        offset = 0
        for w, h, n in layer_params:
            seg = nodes[offset:offset + n]
            seg_len = analysis.path_length_mm(seg)
            total += analysis.dc_resistance_ohm(seg_len, w, h, temp_c=30.0)
            offset += n
        return total

    @staticmethod
    def _compute_coil_length(meta):
        """
        Characteristic trace length:
        - series / single: length of the concatenated path
        - parallel: length of ONE layer's path (same spiral across layers)
        """
        topology = meta.get("topology", "single")
        if topology == "parallel":
            lbl = meta.get("nodes_by_layer", [])
            if not lbl:
                return 0.0
            return analysis.path_length_mm(lbl[0])
        return analysis.path_length_mm(meta.get("nodes", []))

    def _update_f0(self):
        # f0 tied to Coil 1's inductance only.
        if self.last_result is None:
            self.res_f0.set("—"); return
        Zmat = self.last_result["Zmat"]
        f    = self.last_result["frequency"]
        # If coil 1 wasn't registered, skip.
        if self._sources[0] is None:
            self.res_f0.set("—"); return
        # Coil 1 port index = 0 in the Z matrix (always, because we register
        # coil 1 → first .external).
        import math as _m
        L1 = Zmat[0][0].imag / (2 * _m.pi * f)

        txt = self.cap_var.get().strip()
        if not txt:
            self.res_f0.set("—"); return
        try:
            c_nf = float(txt)
        except ValueError:
            self.res_f0.set("—"); return
        if c_nf <= 0 or L1 <= 0:
            self.res_f0.set("—"); return
        f0 = analysis.series_resonant_freq_hz(L1, c_nf * 1e-9)
        self.res_f0.set(f"{f0 / 1000:.3f} kHz")

    # -- reset --
    def reset_this_tab(self):
        # Clear registrations (but DON'T delete the .inp files the source
        # tabs are managing — they'll handle their own cleanup).
        self._sources = [None, None]
        self._inp_paths = [None, None]
        self._metadata = [None, None]
        self._update_source_labels()

        self.last_result = None
        self.elapsed_var.set("Elapsed: —")
        self.sim_status.set("Not started")
        self.freq_var.set(self.DEFAULT_FREQ)
        self.maxiter_var.set("")
        self.tol_var.set("")
        self.cap_var.set("")
        self._clear_coil_result(0); self._clear_coil_result(1)
        self.res_M.set("—"); self.res_k.set("—"); self.res_Zmat.set("—")
        self.res_f0.set("—")

        # Remove our transient run artifacts.
        for f in (os.path.join(TEMP_DIR, "run.inp"),
                  os.path.join(TEMP_DIR, "Zc.mat")):
            _safe_remove(f)


# =========================================================================
#  Placeholder tab
# =========================================================================
class PlaceholderTab(ttk.Frame):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=text, foreground="gray",
                  font=("", 12)).pack(expand=True)


# =========================================================================
#  CoilApp
# =========================================================================
class CoilApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coil2Inductor")
        self.geometry("1400x760")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Sim tab built first so other tabs can call get_target_freq()
        # during their own init.
        self.sim_tab   = SimTab(nb, app=self)
        self.coil1_tab = CoilTab(nb, app=self, coil_index=1)
        # Coil 2 is still a placeholder until its DXF flow is wired up.
        self.coil2_tab = PlaceholderTab(
            nb, "DXF Coil 2 — coming soon.\n\n"
            "Use Parametric Coil 2 as a second coil source for now.")
        self.param1_tab = ParametricCoilTab(
            nb, app=self, coil_index=1, temp_dir=TEMP_DIR)
        self.param2_tab = ParametricCoilTab(
            nb, app=self, coil_index=2, temp_dir=TEMP_DIR)
        self.auto_tab = PlaceholderTab(
            nb, "Automation — coming soon.\n\n"
            "Parametric sweeps: vary turns, layers, spacing, trace width.\n"
            "Drives parametric_coil + port_combiner to batch-run FastHenry.")

        # Order: both DXF coils, both Parametric coils, Simulation, Automation.
        nb.add(self.coil1_tab,  text="  DXF Coil 1  ")
        nb.add(self.coil2_tab,  text="  DXF Coil 2  ")
        nb.add(self.param1_tab, text="  Parametric Coil 1  ")
        nb.add(self.param2_tab, text="  Parametric Coil 2  ")
        nb.add(self.sim_tab,    text="  Simulation  ")
        nb.add(self.auto_tab,   text="  Automation  ")

        self._nb = nb

    def on_coil_dxf_state_changed(self, coil_tab):
        """When a DXF tab gains/loses its DXF, grey the paired parametric tab."""
        loaded = coil_tab.any_dxf_loaded()
        param = self._paired_param_tab(coil_tab)
        if param is None:
            return
        param.set_enabled(not loaded)
        try:
            self._nb.tab(param, state=("disabled" if loaded else "normal"))
        except tk.TclError:
            pass

    def _paired_param_tab(self, coil_tab):
        if coil_tab is self.coil1_tab:
            return self.param1_tab
        # Wire the second pairing when coil2_tab becomes a real CoilTab:
        #   if coil_tab is self.coil2_tab: return self.param2_tab
        return None


def main():
    app = CoilApp()
    app.mainloop()


if __name__ == "__main__":
    main()