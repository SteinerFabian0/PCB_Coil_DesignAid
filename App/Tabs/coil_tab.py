"""DXF-based coil tab (1 or 2 layers, optionally merged via via)."""

import os
import sys
import glob
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
for _p in (_modules, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inp_visualizer as viz
import coil_merger as merger
from paths import TEMP_DIR
from gui_utils import figure_to_photo, safe_remove
from layer_panel import LayerPanel


MAX_LAYERS = 2


class CoilTab(ttk.Frame):
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
        top = ttk.Frame(self); top.pack(fill="both", expand=True, padx=4, pady=4)

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

        bot = ttk.Frame(self); bot.pack(fill="x", padx=8, pady=4)
        ttk.Label(bot, text="Layer spacing mm:").pack(side="left")
        self.spacing_var = tk.StringVar(value="1.4")
        ttk.Entry(bot, textvariable=self.spacing_var, width=6).pack(side="left", padx=4)
        ttk.Label(bot, text="nwinc:").pack(side="left", padx=(12, 0))
        self.nwinc_var = tk.StringVar(value="3")
        ttk.Entry(bot, textvariable=self.nwinc_var, width=4).pack(side="left", padx=4)
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

    def any_dxf_loaded(self):
        return any(p.dxf_path for p in self.panels)

    def _on_layer_dxf_change(self):
        self.app.on_coil_dxf_state_changed(self)

    def convert_and_merge(self):
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Error", "Set target frequency on Simulation tab first.")
            return
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
                messagebox.showerror("Error", "Layer spacing must be numeric."); return
            if spacing <= 0:
                messagebox.showerror("Error", "Spacing must be > 0 for two layers."); return

        self.status_var.set("Converting…"); self.update_idletasks()

        try:
            for p in active:
                p.convert(fmin=fmin, fmax=fmax, nwinc=nwinc)
                p.render_preview(
                    title=p.tag,
                    png_name=f"{p._tag_safe()}_preview.png",
                    highlight={"start": "N0", "end": f"N{len(p.nodes)-1}"})
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
            self.combined_highlight = {"start": "N0", "end": f"N{len(p0.nodes)-1}"}
            self._render_combined("xy", "Single layer")
            self.status_var.set("Single-layer ready.")
            return

        p0, p1 = active[0], active[1]
        match = merger.find_via_match(p0.nodes, p1.nodes)
        if match is None:
            messagebox.showerror("Via match",
                f"No matching endpoints within {merger.DEFAULT_VIA_MATCH_TOL_MM} mm.")
            self.status_var.set("Via match failed."); return

        _, _, via_dist = match
        a_ori, b_ori = merger.orient_layers(p0.nodes, p1.nodes, match)

        try:
            w1, h1 = p0.get_params()
            w2, h2 = p1.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e)); return

        self.combined_inp = os.path.join(TEMP_DIR, f"coil{self.coil_index}_combined.inp")
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
        png = os.path.join(TEMP_DIR, f"coil{self.coil_index}_combined_preview.png")
        self._combined_photo = figure_to_photo(fig, png, max_width=360)
        ttk.Label(self.combined_frame,
                  image=self._combined_photo).pack(fill="both", expand=True)

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
            messagebox.showwarning("Send to Sim", "Convert & Merge first."); return
        topology = "series" if len(self.layer_params) > 1 else "single"
        meta = {"topology": topology,
                "layer_params": self.layer_params,
                "nodes": list(self.combined_nodes or [])}
        self.app.sim_tab.register_coil(
            self.coil_index, "DXF Import", self.combined_inp, meta,
            on_unregister=self.clear_send_status)
        self.status_var.set(f"Sent as Coil {self.coil_index}.")

    def _send_layer_to_sim(self, panel, w, h):
        inp_dest = os.path.join(
            TEMP_DIR, f"coil{self.coil_index}_layer{panel.layer_index}_sent.inp")
        try:
            shutil.copyfile(panel.inp_path, inp_dest)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Copy failed: {e}"); return
        meta = {"topology": "single",
                "layer_params": [(w, h, len(panel.nodes))],
                "nodes": list(panel.nodes)}
        self.app.sim_tab.register_coil(
            self.coil_index, "DXF Import", inp_dest, meta,
            on_unregister=self.clear_send_status)
        self.status_var.set(
            f"Sent Layer {panel.layer_index} as Coil {self.coil_index}.")

    def clear_send_status(self):
        self.status_var.set("Idle")

    def reset_this_tab(self):
        for p in self.panels:
            p._clear()
        prefix = f"coil{self.coil_index}_"
        for f in glob.glob(os.path.join(TEMP_DIR, prefix + "*")):
            safe_remove(f)
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
        self.app.sim_tab.unregister_coil(self.coil_index, "DXF Import")

    def save_state(self):
        return {"type": "dxf",
                "layers": [p.save_state() for p in self.panels],
                "spacing_mm": self.spacing_var.get(),
                "nwinc": self.nwinc_var.get()}

    def load_state(self, d):
        if not d:
            return
        if "spacing_mm" in d: self.spacing_var.set(str(d["spacing_mm"]))
        if "nwinc" in d:      self.nwinc_var.set(str(d["nwinc"]))
        for panel, pd in zip(self.panels, d.get("layers", [])):
            panel.load_state(pd)