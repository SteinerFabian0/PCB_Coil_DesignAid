#!/usr/bin/env python3
"""
DXF coil tab (TX or RX). One tab class reused for both roles.
"""

import os, sys, glob, shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
if _modules not in sys.path:
    sys.path.insert(0, _modules)

import dxf_to_inp_converter as d2i
import inp_visualizer as viz
import coil_merger as merger

MAX_LAYERS = 2
OZ_TO_MM = 0.035

_STATUS_COLORS = {
    "nothing":  ("Nothing loaded in Sim",        "#c01010"),
    "current":  ("Current Coil loaded in Sim",   "#1a8020"),
    "outdated": ("Outdated Coil loaded in Sim",  "#c06010"),
}


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
        if os.path.exists(path): os.remove(path)
    except Exception: pass


class LayerPanel(ttk.LabelFrame):
    def __init__(self, parent, tag, role, layer_index, temp_dir,
                 on_state_change=None, on_send_single=None, **kw):
        super().__init__(parent, text=tag, **kw)
        self.tag = tag
        self.role = role
        self.layer_index = layer_index
        self.temp_dir = temp_dir
        self.dxf_path = None
        self.inp_path = None
        self.nodes = None
        self._photo = None
        self._on_state_change = on_state_change
        self._on_send_single  = on_send_single

        r0 = ttk.Frame(self); r0.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(r0, text="DXF:").pack(side="left")
        self.path_var = tk.StringVar(value="(none)")
        ttk.Label(r0, textvariable=self.path_var, foreground="gray",
                  width=18, anchor="w").pack(side="left", padx=4,
                                              fill="x", expand=True)
        ttk.Button(r0, text="Browse…", command=self._browse
                   ).pack(side="right")
        ttk.Button(r0, text="Clear", command=self._clear
                   ).pack(side="right", padx=(0, 4))

        r1 = ttk.Frame(self); r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="Width mm:").pack(side="left")
        self.w_var = tk.StringVar(value="0.52")
        ttk.Entry(r1, textvariable=self.w_var, width=7
                  ).pack(side="left", padx=4)
        ttk.Label(r1, text="Cu oz:").pack(side="left", padx=(8, 0))
        self.oz_var = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.oz_var, width=5
                  ).pack(side="left", padx=4)

        r2 = ttk.Frame(self); r2.pack(fill="x", padx=6, pady=2)
        ttk.Button(r2, text="Export INP", width=12,
                   command=self._export_inp).pack(side="left")
        ttk.Button(r2, text=f"Send to Sim ({role})", width=18,
                   command=self._send_to_sim).pack(side="left", padx=(6, 0))

        r3 = ttk.Frame(self); r3.pack(fill="x", padx=6, pady=2)
        self.status_var = tk.StringVar(value="—")
        ttk.Label(r3, textvariable=self.status_var,
                  foreground="gray").pack(side="right")

        self.plot_frame = ttk.Frame(self, height=200)
        self.plot_frame.pack(fill="both", expand=True, padx=6, pady=6)

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select DXF",
            filetypes=[("DXF files", "*.dxf"), ("All", "*.*")])
        if not p: return
        try:
            if not d2i.check_dxf_is_mm(p):
                if not messagebox.askyesno(
                        "Units", "DXF units are not mm. Proceed anyway?"):
                    return
        except Exception: pass
        self.dxf_path = p
        self.path_var.set(os.path.basename(p))
        self.status_var.set("Loaded")
        if self._on_state_change: self._on_state_change()
        try: self._quick_preview()
        except Exception as e: self.status_var.set(f"Preview error: {e}")

    def _quick_preview(self):
        w, h = self.get_params()
        tag_safe = self._tag_safe()
        tmp_inp = os.path.join(self.temp_dir, f"{tag_safe}_quick.inp")
        nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, tmp_inp, w=w, h=h, fmin=130000, fmax=145000)
        if len(nodes) < 2:
            self.status_var.set("Too few nodes"); return
        self.nodes = nodes; self.inp_path = tmp_inp
        self.status_var.set(f"{len(nodes)} nodes")
        self.render_preview(self.tag, f"{tag_safe}_preview.png",
                            highlight={"start": "N0",
                                       "end": f"N{len(nodes)-1}"})

    def _clear(self):
        self.dxf_path = self.inp_path = self.nodes = self._photo = None
        self.path_var.set("(none)"); self.status_var.set("—")
        for c in self.plot_frame.winfo_children(): c.destroy()
        if self._on_state_change: self._on_state_change()

    def _tag_safe(self):
        return f"dxf_{self.role.lower()}_layer{self.layer_index}"

    def get_params(self):
        w = float(self.w_var.get())
        h = OZ_TO_MM * float(self.oz_var.get())
        if w <= 0 or h <= 0: raise ValueError("width/oz must be > 0")
        return w, h

    def convert(self, fmin, fmax, nwinc=3, nhinc=1):
        if not self.dxf_path: return False
        w, h = self.get_params()
        tag_safe = self._tag_safe()
        self.inp_path = os.path.join(self.temp_dir, f"{tag_safe}.inp")
        self.nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, self.inp_path, w=w, h=h,
            nwinc=nwinc, nhinc=nhinc, fmin=fmin, fmax=fmax)
        if len(self.nodes) < 2:
            raise RuntimeError(f"Only {len(self.nodes)} node(s)")
        self.status_var.set(f"Converted ({len(self.nodes)} nodes)")
        return True

    def render_preview(self, title, png_name, highlight=None):
        if self.inp_path is None: return
        for c in self.plot_frame.winfo_children(): c.destroy()
        fig, _, _ = viz.build_figure(
            self.inp_path, highlight=highlight, title=title,
            view="xy", figsize=(3.2, 2.8), dpi=90)
        png_path = os.path.join(self.temp_dir, png_name)
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
            messagebox.showwarning("Send to Sim", "Load a DXF first."); return
        if self._on_send_single is None: return
        try: w, h = self.get_params()
        except ValueError as e:
            messagebox.showerror("Send to Sim", str(e)); return
        self._on_send_single(self, w, h)


class DxfCoilTab(ttk.Frame):
    def __init__(self, parent, app, role, temp_dir,
                 on_next_tab=None, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.role = role
        self.temp_dir = temp_dir
        self._on_next_tab = on_next_tab
        self.combined_inp = None
        self.combined_nodes = None
        self.combined_highlight = None
        self.layer_params = []
        self.via_index = None
        self._combined_photo = None
        self._build()
        self.after(150, self._restore_from_savestate)

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="both", expand=True,
                                         padx=4, pady=4)
        self.layers_frame = ttk.Frame(top)
        self.layers_frame.pack(side="left", fill="both", expand=True)
        self.panels = []
        for i in range(MAX_LAYERS):
            opt = "" if i == 0 else " (optional)"
            p = LayerPanel(self.layers_frame,
                           tag=f"Layer {i+1}{opt}",
                           role=self.role, layer_index=i+1,
                           temp_dir=self.temp_dir,
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
        ttk.Entry(bot, textvariable=self.spacing_var, width=6
                  ).pack(side="left", padx=4)
        ttk.Label(bot, text="nwinc:").pack(side="left", padx=(12, 0))
        self.nwinc_var = tk.StringVar(value="3")
        ttk.Entry(bot, textvariable=self.nwinc_var, width=4
                  ).pack(side="left", padx=4)

        ttk.Button(bot, text="Convert & Merge", width=18,
                   command=self.convert_and_merge).pack(side="left", padx=12)
        ttk.Button(bot, text=f"Send Combined to Sim ({self.role})", width=26,
                   command=self._send_combined_to_sim).pack(side="left")
        ttk.Button(bot, text="Export combined INP", width=20,
                   command=self._export_combined
                   ).pack(side="left", padx=(8, 0))
        ttk.Button(bot, text="Next Tab →", width=12,
                   command=lambda: self._on_next_tab and self._on_next_tab()
                   ).pack(side="left", padx=(8, 0))
        ttk.Button(bot, text="Reset This Tab", width=16,
                   command=self.reset_this_tab).pack(side="right")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bot, textvariable=self.status_var,
                  foreground="gray").pack(side="right", padx=8)

        # Sim-loaded status strip.
        self.sim_status_label = tk.Label(
            self, text="", fg=_STATUS_COLORS["nothing"][1], anchor="w")
        self.sim_status_label.pack(fill="x", padx=8)
        self._set_sim_status("nothing")

    def _set_sim_status(self, key):
        text, color = _STATUS_COLORS[key]
        self.sim_status_label.configure(text=text, fg=color)
        self._sim_status_key = key

    def _mark_sim_outdated_if_loaded(self):
        if getattr(self, "_sim_status_key", "nothing") == "current":
            self._set_sim_status("outdated")

    def on_sim_slot_cleared(self):
        self._set_sim_status("nothing")

    def any_dxf_loaded(self):
        return any(p.dxf_path for p in self.panels)

    def _on_layer_dxf_change(self):
        self.app.on_coil_dxf_state_changed(self)
        self._mark_sim_outdated_if_loaded()

    def convert_and_merge(self):
        fmin = self.app.sim_tab.get_target_freq()
        if fmin is None:
            messagebox.showerror("Error",
                                 "Set target frequency in Simulation tab.")
            return
        fmax = fmin + 15000.0
        try: nwinc = max(1, int(self.nwinc_var.get()))
        except ValueError: nwinc = 3
        active = [p for p in self.panels if p.dxf_path is not None]
        if not active:
            messagebox.showerror("Error", "Load at least one DXF."); return
        two_layer = len(active) >= 2
        spacing = 0.0
        if two_layer:
            try: spacing = float(self.spacing_var.get())
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
                p.render_preview(p.tag, f"{p._tag_safe()}_preview.png",
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
            self.combined_highlight = {"start": "N0",
                                       "end": f"N{len(p0.nodes)-1}"}
            self._render_combined("xy", "Single layer")
            self.status_var.set("Single-layer ready."); return

        p0, p1 = active[0], active[1]
        match = merger.find_via_match(p0.nodes, p1.nodes)
        if match is None:
            messagebox.showerror("Via match",
                                 "No matching endpoints within tolerance.")
            self.status_var.set("Via match failed."); return
        _, _, via_dist = match
        a_ori, b_ori = merger.orient_layers(p0.nodes, p1.nodes, match)
        try:
            w1, h1 = p0.get_params(); w2, h2 = p1.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error"); return

        self.combined_inp = os.path.join(
            self.temp_dir, f"dxf_{self.role.lower()}_combined.inp")
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
        self.combined_highlight = {"start": f"N{indices['start_idx']}",
                                   "via":   f"N{indices['via_idx']}",
                                   "end":   f"N{indices['end_idx']}"}
        self._render_combined("iso", "Combined (layer overlay)")
        self.status_var.set(f"Merged. Via gap: {via_dist*1000:.1f} µm.")
        self._mark_sim_outdated_if_loaded()

    def _render_combined(self, view, title):
        for c in self.combined_frame.winfo_children(): c.destroy()
        fig, _, _ = viz.build_figure(
            self.combined_inp, highlight=self.combined_highlight,
            title=title, view=view, figsize=(3.2, 2.8), dpi=90)
        png = os.path.join(self.temp_dir,
                           f"dxf_{self.role.lower()}_combined_preview.png")
        self._combined_photo = _figure_to_photo(fig, png, max_width=360)
        ttk.Label(self.combined_frame,
                  image=self._combined_photo).pack(fill="both", expand=True)

    def _export_combined(self):
        if not self.combined_inp:
            messagebox.showwarning("Export", "Convert & Merge first."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".inp",
            filetypes=[("FastHenry INP", "*.inp")],
            initialfile=f"dxf_{self.role.lower()}_combined.inp")
        if p: shutil.copyfile(self.combined_inp, p)

    def _send_combined_to_sim(self):
        if self.combined_inp is None:
            messagebox.showwarning("Send to Sim", "Convert & Merge first."); return
        topology = "series" if len(self.layer_params) > 1 else "single"
        meta = {"role": self.role, "topology": topology,
                "layer_params": self.layer_params,
                "nodes": list(self.combined_nodes or [])}
        self.app.sim_tab.register_coil(
            self.role, "DXF Import", self.combined_inp, meta)
        self.status_var.set(f"Sent as {self.role}.")
        self._set_sim_status("current")

    def _send_layer_to_sim(self, panel, w, h):
        inp_dest = os.path.join(
            self.temp_dir,
            f"dxf_{self.role.lower()}_layer{panel.layer_index}_sent.inp")
        try: shutil.copyfile(panel.inp_path, inp_dest)
        except Exception as e:
            messagebox.showerror("Send to Sim", f"Copy failed: {e}"); return
        meta = {"role": self.role, "topology": "single",
                "layer_params": [(w, h, len(panel.nodes))],
                "nodes": list(panel.nodes)}
        self.app.sim_tab.register_coil(self.role, "DXF Import", inp_dest, meta)
        self.status_var.set(f"Sent Layer {panel.layer_index}.")
        self._set_sim_status("current")

    def reset_this_tab(self):
        for p in self.panels: p._clear()
        prefix = f"dxf_{self.role.lower()}_"
        for f in glob.glob(os.path.join(self.temp_dir, prefix + "*")):
            _safe_remove(f)
        self.combined_inp = None; self.combined_nodes = None
        self.combined_highlight = None; self.layer_params = []
        self.via_index = None
        self.spacing_var.set("1.4"); self.nwinc_var.set("3")
        for c in self.combined_frame.winfo_children(): c.destroy()
        self.status_var.set("Reset.")
        self.app.sim_tab.unregister_coil(self.role, "DXF Import")
        self._set_sim_status("nothing")

    # ---- savestate ------------------------------------------------------
    def _restore_from_savestate(self):
        st = self.app.load_dxf_tab_state(self.role)
        if not st: return
        try:
            self.spacing_var.set(st.get("spacing", "1.4"))
            self.nwinc_var.set(st.get("nwinc", "3"))
            for i, panel in enumerate(self.panels):
                key = f"layer{i}"
                ld = st.get(key, {})
                panel.w_var.set(ld.get("w", "0.52"))
                panel.oz_var.set(ld.get("oz", "1.0"))
        except Exception:
            pass