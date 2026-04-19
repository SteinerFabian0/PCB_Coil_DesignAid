"""DXF-backed single-layer panel used inside CoilTab."""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Make Modules/ and this Tabs/ dir importable when loaded standalone.
_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
for _p in (_modules, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dxf_to_inp_converter as d2i
import inp_visualizer as viz
from paths import TEMP_DIR
from gui_utils import oz_to_mm, figure_to_photo


class LayerPanel(ttk.LabelFrame):
    def __init__(self, parent, tag, coil_index, layer_index,
                 on_state_change=None, on_send_single=None, **kw):
        super().__init__(parent, text=tag, **kw)
        self.tag = tag
        self.coil_index = coil_index
        self.layer_index = layer_index
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
        ttk.Button(r0, text="Browse…", command=self._browse).pack(side="right")
        ttk.Button(r0, text="Clear", command=self._clear).pack(side="right",
                                                               padx=(0, 4))

        r1 = ttk.Frame(self); r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="Width mm:").pack(side="left")
        self.w_var = tk.StringVar(value="0.52")
        ttk.Entry(r1, textvariable=self.w_var, width=7).pack(side="left", padx=4)
        ttk.Label(r1, text="Cu oz:").pack(side="left", padx=(8, 0))
        self.oz_var = tk.StringVar(value="1.0")
        ttk.Entry(r1, textvariable=self.oz_var, width=5).pack(side="left", padx=4)

        r2 = ttk.Frame(self); r2.pack(fill="x", padx=6, pady=2)
        ttk.Button(r2, text="Export INP", width=12,
                   command=self._export_inp).pack(side="left")
        ttk.Button(r2, text="Send to Simulation", width=20,
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
        if not p:
            return
        self.load_dxf(p, warn_on_bad_units=True)

    def load_dxf(self, p, warn_on_bad_units=False):
        if not os.path.exists(p):
            return False
        try:
            if not d2i.check_dxf_is_mm(p) and warn_on_bad_units:
                if not messagebox.askyesno(
                        "Units", "DXF units are not mm. Proceed anyway?"):
                    return False
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
        return True

    def _quick_preview(self):
        w, h = self.get_params()
        tag_safe = self._tag_safe()
        tmp_inp = os.path.join(TEMP_DIR, f"{tag_safe}_quick.inp")
        nodes = d2i.convert_dxf_to_inp(
            self.dxf_path, tmp_inp, w=w, h=h, fmin=130000, fmax=145000)
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

    def _tag_safe(self):
        return f"coil{self.coil_index}_layer{self.layer_index}"

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
            raise RuntimeError(f"Only {len(self.nodes)} node(s) — check DXF segments")
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
        self._photo = figure_to_photo(fig, png_path, max_width=360)
        ttk.Label(self.plot_frame, image=self._photo).pack(fill="both", expand=True)

    def _export_inp(self):
        import shutil
        if not self.inp_path or not self.dxf_path:
            messagebox.showwarning("Export", "Convert first."); return
        base = os.path.splitext(os.path.basename(self.dxf_path))[0]
        dest = os.path.join(os.path.dirname(self.dxf_path), base + ".inp")
        shutil.copyfile(self.inp_path, dest)
        messagebox.showinfo("Export", f"Wrote:\n{dest}")

    def _send_to_sim(self):
        if self.inp_path is None or self.nodes is None:
            messagebox.showwarning("Send to Sim", "Load a DXF first."); return
        if self._on_send_single is None:
            return
        try:
            w, h = self.get_params()
        except ValueError as e:
            messagebox.showerror("Send to Sim", str(e)); return
        self._on_send_single(self, w, h)

    def save_state(self):
        return {"dxf_path": self.dxf_path,
                "width_mm": self.w_var.get(),
                "copper_oz": self.oz_var.get()}

    def load_state(self, d):
        if not d:
            return
        if "width_mm" in d:  self.w_var.set(str(d["width_mm"]))
        if "copper_oz" in d: self.oz_var.set(str(d["copper_oz"]))
        p = d.get("dxf_path")
        if p and os.path.exists(p):
            self.load_dxf(p, warn_on_bad_units=False)