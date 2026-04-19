"""Simulation tab: solver launch, results, 2-port geometry.

Updates in this revision:
- Coil sources frame + 2-port geometry frame tiled side-by-side; 2-port
  geometry only shown when both slots are loaded.
- Result labels flip red when a new simulation is launched (invalidation
  cue), flip back to default color when results arrive.
"""

import os
import sys
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
for _p in (_modules, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coil_analysis as analysis
import zc_parser
import port_combiner
from paths import TEMP_DIR
from gui_utils import safe_remove, inp_z_bounds

try:
    import fasthenry_runner as runner
    _RUNNER_OK = True
except Exception:
    runner = None
    _RUNNER_OK = False


class SimTab(ttk.Frame):
    POLL_MS = 200
    DEFAULT_FREQ        = "130000"
    DEFAULT_COIL_GAP_MM = "2.6"

    _STALE_FG  = "#c01010"   # results-invalidated red
    _NORMAL_FG = "#000000"   # results-valid default

    def __init__(self, parent, app, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.last_result = None
        self._sim_thread = None
        self._sim_queue = queue.Queue()
        self._result_labels = []   # every ttk.Label we want to recolor on start

        self._sources   = [None, None]
        self._inp_paths = [None, None]
        self._metadata  = [None, None]
        self._on_unregister_cb = [None, None]

        self._build()
        self.after(self.POLL_MS, self._poll)
        self._update_source_labels()

    def _build(self):
        # --- Top row: Coil sources (L) + 2-port geometry (R) side-by-side.
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 4))
        top.grid_columnconfigure(0, weight=1, uniform="top")
        top.grid_columnconfigure(1, weight=1, uniform="top")

        src = ttk.LabelFrame(top, text="Coil sources")
        src.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.coil1_src_var = tk.StringVar(value="Coil 1: None")
        self.coil2_src_var = tk.StringVar(value="Coil 2: None")
        ttk.Label(src, textvariable=self.coil1_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Label(src, textvariable=self.coil2_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(0, 4))

        self.geom_frame = ttk.LabelFrame(top, text="2-port geometry")
        self.geom_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        gr = ttk.Frame(self.geom_frame); gr.pack(fill="x", padx=6, pady=6)
        ttk.Label(gr,
                  text="Coil 1 top → Coil 2 bottom gap (mm):").pack(side="left")
        self.coil_gap_var = tk.StringVar(value=self.DEFAULT_COIL_GAP_MM)
        ttk.Entry(gr, textvariable=self.coil_gap_var,
                  width=8).pack(side="left", padx=4)
        ttk.Label(self.geom_frame,
                  text="Coil 2 sits above Coil 1.",
                  foreground="gray").pack(anchor="w", padx=8, pady=(0, 4))
        self.geom_frame.grid_remove()  # hidden until both slots are loaded

        # --- Solver settings
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

        # --- Run / reset row
        rf = ttk.Frame(self); rf.pack(fill="x", padx=8, pady=4)
        self.start_btn = ttk.Button(rf, text="Start simulation", width=20,
                                     command=self._on_start)
        self.start_btn.pack(side="left")
        self.elapsed_var = tk.StringVar(value="Elapsed: —")
        ttk.Label(rf, textvariable=self.elapsed_var).pack(side="left", padx=12)
        ttk.Button(rf, text="Reset This Tab", width=16,
                   command=self.reset_this_tab).pack(side="right")
        self.sim_status = tk.StringVar(value="Not started")
        ttk.Label(rf, textvariable=self.sim_status,
                  foreground="gray").pack(side="right", padx=8)

        # --- Result frames
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

        extra = ttk.LabelFrame(self, text="Series LC resonance (Coil 1 only)")
        extra.pack(fill="x", padx=8, pady=4)
        eg = ttk.Frame(extra); eg.pack(fill="x", padx=6, pady=6)
        self.res_f0 = tk.StringVar(value="—")
        ttk.Label(eg, text="Capacitance (nF):").pack(side="left")
        self.cap_var = tk.StringVar(value="")
        ttk.Entry(eg, textvariable=self.cap_var,
                  width=10).pack(side="left", padx=4)
        ttk.Label(eg, text="  f₀:").pack(side="left", padx=(12, 0))
        self._put_single(eg, self.res_f0)    # track this label too
        self.cap_var.trace_add("write", lambda *_: self._update_f0())

    def _make_coil_result_frame(self, idx):
        frame = ttk.LabelFrame(self, text=f"Coil {idx} results")
        frame.pack(fill="x", padx=8, pady=4)
        grid = ttk.Frame(frame); grid.pack(fill="x", padx=6, pady=6)
        vars_ = {k: tk.StringVar(value="—")
                 for k in ("L", "Rac", "Rdc", "length", "Q", "ratio")}
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

    def _put(self, parent, r, c, label_text, var):
        """Labelled value cell; value label is registered for stale-recoloring."""
        ttk.Label(parent, text=label_text, font=("", 9, "bold")).grid(
            row=r, column=c, sticky="w", padx=4, pady=2)
        val = ttk.Label(parent, textvariable=var)
        val.grid(row=r, column=c+1, sticky="w", padx=4, pady=2)
        self._result_labels.append(val)

    def _put_single(self, parent, var):
        val = ttk.Label(parent, textvariable=var)
        val.pack(side="left", padx=4)
        self._result_labels.append(val)

    # -- coil-source registration --
    def register_coil(self, coil_index, source_name, inp_path, metadata,
                      on_unregister=None):
        idx = coil_index - 1
        old_cb = self._on_unregister_cb[idx]
        old_src = self._sources[idx]
        if old_cb is not None and old_src != source_name:
            try: old_cb()
            except Exception: pass
        self._sources[idx]         = source_name
        self._inp_paths[idx]       = inp_path
        self._metadata[idx]        = metadata
        self._on_unregister_cb[idx] = on_unregister
        self._update_source_labels()

    def unregister_coil(self, coil_index, source_name):
        idx = coil_index - 1
        if self._sources[idx] == source_name:
            cb = self._on_unregister_cb[idx]
            self._sources[idx]          = None
            self._inp_paths[idx]        = None
            self._metadata[idx]         = None
            self._on_unregister_cb[idx] = None
            if cb is not None:
                try: cb()
                except Exception: pass
            self._update_source_labels()

    def _update_source_labels(self):
        s1 = self._sources[0] or "None"
        s2 = self._sources[1] or "None"
        self.coil1_src_var.set(f"Coil 1: {s1}")
        self.coil2_src_var.set(f"Coil 2: {s2}")
        both = self._sources[0] is not None and self._sources[1] is not None
        try:
            (self.geom_frame.grid() if both else self.geom_frame.grid_remove())
        except tk.TclError:
            pass

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
                "FastHenry runner unavailable (pywin32 + FastHenry2 required, "
                "Windows-only)."); return
        if self._sim_thread and self._sim_thread.is_alive():
            return

        target_f = self.get_target_freq()
        if target_f is None:
            messagebox.showerror("Error", "Invalid target frequency."); return

        n_sources = sum(1 for s in self._sources if s is not None)
        if n_sources == 0:
            messagebox.showwarning("Simulation",
                "No coils registered. Use 'Send to Simulation' on a coil tab first.")
            return

        max_iter = self._parse_opt_int(self.maxiter_var.get(), "Max iter")
        if max_iter is False:
            return
        tol = self._parse_opt_float(self.tol_var.get(), "Tol")
        if tol is False:
            return

        try:
            run_inp = self._prepare_run_inp()
        except Exception as e:
            messagebox.showerror("Simulation", f"Prepare failed: {e}"); return

        # Mark old results as about-to-be-overwritten.
        if self.last_result is not None:
            self._mark_results_stale()

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
            v = int(s)
            if v <= 0:
                raise ValueError()
            return v
        except ValueError:
            messagebox.showerror("Error", f"{label} must be a positive integer.")
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
        run_path = os.path.join(TEMP_DIR, "run.inp")
        both = self._sources[0] and self._sources[1]
        if both:
            gap = self._parse_coil_gap_mm()
            z1_min, z1_max = inp_z_bounds(self._inp_paths[0])
            z2_min, _      = inp_z_bounds(self._inp_paths[1])
            z_offset = (z1_max + gap) - z2_min
            port_combiner.combine_two_port(
                self._inp_paths[0], self._inp_paths[1], run_path,
                z_offset_coil2=z_offset)
        elif self._sources[0]:
            shutil.copyfile(self._inp_paths[0], run_path)
        elif self._sources[1]:
            shutil.copyfile(self._inp_paths[1], run_path)
        else:
            raise RuntimeError("No coil source registered")
        return run_path

    def _parse_coil_gap_mm(self):
        try:
            g = float(self.coil_gap_var.get())
        except ValueError:
            raise ValueError("Coil gap must be numeric")
        if g <= 0:
            raise ValueError("Coil gap must be > 0")
        return g

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
                    self._sim_queue.put(("error", "Zc.mat is empty or unparseable."))
                    return
                f_used, Zmat = zc_parser.matrix_at(blocks, target_f)
                n_ports = zc_parser.port_count(blocks)
                self._sim_queue.put(("done", {"frequency": f_used,
                                               "Zmat": Zmat,
                                               "n_ports": n_ports}))
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

    def _on_done(self, result):
        self.last_result = result
        Zmat = result["Zmat"]; f = result["frequency"]; n = result["n_ports"]
        port_to_coil = [i for i in (0, 1) if self._sources[i] is not None]

        for p_idx in range(n):
            slot = port_to_coil[p_idx] if p_idx < len(port_to_coil) else p_idx
            self._fill_coil_result(slot, Zmat[p_idx][p_idx], f)
        for slot in (0, 1):
            if slot not in port_to_coil[:n]:
                self._clear_coil_result(slot)

        if n == 2:
            import math as _m
            z11 = Zmat[0][0]; z22 = Zmat[1][1]; z12 = Zmat[0][1]
            L1 = z11.imag / (2 * _m.pi * f)
            L2 = z22.imag / (2 * _m.pi * f)
            M  = z12.imag / (2 * _m.pi * f)
            self.res_M.set(f"{M * 1e6:+.4f} µH")
            if L1 > 0 and L2 > 0:
                self.res_k.set(f"{M / ((L1 * L2) ** 0.5):+.4f}")
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
        self._mark_results_fresh()
        self.sim_status.set(f"Done ({n}-port).")
        self.start_btn.config(state="normal")

    def _fill_coil_result(self, coil_slot, z_self, f):
        vars_ = self.res1 if coil_slot == 0 else self.res2
        meta  = self._metadata[coil_slot]
        import math as _m
        L_h  = z_self.imag / (2 * _m.pi * f)
        R_ac = z_self.real
        vars_["L"].set(f"{L_h * 1e6:.4f} µH")
        vars_["Rac"].set(f"{R_ac * 1000:.2f} mΩ  (@ {f/1000:.1f} kHz)")
        vars_["Q"].set(f"{analysis.q_factor(L_h, R_ac, f):.2f}")
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
        topology = meta.get("topology", "single")
        layer_params = meta.get("layer_params", [])
        if not layer_params:
            return 0.0
        if topology == "parallel":
            nbl = meta.get("nodes_by_layer", [])
            resistances = []
            for (w, h, _n), lnodes in zip(layer_params, nbl):
                seg_len = analysis.path_length_mm(lnodes)
                r = analysis.dc_resistance_ohm(seg_len, w, h, temp_c=30.0)
                if r > 0:
                    resistances.append(r)
            if not resistances:
                return 0.0
            return 1.0 / sum(1.0 / r for r in resistances)
        # single / series: accept either 'nodes' (flat) or 'nodes_by_layer'.
        nodes = meta.get("nodes")
        if not nodes:
            nbl = meta.get("nodes_by_layer", [])
            nodes = [pt for layer in nbl for pt in layer]
        if not nodes:
            return 0.0
        total, offset = 0.0, 0
        for w, h, n in layer_params:
            seg = nodes[offset:offset + n]
            if seg:
                total += analysis.dc_resistance_ohm(
                    analysis.path_length_mm(seg), w, h, temp_c=30.0)
            offset += n
        return total

    @staticmethod
    def _compute_coil_length(meta):
        topology = meta.get("topology", "single")
        if topology == "parallel":
            lbl = meta.get("nodes_by_layer", [])
            return analysis.path_length_mm(lbl[0]) if lbl else 0.0
        nodes = meta.get("nodes")
        if not nodes:
            nbl = meta.get("nodes_by_layer", [])
            nodes = [pt for layer in nbl for pt in layer]
        return analysis.path_length_mm(nodes) if nodes else 0.0

    def _update_f0(self):
        if self.last_result is None:
            self.res_f0.set("—"); return
        Zmat = self.last_result["Zmat"]; f = self.last_result["frequency"]
        if self._sources[0] is None:
            self.res_f0.set("—"); return
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

    # -- stale / fresh recolouring --
    def _mark_results_stale(self):
        for lbl in self._result_labels:
            try: lbl.configure(foreground=self._STALE_FG)
            except tk.TclError: pass

    def _mark_results_fresh(self):
        for lbl in self._result_labels:
            try: lbl.configure(foreground=self._NORMAL_FG)
            except tk.TclError: pass

    def reset_this_tab(self):
        for idx in range(2):
            cb = self._on_unregister_cb[idx]
            if cb is not None:
                try: cb()
                except Exception: pass
        self._sources = [None, None]
        self._inp_paths = [None, None]
        self._metadata = [None, None]
        self._on_unregister_cb = [None, None]
        self._update_source_labels()

        self.last_result = None
        self.elapsed_var.set("Elapsed: —")
        self.sim_status.set("Not started")
        self.freq_var.set(self.DEFAULT_FREQ)
        self.maxiter_var.set("")
        self.tol_var.set("")
        self.cap_var.set("")
        self.coil_gap_var.set(self.DEFAULT_COIL_GAP_MM)
        self._clear_coil_result(0); self._clear_coil_result(1)
        self.res_M.set("—"); self.res_k.set("—"); self.res_Zmat.set("—")
        self.res_f0.set("—")
        self._mark_results_fresh()

        for f in (os.path.join(TEMP_DIR, "run.inp"),
                  os.path.join(TEMP_DIR, "Zc.mat")):
            safe_remove(f)

    def save_state(self):
        return {
            "target_freq":     self.freq_var.get(),
            "max_iter":        self.maxiter_var.get(),
            "tol":             self.tol_var.get(),
            "capacitance_nf":  self.cap_var.get(),
            "coil_gap_mm":     self.coil_gap_var.get(),
        }

    def load_state(self, d):
        if not d:
            return
        if "target_freq" in d:    self.freq_var.set(str(d["target_freq"]))
        if "max_iter" in d:       self.maxiter_var.set(str(d["max_iter"]))
        if "tol" in d:            self.tol_var.set(str(d["tol"]))
        if "capacitance_nf" in d: self.cap_var.set(str(d["capacitance_nf"]))
        if "coil_gap_mm" in d:    self.coil_gap_var.set(str(d["coil_gap_mm"]))