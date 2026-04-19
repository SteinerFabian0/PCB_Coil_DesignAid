#!/usr/bin/env python3
"""
SimTab — solver execution + results + system insights.
Layout: TX column left, RX column right, shared globals below.
"""

import os, sys, math, queue, shutil, threading
import tkinter as tk
from tkinter import ttk, messagebox

_here = os.path.dirname(os.path.abspath(__file__))
_modules = os.path.join(os.path.dirname(_here), "Modules")
if _modules not in sys.path:
    sys.path.insert(0, _modules)

import zc_parser
import port_combiner
import coil_analysis as analysis
import cap_combinator

try:
    import fasthenry_runner as runner
    _RUNNER_OK = True
except Exception:
    runner = None
    _RUNNER_OK = False

ROLE_TO_IDX = {"TX": 0, "RX": 1}


class CapField(ttk.Frame):
    def __init__(self, parent, label, on_value_change=None, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=label, width=14, anchor="w").pack(side="left")
        self.var = tk.StringVar(value="")
        ttk.Entry(self, textvariable=self.var, width=9).pack(
            side="left", padx=2)
        ttk.Button(self, text="Find Cap Combo", width=16,
                   command=self._snap).pack(side="left", padx=(4, 2))
        self.combo_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.combo_var,
                  foreground="gray").pack(side="left", padx=4)
        self._on_change = on_value_change
        self.var.trace_add("write",
                           lambda *_a: self._on_change() if self._on_change else None)

    def _snap(self):
        s = self.var.get().strip()
        if not s: return
        try: val = float(s)
        except ValueError:
            messagebox.showerror("Cap", "Numeric nF value required."); return
        if val <= 0:
            messagebox.showerror("Cap", "Must be > 0."); return
        achieved, desc = cap_combinator.find_best_cap(val)
        if desc is None:
            self.combo_var.set(f"(single {achieved:g} nF)")
            return
        self.var.set(f"{achieved:g}")
        self.combo_var.set(f"({desc})")

    def get_value_nf(self):
        s = self.var.get().strip()
        if not s: return None
        try: return float(s)
        except ValueError: return None


class SimTab(ttk.Frame):
    POLL_MS = 200
    DEFAULT_FREQ = "130000"

    def __init__(self, parent, app, temp_dir, **kw):
        super().__init__(parent, **kw)
        self.app = app
        self.temp_dir = temp_dir
        self.last_result = None
        self._sim_thread = None
        self._sim_queue = queue.Queue()
        self._sources   = [None, None]
        self._inp_paths = [None, None]
        self._metadata  = [None, None]
        self._build()
        self.after(self.POLL_MS, self._poll)
        self._update_source_labels()
        self.after(150, self._restore_from_savestate)

    # ---- layout ---------------------------------------------------------
    def _build(self):
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Row 1: Coil sources + 2-port geometry (same row).
        row1 = ttk.Frame(body); row1.pack(fill="x", padx=8, pady=(8, 4))
        src = ttk.LabelFrame(row1, text="Coil sources")
        src.pack(side="left", fill="both", expand=True)
        self.tx_src_var = tk.StringVar(value="TX: None")
        self.rx_src_var = tk.StringVar(value="RX: None")
        ttk.Label(src, textvariable=self.tx_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Label(src, textvariable=self.rx_src_var,
                  font=("", 10, "bold")).pack(anchor="w", padx=8, pady=(0, 4))

        self.geom_frame = ttk.LabelFrame(row1, text="2-port geometry")
        gf = ttk.Frame(self.geom_frame); gf.pack(fill="x", padx=6, pady=6)
        ttk.Label(gf, text="PCB gap TX-top→RX-bottom (mm):").pack(side="left")
        self.pcb_gap_var = tk.StringVar(value="2.5")
        ttk.Entry(gf, textvariable=self.pcb_gap_var, width=8).pack(
            side="left", padx=4)
        self.pcb_gap_var.trace_add("write", lambda *_a: self._save_state())
        self._update_gap_visibility()

        # Row 2: solver settings.
        sf = ttk.LabelFrame(body, text="FastHenry solver settings")
        sf.pack(fill="x", padx=8, pady=4)
        sr = ttk.Frame(sf); sr.pack(fill="x", padx=6, pady=6)
        ttk.Label(sr, text="Target f (Hz):").pack(side="left")
        self.freq_var = tk.StringVar(value=self.DEFAULT_FREQ)
        ttk.Entry(sr, textvariable=self.freq_var, width=10).pack(
            side="left", padx=4)
        ttk.Label(sr, text="Max iter:").pack(side="left", padx=(12, 0))
        self.maxiter_var = tk.StringVar(value="")
        ttk.Entry(sr, textvariable=self.maxiter_var, width=6).pack(
            side="left", padx=4)
        ttk.Label(sr, text="Tol:").pack(side="left", padx=(12, 0))
        self.tol_var = tk.StringVar(value="")
        ttk.Entry(sr, textvariable=self.tol_var, width=8).pack(
            side="left", padx=4)
        for v in (self.freq_var, self.maxiter_var, self.tol_var):
            v.trace_add("write", lambda *_a: self._save_state())

        # Row 3: run controls.
        rf = ttk.Frame(body); rf.pack(fill="x", padx=8, pady=4)
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

        # -------------- Two-column TX / RX split ----------------------
        split = ttk.Frame(body); split.pack(fill="x", padx=8, pady=4)
        tx_col = ttk.Frame(split); tx_col.pack(side="left", fill="both",
                                                expand=True)
        # Subtle vertical separator.
        sep = tk.Frame(split, bg="#b0b0b0", width=2)
        sep.pack(side="left", fill="y", padx=4)
        rx_col = ttk.Frame(split); rx_col.pack(side="left", fill="both",
                                                expand=True)

        # TX column: inputs, then results.
        tx_in = ttk.LabelFrame(tx_col, text="TX inputs")
        tx_in.pack(fill="x", pady=(0, 4))
        self.cap_tx = CapField(tx_in, "C_TX (nF):",
                               on_value_change=self._update_insights)
        self.cap_tx.pack(fill="x", padx=4, pady=2)
        r = ttk.Frame(tx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="V_supply min (V):",
                  width=18, anchor="w").pack(side="left")
        self.v_min_var = tk.StringVar(value="3.3")
        ttk.Entry(r, textvariable=self.v_min_var, width=8).pack(
            side="left", padx=2)
        r = ttk.Frame(tx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="V_supply max (V):",
                  width=18, anchor="w").pack(side="left")
        self.v_max_var = tk.StringVar(value="4.2")
        ttk.Entry(r, textvariable=self.v_max_var, width=8).pack(
            side="left", padx=2)
        r = ttk.Frame(tx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="Envelope duty %:",
                  width=18, anchor="w").pack(side="left")
        self.duty_var = tk.StringVar(value="50")
        ttk.Entry(r, textvariable=self.duty_var, width=8).pack(
            side="left", padx=2)
        r = ttk.Frame(tx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="Drive center f (Hz):",
                  width=18, anchor="w").pack(side="left")
        self.fc_var = tk.StringVar(value="130000")
        ttk.Entry(r, textvariable=self.fc_var, width=10).pack(
            side="left", padx=2)

        self.tx_frame, self.res_tx = self._make_coil_result_frame(tx_col, "TX")

        # RX column: inputs, then results.
        rx_in = ttk.LabelFrame(rx_col, text="RX inputs")
        rx_in.pack(fill="x", pady=(0, 4))
        self.cap_rx = CapField(rx_in, "C_RX (nF):",
                               on_value_change=self._update_insights)
        self.cap_rx.pack(fill="x", padx=4, pady=2)
        r = ttk.Frame(rx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="Avg P consumption (W):",
                  width=22, anchor="w").pack(side="left")
        self.p_avg_var = tk.StringVar(value="0.1")
        ttk.Entry(r, textvariable=self.p_avg_var, width=8).pack(
            side="left", padx=2)
        r = ttk.Frame(rx_in); r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="Min V_rectified (V):",
                  width=22, anchor="w").pack(side="left")
        self.v_rect_min_var = tk.StringVar(value="3.4")
        ttk.Entry(r, textvariable=self.v_rect_min_var, width=8).pack(
            side="left", padx=2)

        self.rx_frame, self.res_rx = self._make_coil_result_frame(rx_col, "RX")

        # Trace all the user-fillable insight inputs.
        for v in (self.v_min_var, self.v_max_var, self.duty_var,
                  self.fc_var, self.p_avg_var, self.v_rect_min_var):
            v.trace_add("write",
                        lambda *_a: (self._update_insights(),
                                     self._save_state()))

        # ---- Shared block: coupling + derived ----------------------------
        coup = ttk.LabelFrame(body, text="Coupling (2-port only)")
        coup.pack(fill="x", padx=8, pady=4)
        cg = ttk.Frame(coup); cg.pack(fill="x", padx=6, pady=6)
        self.res_M = tk.StringVar(value="—")
        self.res_k = tk.StringVar(value="—")
        self.res_Zmat = tk.StringVar(value="—")
        self._put(cg, 0, 0, "Mutual inductance M:", self.res_M)
        self._put(cg, 0, 2, "Coupling k:",          self.res_k)
        self._put(cg, 1, 0, "Z-matrix (Ω):",        self.res_Zmat)

        ins = ttk.LabelFrame(body, text="Derived system insights")
        ins.pack(fill="x", padx=8, pady=4)
        self.ins_vars = {k: tk.StringVar(value="—") for k in (
            "f0_tx", "f0_rx", "df_tx", "df_rx",
            "z_tx_drive", "p_on", "p_avg", "q_tx_rx", "k",
            "p_rect_on", "v_rect_peak", "goal_status")}
        og = ttk.Frame(ins); og.pack(fill="x", padx=6, pady=4)
        items = [
            ("f₀ TX:",               "f0_tx"),
            ("Δf TX (fc-f0):",       "df_tx"),
            ("f₀ RX:",               "f0_rx"),
            ("Δf RX (target-f0):",   "df_rx"),
            ("|Z_TX| at drive:",     "z_tx_drive"),
            ("Coupling k:",          "k"),
            ("Q_TX / Q_RX:",         "q_tx_rx"),
            ("P_in ON  (Vmin→max):", "p_on"),
            ("P_in avg (Vmin→max):", "p_avg"),
            ("P_rect ON (Vmin→max):","p_rect_on"),
            ("V_rect pk (Vmin→max):","v_rect_peak"),
        ]
        for i, (lbl, key) in enumerate(items):
            r, c = divmod(i, 2)
            ttk.Label(og, text=lbl, font=("", 9, "bold")).grid(
                row=r, column=c*2, sticky="w", padx=4, pady=2)
            ttk.Label(og, textvariable=self.ins_vars[key]).grid(
                row=r, column=c*2+1, sticky="w", padx=4, pady=2)
        ttk.Label(ins, textvariable=self.ins_vars["goal_status"],
                  foreground="#206020",
                  font=("", 9, "bold")).pack(anchor="w", padx=6, pady=(0, 4))

    def _make_coil_result_frame(self, parent, role):
        frame = ttk.LabelFrame(parent, text=f"{role} coil results")
        frame.pack(fill="x", pady=(0, 4))
        grid = ttk.Frame(frame); grid.pack(fill="x", padx=6, pady=6)
        vars_ = {k: tk.StringVar(value="—")
                 for k in ("L", "Rac", "Rdc", "length", "Q", "ratio")}
        self._put(grid, 0, 0, "Inductance:",             vars_["L"])
        self._put(grid, 1, 0, "AC resistance (solver):", vars_["Rac"])
        self._put(grid, 2, 0, "DC resistance (30 °C):",  vars_["Rdc"])
        self._put(grid, 3, 0, "Total path length:",      vars_["length"])
        self._put(grid, 4, 0, "Q at target f:",          vars_["Q"])
        self._put(grid, 5, 0, "µH / mΩ ratio:",          vars_["ratio"])
        return frame, vars_

    @staticmethod
    def _put(parent, r, c, label, var):
        ttk.Label(parent, text=label, font=("", 9, "bold")).grid(
            row=r, column=c, sticky="w", padx=4, pady=2)
        ttk.Label(parent, textvariable=var).grid(
            row=r, column=c+1, sticky="w", padx=4, pady=2)

    # ---- registration ---------------------------------------------------
    def register_coil(self, role, source_name, inp_path, metadata):
        idx = ROLE_TO_IDX[role]
        self._sources[idx]   = source_name
        self._inp_paths[idx] = inp_path
        self._metadata[idx]  = metadata
        self._update_source_labels()
        self._update_gap_visibility()

    def unregister_coil(self, role, source_name):
        idx = ROLE_TO_IDX[role]
        if self._sources[idx] == source_name:
            self._sources[idx]   = None
            self._inp_paths[idx] = None
            self._metadata[idx]  = None
            self._update_source_labels()
            self._update_gap_visibility()
            self.app.on_sim_slot_cleared(role, source_name)

    def _update_source_labels(self):
        self.tx_src_var.set(f"TX: {self._sources[0] or 'None'}")
        self.rx_src_var.set(f"RX: {self._sources[1] or 'None'}")

    def _update_gap_visibility(self):
        if self._sources[0] and self._sources[1]:
            self.geom_frame.pack(side="left", fill="both", expand=True,
                                 padx=(8, 0))
        else:
            self.geom_frame.pack_forget()

    def get_target_freq(self):
        try:
            f = float(self.freq_var.get())
            return f if f > 0 else None
        except ValueError:
            return None

    # ---- run ------------------------------------------------------------
    def _on_start(self):
        if not _RUNNER_OK:
            messagebox.showerror("Simulation",
                                 "FastHenry runner unavailable."); return
        if self._sim_thread and self._sim_thread.is_alive(): return
        target_f = self.get_target_freq()
        if target_f is None:
            messagebox.showerror("Error", "Invalid target frequency."); return
        if sum(1 for s in self._sources if s is not None) == 0:
            messagebox.showwarning("Simulation",
                                   "No coils registered."); return
        try: run_inp = self._prepare_run_inp()
        except Exception as e:
            messagebox.showerror("Simulation", f"Prepare failed: {e}"); return
        max_iter = self._parse_opt_int(self.maxiter_var.get(), "Max iter")
        if max_iter is False: return
        tol = self._parse_opt_float(self.tol_var.get(), "Tol")
        if tol is False: return

        self.start_btn.config(state="disabled")
        self.sim_status.set("Running…")
        self.elapsed_var.set("Elapsed: 00:00")
        self._sim_thread = threading.Thread(
            target=self._worker,
            args=(run_inp, max_iter, tol, target_f),
            daemon=True)
        self._sim_thread.start()

    def _parse_opt_int(self, s, label):
        if not s.strip(): return None
        try: return int(s)
        except ValueError:
            messagebox.showerror("Error", f"{label} must be integer.")
            return False

    def _parse_opt_float(self, s, label):
        if not s.strip(): return None
        try: return float(s)
        except ValueError:
            messagebox.showerror("Error", f"{label} must be numeric.")
            return False

    def _prepare_run_inp(self):
        run_path = os.path.join(self.temp_dir, "run.inp")
        if self._sources[0] and self._sources[1]:
            try: gap = float(self.pcb_gap_var.get())
            except ValueError:
                raise RuntimeError("PCB gap must be numeric.")
            if gap < 0: raise RuntimeError("PCB gap must be >= 0.")
            port_combiner.combine_two_port(
                self._inp_paths[0], self._inp_paths[1], run_path,
                pcb_gap_mm=gap)
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
                ok = fh.run(inp_path, max_iter=max_iter, tol=tol,
                            progress_cb=lambda e: self._sim_queue.put(("tick", e)))
                if not ok:
                    self._sim_queue.put(("error", "Timed out.")); return
                zc_path = os.path.join(self.temp_dir, "Zc.mat")
                fh.export_zc_mat(zc_path)
                blocks = zc_parser.parse_zc_mat(zc_path)
                if not blocks:
                    self._sim_queue.put(("error", "Zc.mat empty.")); return
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
                elif kind == "done": self._on_done(payload)
                elif kind == "error":
                    messagebox.showerror("Simulation", payload)
                    self.sim_status.set("Error")
                    self.start_btn.config(state="normal")
        except queue.Empty: pass
        finally: self.after(self.POLL_MS, self._poll)

    def _on_done(self, result):
        self.last_result = result
        Zmat = result["Zmat"]; f = result["frequency"]; n = result["n_ports"]
        port_to_role_idx = [i for i in (0, 1) if self._sources[i] is not None]
        for p_idx in range(n):
            slot = port_to_role_idx[p_idx] if p_idx < len(port_to_role_idx) else p_idx
            self._fill_coil_result(slot, Zmat[p_idx][p_idx], f)
        for slot in (0, 1):
            if slot not in port_to_role_idx[:n]:
                self._clear_coil_result(slot)

        if n == 2:
            z11, z22, z12 = Zmat[0][0], Zmat[1][1], Zmat[0][1]
            w = 2 * math.pi * f
            L1, L2, M = z11.imag/w, z22.imag/w, z12.imag/w
            self.res_M.set(f"{M*1e6:+.4f} µH")
            if L1 > 0 and L2 > 0:
                self.res_k.set(f"{M/math.sqrt(L1*L2):+.4f}")
            else: self.res_k.set("—")
            self.res_Zmat.set(
                f"[{Zmat[0][0].real:.3f}+{Zmat[0][0].imag:.3f}j, "
                f"{Zmat[0][1].real:.3f}+{Zmat[0][1].imag:.3f}j; "
                f"{Zmat[1][0].real:.3f}+{Zmat[1][0].imag:.3f}j, "
                f"{Zmat[1][1].real:.3f}+{Zmat[1][1].imag:.3f}j]")
        else:
            self.res_M.set("—"); self.res_k.set("—"); self.res_Zmat.set("—")
        self.sim_status.set(f"Done ({n}-port).")
        self.start_btn.config(state="normal")
        self._update_insights()

    def _fill_coil_result(self, slot, z_self, f):
        vars_ = self.res_tx if slot == 0 else self.res_rx
        meta  = self._metadata[slot]
        L_h = z_self.imag / (2 * math.pi * f)
        R_ac = z_self.real
        vars_["L"].set(f"{L_h*1e6:.4f} µH")
        vars_["Rac"].set(f"{R_ac*1000:.2f} mΩ (@ {f/1000:.1f} kHz)")
        vars_["Q"].set(f"{analysis.q_factor(L_h, R_ac, f):.2f}")
        R_dc = self._compute_coil_dcr(meta) if meta else 0.0
        vars_["Rdc"].set(f"{R_dc*1000:.2f} mΩ" if R_dc > 0 else "—")
        length_mm = self._compute_coil_length(meta) if meta else 0.0
        vars_["length"].set(f"{length_mm:.1f} mm" if length_mm > 0 else "—")
        ratio = analysis.quality_ratio_uh_per_mohm(L_h, R_dc)
        vars_["ratio"].set(f"{ratio:.3f}" if R_dc > 0 else "—")

    def _clear_coil_result(self, slot):
        vars_ = self.res_tx if slot == 0 else self.res_rx
        for v in vars_.values(): v.set("—")

    @staticmethod
    def _compute_coil_dcr(meta):
        topology = meta.get("topology", "single")
        layer_params = meta.get("layer_params", [])
        if not layer_params: return 0.0
        if topology == "parallel":
            nbl = meta.get("nodes_by_layer", [])
            rs = []
            for (w, h, _), lnodes in zip(layer_params, nbl):
                r = analysis.dc_resistance_ohm(
                    analysis.path_length_mm(lnodes), w, h, 30.0)
                if r > 0: rs.append(r)
            return 1.0/sum(1.0/r for r in rs) if rs else 0.0
        if topology in ("series_pairs_par", "parallel_pairs_ser"):
            nbl = meta.get("nodes_by_layer", [])
            if len(layer_params) != 4 or len(nbl) != 4: return 0.0
            def rl(i):
                w, h, _ = layer_params[i]
                return analysis.dc_resistance_ohm(
                    analysis.path_length_mm(nbl[i]), w, h, 30.0)
            R = [rl(i) for i in range(4)]
            if topology == "series_pairs_par":
                ra = R[0]+R[1]; rb = R[2]+R[3]
                return 1.0/(1.0/ra + 1.0/rb) if ra>0 and rb>0 else 0.0
            ra = 1.0/(1.0/R[0] + 1.0/R[1]) if R[0]>0 and R[1]>0 else 0.0
            rb = 1.0/(1.0/R[2] + 1.0/R[3]) if R[2]>0 and R[3]>0 else 0.0
            return ra + rb
        # series / single:
        nodes = meta.get("nodes", [])
        if not nodes: return 0.0
        total = 0.0; off = 0
        for w, h, n in layer_params:
            total += analysis.dc_resistance_ohm(
                analysis.path_length_mm(nodes[off:off+n]), w, h, 30.0)
            off += n
        return total

    @staticmethod
    def _compute_coil_length(meta):
        topology = meta.get("topology", "single")
        if topology in ("parallel", "series_pairs_par", "parallel_pairs_ser"):
            nbl = meta.get("nodes_by_layer", [])
            return analysis.path_length_mm(nbl[0]) if nbl else 0.0
        return analysis.path_length_mm(meta.get("nodes", []))

    # ---- insights (unchanged logic, just smaller display) --------------
    def _update_insights(self):
        g = self.ins_vars
        for k in g: g[k].set("—")
        g["goal_status"].set("")
        c_tx_nf = self.cap_tx.get_value_nf()
        c_rx_nf = self.cap_rx.get_value_nf()
        try:
            fc = float(self.fc_var.get())
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            duty_pct = float(self.duty_var.get())
            p_avg_target = float(self.p_avg_var.get())
            v_rect_min_needed = float(self.v_rect_min_var.get())
        except ValueError:
            return
        duty = max(0.0, min(1.0, duty_pct / 100.0))

        L_tx = R_tx = L_rx = R_rx = M = None
        if self.last_result is not None:
            Zmat = self.last_result["Zmat"]
            f = self.last_result["frequency"]
            w = 2 * math.pi * f
            n = self.last_result["n_ports"]
            port_to_role_idx = [i for i in (0, 1) if self._sources[i] is not None]
            for p_idx in range(n):
                slot = port_to_role_idx[p_idx] if p_idx < len(port_to_role_idx) else p_idx
                z = Zmat[p_idx][p_idx]
                if slot == 0: L_tx, R_tx = z.imag/w, z.real
                else:         L_rx, R_rx = z.imag/w, z.real
            if n == 2: M = Zmat[0][1].imag / w

        if L_tx is not None and c_tx_nf and c_tx_nf > 0:
            f0_tx = analysis.series_resonant_freq_hz(L_tx, c_tx_nf*1e-9)
            g["f0_tx"].set(f"{f0_tx/1000:.3f} kHz")
            g["df_tx"].set(f"{(fc - f0_tx)/1000:+.3f} kHz")
        if L_rx is not None and c_rx_nf and c_rx_nf > 0:
            f0_rx = analysis.series_resonant_freq_hz(L_rx, c_rx_nf*1e-9)
            g["f0_rx"].set(f"{f0_rx/1000:.3f} kHz")
            target_f = self.get_target_freq() or fc
            g["df_rx"].set(f"{(target_f - f0_rx)/1000:+.3f} kHz")

        if L_tx is not None and R_tx and R_tx > 0:
            qtx = analysis.q_factor(L_tx, R_tx, fc)
            qrx = analysis.q_factor(L_rx, R_rx, fc) if L_rx and R_rx else None
            g["q_tx_rx"].set(f"{qtx:.1f} / "
                             f"{qrx:.1f}" if qrx is not None else f"{qtx:.1f} / —")

        if None in (L_tx, R_tx, L_rx, R_rx, M) or not c_tx_nf or not c_rx_nf:
            return
        if duty > 0 and p_avg_target > 0 and v_rect_min_needed > 0:
            p_on_needed = p_avg_target / duty
            r_load_guess = (v_rect_min_needed ** 2) / p_on_needed
        else:
            r_load_guess = 1000.0
        res = analysis.tx_rx_system_analysis(
            L_tx=L_tx, R_tx=R_tx, L_rx=L_rx, R_rx=R_rx, M=M, f_hz=fc,
            C_tx=c_tx_nf*1e-9, C_rx=c_rx_nf*1e-9,
            R_load=r_load_guess,
            V_supply_min=v_min, V_supply_max=v_max,
            duty_fraction=duty)
        g["k"].set(f"{analysis.coupling_k(L_tx, L_rx, M):+.4f}")
        Zt = res["Z_tx_total"]
        g["z_tx_drive"].set(
            f"{abs(Zt):.2f} Ω  "
            f"(arg {math.degrees(math.atan2(Zt.imag, Zt.real)):+.1f}°)")
        g["p_on"].set(f"{res['P_in_on_Vmin']*1000:.1f} → "
                      f"{res['P_in_on_Vmax']*1000:.1f} mW")
        g["p_avg"].set(f"{res['P_in_avg_Vmin']*1000:.1f} → "
                       f"{res['P_in_avg_Vmax']*1000:.1f} mW")
        g["p_rect_on"].set(f"{res['P_rect_on_Vmin']*1000:.1f} → "
                           f"{res['P_rect_on_Vmax']*1000:.1f} mW")
        g["v_rect_peak"].set(f"{res['V_rect_peak_Vmin']:.2f} → "
                             f"{res['V_rect_peak_Vmax']:.2f} V")
        goal_ok = (res['P_rect_avg_Vmin'] >= p_avg_target
                   and res['V_rect_peak_Vmin'] >= v_rect_min_needed)
        if goal_ok:
            g["goal_status"].set("✓ Target P and V_rect met at V_supply_min.")
        else:
            sp = max(0.0, p_avg_target - res['P_rect_avg_Vmin'])
            sv = max(0.0, v_rect_min_needed - res['V_rect_peak_Vmin'])
            g["goal_status"].set(
                f"✗ At V_supply_min: short P by {sp*1000:.1f} mW, "
                f"V by {sv:.2f} V.")

    # ---- reset / savestate ---------------------------------------------
    def reset_this_tab(self):
        self._sources = [None, None]; self._inp_paths = [None, None]
        self._metadata = [None, None]
        self._update_source_labels(); self._update_gap_visibility()
        self.last_result = None
        self.elapsed_var.set("Elapsed: —")
        self.sim_status.set("Not started")
        self.freq_var.set(self.DEFAULT_FREQ)
        self.maxiter_var.set(""); self.tol_var.set("")
        self.pcb_gap_var.set("2.5")
        self.cap_tx.var.set(""); self.cap_tx.combo_var.set("")
        self.cap_rx.var.set(""); self.cap_rx.combo_var.set("")
        self._clear_coil_result(0); self._clear_coil_result(1)
        self.res_M.set("—"); self.res_k.set("—"); self.res_Zmat.set("—")
        for v in self.ins_vars.values(): v.set("—")
        for fn in (os.path.join(self.temp_dir, "run.inp"),
                   os.path.join(self.temp_dir, "Zc.mat")):
            try:
                if os.path.exists(fn): os.remove(fn)
            except Exception: pass
        self._save_state()
        self.app.on_sim_tab_reset()

    def to_state(self):
        return {
            "target_f":     self.freq_var.get(),
            "max_iter":     self.maxiter_var.get(),
            "tol":          self.tol_var.get(),
            "pcb_gap":      self.pcb_gap_var.get(),
            "c_tx":         self.cap_tx.var.get(),
            "c_rx":         self.cap_rx.var.get(),
            "v_min":        self.v_min_var.get(),
            "v_max":        self.v_max_var.get(),
            "duty":         self.duty_var.get(),
            "fc":           self.fc_var.get(),
            "p_avg":        self.p_avg_var.get(),
            "v_rect_min":   self.v_rect_min_var.get(),
        }

    def _restore_from_savestate(self):
        st = self.app.load_sim_tab_state()
        if not st: return
        try:
            self.freq_var.set(st.get("target_f", self.DEFAULT_FREQ))
            self.maxiter_var.set(st.get("max_iter", ""))
            self.tol_var.set(st.get("tol", ""))
            self.pcb_gap_var.set(st.get("pcb_gap", "2.5"))
            self.cap_tx.var.set(st.get("c_tx", ""))
            self.cap_rx.var.set(st.get("c_rx", ""))
            self.v_min_var.set(st.get("v_min", "3.3"))
            self.v_max_var.set(st.get("v_max", "4.2"))
            self.duty_var.set(st.get("duty", "50"))
            self.fc_var.set(st.get("fc", "130000"))
            self.p_avg_var.set(st.get("p_avg", "0.1"))
            self.v_rect_min_var.set(st.get("v_rect_min", "3.4"))
        except Exception:
            pass

    def _save_state(self):
        self.app.persist_sim_tab(self.to_state())