#!/usr/bin/env python3
"""
NNAnalysisTab — power-efficiency analysis of bulk NN simulation results.

Physics kernel (vectorized):
  • R_ac(f) = DCR + R_skin(f0)·√(f/f0)   — skin effect scaling
  • C_tx = 1/(ω²L_tx)                      — series resonance
  • C_rx = 1/(ω²L_rx)                      — parallel resonance
  • U = k·√(Q_tx·Q_rx)                     — figure of merit
  • η_max = U²/(1+√(1+U²))²               — max achievable efficiency
  • Z_tx_opt = R_tx·√(1+U²)               — TX impedance at optimal load
  • P_rx_on = η_max·V_rms²/Z_tx_opt       — ON-state RX power
  • D = P_target / P_rx_on                  — required duty cycle

Frequency sweep: evaluates N_FREQ points in [f_min, f_max] from RAM data.
Ranking: highest η_max over all frequencies where the combo is feasible.
"""

import os, sys, math, json, glob, threading, traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES  = os.path.join(_APP_ROOT, "Modules")
_NN_DIR   = os.path.join(_APP_ROOT, "NeuralNetwork")
for _p in (_MODULES, _NN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SIMDATA_DIR     = os.path.join(_APP_ROOT, "SimulationData")
_EXPORT_BASENAME = "nn_simulation_results"
_SPACING_MM      = 0.16
_TX_N_LAYERS     = 3

try:
    from cap_combinator import find_best_cap as _find_best_cap
    _HAS_CAP = True
except ImportError:
    _HAS_CAP = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spiral_length_m(od_mm, w_mm, turns):
    pitch   = w_mm + _SPACING_MM
    R0      = od_mm / 2.0 - w_mm / 2.0
    R_inner = R0 - (2.0 * turns - 1.0) * pitch / 2.0
    return math.pi * turns * (R0 + R_inner) * 1e-3


def _cap_label(c_nf, max_caps=2):
    if not _HAS_CAP or c_nf <= 0:
        return f"{c_nf:.1f} nF"
    try:
        _, desc = _find_best_cap(c_nf, max_caps=max_caps)
        return desc if desc else f"{c_nf:.1f} nF"
    except Exception:
        return f"{c_nf:.1f} nF"


# Full-bridge rectifier constants (2 Schottky diodes in conduction at any time)
_V_DIODE = 0.35   # forward voltage per diode (V), typical Schottky
_V_DROP  = 2.0 * _V_DIODE


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized physics kernel
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_batch(res, freq_hz, V_min, V_max, P_target_w, D_min):
    omega   = 2.0 * math.pi * freq_hz
    f_ref   = float(res["_freq_ref_hz"])
    skin_sc = math.sqrt(freq_hz / f_ref)

    L_tx = np.maximum(res["L_tx"].astype(np.float64) * 1e-6, 1e-9)
    L_rx = np.maximum(res["L_rx"].astype(np.float64) * 1e-6, 1e-9)
    M    = np.maximum(res["M"].astype(np.float64)    * 1e-6, 0.0)

    DCR_tx  = res["DCR_tx"].astype(np.float64)
    DCR_rx  = res["DCR_rx"].astype(np.float64)
    # AC resistance = DCR + skin-effect excess scaled by sqrt(f/f_ref)
    R_tx_ac = DCR_tx + np.maximum(0.0, res["R_tx"].astype(np.float64) - DCR_tx) * skin_sc
    R_rx_ac = DCR_rx + np.maximum(0.0, res["R_rx"].astype(np.float64) - DCR_rx) * skin_sc

    Q_tx = omega * L_tx / np.maximum(R_tx_ac, 1e-12)
    Q_rx = omega * L_rx / np.maximum(R_rx_ac, 1e-12)
    k    = np.clip(M / np.sqrt(L_tx * L_rx), 0.0, 1.0)
    U    = k * np.sqrt(np.maximum(Q_tx * Q_rx, 0.0))

    sq       = np.sqrt(1.0 + U * U)
    # η_link: maximum coupled-inductor link efficiency (accounts for I²R in both coils)
    eta_link = U * U / np.maximum((1.0 + sq) ** 2, 1e-18)
    Z_tx_opt = np.maximum(R_tx_ac * sq, 1e-12)

    V_rms_min = V_min * math.sqrt(2.0) / math.pi
    V_rms_max = V_max * math.sqrt(2.0) / math.pi

    # Power at RX coil terminals (before rectifier)
    P_rx_coil_min = eta_link * (V_rms_min ** 2) / Z_tx_opt
    P_rx_coil_max = eta_link * (V_rms_max ** 2) / Z_tx_opt

    # Full-bridge rectifier efficiency: η_rect = (V_load_peak - 2·V_diode) / V_load_peak
    # V_load_rms at optimal operating point = U · V_rms · √(R_rx/R_tx) / (1 + √(1+U²))
    R_ratio = np.sqrt(np.maximum(R_rx_ac / np.maximum(R_tx_ac, 1e-12), 0.0))

    def _eta_rect(V_rms):
        V_load_pk = (U * V_rms * R_ratio
                     / np.maximum(1.0 + sq, 1e-12) * math.sqrt(2.0))
        return np.where(V_load_pk > _V_DROP,
                        (V_load_pk - _V_DROP) / np.maximum(V_load_pk, 1e-12),
                        0.0)

    eta_rect_min = _eta_rect(V_rms_min)   # worst case (lower voltage)
    eta_rect_max = _eta_rect(V_rms_max)

    # System efficiency = link × rectifier; use V_min for ranking (worst case)
    eta_sys  = eta_link * eta_rect_min

    # DC power delivered to load (after rectifier)
    P_dc_min = P_rx_coil_min * eta_rect_min
    P_dc_max = P_rx_coil_max * eta_rect_max

    D_vmin = np.where(P_dc_min > 1e-18, P_target_w / P_dc_min, np.inf)
    D_vmax = np.where(P_dc_max > 1e-18, P_target_w / P_dc_max, np.inf)

    C_tx_nf = np.where(L_tx > 0, 1e9 / (omega ** 2 * L_tx), 0.0)
    C_rx_nf = np.where(L_rx > 0, 1e9 / (omega ** 2 * L_rx), 0.0)

    if D_min > 0.0:
        feasible = (P_dc_min * D_min >= P_target_w)
    else:
        feasible = (D_vmin <= 1.0)

    feasible &= (R_tx_ac > 0) & (R_rx_ac > 0) & (M > 0)

    return dict(
        eta_max  = eta_sys.astype(np.float32),          # system eff (link×rect, Vmin)
        eta_link = eta_link.astype(np.float32),         # magnetic link eff only
        eta_rect = eta_rect_min.astype(np.float32),     # rectifier eff at Vmin
        U        = U.astype(np.float32),
        k        = k.astype(np.float32),
        Q_tx     = Q_tx.astype(np.float32),
        Q_rx     = Q_rx.astype(np.float32),
        R_tx_ac  = R_tx_ac.astype(np.float32),
        R_rx_ac  = R_rx_ac.astype(np.float32),
        P_rx_min = P_dc_min.astype(np.float32),         # DC power at Vmin
        P_rx_max = P_dc_max.astype(np.float32),         # DC power at Vmax
        D_vmin   = D_vmin.astype(np.float32),
        D_vmax   = D_vmax.astype(np.float32),
        C_tx_nf  = C_tx_nf.astype(np.float32),
        C_rx_nf  = C_rx_nf.astype(np.float32),
        feasible = feasible,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

_MET_KEYS = ["eta_max", "eta_link", "eta_rect", "U", "k", "Q_tx", "Q_rx",
             "R_tx_ac", "R_rx_ac", "P_rx_min", "P_rx_max",
             "D_vmin", "D_vmax", "C_tx_nf", "C_rx_nf"]

CHUNK = 2_000_000


def _analysis_worker(raw, params, log_cb, done_cb):
    try:
        V_min      = params["V_min"]
        V_max      = params["V_max"]
        P_target_w = params["P_target_mw"] * 1e-3
        D_min      = params["D_min"]
        incl_gc    = params["include_gc"]
        N_top      = params["n_top"]

        topologies = raw.get("_topologies",
                              np.array(["parallel", "series", "parallel_pairs_ser"]))

        # Frequency sweep range from RAM metadata
        freq_ref = float(raw.get("_freq_ref_hz", 125_000.0))
        freq_min = float(raw.get("_freq_min_hz", freq_ref))
        freq_max = float(raw.get("_freq_max_hz", freq_ref))
        if freq_max < freq_min:
            freq_max = freq_min
        N_FREQ = 10 if freq_max > freq_min else 1
        freq_points = np.linspace(freq_min, freq_max, N_FREQ)

        n_total = len(raw["L_tx"])
        log_cb(f"Sweeping {N_FREQ} frequencies: {freq_min/1e3:.1f}–{freq_max/1e3:.1f} kHz")
        log_cb(f"Total combinations: {n_total:,}")

        # GC filter — only index arrays that are the same length as the data
        gc_mask = np.ones(n_total, dtype=bool)
        if "gc_dia_mm" in raw:
            if incl_gc:
                gc_mask = (raw["gc_dia_mm"] > 0.0)   # GC-only
            else:
                gc_mask = (raw["gc_dia_mm"] == 0.0)  # no-GC only

        def _sl(arr):
            if isinstance(arr, np.ndarray) and arr.ndim > 0 and len(arr) == n_total:
                return arr[gc_mask]
            return arr

        res_f = {k: _sl(v) for k, v in raw.items()}
        n_filt = int(gc_mask.sum())

        if "gc_dia_mm" in raw:
            if incl_gc:
                log_cb(f"  GC only: {n_filt:,} ground-circle combinations")
            else:
                log_cb(f"  No-GC only: {n_filt:,} combinations (no ground circle)")
        else:
            log_cb(f"  {n_filt:,} combinations (no GC column in data)")

        # Phase 1: frequency sweep — track best feasible η per combo
        # Only store best_eta and best_freq (400 MB for 50 M combos) to minimise RAM.
        # Per-combo metrics are recomputed in Phase 2 for the top-K only.
        best_eta      = np.full(n_filt, -1.0, dtype=np.float32)
        best_freq_arr = np.full(n_filt, np.float32(freq_min), dtype=np.float32)

        for fi, freq_hz in enumerate(freq_points):
            for cs in range(0, n_filt, CHUNK):
                ce = min(cs + CHUNK, n_filt)
                res_chunk = {
                    k: (v[cs:ce] if (isinstance(v, np.ndarray) and
                                     v.ndim > 0 and len(v) == n_filt)
                        else v)
                    for k, v in res_f.items()
                }
                met = _analyse_batch(res_chunk, freq_hz, V_min, V_max, P_target_w, D_min)
                improve = met["feasible"] & (met["eta_max"] > best_eta[cs:ce])
                best_eta[cs:ce]      = np.where(improve, met["eta_max"], best_eta[cs:ce])
                best_freq_arr[cs:ce] = np.where(improve, np.float32(freq_hz),
                                                best_freq_arr[cs:ce])

            n_feas_so_far = int((best_eta > -1.0).sum())
            log_cb(f"  [{fi+1}/{N_FREQ}] {freq_hz/1e3:.1f} kHz — "
                   f"{n_feas_so_far:,} feasible so far")

        global_feasible = best_eta > -1.0
        n_feas = int(global_feasible.sum())
        pct    = 100.0 * n_feas / max(n_filt, 1)
        log_cb(f"Feasible combinations: {n_feas:,}  ({pct:.1f}%)")

        if n_feas == 0:
            log_cb("No feasible combinations — try higher voltages, lower target "
                   "power, or a smaller min-duty-cycle.")
            done_cb([], params)
            return

        feas_idx = np.where(global_feasible)[0]
        order    = np.argsort(-best_eta[feas_idx])
        top_idx  = feas_idx[order[:N_top]]
        log_cb(f"Building top {len(top_idx)} result rows …")

        # Phase 2: recompute full metrics for top-K combos at their optimal frequency
        topo_arr    = res_f["rx_topo"]
        rx_max_caps = params.get("rx_max_caps", 2)
        rows = []
        for rank_i, idx in enumerate(top_idx):
            freq_hz_i = float(best_freq_arr[idx])
            ti        = int(topo_arr[idx])
            topo_str  = str(topologies[ti]) if ti < len(topologies) else "unknown"

            res_single = {
                k: (v[idx:idx+1] if (isinstance(v, np.ndarray) and
                                     v.ndim > 0 and len(v) == n_filt)
                    else v)
                for k, v in res_f.items()
            }
            met = _analyse_batch(res_single, freq_hz_i, V_min, V_max, P_target_w, D_min)

            c_tx_nf   = float(met["C_tx_nf"][0])
            c_rx_nf   = float(met["C_rx_nf"][0])
            d_vmin    = float(met["D_vmin"][0])
            d_vmax    = float(met["D_vmax"][0])
            eta_sys   = float(met["eta_max"][0])
            eta_link  = float(met["eta_link"][0])
            eta_rect  = float(met["eta_rect"][0])

            if D_min > 0.0:
                d_show_min = max(d_vmin, D_min)
                d_show_max = max(d_vmax, D_min)
            else:
                d_show_min = d_vmin
                d_show_max = d_vmax

            rows.append(dict(
                rank         = rank_i + 1,
                eta_max      = eta_sys,          # system eff (link × rectifier @ Vmin)
                eta_link     = eta_link,         # pure magnetic link efficiency
                eta_rect     = eta_rect,         # full-bridge rectifier efficiency @ Vmin
                U            = float(met["U"][0]),
                k            = float(met["k"][0]),
                Q_tx         = float(met["Q_tx"][0]),
                Q_rx         = float(met["Q_rx"][0]),
                freq_hz      = freq_hz_i,
                tx_od_mm     = float(res_f["tx_od_mm"][idx]),
                tx_turns     = int(res_f["tx_turns"][idx]),
                tx_width     = float(res_f["tx_width"][idx]),
                tx_id_mm     = float(res_f["tx_id_mm"][idx]),
                rx_od_mm     = float(res_f["rx_od_mm"][idx]),
                rx_turns     = int(res_f["rx_turns"][idx]),
                rx_width     = float(res_f["rx_width"][idx]),
                rx_id_mm     = float(res_f["rx_id_mm"][idx]),
                rx_topology  = topo_str,
                gc_dia_mm    = float(res_f["gc_dia_mm"][idx]) if "gc_dia_mm" in res_f else 0.0,
                L_tx_uH      = float(res_f["L_tx"][idx]),
                L_rx_uH      = float(res_f["L_rx"][idx]),
                M_uH         = float(res_f["M"][idx]),
                R_tx_ohm     = float(met["R_tx_ac"][0]),
                R_rx_ohm     = float(met["R_rx_ac"][0]),
                DCR_tx_ohm   = float(res_f["DCR_tx"][idx]),
                DCR_rx_ohm   = float(res_f["DCR_rx"][idx]),
                P_rx_mw_vmin = float(met["P_rx_min"][0]) * 1e3,  # DC power @ Vmin
                P_rx_mw_vmax = float(met["P_rx_max"][0]) * 1e3,  # DC power @ Vmax
                D_vmin       = d_show_min,
                D_vmax       = d_show_max,
                D_natural_vmin = d_vmin,
                D_natural_vmax = d_vmax,
                C_tx_nf      = c_tx_nf,
                C_rx_nf      = c_rx_nf,
                C_tx_label   = _cap_label(c_tx_nf, max_caps=2),       # TX: always allow 2 caps
                C_rx_label   = _cap_label(c_rx_nf, max_caps=rx_max_caps),
                # aliases expected by nn_optimisation._on_send_to_sim
                eff_mid      = eta_sys,
                Duty_vmin    = d_show_min,
                Duty_vmax    = d_show_max,
                V_ind_min_V  = 0.0,
                Zin_re       = float(res_f["DCR_tx"][idx]),
                Zin_im       = 0.0,
                rx_eff_turns = int(res_f["rx_turns"][idx]),
            ))

        log_cb(f"Done.  Best η_sys = {rows[0]['eta_max']*100:.2f}%  "
               f"(link={rows[0].get('eta_link', rows[0]['eta_max'])*100:.2f}%  "
               f"rect={rows[0].get('eta_rect', 1.0)*100:.2f}%)  "
               f"@ {rows[0]['freq_hz']/1e3:.1f} kHz  "
               f"D@Vmin = {rows[0]['D_vmin']*100:.1f}%")
        done_cb(rows, params)

    except Exception as e:
        log_cb(f"Error: {e}\n{traceback.format_exc()}")
        done_cb([], params)


# ─────────────────────────────────────────────────────────────────────────────
# Tab widget
# ─────────────────────────────────────────────────────────────────────────────

class NNAnalysisTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app           = app
        self._results      = []
        self._run_params   = {}
        self._selected_idx = -1
        self._running      = False
        self._raw_results  = None
        self._build()
        self.after(300, self._auto_load)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        self._build_params(left)
        self._build_results(right)

    def _build_params(self, parent):
        src_lf = ttk.LabelFrame(parent, text="Data Source")
        src_lf.pack(fill="x", pady=(0, 6))

        self._src_label_var = tk.StringVar(value="(none)")
        ttk.Label(src_lf, textvariable=self._src_label_var,
                  foreground="#555", font=("TkDefaultFont", 8, "italic"),
                  wraplength=280, justify="left").pack(fill="x", padx=6, pady=(4, 2))

        btn_row = ttk.Frame(src_lf)
        btn_row.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Button(btn_row, text="Use RAM Data",
                   command=self._on_use_ram).pack(side="left", padx=(0, 4))

        filt_lf = ttk.LabelFrame(parent, text="Filters")
        filt_lf.pack(fill="x", pady=(0, 6))
        self._incl_gc = tk.BooleanVar(value=False)
        ttk.Checkbutton(filt_lf, text="Add Ground Circle  (GC-only combinations)",
                        variable=self._incl_gc).pack(anchor="w", padx=6, pady=(6, 6))

        pwr_lf = ttk.LabelFrame(parent, text="Power Target")
        pwr_lf.pack(fill="x", pady=(0, 6))
        self._p_target = self._prow(pwr_lf, "Target RX power (mW):", "50")

        volt_lf = ttk.LabelFrame(parent, text="TX Supply Voltage")
        volt_lf.pack(fill="x", pady=(0, 6))
        self._v_min = self._prow(volt_lf, "V_min (V):", "3.2")
        self._v_max = self._prow(volt_lf, "V_max (V):", "4.4")

        duty_lf = ttk.LabelFrame(parent, text="Envelope Duty Cycle")
        duty_lf.pack(fill="x", pady=(0, 6))
        self._d_min = self._prow(duty_lf, "Min duty cycle (%):", "")
        ttk.Label(duty_lf,
                  text="Blank or 0 = no floor (go as low as needed).\n"
                       "e.g. 10 → top result is most efficient combo\n"
                       "     that delivers target power at ≥10% duty.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  justify="left").pack(anchor="w", padx=6, pady=(0, 6))

        adv_lf = ttk.LabelFrame(parent, text="Display")
        adv_lf.pack(fill="x", pady=(0, 6))
        self._n_top      = self._prow(adv_lf, "Top results to show:", "10")
        self._rx_max_caps = self._prow(adv_lf, "RX max caps (1 or 2):", "2")

        self._run_btn = ttk.Button(parent, text="▶  Run Analysis",
                                   command=self._on_run)
        self._run_btn.pack(fill="x", pady=(4, 2))

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self._status_var,
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=280, justify="left").pack(fill="x", padx=2)

        log_lf = ttk.LabelFrame(parent, text="Log")
        log_lf.pack(fill="both", expand=True, pady=(6, 0))
        log_lf.columnconfigure(0, weight=1)
        log_lf.rowconfigure(0, weight=1)
        self._log = tk.Text(log_lf, height=8, state="disabled",
                            font=("Consolas", 8), wrap="word",
                            background="#f8f8f8")
        lsb = ttk.Scrollbar(log_lf, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=lsb.set)
        lsb.grid(row=0, column=1, sticky="ns", pady=4)
        self._log.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)

    def _prow(self, parent, label, default, lw=22):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text=label, width=lw, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=10).pack(side="left", padx=4)
        return var

    def _build_results(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=0)
        parent.columnconfigure(0, weight=1)

        res_lf = ttk.LabelFrame(parent, text="Top Configurations (ranked by η_max)")
        res_lf.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self._build_tree(res_lf)

        det_lf = ttk.LabelFrame(parent, text="Detail  (click a row)")
        det_lf.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self._build_detail(det_lf)

        btn_row = ttk.Frame(parent)
        btn_row.grid(row=2, column=0, sticky="ew")
        self._send_btn = ttk.Button(btn_row, text="Send Selected to Simulation",
                                    command=self._on_send_to_sim,
                                    state="disabled", width=30)
        self._send_btn.pack(side="left")

    def _build_tree(self, parent):
        cols = (
            "#", "η_sys%", "η_link%", "η_rect%", "U", "k",
            "Q_tx", "Q_rx", "f kHz",
            "TX OD", "TX T", "TX W", "TX ID",
            "RX OD", "RX T", "RX W", "RX ID", "RX Topo",
            "GC mm",
            "TX L µH", "RX L µH", "M µH",
            "TX Rac mΩ", "RX Rac mΩ",
            "TX DCR mΩ", "RX DCR mΩ",
            "C_tx", "C_rx",
            "P@Vmin mW", "P@Vmax mW",
            "D@Vmin %", "D@Vmax %",
        )
        widths = {
            "#": 28,
            "η_sys%": 54, "η_link%": 54, "η_rect%": 54,
            "U": 42, "k": 46,
            "Q_tx": 44, "Q_rx": 44, "f kHz": 54,
            "TX OD": 48, "TX T": 34, "TX W": 42, "TX ID": 44,
            "RX OD": 48, "RX T": 34, "RX W": 42, "RX ID": 44, "RX Topo": 80,
            "GC mm": 44,
            "TX L µH": 56, "RX L µH": 56, "M µH": 52,
            "TX Rac mΩ": 72, "RX Rac mΩ": 72,
            "TX DCR mΩ": 72, "RX DCR mΩ": 72,
            "C_tx": 120, "C_rx": 120,
            "P@Vmin mW": 72, "P@Vmax mW": 72,
            "D@Vmin %": 60, "D@Vmax %": 60,
        }

        tf = ttk.Frame(parent)
        tf.pack(fill="both", expand=True, padx=4, pady=4)
        xsb = ttk.Scrollbar(tf, orient="horizontal")
        ysb = ttk.Scrollbar(tf, orient="vertical")
        self._tree = ttk.Treeview(tf, columns=cols, show="headings",
                                  yscrollcommand=ysb.set,
                                  xscrollcommand=xsb.set, height=12)
        xsb.configure(command=self._tree.xview)
        ysb.configure(command=self._tree.yview)
        xsb.pack(side="bottom", fill="x")
        ysb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths.get(c, 72), minwidth=30, stretch=False)

        self._tree.tag_configure("rank1", background="#d4edda")
        self._tree.tag_configure("rank2", background="#e8f4fd")
        self._tree.tag_configure("rank3", background="#fff3cd")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_detail(self, parent):
        FONT = ("Consolas", 9)
        BG   = "#f8f8f8"
        H    = 18

        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        def _scroll_all(*args):
            for t in (self._det_l, self._det_r):
                t.yview(*args)

        vsb.configure(command=_scroll_all)

        cf = ttk.Frame(outer)
        cf.pack(fill="both", expand=True)
        cf.columnconfigure(0, weight=1, uniform="dc")
        cf.columnconfigure(2, weight=1, uniform="dc")

        def _make_col(col_idx, title):
            hdr = ttk.Frame(cf)
            hdr.grid(row=0, column=col_idx, sticky="ew")
            ttk.Label(hdr, text=title, font=("TkDefaultFont", 9, "bold"),
                      anchor="center", background="#e8eef8").pack(fill="x", padx=1, pady=(2, 0))
            t = tk.Text(cf, height=H, state="disabled",
                        font=FONT, wrap="none", background=BG,
                        borderwidth=0, highlightthickness=0)
            t.grid(row=1, column=col_idx, sticky="nsew")
            t.configure(yscrollcommand=vsb.set)

            def _wheel(event):
                delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
                _scroll_all("scroll", delta, "units")
                return "break"

            t.bind("<MouseWheel>", _wheel)
            t.bind("<Button-4>", _wheel)
            t.bind("<Button-5>", _wheel)
            return t

        self._det_l = _make_col(0, "NN + Analysis")
        tk.Frame(cf, bg="#b0b0b0", width=1).grid(row=0, column=1, rowspan=2, sticky="ns")
        self._det_r = _make_col(2, "FastHenry Simulation  (run sim to populate)")

    # ─────────────────────────────────────────────────────────────────────────
    # Data source
    # ─────────────────────────────────────────────────────────────────────────

    def _auto_load(self):
        raw = self._get_ram()
        if raw:
            self._raw_results = raw
            n     = len(raw.get("L_tx", []))
            f_min = float(raw.get("_freq_min_hz", raw.get("_freq_ref_hz", 125000.0)))
            f_max = float(raw.get("_freq_max_hz", f_min))
            self._src_label_var.set(
                f"RAM  ({n:,} combinations, "
                f"f={f_min/1e3:.1f}–{f_max/1e3:.1f} kHz)")
            self._log_append(f"RAM data detected: {n:,} combinations.")

    def _get_ram(self):
        auto = getattr(self.app, "nn_optim_tab", None)
        if auto is None:
            return None
        r = getattr(auto, "_results", {})
        if r and len(r.get("L_tx", [])) > 0:
            return r
        return None

    def _on_use_ram(self):
        raw = self._get_ram()
        if raw is None:
            messagebox.showinfo("RAM Data",
                                "No sweep results in RAM.\n"
                                "Run the Automation NN sweep first.")
            return
        self._raw_results = raw
        n     = len(raw.get("L_tx", []))
        f_min = float(raw.get("_freq_min_hz", raw.get("_freq_ref_hz", 125000.0)))
        f_max = float(raw.get("_freq_max_hz", f_min))
        self._src_label_var.set(
            f"RAM  ({n:,} combinations, "
            f"f={f_min/1e3:.1f}–{f_max/1e3:.1f} kHz)")
        self._log_append(f"Using RAM data: {n:,} combinations.")

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis run
    # ─────────────────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        raw = self._raw_results or self._get_ram()
        if raw is None:
            messagebox.showwarning("No Data",
                                   "No sweep data available.\n"
                                   "Run Automation NN sweep first.")
            return

        try:
            params = self._parse_params()
        except ValueError as e:
            self._set_status(str(e), "red")
            return

        self._running = True
        self._run_btn.configure(state="disabled")
        self._log_clear()
        self._set_status("Running…")
        self._results = []
        self._clear_tree()
        self._clear_detail()

        t = threading.Thread(
            target=_analysis_worker,
            args=(raw, params, self._log_cb, self._done_cb),
            daemon=True,
        )
        t.start()

    def _log_cb(self, msg):
        self.after(0, lambda m=msg: self._log_append(m))

    def _done_cb(self, rows, params):
        self.after(0, lambda: self._on_done(rows, params))

    def _on_done(self, rows, params):
        self._running = False
        self._run_btn.configure(state="normal")
        self._results    = rows
        self._run_params = params
        if rows:
            self._populate_tree(rows)
            best = rows[0]
            self._set_status(
                f"{len(rows)} results  |  best η_sys = {best['eta_max']*100:.2f}%"
                f" (link={best.get('eta_link', best['eta_max'])*100:.2f}%)"
                f" @ {best['freq_hz']/1e3:.1f} kHz  "
                f"D@Vmin = {best['D_vmin']*100:.1f}%", "green")
        else:
            self._set_status("No feasible combinations found.", "red")

    def _parse_params(self):
        def _f(var, name, lo=None):
            s = var.get().strip()
            if not s:
                return None
            try:
                v = float(s)
            except ValueError:
                raise ValueError(f"'{name}' must be a number.")
            if lo is not None and v < lo:
                raise ValueError(f"'{name}' must be ≥ {lo}.")
            return v

        p_mw    = _f(self._p_target,   "Target RX power", lo=0.001)
        v_min   = _f(self._v_min,      "V_min",           lo=0.1)
        v_max   = _f(self._v_max,      "V_max",           lo=0.1)
        d_pct   = _f(self._d_min,      "Min duty %",      lo=0.0)
        n_top   = _f(self._n_top,      "Top N",           lo=1)
        rx_caps = _f(self._rx_max_caps,"RX max caps",     lo=1)

        if p_mw  is None: raise ValueError("Target RX power is required.")
        if v_min is None: raise ValueError("V_min is required.")
        if v_max is None: raise ValueError("V_max is required.")
        if v_max < v_min: raise ValueError("V_max must be ≥ V_min.")

        rx_caps_int = max(1, min(2, int(rx_caps) if rx_caps else 2))

        return dict(
            V_min        = v_min,
            V_max        = v_max,
            P_target_mw  = p_mw,
            D_min        = (d_pct / 100.0) if (d_pct and d_pct > 0) else 0.0,
            include_gc   = self._incl_gc.get(),
            n_top        = int(n_top) if n_top else 10,
            rx_max_caps  = rx_caps_int,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tree population
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_tree(self):
        for iid in self._tree.get_children():
            self._tree.delete(iid)

    def _populate_tree(self, rows):
        self._clear_tree()
        tag_map = {0: "rank1", 1: "rank2", 2: "rank3"}
        for i, r in enumerate(rows):
            tag = tag_map.get(i, "")
            gc  = r.get("gc_dia_mm", 0.0)
            vals = (
                i + 1,
                f"{r['eta_max']*100:.2f}",
                f"{r.get('eta_link', r['eta_max'])*100:.2f}",
                f"{r.get('eta_rect', 1.0)*100:.2f}",
                f"{r['U']:.2f}",
                f"{r['k']:.4f}",
                f"{r['Q_tx']:.1f}",
                f"{r['Q_rx']:.1f}",
                f"{r['freq_hz']/1e3:.2f}",
                f"{r['tx_od_mm']:.1f}",
                r["tx_turns"],
                f"{r['tx_width']:.3f}",
                f"{r['tx_id_mm']:.1f}",
                f"{r['rx_od_mm']:.1f}",
                r["rx_turns"],
                f"{r['rx_width']:.3f}",
                f"{r['rx_id_mm']:.1f}",
                r["rx_topology"],
                f"{gc:.1f}" if gc > 0 else "—",
                f"{r['L_tx_uH']:.3f}",
                f"{r['L_rx_uH']:.3f}",
                f"{r['M_uH']:.4f}",
                f"{r['R_tx_ohm']*1e3:.2f}",
                f"{r['R_rx_ohm']*1e3:.2f}",
                f"{r['DCR_tx_ohm']*1e3:.2f}",
                f"{r['DCR_rx_ohm']*1e3:.2f}",
                r.get("C_tx_label", f"{r['C_tx_nf']:.2f} nF"),
                r.get("C_rx_label", f"{r['C_rx_nf']:.2f} nF"),
                f"{r['P_rx_mw_vmin']:.2f}",
                f"{r['P_rx_mw_vmax']:.2f}",
                f"{r['D_vmin']*100:.2f}",
                f"{r['D_vmax']*100:.2f}",
            )
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

    # ─────────────────────────────────────────────────────────────────────────
    # Tree selection → detail panels
    # ─────────────────────────────────────────────────────────────────────────

    def _on_tree_select(self, _=None):
        sel = self._tree.selection()
        if not sel:
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        items = self._tree.get_children()
        idx   = list(items).index(sel[0])
        if idx >= len(self._results):
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        self._send_btn.configure(state="normal")
        self._selected_idx = idx
        r       = self._results[idx]
        sim_res = None
        sim_tab = getattr(self.app, "sim_tab", None)
        if sim_tab:
            sim_res = getattr(sim_tab, "last_result", None)
        L, R = self._make_detail_cols(idx + 1, r, sim_res)
        self._write_col(self._det_l, L)
        self._write_col(self._det_r, R)

    def on_sim_done(self, r_nn, sim_result):
        if self._selected_idx < 0 or self._selected_idx >= len(self._results):
            return
        if self._results[self._selected_idx] is not r_nn:
            return
        L, R = self._make_detail_cols(self._selected_idx + 1, r_nn, sim_result)
        self._write_col(self._det_l, L)
        self._write_col(self._det_r, R)

    # ─────────────────────────────────────────────────────────────────────────
    # Detail column builder
    # ─────────────────────────────────────────────────────────────────────────

    def _make_detail_cols(self, rank, r, sim_res=None):
        SEP = "─" * 50
        topo    = r["rx_topology"]
        gc      = r.get("gc_dia_mm", 0.0)
        tx_l_mm = _spiral_length_m(r["tx_od_mm"], r["tx_width"], r["tx_turns"]) * 1e3 * _TX_N_LAYERS
        rx_l_mm = _spiral_length_m(r["rx_od_mm"], r["rx_width"], r["rx_turns"]) * 1e3 * 4
        k        = r["k"]
        U        = r["U"]
        eta      = r["eta_max"]          # system efficiency (link × rectifier)
        eta_link = r.get("eta_link", eta)
        eta_rect = r.get("eta_rect", 1.0)
        d_nat_min = r.get("D_natural_vmin", r["D_vmin"])
        d_nat_max = r.get("D_natural_vmax", r["D_vmax"])
        d_min_p   = self._run_params.get("D_min", 0.0) * 100.0

        p_target = self._run_params.get("P_target_mw", 0.0)
        V_min    = self._run_params.get("V_min", 0.0)
        V_max    = self._run_params.get("V_max", 0.0)
        freq_hz  = r["freq_hz"]

        L_lines = [
            SEP,
            f"  Rank #{rank}   η_sys = {eta*100:.2f}%   U = {U:.3f}",
            SEP,
            "",
            "  GEOMETRY",
            f"    TX: OD={r['tx_od_mm']:.1f} mm  T={r['tx_turns']}  W={r['tx_width']:.3f} mm",
            f"        ID={r['tx_id_mm']:.1f} mm  wire≈{tx_l_mm:.0f} mm  (3 layers ∥)",
            f"    RX: OD={r['rx_od_mm']:.1f} mm  T={r['rx_turns']}  W={r['rx_width']:.3f} mm  topo={topo}",
            f"        ID={r['rx_id_mm']:.1f} mm  wire≈{rx_l_mm:.0f} mm",
            f"    GC: {gc:.1f} mm" if gc > 0 else "    GC: none",
            "",
            "  EM  (NN predicted at f_ref, AC-scaled to optimal freq)",
            f"    Optimal freq  {freq_hz/1e3:.3f} kHz",
            f"    L_TX          {r['L_tx_uH']:.4f} µH",
            f"    L_RX          {r['L_rx_uH']:.4f} µH",
            f"    M             {r['M_uH']:.5f} µH",
            f"    k             {k:.4f}",
            f"    R_TX_ac       {r['R_tx_ohm']*1e3:.3f} mΩ"
            f"  (DCR={r['DCR_tx_ohm']*1e3:.3f}  skin={max(0,r['R_tx_ohm']-r['DCR_tx_ohm'])*1e3:.3f})",
            f"    R_RX_ac       {r['R_rx_ohm']*1e3:.3f} mΩ"
            f"  (DCR={r['DCR_rx_ohm']*1e3:.3f}  skin={max(0,r['R_rx_ohm']-r['DCR_rx_ohm'])*1e3:.3f})",
            f"    Q_TX          {r['Q_tx']:.1f}",
            f"    Q_RX          {r['Q_rx']:.1f}",
            "",
            "  RESONANCE",
            f"    C_TX          {r['C_tx_nf']:.3f} nF  →  {r['C_tx_label']}",
            f"    C_RX          {r['C_rx_nf']:.3f} nF  →  {r['C_rx_label']}",
            "",
            "  POWER  (efficiency-optimal load, half-bridge drive)",
            f"    η_link (mag)   {eta_link*100:.3f}%  ← coupled-inductor limit (I²R in both coils)",
            f"    η_rect (FBR)   {eta_rect*100:.3f}%  ← full-bridge rectifier (2×0.35V Schottky)",
            f"    η_system       {eta*100:.3f}%  ← η_link × η_rect @ V_min",
            f"    U = k√(QQ)    {U:.4f}",
            f"    Target Pavg   {p_target:.2f} mW",
            f"    V_min = {V_min:.2f} V",
            f"      P_rx ON     {r['P_rx_mw_vmin']:.3f} mW",
            f"      D natural   {d_nat_min*100:.2f}%"
            + (f"  →  floored to {d_min_p:.1f}%" if d_min_p > 0 and d_nat_min * 100 < d_min_p else ""),
            f"      D applied   {r['D_vmin']*100:.2f}%",
            f"    V_max = {V_max:.2f} V",
            f"      P_rx ON     {r['P_rx_mw_vmax']:.3f} mW",
            f"      D natural   {d_nat_max*100:.2f}%"
            + (f"  →  floored to {d_min_p:.1f}%" if d_min_p > 0 and d_nat_max * 100 < d_min_p else ""),
            f"      D applied   {r['D_vmax']*100:.2f}%",
        ]

        have_sim = (sim_res is not None and sim_res.get("n_ports", 0) == 2)
        if have_sim:
            Zmat  = sim_res["Zmat"]
            f_sim = sim_res["frequency"]
            w_s   = 2.0 * math.pi * f_sim
            sL_tx = Zmat[0][0].imag / w_s * 1e6
            sL_rx = Zmat[1][1].imag / w_s * 1e6
            sM    = Zmat[0][1].imag / w_s * 1e6
            sR_tx = Zmat[0][0].real
            sR_rx = Zmat[1][1].real
            sk    = sM / math.sqrt(sL_tx * sL_rx) if sL_tx > 0 and sL_rx > 0 else 0.0
            sQ_tx = w_s * sL_tx * 1e-6 / sR_tx if sR_tx > 0 else 0.0
            sQ_rx = w_s * sL_rx * 1e-6 / sR_rx if sR_rx > 0 else 0.0

            def pct(nn, s):
                return f" ({(s - nn) / nn * 100:+.1f}%)" if abs(nn) > 1e-12 else ""

            R_lines = [
                SEP,
                "  FastHenry  —  Simulated values",
                SEP,
                "",
                "  EM",
                "", "", "", "", "", "", "",
                f"    Frequency     {f_sim/1e3:.3f} kHz",
                f"    L_TX          {sL_tx:.4f} µH" + pct(r["L_tx_uH"], sL_tx),
                f"    L_RX          {sL_rx:.4f} µH" + pct(r["L_rx_uH"], sL_rx),
                f"    M             {sM:.5f} µH"    + pct(r["M_uH"],    sM),
                f"    k             {sk:.4f}"        + pct(r["k"],       sk),
                f"    R_TX_ac       {sR_tx*1e3:.3f} mΩ" + pct(r["R_tx_ohm"]*1e3, sR_tx*1e3),
                f"    R_RX_ac       {sR_rx*1e3:.3f} mΩ" + pct(r["R_rx_ohm"]*1e3, sR_rx*1e3),
                f"    Q_TX          {sQ_tx:.1f}"    + pct(r["Q_tx"],    sQ_tx),
                f"    Q_RX          {sQ_rx:.1f}"    + pct(r["Q_rx"],    sQ_rx),
            ]
        else:
            R_lines = [SEP, "  Run simulation to see FastHenry comparison.", SEP]

        n = max(len(L_lines), len(R_lines))
        L_lines += [""] * (n - len(L_lines))
        R_lines += [""] * (n - len(R_lines))
        return L_lines, R_lines

    # ─────────────────────────────────────────────────────────────────────────
    # Send to simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _on_send_to_sim(self):
        if self._selected_idx < 0 or self._selected_idx >= len(self._results):
            return
        r = self._results[self._selected_idx]
        auto_nn = getattr(self.app, "nn_optim_tab", None)
        if auto_nn is None:
            messagebox.showerror("Send to Sim", "NNOptimisationTab not available.")
            return
        auto_nn._last_run_params = self._run_params
        auto_nn._on_send_to_sim(r)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_col(self, widget, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines))
        widget.configure(state="disabled")

    def _clear_detail(self):
        for w in (self._det_l, self._det_r):
            w.configure(state="normal")
            w.delete("1.0", "end")
            w.configure(state="disabled")

    def _log_clear(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _log_append(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_var.set(msg)
