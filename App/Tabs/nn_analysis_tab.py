#!/usr/bin/env python3
"""
NNAnalysisTab — displays results from nn_simulation_results_*.json files
produced by the Automation NN tab optimizer.

Auto-loads nn_simulation_results_0.json on startup; falls back to the
smallest present index if 0 is absent.
"""

import os, sys, math, json, glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES  = os.path.join(_APP_ROOT, "Modules")
_NN_DIR   = os.path.join(_APP_ROOT, "NeuralNetwork")
for _p in (_MODULES, _NN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SIMDATA_DIR  = os.path.join(_APP_ROOT, "SimulationData")
_EXPORT_BASENAME = "nn_simulation_results"

# PCB constants (same as automation_nn_tab — kept local so this tab is standalone)
_SPACING_MM  = 0.16
_TX_N_LAYERS = 3


def _spiral_length_m(od_mm, w_mm, turns):
    pitch   = w_mm + _SPACING_MM
    R0      = od_mm / 2.0 - w_mm / 2.0
    R_inner = R0 - (2.0 * turns - 1.0) * pitch / 2.0
    return math.pi * turns * (R0 + R_inner) * 1e-3


def _find_default_results_file():
    """Return path to the smallest-index nn_simulation_results_*.json, preferring 0."""
    path0 = os.path.join(_SIMDATA_DIR, f"{_EXPORT_BASENAME}_0.json")
    if os.path.exists(path0):
        return path0
    pattern = os.path.join(_SIMDATA_DIR, f"{_EXPORT_BASENAME}_*.json")
    candidates = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        prefix = f"{_EXPORT_BASENAME}_"
        suffix = ".json"
        if base.startswith(prefix) and base.endswith(suffix):
            try:
                idx = int(base[len(prefix):-len(suffix)])
                candidates.append((idx, p))
            except ValueError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


class NNAnalysisTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app             = app
        self._results        = []
        self._run_params     = {}
        self._selected_idx   = -1
        self._loaded_file    = None
        self._build()
        self.after(200, self._auto_load)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build(self):
        # Top bar: loaded-file label + Load button
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=8, pady=(6, 0))

        ttk.Label(top_bar, text="Results file:", font=("TkDefaultFont", 9)).pack(side="left")
        self._file_label_var = tk.StringVar(value="(none loaded)")
        ttk.Label(top_bar, textvariable=self._file_label_var,
                  foreground="#555555", font=("TkDefaultFont", 9, "italic"),
                  wraplength=900, justify="left").pack(side="left", padx=6)

        ttk.Button(top_bar, text="Load Results…",
                   command=self._on_load_file).pack(side="right")

        # Results table
        res_lf = ttk.LabelFrame(self, text="Top Configurations")
        res_lf.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self._build_tree(res_lf)

        # Detail panel
        det_lf = ttk.LabelFrame(self, text="Detail (click a row)")
        det_lf.pack(fill="x", padx=8, pady=(0, 4))
        self._build_detail_panels(det_lf)

        # Bottom action row
        send_row = ttk.Frame(self)
        send_row.pack(fill="x", padx=8, pady=(0, 8))
        self._send_btn = ttk.Button(send_row, text="Send Selected to Simulation",
                                    command=self._on_send_to_sim, state="disabled",
                                    width=30)
        self._send_btn.pack(side="left")

    # -----------------------------------------------------------------------
    # Tree
    # -----------------------------------------------------------------------

    def _build_tree(self, parent):
        cols = (
            "#", "Fitness", "Eff%", "f kHz",
            "TX T/Ly", "TX width", "RX OD", "RX T/Ly", "RX width", "RX Topo",
            "TX L µH", "RX L µH", "M L µH", "TX R Ω", "RX R Ω",
            "TX DCR mΩ", "RX DCR mΩ", "TX Q", "RX Q",
            "TX cap", "RX cap",
            "DutyVmin", "DutyVmax", "V_ind",
        )
        widths = {
            "#": 26,
            "Fitness": 50, "Eff%": 34, "f kHz": 42,
            "TX T/Ly": 55, "TX width": 58,
            "RX OD": 50,  "RX T/Ly": 55, "RX width": 58, "RX Topo": 76,
            "TX L µH": 58, "RX L µH": 58, "M L µH": 52,
            "TX R Ω": 52,  "RX R Ω": 52,
            "TX DCR mΩ": 75, "RX DCR mΩ": 75, "TX Q": 40, "RX Q": 40,
            "TX cap": 85,  "RX cap": 50,
            "DutyVmin": 70, "DutyVmax": 70, "V_ind": 40,
        }

        tree_f = ttk.Frame(parent)
        tree_f.pack(fill="both", expand=True, padx=4, pady=4)

        xsb = ttk.Scrollbar(tree_f, orient="horizontal")
        ysb = ttk.Scrollbar(tree_f, orient="vertical")
        self._tree = ttk.Treeview(tree_f, columns=cols, show="headings",
                                   yscrollcommand=ysb.set,
                                   xscrollcommand=xsb.set, height=12)
        xsb.configure(command=self._tree.xview)
        ysb.configure(command=self._tree.yview)
        xsb.pack(side="bottom", fill="x")
        ysb.pack(side="right",  fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths.get(c, 80), minwidth=36, stretch=False)

        self._tree.tag_configure("rank1", background="#d4edda")
        self._tree.tag_configure("rank2", background="#e8f4fd")
        self._tree.tag_configure("rank3", background="#fff3cd")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # -----------------------------------------------------------------------
    # Detail panels
    # -----------------------------------------------------------------------

    def _build_detail_panels(self, parent):
        FONT = ("Consolas", 9)
        BG   = "#f8f8f8"
        H    = 20

        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        def _scroll_all(*args):
            for t in (self._det_nn, self._det_sim, self._det_delta):
                t.yview(*args)

        vsb.configure(command=_scroll_all)

        cols_frame = ttk.Frame(outer)
        cols_frame.pack(fill="both", expand=True)
        cols_frame.columnconfigure(0, weight=1, uniform="dc")
        cols_frame.columnconfigure(2, weight=1, uniform="dc")
        cols_frame.columnconfigure(4, weight=1, uniform="dc")

        def _make_col(col_idx, title):
            hdr = ttk.Frame(cols_frame)
            hdr.grid(row=0, column=col_idx, sticky="ew")
            ttk.Label(hdr, text=title, font=("TkDefaultFont", 9, "bold"),
                      anchor="center", background="#e8eef8"
                      ).pack(fill="x", padx=1, pady=(2, 0))
            t = tk.Text(cols_frame, height=H, state="disabled",
                        font=FONT, wrap="none", background=BG,
                        borderwidth=0, highlightthickness=0)
            t.grid(row=1, column=col_idx, sticky="nsew")
            t.configure(yscrollcommand=vsb.set)

            def _on_wheel(event):
                delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
                _scroll_all("scroll", delta, "units")
                return "break"

            t.bind("<MouseWheel>", _on_wheel)
            t.bind("<Button-4>", _on_wheel)
            t.bind("<Button-5>", _on_wheel)
            return t

        self._det_nn    = _make_col(0, "NN Prediction")
        tk.Frame(cols_frame, bg="#b0b0b0", width=1).grid(row=0, column=1, rowspan=2, sticky="ns")
        self._det_sim   = _make_col(2, "Simulation Result")
        tk.Frame(cols_frame, bg="#b0b0b0", width=1).grid(row=0, column=3, rowspan=2, sticky="ns")
        self._det_delta = _make_col(4, "Delta  (Sim − NN)")

    # -----------------------------------------------------------------------
    # Load logic
    # -----------------------------------------------------------------------

    def _auto_load(self):
        path = _find_default_results_file()
        if path:
            self._load_file(path, auto=True)

    def _on_load_file(self):
        path = filedialog.askopenfilename(
            title="Load NN Simulation Results",
            initialdir=_SIMDATA_DIR if os.path.isdir(_SIMDATA_DIR) else os.path.expanduser("~"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._load_file(path, auto=False)

    def _load_file(self, path, auto=False):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read file:\n{e}")
            return

        results = data.get("results", data) if isinstance(data, dict) else data
        meta    = data.get("metadata", {}) if isinstance(data, dict) else {}

        if not isinstance(results, list) or not results:
            messagebox.showerror("Load Error", "File contains no result entries.")
            return

        self._results    = results
        self._run_params = meta.get("run_params", {})
        self._loaded_file = path

        fname   = os.path.basename(path)
        prefix  = "(auto-loaded)" if auto else "(loaded)"
        self._file_label_var.set(f"{prefix}  {fname}")

        self._populate_tree(results)
        self._clear_detail()

    # -----------------------------------------------------------------------
    # Tree population
    # -----------------------------------------------------------------------

    def _clear_tree(self):
        for iid in self._tree.get_children():
            self._tree.delete(iid)

    def _populate_tree(self, rows):
        self._clear_tree()
        tag_map = {0: "rank1", 1: "rank2", 2: "rank3"}
        for i, r in enumerate(rows):
            tag  = tag_map.get(i, "")
            vals = (
                i + 1,
                f"{r['fitness']:.4f}",
                f"{r['eff_mid']*100:.1f}",
                f"{r['freq_hz']/1e3:.2f}",
                r["tx_turns"],
                f"{r['tx_width']:.2f}",
                f"{r['rx_od_mm']:.1f}",
                r["rx_turns"],
                f"{r['rx_width']:.2f}",
                r["rx_topology"],
                f"{r['L_tx_uH']:.3f}",
                f"{r['L_rx_uH']:.3f}",
                f"{r['M_uH']:.4f}",
                f"{r['R_tx_ohm']:.3f}",
                f"{r['R_rx_ohm']:.3f}",
                f"{r['DCR_tx_ohm']*1e3:.1f}",
                f"{r['DCR_rx_ohm']*1e3:.1f}",
                f"{r['Q_tx']:.1f}",
                f"{r['Q_rx']:.1f}",
                r["C_tx_label"],
                r["C_rx_label"],
                f"{r['Duty_vmin']:.3f}",
                f"{r['Duty_vmax']:.3f}",
                f"{r['V_ind_min_V']:.2f}",
            )
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

    # -----------------------------------------------------------------------
    # Tree selection → detail panels
    # -----------------------------------------------------------------------

    def _on_tree_select(self, _=None):
        sel = self._tree.selection()
        if not sel:
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        iid   = sel[0]
        items = self._tree.get_children()
        idx   = list(items).index(iid)
        if idx >= len(self._results):
            self._send_btn.configure(state="disabled")
            self._selected_idx = -1
            return
        self._send_btn.configure(state="normal")
        self._selected_idx = idx
        sim_res = getattr(self.app, "sim_tab", None)
        sim_res = sim_res.last_result if sim_res else None
        L, M, R = self._make_detail_columns(idx + 1, self._results[idx], sim_res)
        self._write_col(self._det_nn,    L)
        self._write_col(self._det_sim,   M)
        self._write_col(self._det_delta, R)

    def on_sim_done(self, r_nn, sim_result):
        """Called by AutomationNNTab after FastHenry finishes for a selected row."""
        if self._selected_idx < 0 or self._selected_idx >= len(self._results):
            return
        if self._results[self._selected_idx] is not r_nn:
            return
        L, M, R = self._make_detail_columns(self._selected_idx + 1, r_nn, sim_result)
        self._write_col(self._det_nn,    L)
        self._write_col(self._det_sim,   M)
        self._write_col(self._det_delta, R)

    # -----------------------------------------------------------------------
    # Detail column builder
    # -----------------------------------------------------------------------

    def _make_detail_columns(self, rank, r, sim_res=None):
        topo      = r["rx_topology"]
        eff_turns = r["rx_eff_turns"]
        tx_l_mm   = (_spiral_length_m(r["tx_od_mm"], r["tx_width"], r["tx_turns"])
                     * 1e3 * _TX_N_LAYERS)
        rx_l_mm   = (_spiral_length_m(r["rx_od_mm"], r["rx_width"], r["rx_turns"])
                     * 1e3 * 4)

        have_sim = (sim_res is not None and sim_res.get("n_ports", 0) == 2)
        if have_sim:
            Zmat  = sim_res["Zmat"]
            f_sim = sim_res["frequency"]
            w_s   = 2.0 * math.pi * f_sim
            sL_tx = Zmat[0][0].imag / w_s * 1e6
            sL_rx = Zmat[1][1].imag / w_s * 1e6
            sM    = Zmat[0][1].imag / w_s * 1e6
            sR_tx = Zmat[0][0].real * 1e3
            sR_rx = Zmat[1][1].real * 1e3
            sk    = sM / math.sqrt(sL_tx * sL_rx) if sL_tx > 0 and sL_rx > 0 else 0.0
            sQ_tx = w_s * sL_tx * 1e-6 / Zmat[0][0].real if Zmat[0][0].real > 0 else 0.0
            sQ_rx = w_s * sL_rx * 1e-6 / Zmat[1][1].real if Zmat[1][1].real > 0 else 0.0
        else:
            f_sim = sL_tx = sL_rx = sM = sR_tx = sR_rx = sk = sQ_tx = sQ_rx = None

        nn_k = r["M_uH"] / math.sqrt(r["L_tx_uH"] * r["L_rx_uH"])

        SEP = "─" * 44

        def _pct(nn, s):
            return f" ({(s - nn) / nn * 100:+.1f}%)" if nn != 0 else ""

        def _d(nn_val, s_val, label, fmt, unit):
            if not have_sim:
                return ""
            d = s_val - nn_val
            return f"    {label:<14}{fmt.format(d)}{unit}{_pct(nn_val, s_val)}"

        s_freq = f"    {'Frequency':<16}{f_sim/1e3:.3f} kHz"      if have_sim else "    Frequency        —"
        s_ltx  = f"    {'L_TX':<16}{sL_tx:.4f} µH"                if have_sim else "  —"
        s_lrx  = f"    {'L_RX':<16}{sL_rx:.4f} µH"                if have_sim else "  —"
        s_m    = f"    {'M':<16}{sM:.5f} µH"                       if have_sim else "  —"
        s_k    = f"    {'k':<16}{sk:.4f}"                          if have_sim else "  —"
        s_rtx  = f"    {'R_TX':<16}{sR_tx:.2f} mΩ"                 if have_sim else "  —"
        s_rrx  = f"    {'R_RX':<16}{sR_rx:.2f} mΩ"                 if have_sim else "  —"
        s_qtx  = f"    {'Q_TX':<16}{sQ_tx:.1f}"                    if have_sim else "  —"
        s_qrx  = f"    {'Q_RX':<16}{sQ_rx:.1f}"                    if have_sim else "  —"

        em_hdr_m = "  EM PARAMETERS  (Simulated)"    if have_sim else "  EM PARAMETERS  (run sim…)"
        em_hdr_r = "  DELTA  (Sim − NN)"             if have_sim else "  DELTA  (run sim…)"

        rows = [
            (SEP, SEP, SEP),
            (f"  Rank #{rank}  —  Fitness {r['fitness']:.6f}",
             "  FastHenry Sim Results",
             "  Delta  (Sim − NN)"),
            (SEP, SEP, SEP),
            ("", "", ""),
            ("  GEOMETRY  (T/Ly = turns/layer)", "", ""),
            (f"    TX: OD={r['tx_od_mm']:.1f}  T={r['tx_turns']}  W={r['tx_width']:.3f} mm", "", ""),
            (f"        ID={r['tx_id_mm']:.1f} mm  wire={tx_l_mm:.0f} mm  (3 layers ||)", "", ""),
            (f"    RX: OD={r['rx_od_mm']:.1f}  T={r['rx_turns']}  W={r['rx_width']:.3f} mm  topo={topo}", "", ""),
            (f"        ID={r['rx_id_mm']:.1f} mm  wire={rx_l_mm:.0f} mm"
             + (f"  eff.turns={eff_turns}" if topo != "parallel" else ""),
             "", ""),
            ("", "", ""),
            ("  DRIVE", "", ""),
            (f"    {'Frequency':<16}{r['freq_hz']/1e3:.3f} kHz", s_freq, ""),
            (f"    {'C_TX':<16}{r['C_tx_nf']:.4f} nF  ({r['C_tx_label']})", "", ""),
            (f"    {'C_RX':<16}{r['C_rx_nf']:.4f} nF  ({r['C_rx_label']})", "", ""),
            ("", "", ""),
            ("  EM PARAMETERS  (NN predicted)", em_hdr_m, em_hdr_r),
            (f"    {'L_TX':<16}{r['L_tx_uH']:.4f} µH",
             s_ltx,
             _d(r["L_tx_uH"], sL_tx, "ΔL_TX", "{:+.4f}", " µH")),
            (f"    {'L_RX':<16}{r['L_rx_uH']:.4f} µH",
             s_lrx,
             _d(r["L_rx_uH"], sL_rx, "ΔL_RX", "{:+.4f}", " µH")),
            (f"    {'M':<16}{r['M_uH']:.5f} µH",
             s_m,
             _d(r["M_uH"], sM, "ΔM", "{:+.5f}", " µH")),
            (f"    {'k':<16}{nn_k:.4f}",
             s_k,
             _d(nn_k, sk, "Δk", "{:+.4f}", "")),
            (f"    {'R_TX':<16}{r['R_tx_ohm']*1e3:.2f} mΩ"
             f"  (DCR={r['DCR_tx_ohm']*1e3:.2f}  skin={max(0,r['R_tx_ohm']-r['DCR_tx_ohm'])*1e3:.2f})",
             s_rtx,
             _d(r["R_tx_ohm"]*1e3, sR_tx, "ΔR_TX", "{:+.2f}", " mΩ")),
            (f"    {'R_RX':<16}{r['R_rx_ohm']*1e3:.2f} mΩ"
             f"  (DCR={r['DCR_rx_ohm']*1e3:.2f}  skin={max(0,r['R_rx_ohm']-r['DCR_rx_ohm'])*1e3:.2f})",
             s_rrx,
             _d(r["R_rx_ohm"]*1e3, sR_rx, "ΔR_RX", "{:+.2f}", " mΩ")),
            (f"    {'Q_TX':<16}{r['Q_tx']:.1f}", s_qtx, ""),
            (f"    {'Q_RX':<16}{r['Q_rx']:.1f}", s_qrx, ""),
            ("", "", ""),
            ("  CIRCUIT", "", ""),
            (f"    Z_in = {r['Zin_re']*1e3:.2f}+j{r['Zin_im']*1e3:.2f} mΩ"
             f"  ({'inductive ✓' if r['Zin_im'] > 0 else 'CAPACITIVE ✗'})",
             "", ""),
            (f"    V_induced_min = {r['V_ind_min_V']:.3f} V", "", ""),
            ("", "", ""),
            ("  PERFORMANCE", "", ""),
            (f"    Duty @ V_min   {r['Duty_vmin']:.4f}  ({r['Duty_vmin']*100:.1f}%)", "", ""),
            (f"    Duty @ V_max   {r['Duty_vmax']:.4f}  ({r['Duty_vmax']*100:.1f}%)", "", ""),
            (f"    Efficiency     {r['eff_mid']*100:.2f}%  (at mid supply)", "", ""),
        ]

        L = [row[0] for row in rows]
        M = [row[1] for row in rows]
        R = [row[2] for row in rows]
        n = max(len(L), len(M), len(R))
        L += [""] * (n - len(L))
        M += [""] * (n - len(M))
        R += [""] * (n - len(R))
        return L, M, R

    # -----------------------------------------------------------------------
    # Send to sim
    # -----------------------------------------------------------------------

    def _on_send_to_sim(self):
        if self._selected_idx < 0 or self._selected_idx >= len(self._results):
            return
        r = self._results[self._selected_idx]
        auto_nn = getattr(self.app, "auto_nn_tab", None)
        if auto_nn is None:
            messagebox.showerror("Send to Sim", "AutomationNNTab not available.")
            return
        # Sync run params so p_target propagates correctly
        auto_nn._last_run_params = self._run_params
        auto_nn._on_send_to_sim(r)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _write_col(self, widget, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", "\n".join(lines))
        widget.configure(state="disabled")

    def _clear_detail(self):
        for w in (self._det_nn, self._det_sim, self._det_delta):
            w.configure(state="normal")
            w.delete("1.0", "end")
            w.configure(state="disabled")
