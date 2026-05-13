#!/usr/bin/env python3
"""
ResultsTab — top 15 FastHenry-validated combinations from refined_results.json,
scored with the identical physics as the NN Optimisation sweep.

No inputs: all scoring parameters are read from the NN Optimisation tab.
Refresh is triggered on startup or via the Refresh button.
"""

import json
import math
import os
import sys
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk

_HERE     = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES  = os.path.join(_APP_ROOT, "Modules")
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

_SPACING_MM       = 0.16
_OZ_MM            = 0.035
_RHO_30C          = 1.724e-8 * (1 + 0.00393 * 10)
_V_MIN_INDUCED_DC = 3.6
_V_ZENER_DC       = 6.8
TOP_N             = 50


# ─── Physics helpers (mirrors nn_optimisation to avoid circular import) ───────

def _spiral_length_m(od_mm, w_mm, turns):
    pitch   = w_mm + _SPACING_MM
    R0      = od_mm / 2.0 - w_mm / 2.0
    R_inner = R0 - (2.0 * turns - 1.0) * pitch / 2.0
    return np.pi * turns * (R0 + R_inner) * 1e-3


def _dcr_ohm(length_m, width_mm, h_total_m):
    return _RHO_30C * length_m / np.maximum(width_mm * 1e-3 * h_total_m, 1e-15)


def _cap_label(c_nf):
    try:
        from cap_combinator import find_best_cap
        _, desc = find_best_cap(c_nf, max_caps=2)
        return desc if desc else f"{c_nf:.1f} nF"
    except Exception:
        return f"{c_nf:.1f} nF"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — identical pipeline to _nn_sweep, applied to JSON entries directly
# ─────────────────────────────────────────────────────────────────────────────

def _score_entries(entries, params):
    """
    Score refined_results entries using the identical _score_torch pipeline:
      • DCR computed from geometry + stackup oz (same formula as _nn_sweep)
      • R_tx_ac / R_rx_ac from FastHenry used as the 'NN prediction'
      • _score_torch evaluates all C_rx options, picks best per entry
    Returns a list of result dicts sorted best-first, length ≤ TOP_N.
    """
    from nn_optimisation import _score_torch   # lazy import avoids circular dep
    import torch

    tx_oz        = params["tx_oz_per_layer"]
    rx_oz        = params["rx_oz_per_layer"]
    tx_h_total_m = sum(oz * _OZ_MM * 1e-3 for oz in tx_oz if oz > 0) or 1e-9
    _h_outer     = rx_oz[0] * _OZ_MM * 1e-3
    _h_inner     = rx_oz[1] * _OZ_MM * 1e-3

    # ── Extract arrays from JSON entries ──────────────────────────────────────
    def _col(key, fallback=None):
        if fallback is None:
            return np.array([e[key] for e in entries], np.float32)
        return np.array([e.get(key, e[fallback] if fallback in e else 0)
                         for e in entries], np.float32)

    tx_turns_s       = _col("tx_turns")
    tx_width_s       = _col("tx_width")
    tx_od_s          = _col("tx_od_mm")
    rx_turns_s       = _col("rx_turns")
    rx_inner_turns_s = np.array([e.get("rx_inner_turns", e["rx_turns"])
                                  for e in entries], np.float32)
    rx_width_s       = _col("rx_width")
    rx_od_s          = _col("rx_od_mm")
    L_tx_np          = _col("L_tx_uH")
    L_rx_np          = _col("L_rx_uH")
    M_np             = _col("M_uH")
    R_tx_np          = _col("R_tx_ac")
    R_rx_np          = _col("R_rx_ac")

    # ── DCR — same formula as _nn_sweep ───────────────────────────────────────
    tx_len_m  = _spiral_length_m(tx_od_s, tx_width_s, tx_turns_s)
    DCR_tx_np = _dcr_ohm(tx_len_m, tx_width_s, tx_h_total_m).astype(np.float32)

    _pitch_outer_rx = rx_width_s + _SPACING_MM
    _inner_safe     = np.maximum(rx_inner_turns_s, 1.0)
    _pitch_inner_rx = (rx_turns_s / _inner_safe) * _pitch_outer_rx
    rx_w_inner_s    = np.maximum(_pitch_inner_rx - _SPACING_MM, 1e-6).astype(np.float32)
    rx_len_outer_m  = _spiral_length_m(rx_od_s, rx_width_s,   rx_turns_s)
    rx_len_inner_m  = _spiral_length_m(rx_od_s, rx_w_inner_s, rx_inner_turns_s)
    DCR_rx_np = (
        2.0 * _dcr_ohm(rx_len_outer_m, rx_width_s,   _h_outer)
      + 2.0 * _dcr_ohm(rx_len_inner_m, rx_w_inner_s, _h_inner)
    ).astype(np.float32)

    # ── Score on GPU/CPU via _score_torch ─────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crx_arr = np.array(params["c_rx_options_f"], dtype=np.float32)
    crx_g   = torch.as_tensor(crx_arr, device=device)

    def _t(a): return torch.from_numpy(a).to(device)

    (score_2d, eta_2d, _, feas_rel_2d,
     D_vmax_g, D_vmin_g, _,
     f0_tx_g, f0_rx_2d,
     I_tx_pk_g, _) = _score_torch(
        _t(L_tx_np), _t(L_rx_np), _t(M_np),
        _t(R_tx_np), _t(R_rx_np),
        _t(DCR_tx_np), _t(DCR_rx_np),
        params["c_tx_f"], crx_g,
        params["freq_min_hz"], params["freq_max_hz"], params["dither_amp_hz"],
        params["v_min"], params["v_max"], params["p_target_w"],
        params["d_min"], params["d_max"],
        params["zvs_margin"], _V_MIN_INDUCED_DC, _V_ZENER_DC,
        params["v_cap_target_dc"], params["v_ldo_out"], params["v_rx_cap_rating"],
        torch)

    inf_neg  = float("-inf")
    inf_t    = torch.tensor(inf_neg, dtype=score_2d.dtype, device=device)
    score_2d = torch.where(feas_rel_2d, score_2d, inf_t)
    eta_2d   = torch.where(feas_rel_2d, eta_2d,   inf_t)

    n                = len(entries)
    best_score, best_cap_idx = score_2d.max(dim=1)
    best_eta,   _            = eta_2d.max(dim=1)

    ri              = torch.arange(n, device=device)
    best_score_np   = best_score.cpu().numpy()
    best_eta_np     = best_eta.cpu().numpy()
    best_cap_idx_np = best_cap_idx.cpu().numpy()
    best_D_vmax     = D_vmax_g[ri, best_cap_idx].cpu().numpy()
    best_D_vmin     = D_vmin_g[ri, best_cap_idx].cpu().numpy()
    best_f0_rx      = f0_rx_2d[ri, best_cap_idx].cpu().numpy()
    best_I_tx_pk_mA = I_tx_pk_g[ri, best_cap_idx].cpu().numpy() * 1e3
    f0_tx_np        = f0_tx_g.cpu().numpy()
    best_cap_nf     = crx_arr[best_cap_idx_np] * 1e9
    c_tx_nf         = params["c_tx_f"] * 1e9

    # ── Build result rows — sorted by system efficiency ───────────────────────
    order = np.argsort(-best_eta_np)
    rows  = []
    for rank_i, idx in enumerate(order):
        if len(rows) >= TOP_N:
            break
        if best_score_np[idx] <= inf_neg:
            break
        e = entries[idx]

        # Effective AC resistance after DCR clamp (same as _score_torch)
        R_tx_eff = float(DCR_tx_np[idx]) + max(0.0, float(R_tx_np[idx]) - float(DCR_tx_np[idx]))
        R_rx_eff = float(DCR_rx_np[idx]) + max(0.0, float(R_rx_np[idx]) - float(DCR_rx_np[idx]))

        L_tx_v = max(float(L_tx_np[idx]) * 1e-6, 1e-9)
        L_rx_v = max(float(L_rx_np[idx]) * 1e-6, 1e-9)
        M_v    = max(float(M_np[idx]) * 1e-6, 0.0)
        k      = min(M_v / math.sqrt(L_tx_v * L_rx_v), 1.0)
        f_op   = float(best_f0_rx[idx])
        omega  = 2.0 * math.pi * f_op
        Q_tx   = omega * L_tx_v / max(R_tx_eff, 1e-12)
        Q_rx   = omega * L_rx_v / max(R_rx_eff, 1e-12)
        U      = k * math.sqrt(max(Q_tx * Q_rx, 0.0))

        rows.append(dict(
            rank          = rank_i + 1,
            eta_sys       = float(best_eta_np[idx]),
            U=U, k=k, Q_tx=Q_tx, Q_rx=Q_rx,
            f0_tx_hz      = float(f0_tx_np[idx]),
            f_op_hz       = f_op,
            # geometry
            tx_turns      = int(e["tx_turns"]),
            tx_l2_turns   = int(e.get("tx_l2_turns", e["tx_turns"])),
            tx_width      = float(e["tx_width"]),
            tx_od_mm      = float(e["tx_od_mm"]),
            rx_turns      = int(e["rx_turns"]),
            rx_inner_turns= int(e.get("rx_inner_turns", e["rx_turns"])),
            rx_width      = float(e["rx_width"]),
            rx_od_mm      = float(e["rx_od_mm"]),
            # EM
            L_tx_uH       = float(e["L_tx_uH"]),
            L_rx_uH       = float(e["L_rx_uH"]),
            M_uH          = float(e["M_uH"]),
            R_tx_ac       = R_tx_eff,
            R_rx_ac       = R_rx_eff,
            DCR_tx        = float(DCR_tx_np[idx]),
            DCR_rx        = float(DCR_rx_np[idx]),
            D_vmax        = float(best_D_vmax[idx]),
            D_vmin        = float(best_D_vmin[idx]),
            I_tx_pk_mA    = float(best_I_tx_pk_mA[idx]),
            C_tx_nf       = c_tx_nf,
            C_rx_nf       = float(best_cap_nf[idx]),
            C_tx_label    = _cap_label(c_tx_nf),
            C_rx_label    = _cap_label(float(best_cap_nf[idx])),
        ))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(optim_tab, done_cb, error_cb):
    try:
        params = optim_tab._parse_params()
    except Exception as e:
        error_cb(str(e))
        return

    refined_path = os.path.join(params["model_dir"], "refined_results.json")
    if not os.path.isfile(refined_path):
        error_cb(
            f"refined_results.json not found in model folder.\n"
            f"Run at least one NN Optimisation iteration first.\n({refined_path})")
        return

    try:
        with open(refined_path) as f:
            data = json.load(f)
    except Exception as e:
        error_cb(f"Failed to read refined_results.json:\n{e}")
        return

    entries = data.get("results", [])
    if not entries:
        error_cb("refined_results.json contains no result rows.")
        return

    try:
        rows = _score_entries(entries, params)
    except Exception as e:
        error_cb(f"Scoring failed:\n{e}\n{traceback.format_exc()}")
        return

    done_cb(rows, len(entries), params)


# ─────────────────────────────────────────────────────────────────────────────
# Tab widget
# ─────────────────────────────────────────────────────────────────────────────

# Map column header → result-dict key (all numeric so sort works correctly)
_SORT_KEY = {
    "#":          "rank",
    "η_sys%":     "eta_sys",
    "U":          "U",
    "k":          "k",
    "f_op kHz":   "f_op_hz",
    "f0_tx kHz":  "f0_tx_hz",
    "TX OD":      "tx_od_mm",
    "TX T":       "tx_turns",
    "TX L2":      "tx_l2_turns",
    "TX W":       "tx_width",
    "RX OD":      "rx_od_mm",
    "RX T":       "rx_turns",
    "RX W":       "rx_width",
    "RX In":      "rx_inner_turns",
    "L_tx µH":    "L_tx_uH",
    "L_rx µH":    "L_rx_uH",
    "M µH":       "M_uH",
    "R_tx mΩ":    "R_tx_ac",
    "R_rx mΩ":    "R_rx_ac",
    "DCR_tx mΩ":  "DCR_tx",
    "DCR_rx mΩ":  "DCR_rx",
    "C_tx":       "C_tx_nf",
    "C_rx":       "C_rx_nf",
    "D@Vmin%":    "D_vmin",
    "D@Vmax%":    "D_vmax",
    "TX Ipk mA":  "I_tx_pk_mA",
}


class ResultsTab(ttk.Frame):

    def __init__(self, parent, app=None, **kw):
        super().__init__(parent, **kw)
        self.app           = app
        self._results      = []   # always eta-sorted
        self._params       = {}
        self._running      = False
        self._selected_idx = -1
        self._sort_col     = "η_sys%"   # default: efficiency descending
        self._sort_asc     = False
        self._build()
        self.after(600, self._refresh)

    # ─── UI construction ─────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(2, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ttk.Frame(self)
        hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        self._refresh_btn = ttk.Button(hdr, text="Refresh Results",
                                       command=self._refresh, width=16)
        self._refresh_btn.pack(side="left", padx=(0, 10))

        self._status_lbl = ttk.Label(hdr, text="Click Refresh to score refined_results.json.",
                                     foreground="gray", font=("TkDefaultFont", 9))
        self._status_lbl.pack(side="left")

        # ── Table ─────────────────────────────────────────────────────────────
        tbl_frame = ttk.LabelFrame(self, text="Top Results  (ranked by system efficiency — "
                                              "parameters from NN Optimisation tab)")
        tbl_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 3))
        self._build_table(tbl_frame)

        # ── Detail ────────────────────────────────────────────────────────────
        det_frame = ttk.LabelFrame(self, text="Detail  (click a row)")
        det_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 6))
        self._build_detail(det_frame)

    def _build_table(self, parent):
        cols = (
            "#", "η_sys%", "U", "k",
            "f_op kHz", "f0_tx kHz",
            "TX OD", "TX T", "TX L2", "TX W",
            "RX OD", "RX T", "RX W", "RX In",
            "L_tx µH", "L_rx µH", "M µH",
            "R_tx mΩ", "R_rx mΩ", "DCR_tx mΩ", "DCR_rx mΩ",
            "C_tx", "C_rx",
            "D@Vmin%", "D@Vmax%",
            "TX Ipk mA",
        )
        widths = {
            "#": 28,
            "η_sys%": 56, "U": 46, "k": 52,
            "f_op kHz": 66, "f0_tx kHz": 68,
            "TX OD": 50, "TX T": 36, "TX L2": 40, "TX W": 48,
            "RX OD": 50, "RX T": 36, "RX W": 48, "RX In": 44,
            "L_tx µH": 58, "L_rx µH": 58, "M µH": 52,
            "R_tx mΩ": 62, "R_rx mΩ": 62, "DCR_tx mΩ": 68, "DCR_rx mΩ": 68,
            "C_tx": 110, "C_rx": 110,
            "D@Vmin%": 66, "D@Vmax%": 66,
            "TX Ipk mA": 72,
        }

        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=4, pady=4)
        xsb = ttk.Scrollbar(frm, orient="horizontal")
        ysb = ttk.Scrollbar(frm, orient="vertical")
        self._tree = ttk.Treeview(frm, columns=cols, show="headings",
                                  yscrollcommand=ysb.set, xscrollcommand=xsb.set,
                                  height=13)
        xsb.configure(command=self._tree.xview)
        ysb.configure(command=self._tree.yview)
        xsb.pack(side="bottom", fill="x")
        ysb.pack(side="right",  fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        for c in cols:
            self._tree.heading(c, text=c,
                               command=lambda col=c: self._on_heading_click(col))
            self._tree.column(c, width=widths.get(c, 64), minwidth=28, stretch=False)

        self._tree.tag_configure("rank1", background="#d4edda")
        self._tree.tag_configure("rank2", background="#e8f4fd")
        self._tree.tag_configure("rank3", background="#fff3cd")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_detail(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=4, pady=4)
        vsb = ttk.Scrollbar(frm, orient="vertical")
        vsb.pack(side="right", fill="y")
        self._det = tk.Text(frm, height=8, state="disabled",
                            font=("Consolas", 9), wrap="none",
                            background="#f8f8f8", borderwidth=0,
                            highlightthickness=0,
                            yscrollcommand=vsb.set)
        vsb.configure(command=self._det.yview)
        self._det.pack(side="left", fill="both", expand=True)

    # ─── Refresh logic ───────────────────────────────────────────────────────

    def _refresh(self):
        if self._running:
            return
        optim_tab = getattr(self.app, "nn_optim_tab", None)
        if optim_tab is None:
            self._set_status("NN Optimisation tab not found.", "red")
            return
        self._running = True
        self._refresh_btn.configure(state="disabled")
        self._set_status("Scoring refined_results.json…")
        threading.Thread(
            target=_worker,
            args=(optim_tab, self._done_cb, self._error_cb),
            daemon=True,
        ).start()

    def _done_cb(self, rows, n_total, params):
        self.after(0, lambda: self._on_done(rows, n_total, params))

    def _error_cb(self, msg):
        self.after(0, lambda: self._on_error(msg))

    def _on_done(self, rows, n_total, params):
        self._running      = False
        self._results      = rows
        self._params       = params
        self._selected_idx = -1
        self._refresh_btn.configure(state="normal")
        self._clear_detail()
        self._populate_tree(rows)

        if rows:
            cap_str = ", ".join(f"{c*1e9:g}" for c in params["c_rx_options_f"])
            self._set_status(
                f"Showing {len(rows)} / {n_total} results  |  "
                f"best η = {rows[0]['eta_sys']*100:.2f}%  "
                f"@ {rows[0]['f_op_hz']/1e3:.1f} kHz  |  "
                f"C_rx options: [{cap_str}] nF", "green")
        else:
            self._set_status(
                f"No feasible combinations found in {n_total} results — "
                "check power / voltage / frequency settings in NN Optimisation tab.", "orange")

    def _on_error(self, msg):
        self._running = False
        self._refresh_btn.configure(state="normal")
        self._set_status(msg, "red")

    # ─── Column sort ─────────────────────────────────────────────────────────

    def _on_heading_click(self, col):
        if col not in _SORT_KEY:
            return
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            # numeric columns: default descending (biggest first); rank asc
            self._sort_asc = (col == "#")
        self._populate_tree(self._results)

    def _sorted_rows(self, rows):
        key = _SORT_KEY.get(self._sort_col, "rank")
        return sorted(rows, key=lambda r: r[key], reverse=not self._sort_asc)

    def _update_headings(self):
        for c in self._tree["columns"]:
            label = c
            if c == self._sort_col:
                label = c + ("  ▲" if self._sort_asc else "  ▼")
            self._tree.heading(c, text=label,
                               command=lambda col=c: self._on_heading_click(col))

    # ─── Table population ────────────────────────────────────────────────────

    def _populate_tree(self, rows):
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        self._update_headings()
        sorted_rows = self._sorted_rows(rows)
        # rank tags follow efficiency rank, not display position
        rank_tag = {1: "rank1", 2: "rank2", 3: "rank3"}
        for r in sorted_rows:
            tag  = rank_tag.get(r["rank"], "")
            vals = (
                r["rank"],   # always the efficiency rank
                f"{r['eta_sys']*100:.2f}",
                f"{r['U']:.2f}",
                f"{r['k']:.4f}",
                f"{r['f_op_hz']/1e3:.2f}",
                f"{r['f0_tx_hz']/1e3:.2f}",
                f"{r['tx_od_mm']:.1f}",
                r["tx_turns"],
                r["tx_l2_turns"],
                f"{r['tx_width']:.3f}",
                f"{r['rx_od_mm']:.1f}",
                r["rx_turns"],
                f"{r['rx_width']:.3f}",
                r["rx_inner_turns"],
                f"{r['L_tx_uH']:.1f}",
                f"{r['L_rx_uH']:.1f}",
                f"{r['M_uH']:.1f}",
                f"{r['R_tx_ac']*1e3:.0f}",
                f"{r['R_rx_ac']*1e3:.0f}",
                f"{r['DCR_tx']*1e3:.0f}",
                f"{r['DCR_rx']*1e3:.0f}",
                r["C_tx_label"],
                r["C_rx_label"],
                f"{r['D_vmin']*100:.1f}",
                f"{r['D_vmax']*100:.1f}",
                f"{r['I_tx_pk_mA']:.0f}",
            )
            self._tree.insert("", "end", values=vals, tags=(tag,) if tag else ())

    # ─── Row selection → detail ──────────────────────────────────────────────

    def _on_select(self, _=None):
        sel = self._tree.selection()
        if not sel:
            self._selected_idx = -1
            return
        # First value in the row is the efficiency rank (1-based); map to list index
        rank = self._tree.item(sel[0], "values")[0]
        try:
            rank = int(rank)
        except (ValueError, TypeError):
            self._selected_idx = -1
            return
        idx = rank - 1
        if idx < 0 or idx >= len(self._results):
            self._selected_idx = -1
            return
        self._selected_idx = idx
        self._show_detail(self._results[idx])

    def _show_detail(self, r):
        p   = self._params
        SEP = "─" * 72
        skin_tx = max(0.0, r["R_tx_ac"] - r["DCR_tx"])
        skin_rx = max(0.0, r["R_rx_ac"] - r["DCR_rx"])
        lines = [
            SEP,
            (f"  Rank #{r['rank']}   η_sys = {r['eta_sys']*100:.3f}%   "
             f"U = {r['U']:.3f}   k = {r['k']:.5f}   "
             f"Q_tx = {r['Q_tx']:.1f}   Q_rx = {r['Q_rx']:.1f}"),
            SEP,
            "",
            "  GEOMETRY",
            (f"    TX   OD = {r['tx_od_mm']:.2f} mm   turns = {r['tx_turns']}"
             f"   L2 = {r['tx_l2_turns']}   W = {r['tx_width']:.4f} mm"),
            (f"    RX   OD = {r['rx_od_mm']:.2f} mm   outer = {r['rx_turns']}"
             f"   inner = {r['rx_inner_turns']}   W = {r['rx_width']:.4f} mm"),
            "",
            "  EM  (FastHenry validated — scored at operating frequency)",
            (f"    f0_tx    {r['f0_tx_hz']/1e3:.3f} kHz    "
             f"C_tx = {r['C_tx_nf']:.1f} nF  →  {r['C_tx_label']}"),
            (f"    f_op     {r['f_op_hz']/1e3:.3f} kHz    "
             f"C_rx = {r['C_rx_nf']:.1f} nF  →  {r['C_rx_label']}"),
            f"    L_tx     {r['L_tx_uH']:.1f} µH",
            f"    L_rx     {r['L_rx_uH']:.1f} µH",
            f"    M        {r['M_uH']:.1f} µH",
            (f"    R_tx_ac  {r['R_tx_ac']*1e3:.0f} mΩ   "
             f"(DCR = {r['DCR_tx']*1e3:.0f} mΩ   skin = {skin_tx*1e3:.0f} mΩ)"),
            (f"    R_rx_ac  {r['R_rx_ac']*1e3:.0f} mΩ   "
             f"(DCR = {r['DCR_rx']*1e3:.0f} mΩ   skin = {skin_rx*1e3:.0f} mΩ)"),
            f"    TX Ipk   {r['I_tx_pk_mA']:.0f} mA  (peak, at Vmax)",
            "",
            "  SCORING  (parameters from NN Optimisation tab)",
            (f"    V_min = {p.get('v_min', 0):.2f} V   "
             f"V_max = {p.get('v_max', 0):.2f} V   "
             f"P_target = {p.get('p_target_w', 0)*1000:.1f} mW"),
            (f"    D_min = {p.get('d_min', 0)*100:.1f}%   "
             f"D_max = {p.get('d_max', 1.0)*100:.0f}%   "
             f"ZVS margin = {p.get('zvs_margin', 0)*100:.0f}%"),
            (f"    D@Vmin = {r['D_vmin']*100:.1f}%   "
             f"D@Vmax = {r['D_vmax']*100:.1f}%"),
        ]
        self._det.configure(state="normal")
        self._det.delete("1.0", "end")
        self._det.insert("end", "\n".join(lines))
        self._det.configure(state="disabled")

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _clear_detail(self):
        self._det.configure(state="normal")
        self._det.delete("1.0", "end")
        self._det.configure(state="disabled")

    def _set_status(self, msg, color="gray"):
        self._status_lbl.configure(text=msg, foreground=color)
