#!/usr/bin/env python3
"""
Coil2Inductor — master coordinator (slim).
Tab order: DXF TX | DXF RX | Parametric TX | Parametric RX | Simulation | Automation.
Opens on Parametric TX.
"""

import os, sys, glob, shutil
import tkinter as tk
from tkinter import ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
_MODULES_DIR = os.path.join(_APP_ROOT, "Modules")
_TABS_DIR = os.path.join(_APP_ROOT, "Tabs")
_SAVESTATE_DIR = os.path.join(_APP_ROOT, "Savestate")
for _p in (_MODULES_DIR, _TABS_DIR, _SAVESTATE_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import savestate
from parametric_tab import ParametricCoilTab
from sim_tab import SimTab
from dxf_coil_tab import DxfCoilTab
from automation_tab import AutomationTab


PROJECT_ROOT = os.path.dirname(_APP_ROOT)
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")


def _clean_temp():
    if os.path.isdir(TEMP_DIR):
        for entry in glob.glob(os.path.join(TEMP_DIR, "*")):
            try:
                if os.path.isfile(entry): os.remove(entry)
                elif os.path.isdir(entry): shutil.rmtree(entry)
            except Exception: pass
    os.makedirs(TEMP_DIR, exist_ok=True)
_clean_temp()


class PlaceholderTab(ttk.Frame):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=text, foreground="gray",
                  font=("", 12)).pack(expand=True)


class CoilApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coil2Inductor")
        self.geometry("1200x850")

        # Load savestate once, keep in memory; writes go through helpers.
        self._state = savestate.load(PROJECT_ROOT)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Sim first so coil tabs can call get_target_freq() during init.
        self.sim_tab = SimTab(nb, app=self, temp_dir=TEMP_DIR)

        # Coil tabs — built as ordered list so Next Tab can advance
        # without each tab knowing its neighbor.
        self._ordered_tabs = []

        self.dxf_tx_tab   = DxfCoilTab(nb, app=self, role="TX",
                                        temp_dir=TEMP_DIR,
                                        on_next_tab=lambda: self._advance_tab(0))
        self.dxf_rx_tab   = DxfCoilTab(nb, app=self, role="RX",
                                        temp_dir=TEMP_DIR,
                                        on_next_tab=lambda: self._advance_tab(1))
        self.param_tx_tab = ParametricCoilTab(nb, app=self, role="TX",
                                               coil_index=1,
                                               temp_dir=TEMP_DIR,
                                               on_next_tab=lambda: self._advance_tab(2))
        self.param_rx_tab = ParametricCoilTab(nb, app=self, role="RX",
                                               coil_index=2,
                                               temp_dir=TEMP_DIR,
                                               on_next_tab=lambda: self._advance_tab(3))
        self.auto_tab = AutomationTab(nb, app=self)

        nb.add(self.dxf_tx_tab,   text="  DXF TX  ")
        nb.add(self.dxf_rx_tab,   text="  DXF RX  ")
        nb.add(self.param_tx_tab, text="  Parametric TX  ")
        nb.add(self.param_rx_tab, text="  Parametric RX  ")
        nb.add(self.sim_tab,      text="  Simulation  ")
        nb.add(self.auto_tab,     text="  Automation  ")

        self._ordered_tabs = [self.dxf_tx_tab, self.dxf_rx_tab,
                              self.param_tx_tab, self.param_rx_tab,
                              self.sim_tab]

        nb.select(self.param_tx_tab)
        self._nb = nb

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- tab navigation -------------------------------------------------
    def _advance_tab(self, from_idx):
        nxt = min(from_idx + 1, len(self._ordered_tabs) - 1)
        self._nb.select(self._ordered_tabs[nxt])

    # ---- DXF/parametric pairing ----------------------------------------
    def on_coil_dxf_state_changed(self, coil_tab):
        loaded = coil_tab.any_dxf_loaded()
        param = self._paired_param_tab(coil_tab)
        if param is None: return
        param.set_enabled(not loaded)
        try:
            self._nb.tab(param,
                         state=("disabled" if loaded else "normal"))
        except tk.TclError: pass

    def _paired_param_tab(self, coil_tab):
        if coil_tab is self.dxf_tx_tab: return self.param_tx_tab
        if coil_tab is self.dxf_rx_tab: return self.param_rx_tab
        return None

    # ---- sim-tab → coil-tab notifications -------------------------------
    def on_sim_slot_cleared(self, role, source_name):
        """SimTab cleared a source — update the owner's status strip."""
        tabs = {"TX": (self.dxf_tx_tab, self.param_tx_tab),
                "RX": (self.dxf_rx_tab, self.param_rx_tab)}.get(role, ())
        for t in tabs:
            try: t.on_sim_slot_cleared()
            except Exception: pass

    def on_sim_tab_reset(self):
        """Full sim reset — everyone back to 'nothing loaded'."""
        for t in (self.dxf_tx_tab, self.dxf_rx_tab,
                  self.param_tx_tab, self.param_rx_tab):
            try: t.on_sim_slot_cleared()
            except Exception: pass

    # ---- savestate helpers ---------------------------------------------
    def persist_parametric_tab(self, role, state_dict):
        savestate.set_section(self._state, "param", role, state_dict)
        savestate.save(PROJECT_ROOT, self._state)

    def load_parametric_tab_state(self, role):
        return savestate.get_section(self._state, "param", role)

    def persist_dxf_tab(self, role, state_dict):
        savestate.set_section(self._state, "dxf", role, state_dict)
        savestate.save(PROJECT_ROOT, self._state)

    def load_dxf_tab_state(self, role):
        return savestate.get_section(self._state, "dxf", role)

    def persist_sim_tab(self, state_dict):
        self._state.setdefault("sim", {}).update(state_dict)
        savestate.save(PROJECT_ROOT, self._state)

    def load_sim_tab_state(self):
        return self._state.get("sim", {})

    def _on_close(self):
        # Final flush.
        savestate.save(PROJECT_ROOT, self._state)
        self.destroy()


def main():
    CoilApp().mainloop()


if __name__ == "__main__":
    main()