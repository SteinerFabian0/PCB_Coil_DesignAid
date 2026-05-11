#!/usr/bin/env python3
"""
Coil2Inductor — master coordinator (slim).
Tab order: DXF TX | DXF RX | Parametric TX | Parametric RX | Simulation | Automation | Automation NN.
Opens on Parametric TX.
"""

import os, sys, glob, shutil
import tkinter as tk
from tkinter import ttk

# Must be set before any runner import so child processes inherit it too.
_LINUX_MODE = "--linux" in sys.argv
if _LINUX_MODE:
    os.environ["FASTHENRY_BACKEND"] = "linux"
    sys.argv.remove("--linux")

# Scale factor applied to all tkinter fonts, window geometry, and fixed pixel
# widths on Linux. Increase if widgets are still too small, decrease if overflow.
LINUX_SCALE = 1.75
if _LINUX_MODE:
    os.environ["COIL_GUI_SCALE"] = str(LINUX_SCALE)

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
from sim_nn_tab import SimNNTab
from dxf_coil_tab import DxfCoilTab
from nn_setup import AutomationTab
from nn_optimisation import NNOptimisationTab
from nn_analysis_tab import NNAnalysisTab

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
        if _LINUX_MODE:
            self.tk.call("tk", "scaling", LINUX_SCALE)
            self.geometry(f"{int(1600 * LINUX_SCALE)}x{int(850 * LINUX_SCALE)}")
        else:
            self.geometry("1600x850")

        # Load savestate once, keep in memory; writes go through helpers.
        self._state = savestate.load(PROJECT_ROOT)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Apply custom entry styles for the NN tab range validation.
        SimNNTab.configure_styles()

        # Sim first so coil tabs can call get_target_freq() during init.
        self.sim_tab = SimTab(nb, app=self, temp_dir=TEMP_DIR)
        self.sim_nn_tab = SimNNTab(nb, app=self)

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
        self.auto_tab    = AutomationTab(nb, app=self)
        self.nn_optim_tab = NNOptimisationTab(nb, app=self,
                                             on_next_tab=lambda: self._advance_tab(6))
        self.nn_analysis_tab = NNAnalysisTab(nb, app=self)

        nb.add(self.dxf_tx_tab,      text="  DXF TX  ")
        nb.add(self.dxf_rx_tab,      text="  DXF RX  ")
        nb.add(self.param_tx_tab,    text="  Parametric TX  ")
        nb.add(self.param_rx_tab,    text="  Parametric RX  ")
        nb.add(self.sim_tab,         text="  Simulation  ")
        nb.add(self.sim_nn_tab,      text="  Simulation NN  ")
        nb.add(self.auto_tab,        text="  NN Setup  ")
        nb.add(self.nn_optim_tab,    text="  NN Optimisation  ", state="disabled")
        nb.add(self.nn_analysis_tab, text="  NN Analysis  ")

        self._ordered_tabs = [self.dxf_tx_tab, self.dxf_rx_tab,
                              self.param_tx_tab, self.param_rx_tab,
                              self.sim_tab, self.auto_tab,
                              self.nn_optim_tab, self.nn_analysis_tab]

        # Restore active tab from savestate, fall back to Parametric TX.
        saved_tab_name = self._state.get("ui", {}).get("active_tab", "")
        _tab_map = {
            "dxf_tx":       self.dxf_tx_tab,
            "dxf_rx":       self.dxf_rx_tab,
            "param_tx":     self.param_tx_tab,
            "param_rx":     self.param_rx_tab,
            "simulation":   self.sim_tab,
            "sim_nn":       self.sim_nn_tab,
            "nn_setup":     self.auto_tab,
            "nn_optim":     self.nn_optim_tab,
            "nn_analysis":  self.nn_analysis_tab,
        }
        _start_tab = _tab_map.get(saved_tab_name, self.param_tx_tab)
        nb.select(_start_tab)
        self._nb = nb

        # Save active tab whenever the user switches.
        _tab_names = {id(v): k for k, v in _tab_map.items()}
        def _on_tab_changed(event):
            try:
                widget = nb.nametowidget(nb.select())
                name = _tab_names.get(id(widget), "")
                if name:
                    self._state.setdefault("ui", {})["active_tab"] = name
                    savestate.save(PROJECT_ROOT, self._state)
            except Exception:
                pass
        nb.bind("<<NotebookTabChanged>>", _on_tab_changed)

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

    def persist_nn_setup_folder(self, folder: str):
        self._state.setdefault("nn_setup", {})["folder"] = folder
        savestate.save(PROJECT_ROOT, self._state)

    def load_nn_setup_folder(self) -> str:
        return self._state.get("nn_setup", {}).get("folder", "")

    def persist_nn_setup_inputs(self, inputs: dict):
        self._state.setdefault("nn_setup", {})["inputs"] = inputs
        savestate.save(PROJECT_ROOT, self._state)

    def load_nn_setup_inputs(self) -> dict:
        return self._state.get("nn_setup", {}).get("inputs", {})

    def persist_nn_optim_tab(self, state: dict):
        self._state.setdefault("nn_optim", {}).update(state)
        savestate.save(PROJECT_ROOT, self._state)

    def load_nn_optim_tab_state(self) -> dict:
        return self._state.get("nn_optim", {})

    def set_nn_optim_tab_visible(self, visible: bool):
        """Show or hide (disable) the NN Optimisation tab."""
        try:
            state = "normal" if visible else "disabled"
            self._nb.tab(self.nn_optim_tab, state=state)
        except Exception:
            pass

    def _on_close(self):
        # Final flush.
        savestate.save(PROJECT_ROOT, self._state)
        self.destroy()


def main():
    CoilApp().mainloop()


if __name__ == "__main__":
    main()