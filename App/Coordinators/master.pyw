#!/usr/bin/env python3
"""Coil2Inductor — master GUI shell. Tab classes live in App/Tabs/."""

import os
import sys
import glob
import shutil
import tkinter as tk
from tkinter import ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_APP_ROOT, "Modules"),
           os.path.join(_APP_ROOT, "Tabs"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import savestate
from paths import TEMP_DIR, SAVESTATE_DIR, SAVESTATE_FILE

from coil_tab import CoilTab
from sim_tab import SimTab
from parametric_tab import ParametricCoilTab
from placeholder_tab import PlaceholderTab


def _clean_temp():
    if os.path.isdir(TEMP_DIR):
        for entry in glob.glob(os.path.join(TEMP_DIR, "*")):
            try:
                if os.path.isfile(entry):
                    os.remove(entry)
                elif os.path.isdir(entry):
                    shutil.rmtree(entry)
            except Exception:
                pass
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SAVESTATE_DIR, exist_ok=True)


_clean_temp()


class CoilApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coil2Inductor")
        self.geometry("1400x760")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Sim tab first so other tabs can call get_target_freq() in init.
        self.sim_tab   = SimTab(nb, app=self)
        self.coil1_tab = CoilTab(nb, app=self, coil_index=1)
        self.coil2_tab = PlaceholderTab(
            nb, "DXF Coil 2 — coming soon.\n\n"
            "Use Parametric Coil 2 as a second coil source for now.")
        self.param1_tab = ParametricCoilTab(
            nb, app=self, coil_index=1, temp_dir=TEMP_DIR)
        self.param2_tab = ParametricCoilTab(
            nb, app=self, coil_index=2, temp_dir=TEMP_DIR)
        self.auto_tab = PlaceholderTab(
            nb, "Automation — coming soon.\n\n"
            "Parametric sweeps: vary turns, layers, spacing, trace width.\n"
            "Drives parametric_coil + port_combiner to batch-run FastHenry.")

        nb.add(self.coil1_tab,  text="  DXF Coil 1  ")
        nb.add(self.coil2_tab,  text="  DXF Coil 2  ")
        nb.add(self.param1_tab, text="  Parametric Coil 1  ")
        nb.add(self.param2_tab, text="  Parametric Coil 2  ")
        nb.add(self.sim_tab,    text="  Simulation  ")
        nb.add(self.auto_tab,   text="  Automation  ")
        self._nb = nb

        self._restore_session()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def on_coil_dxf_state_changed(self, coil_tab):
        loaded = coil_tab.any_dxf_loaded()
        param = self._paired_param_tab(coil_tab)
        if param is None:
            return
        param.set_enabled(not loaded)
        try:
            self._nb.tab(param, state=("disabled" if loaded else "normal"))
        except tk.TclError:
            pass

    def _paired_param_tab(self, coil_tab):
        if coil_tab is self.coil1_tab:
            return self.param1_tab
        return None

    def go_to_next_tab(self):
        """Advance to the next non-disabled tab. No-op at the end."""
        idx = self._nb.index("current")
        n = self._nb.index("end")
        for i in range(idx + 1, n):
            try:
                if self._nb.tab(i, "state") != "disabled":
                    self._nb.select(i); return
            except tk.TclError:
                continue

    def _restore_session(self):
        state = savestate.load_state(SAVESTATE_FILE)
        if not state:
            return
        try:
            self.sim_tab.load_state(state.get("sim", {}))
            coils = state.get("coils", {})
            if "1" in coils and isinstance(self.coil1_tab, CoilTab):
                self.coil1_tab.load_state(coils["1"])
            params = state.get("parametric", {})
            if "1" in params: self.param1_tab.load_state(params["1"])
            if "2" in params: self.param2_tab.load_state(params["2"])
        except Exception as e:
            print(f"[savestate] restore failed: {e}")

    def _snapshot_session(self):
        coils = {}
        if isinstance(self.coil1_tab, CoilTab):
            coils["1"] = self.coil1_tab.save_state()
        return {
            "sim":        self.sim_tab.save_state(),
            "coils":      coils,
            "parametric": {
                "1": self.param1_tab.save_state(),
                "2": self.param2_tab.save_state(),
            },
        }

    def _on_close(self):
        try:
            savestate.save_state(SAVESTATE_FILE, self._snapshot_session())
        except Exception as e:
            print(f"[savestate] save failed: {e}")
        self.destroy()


def main():
    app = CoilApp()
    app.mainloop()


if __name__ == "__main__":
    main()