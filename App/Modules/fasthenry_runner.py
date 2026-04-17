#!/usr/bin/env python3
"""
Wrapper around FastHenry2's OLE Automation interface.

Starts the FastHenry2 COM object (console window stays hidden), runs a
simulation, polls for completion, and pulls results directly without
touching Zc.mat. Zc.mat still lands on disk - we optionally copy it out
to the user's requested location.

Requires pywin32:  pip install pywin32

The API is Windows-only. On other platforms this module will import
but fail at dispatch time.
"""

import os
import time


# Imported lazily so the module can at least be imported on non-Windows
# systems for test discovery etc.
_win32com = None


def _ensure_win32():
    global _win32com
    if _win32com is None:
        import win32com.client as w32c  # noqa: F401
        _win32com = w32c
    return _win32com


# -------------------------------------------------------------------------
# Command-line option assembly
# -------------------------------------------------------------------------

def build_command_line(inp_path, max_iter=180, tol=None, extra_args=None):
    """
    Build the string passed to FastHenry2.Run().

    The Run() method takes "a FastHenry2 command line, as would be given
    from a shell console" (per the docs you shared), so the first token
    is the .inp file and the rest are the usual FastHenry flags.

    Quoting: paths with spaces must be quoted. We always quote.
    """
    parts = [f'"{inp_path}"']
    if max_iter is not None:
        parts.append(f"-i {int(max_iter)}")
    if tol is not None:
        parts.append(f"-t {tol:g}")
    if extra_args:
        parts.append(extra_args)
    return " ".join(parts)


# -------------------------------------------------------------------------
# Result unwrapping
# -------------------------------------------------------------------------

def _safearray_to_nested_list(sa):
    """
    pywin32 unwraps COM SAFEARRAYs to nested tuples. For our 3D
    resistance/inductance arrays we want plain Python lists indexed as
    [freq][row][col], so this normalizes the tuple-of-tuples to
    list-of-lists.

    A 1D SAFEARRAY (e.g. frequencies) comes through as a flat tuple and
    is handled separately.
    """
    if isinstance(sa, tuple):
        return [_safearray_to_nested_list(x) for x in sa]
    return sa


def _extract_single_port_scalar(matrix_3d, freq_index):
    """
    For a single-port extraction, the 2D slice per frequency is 1x1.
    Reach in and pull the scalar.
    """
    return matrix_3d[freq_index][0][0]


# -------------------------------------------------------------------------
# The runner
# -------------------------------------------------------------------------

class FastHenryRunner:
    """
    Context-manager-friendly wrapper.

        with FastHenryRunner() as fh:
            fh.run("combined.inp", max_iter=180)
            freqs, R, L = fh.results()

    The COM server is released on __exit__. The console window is never
    shown (we never call ShowWindow()).
    """

    POLL_INTERVAL_SEC = 0.5
    DEFAULT_TIMEOUT_SEC = 60 * 30   # 30 min is absurdly generous

    def __init__(self):
        self._obj = None
        self._work_dir = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        return False   # don't suppress exceptions

    def _start(self):
        w32c = _ensure_win32()
        import pythoncom
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        self._obj = w32c.Dispatch("FastHenry2.Document")

    def _stop(self):
        if self._obj is None:
            return
        try:
            if self._obj.IsRunning:
                self._obj.Stop
                for _ in range(10):
                    if not self._obj.IsRunning:
                        break
                    time.sleep(0.2)
            self._obj.Quit
        except Exception:
            pass
        self._obj = None

    def run(self, inp_path, max_iter=180, tol=None, extra_args=None,
            timeout_sec=DEFAULT_TIMEOUT_SEC, progress_cb=None):
        """
        Kick off a simulation and block until it finishes or times out.

        progress_cb(elapsed_sec) is called roughly once per POLL_INTERVAL
        so the GUI can update an elapsed timer. It's allowed to be None.

        Returns True on normal completion, False on timeout.
        """
        if self._obj is None:
            self._start()

        # Remember the .inp's directory — Zc.mat is written to CWD of
        # the FastHenry process, which for automation is the folder
        # containing the .inp (observed behavior, not documented).
        self._work_dir = os.path.dirname(os.path.abspath(inp_path))

        cmdline = build_command_line(inp_path,
                                     max_iter=max_iter,
                                     tol=tol,
                                     extra_args=extra_args)

        started = self._obj.Run(cmdline)
        if not started:
            # Most common cause: a previous sim is still running.
            # Run() returns False and doesn't raise. We mirror that.
            raise RuntimeError(
                "FastHenry2.Run() returned False. "
                "A simulation may already be in progress."
            )

        t0 = time.time()
        while self._obj.IsRunning:
            elapsed = time.time() - t0
            if elapsed > timeout_sec:
                self._obj.Stop()
                return False
            if progress_cb is not None:
                try:
                    progress_cb(elapsed)
                except Exception:
                    # Progress callback failures must not kill the sim.
                    pass
            time.sleep(self.POLL_INTERVAL_SEC)

        return True

    # ----- Result pulling -----

    def frequencies(self):
        """Return the sweep frequency list as a tuple of floats."""
        if self._obj is None:
            raise RuntimeError("FastHenry object not initialized")
        raw = self._obj.GetFrequencies
        return tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)

    def inductance(self):
        """Return 3D nested list [freq_idx][row][col] of L in henries."""
        if self._obj is None:
            raise RuntimeError("FastHenry object not initialized")
        return _safearray_to_nested_list(self._obj.GetInductance)

    def resistance(self):
        """Return 3D nested list [freq_idx][row][col] of R in ohms."""
        if self._obj is None:
            raise RuntimeError("FastHenry object not initialized")
        return _safearray_to_nested_list(self._obj.GetResistance)

    def single_port_result(self, target_freq_hz):
        """
        Convenience for our single-port coil case. Pulls the scalar L and
        R at the frequency closest to target_freq_hz.

        Returns dict:
            {"frequency": <actual f used>,
             "L_henry":   <inductance>,
             "R_ohm":     <AC resistance at that f>}
        """
        freqs = self.frequencies()
        if not freqs:
            raise RuntimeError("No frequency data available")
        L = self.inductance()
        R = self.resistance()

        # Pick nearest frequency.
        best_idx = min(range(len(freqs)),
                       key=lambda i: abs(freqs[i] - target_freq_hz))
        return {
            "frequency": freqs[best_idx],
            "L_henry":   _extract_single_port_scalar(L, best_idx),
            "R_ohm":     _extract_single_port_scalar(R, best_idx),
        }

    # ----- Zc.mat export helper -----

    def export_zc_mat(self, dest_path):
        import shutil
        if self._work_dir is None:
            raise RuntimeError("No simulation has been run yet")
        src = os.path.join(self._work_dir, "Zc.mat")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Zc.mat not found in {self._work_dir}")
        # If dest is already the source, nothing to do.
        if os.path.abspath(src) == os.path.abspath(dest_path):
            return
        shutil.copyfile(src, dest_path)