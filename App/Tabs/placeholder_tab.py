import tkinter as tk
from tkinter import ttk


class PlaceholderTab(ttk.Frame):
    """Greyed-out message tab for unfinished features."""
    def __init__(self, parent, text, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=text, foreground="gray",
                  font=("", 12)).pack(expand=True)