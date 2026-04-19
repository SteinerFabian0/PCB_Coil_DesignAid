"""Shared GUI helpers. Depends on Modules/ being on sys.path."""
import os
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg


OZ_TO_MM = 0.035


def oz_to_mm(oz):
    return oz * OZ_TO_MM


def figure_to_photo(fig, png_path, max_width=None):
    """matplotlib Figure -> Tk PhotoImage via an intermediate PNG."""
    import matplotlib.pyplot as plt
    FigureCanvasAgg(fig)
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    img = Image.open(png_path)
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def inp_z_bounds(inp_path):
    """Returns (z_min, z_max) across all nodes of a FastHenry .inp."""
    import inp_visualizer as viz
    nodes, _ = viz.parse_inp(inp_path)
    if not nodes:
        raise RuntimeError(f"{inp_path}: no nodes parsed")
    zs = [p[2] for p in nodes.values()]
    return min(zs), max(zs)