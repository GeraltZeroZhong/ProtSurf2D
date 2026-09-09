"""Consistent vector and raster exports for CLI and GUI maps."""

from pathlib import Path

import matplotlib as mpl


def save_figure(figure, path):
    """Save editable SVG, embedded-font PDF, or publication-resolution raster."""
    suffix = Path(path).suffix.lower()
    if suffix not in {".svg", ".pdf", ".png", ".tif", ".tiff"}:
        raise ValueError("Map exports must use PNG, TIFF, SVG, or PDF.")
    options = {"dpi": 600 if suffix in {".tif", ".tiff"} else 300, "facecolor": "white"}
    if suffix in {".tif", ".tiff"}:
        options["pil_kwargs"] = {"compression": "tiff_lzw"}
    with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(path, **options)
