"""GUI entry point for TopoPPI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from topoppi import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topoppi-gui",
        description="Open the TopoPPI desktop application.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _set_window_icon(root) -> None:
    """Set the Tk window icon when packaged assets are available."""
    import sys
    import tkinter as tk
    from importlib import resources

    try:
        asset_files = resources.files("topoppi.assets")
        if sys.platform.startswith("win"):
            with resources.as_file(asset_files / "topoppi.ico") as icon_path:
                root.iconbitmap(default=str(icon_path))

        with resources.as_file(asset_files / "topoppi.png") as icon_path:
            icon_image = tk.PhotoImage(file=str(icon_path))
        root.iconphoto(True, icon_image)
        root._topoppi_icon_image = icon_image
    except (FileNotFoundError, ModuleNotFoundError, OSError, tk.TclError):
        return


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)

    import tkinter as tk

    import matplotlib

    from topoppi.config import DEFAULT_GUI_CONFIG

    matplotlib.use("TkAgg", force=True)

    from topoppi.gui_app import ProtSurfApp

    root = tk.Tk(className="TopoPPI")
    _set_window_icon(root)
    root.tk.call("tk", "scaling", DEFAULT_GUI_CONFIG.tk_scaling)
    ProtSurfApp(root, config=DEFAULT_GUI_CONFIG)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
