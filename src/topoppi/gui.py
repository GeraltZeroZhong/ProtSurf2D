"""GUI entry point for TopoPPI."""

from __future__ import annotations


def main() -> int:
    import tkinter as tk

    import matplotlib

    from topoppi.config import DEFAULT_GUI_CONFIG

    matplotlib.use("TkAgg", force=True)

    from topoppi.gui_app import ProtSurfApp

    root = tk.Tk(className="TopoPPI")
    root.tk.call("tk", "scaling", DEFAULT_GUI_CONFIG.tk_scaling)
    ProtSurfApp(root, config=DEFAULT_GUI_CONFIG)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
