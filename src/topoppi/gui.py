"""GUI entry point for TopoPPI."""

from __future__ import annotations

def main() -> int:
    import tkinter as tk

    from topoppi.gui_app import ProtSurfApp

    root = tk.Tk()
    ProtSurfApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
