"""Start the packaged TopoPPI GUI without opening a console window."""

from __future__ import annotations

import ctypes
import os
import traceback
from pathlib import Path


def run() -> int:
    install_dir = Path(__file__).resolve().parent
    os.environ["TOPOPPI_HOME"] = str(install_dir)
    os.environ["TOPOPPI_OPTCUTS_BIN"] = str(install_dir / "bin" / "OptCuts_bin.exe")

    try:
        from topoppi.gui import main

        return main()
    except Exception:
        log_path = install_dir / "gui-startup.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        ctypes.windll.user32.MessageBoxW(
            0,
            f"TopoPPI could not start. Details were written to:\n\n{log_path}\n\n"
            "Run the TopoPPI installer again to repair the application.",
            "TopoPPI",
            0x10,
        )
        return 1


raise SystemExit(run())
