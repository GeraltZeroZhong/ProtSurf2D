import os
import sys
import tkinter as tk

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gui_app import ProtSurfApp


if __name__ == "__main__":
    root = tk.Tk()
    app = ProtSurfApp(root)
    root.mainloop()
