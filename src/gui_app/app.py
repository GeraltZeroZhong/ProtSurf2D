import tkinter as tk
from tkinter import ttk

from .constants import INTERACTION_TYPES, DEFAULT_ACTIVE_TYPES
from .ui_mixin import UIMixin
from .workflow_mixin import WorkflowMixin
from .plot_mixin import PlotMixin


class ProtSurfApp(UIMixin, WorkflowMixin, PlotMixin):
    def __init__(self, root):
        self.root = root
        self.root.title("TopoPPI - Mapping Protein Interaction Surfaces")
        self.root.geometry("1200x950")

        self.cached_viz = None
        self.cached_patches = None
        self.current_fig = None
        self._picking = False
        self._drag_state = None
        self.label_offsets = {}

        self.interaction_types_list = list(INTERACTION_TYPES)
        self.default_active = set(DEFAULT_ACTIVE_TYPES)
        self.interaction_vars = {}

        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.left_frame = ttk.Frame(self.paned_window, width=340)
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        self.paned_window.add(self.right_frame, weight=4)

        self._init_controls()
        self._init_plot_area()
        self._init_status_bar()
