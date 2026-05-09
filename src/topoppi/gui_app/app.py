import queue
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from topoppi.config import DEFAULT_GUI_CONFIG

from .constants import DEFAULT_ACTIVE_TYPES, INTERACTION_COLORS, INTERACTION_TYPES
from .plot_mixin import PlotMixin
from .ui_mixin import UIMixin
from .workflow_mixin import WorkflowMixin


class ProtSurfApp(UIMixin, WorkflowMixin, PlotMixin):
    def __init__(self, root, config=DEFAULT_GUI_CONFIG):
        self.root = root
        self.config = config
        self.root.title("TopoPPI - Mapping Protein Interaction Surfaces")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.minsize(self.config.min_window_width, self.config.min_window_height)

        self._ui_thread_id = threading.get_ident()
        self._ui_queue = queue.Queue()
        self._busy = False
        self._closed = False
        self._cancel_event = threading.Event()

        self.cached_viz = None
        self.cached_patches = None
        self.current_fig = None
        self.current_toolbar = None
        self._picking = False
        self._drag_state = None
        self.label_offsets = {}
        self.last_run_manifest = {}
        self.last_run_params = {}
        self.log_history = []
        self.current_run_log = []
        self.chain_residue_counts = {}
        self.recent_files = []
        self.recent_output_dirs = []

        self.interaction_types_list = list(INTERACTION_TYPES)
        self.default_active = set(DEFAULT_ACTIVE_TYPES)
        self.interaction_vars = {}
        self.interaction_colors = dict(INTERACTION_COLORS)
        self.interaction_color_swatches = {}

        self._configure_style()
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._init_sidebar()
        self.right_frame = ttk.Frame(self.paned_window, style="Workspace.TFrame")
        self.paned_window.add(self.sidebar_outer, weight=0)
        self.paned_window.add(self.right_frame, weight=5)

        self._init_controls()
        self._init_plot_area()
        self._init_status_bar()
        self._refresh_action_states()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.config.ui_poll_interval_ms, self._drain_ui_queue)

    def _configure_style(self):
        font_family = self.config.font_family
        try:
            base_font = tkfont.nametofont("TkDefaultFont")
            available = set(tkfont.families(self.root))
            family = next((name for name in self.config.font_fallbacks if name in available), self.config.font_family)
            base_font.configure(family=family, size=self.config.font_size)
            self.root.option_add("*Font", base_font)
            font_family = base_font.actual("family")
        except tk.TclError:
            pass

        style = ttk.Style(self.root)
        if self.config.ttk_theme in style.theme_names():
            style.theme_use(self.config.ttk_theme)
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Workspace.TFrame", background="#ffffff")
        style.configure("Sidebar.TFrame", background="#f5f7fb")
        style.configure("TLabel", background="#f5f7fb", foreground="#1f2937")
        style.configure("Workspace.TLabel", background="#ffffff", foreground="#1f2937")
        style.configure(
            "Header.TLabel",
            background="#f5f7fb",
            foreground="#111827",
            font=(font_family, self.config.header_font_size, "bold"),
        )
        style.configure("Muted.TLabel", background="#f5f7fb", foreground="#6b7280")
        style.configure("Error.TLabel", background="#f5f7fb", foreground="#b91c1c")
        style.configure("TLabelframe", background="#f5f7fb", borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background="#f5f7fb", foreground="#111827")
        style.configure("TButton", padding=(8, 5))
        style.configure("Invalid.TEntry", fieldbackground="#fff1f2")
        style.configure("Invalid.TCombobox", fieldbackground="#fff1f2")
        style.configure(
            "Primary.TButton",
            padding=(12, 7),
            font=(font_family, self.config.font_size, "bold"),
        )
        style.configure("Tool.TButton", padding=(6, 4))
        style.configure("Status.TLabel", background="#edf2f7", foreground="#1f2937")

    def _init_sidebar(self):
        self.sidebar_outer = ttk.Frame(self.paned_window, width=self.config.sidebar_width, style="Sidebar.TFrame")
        self.sidebar_outer.pack_propagate(False)
        self.sidebar_action_frame = ttk.Frame(self.sidebar_outer, padding=(12, 8), style="Sidebar.TFrame")
        self.sidebar_action_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.sidebar_scroll_area = ttk.Frame(self.sidebar_outer, style="Sidebar.TFrame")
        self.sidebar_scroll_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.sidebar_canvas = tk.Canvas(
            self.sidebar_scroll_area,
            borderwidth=0,
            highlightthickness=0,
            background="#f5f7fb",
        )
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_scroll_area, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.sidebar_canvas, padding=(12, 8), style="Sidebar.TFrame")
        self._sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.left_frame, anchor=tk.NW)
        self.left_frame.bind(
            "<Configure>",
            lambda _event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")),
        )
        self.sidebar_canvas.bind(
            "<Configure>",
            lambda event: self.sidebar_canvas.itemconfigure(self._sidebar_window, width=event.width),
        )
        self.sidebar_canvas.bind("<Enter>", self._bind_sidebar_wheel)
        self.sidebar_canvas.bind("<Leave>", self._unbind_sidebar_wheel)
        self.sidebar_canvas.bind("<Prior>", lambda _event: self.sidebar_canvas.yview_scroll(-1, "pages"))
        self.sidebar_canvas.bind("<Next>", lambda _event: self.sidebar_canvas.yview_scroll(1, "pages"))

    def _bind_sidebar_wheel(self, _event):
        self.sidebar_canvas.focus_set()
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_sidebar_mousewheel)
        self.sidebar_canvas.bind_all("<Button-4>", self._on_sidebar_mousewheel)
        self.sidebar_canvas.bind_all("<Button-5>", self._on_sidebar_mousewheel)

    def _unbind_sidebar_wheel(self, _event):
        self.sidebar_canvas.unbind_all("<MouseWheel>")
        self.sidebar_canvas.unbind_all("<Button-4>")
        self.sidebar_canvas.unbind_all("<Button-5>")

    def _on_sidebar_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(delta, "units")

    def close(self):
        self._closed = True
        try:
            self._unbind_sidebar_wheel(None)
        except Exception:
            pass
        try:
            self._close_current_figure()
        except Exception:
            pass
        self.root.destroy()
