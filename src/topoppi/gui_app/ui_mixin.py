import json
import os
import threading
import tkinter as tk
import webbrowser
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, scrolledtext, ttk

from topoppi import __version__
from topoppi.config import DEFAULT_BENCHMARK_CONFIG, DEFAULT_RUN_CONFIG
from topoppi.errors import ConfigurationError
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

from .constants import INTERACTION_COLORS
from .forms import parse_benchmark_form, parse_single_run_form


class UIMixin:
    def _init_menu(self):
        menu = tk.Menu(self.root)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(
            label="User guide",
            command=lambda: webbrowser.open("https://github.com/GeraltZeroZhong/TopoPPI#readme"),
        )
        help_menu.add_command(
            label="Report an issue",
            command=lambda: webbrowser.open("https://github.com/GeraltZeroZhong/TopoPPI/issues"),
        )
        help_menu.add_separator()
        help_menu.add_command(label="About TopoPPI", command=self._show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _show_about(self):
        messagebox.showinfo(
            "About TopoPPI",
            f"TopoPPI {__version__}\n\n"
            "Create annotated 2D maps of protein-protein interfaces.\n\n"
            "Project: github.com/GeraltZeroZhong/TopoPPI",
            parent=self.root,
        )

    def _init_controls(self):
        header = ttk.Frame(self.left_frame, style="Sidebar.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))

        root_icon = getattr(self.root, "_topoppi_icon_image", None)
        if root_icon is not None:
            self.sidebar_icon_image = root_icon.subsample(5, 5)
            tk.Label(
                header,
                image=self.sidebar_icon_image,
                background="#f5f7fb",
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(0, 10))

        header_text = ttk.Frame(header, style="Sidebar.TFrame")
        header_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(header_text, text=f"TopoPPI {__version__}", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(
            header_text,
            text="Protein interface mapping for reproducible UV atlas figures",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 104,
        ).pack(anchor=tk.W)

        self._init_settings_tabs()
        self._init_basic_input_controls(self.basic_tab)
        self._init_interaction_controls(self.basic_tab)
        self._init_mode_controls(self.advanced_tab)
        self._init_advanced_input_controls(self.advanced_tab)
        self._init_core_controls(self.advanced_tab)
        self._init_benchmark_controls(self.advanced_tab)
        self._init_optcuts_controls(self.advanced_tab)
        self._init_label_layout_controls(self.advanced_tab)
        self.apply_style_preset()
        self._init_run_summary()
        self._init_run_controls()
        self._init_log_panel()
        self.toggle_color_mode(mark_custom=False)

    def _init_settings_tabs(self):
        self.var_input_path = tk.StringVar(value="")
        self.var_prolif_path = tk.StringVar(value="")
        self.var_chain_a = tk.StringVar(value=str(DEFAULT_RUN_CONFIG.chain_a))
        self.var_chain_b = tk.StringVar(value=str(DEFAULT_RUN_CONFIG.chain_b))
        self.var_output_dir = tk.StringVar(value="")
        self.var_recent_file = tk.StringVar(value="")
        self.var_recent_output_dir = tk.StringVar(value="")
        self._track_prolif_override_context()
        self._load_recent_items()
        self.var_settings_page = tk.StringVar(value="basic")
        tab_bar = ttk.Frame(self.left_frame)
        tab_bar.pack(fill=tk.X, pady=(0, 5))
        ttk.Radiobutton(
            tab_bar,
            text="Basic",
            value="basic",
            variable=self.var_settings_page,
            command=self._sync_settings_page,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            tab_bar,
            text="Advanced",
            value="advanced",
            variable=self.var_settings_page,
            command=self._sync_settings_page,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.settings_page_container = ttk.Frame(self.left_frame)
        self.settings_page_container.pack(fill=tk.X)
        self.basic_tab = ttk.Frame(self.settings_page_container)
        self.advanced_tab = ttk.Frame(self.settings_page_container)
        self.basic_tab.pack(fill=tk.X)

    def _track_prolif_override_context(self):
        for variable in (self.var_input_path, self.var_chain_a, self.var_chain_b):
            variable.trace_add("write", self._clear_prolif_override)

    def _clear_prolif_override(self, *_args):
        if self.var_prolif_path.get():
            self.var_prolif_path.set("")

    def _sync_settings_page(self):
        self.basic_tab.pack_forget()
        self.advanced_tab.pack_forget()
        if self.var_settings_page.get() == "advanced":
            self.advanced_tab.pack(fill=tk.X)
        else:
            self.basic_tab.pack(fill=tk.X)
        if self.var_settings_page.get() == "basic" and self.var_run_mode.get() == "benchmark":
            self.var_run_mode.set("single")
            self._sync_mode_controls()

    def _init_mode_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Run Mode", padding=10)
        frame.pack(fill=tk.X, pady=5)
        self.var_run_mode = tk.StringVar(value="single")
        ttk.Radiobutton(
            frame,
            text="Single analysis",
            value="single",
            variable=self.var_run_mode,
            command=self._sync_mode_controls,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            frame,
            text="Benchmark",
            value="benchmark",
            variable=self.var_run_mode,
            command=self._sync_mode_controls,
        ).pack(side=tk.LEFT, padx=(12, 0))

    def _init_basic_input_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Input", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(0, weight=1)

        self.lbl_basic_input_path = ttk.Label(frame, text="Structure file")
        self.lbl_basic_input_path.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.entry_file = ttk.Entry(frame, textvariable=self.var_input_path)
        self.entry_file.grid(row=1, column=0, sticky=tk.EW, pady=(2, 0))
        self._bind_field_update(self.entry_file)
        ttk.Button(frame, text="File...", width=9, command=self.browse_file).grid(
            row=1, column=1, padx=(6, 0), pady=(2, 0)
        )

        ttk.Label(frame, text="Recent input").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        self.combo_recent_file = ttk.Combobox(
            frame,
            values=self.recent_files,
            textvariable=self.var_recent_file,
            state="readonly",
        )
        self.combo_recent_file.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
        self.combo_recent_file.bind("<<ComboboxSelected>>", self._select_recent_file)

        chain_frame = ttk.LabelFrame(parent, text="Chains", padding=10)
        chain_frame.pack(fill=tk.X, pady=5)
        chain_frame.columnconfigure(1, weight=1)
        self.combo_basic_chain_a = self._chain_row(
            chain_frame, 0, "Surface chain", DEFAULT_RUN_CONFIG.chain_a, variable=self.var_chain_a
        )
        self.combo_basic_chain_b = self._chain_row(
            chain_frame, 1, "Partner chain", DEFAULT_RUN_CONFIG.chain_b, variable=self.var_chain_b
        )
        self.btn_swap_chains_basic = ttk.Button(
            chain_frame, text="Swap A/B", style="Tool.TButton", command=self.swap_chains
        )
        self.btn_swap_chains_basic.grid(row=0, column=2, rowspan=2, sticky=tk.NS, padx=(6, 0), pady=3)
        self.chain_preview_var = tk.StringVar(value="Select a structure to detect chains.")
        ttk.Label(
            chain_frame,
            textvariable=self.chain_preview_var,
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 86,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        output_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        output_frame.pack(fill=tk.X, pady=5)
        output_frame.columnconfigure(0, weight=1)
        ttk.Label(output_frame, text="Save figures and run details to").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.entry_output_dir_basic = ttk.Entry(output_frame, textvariable=self.var_output_dir)
        self.entry_output_dir_basic.grid(row=1, column=0, sticky=tk.EW, pady=(2, 0))
        self._bind_field_update(self.entry_output_dir_basic)
        ttk.Button(output_frame, text="Browse...", width=9, command=self.browse_output_dir).grid(
            row=1, column=1, padx=(6, 0), pady=(2, 0)
        )
        ttk.Label(
            output_frame,
            text="Leave blank to use the structure folder.",
            style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

    def _init_advanced_input_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Files and Output", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(0, weight=1)

        self.lbl_input_path = ttk.Label(frame, text="Structure file")
        self.lbl_input_path.grid(row=0, column=0, columnspan=3, sticky=tk.W)
        self.entry_file_advanced = ttk.Entry(frame, textvariable=self.var_input_path)
        self.entry_file_advanced.grid(row=1, column=0, sticky=tk.EW, pady=(2, 6))
        self._bind_field_update(self.entry_file_advanced)
        ttk.Button(frame, text="File...", width=9, command=self.browse_file).grid(
            row=1, column=1, padx=(6, 0), pady=(2, 6)
        )
        ttk.Button(frame, text="Folder...", width=9, command=self.browse_folder).grid(
            row=1, column=2, padx=(4, 0), pady=(2, 6)
        )

        ttk.Label(frame, text="Recent file/folder").grid(row=3, column=0, columnspan=3, sticky=tk.W)
        self.combo_recent_file_advanced = ttk.Combobox(
            frame,
            values=self.recent_files,
            textvariable=self.var_recent_file,
            state="readonly",
        )
        self.combo_recent_file_advanced.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(2, 6))
        self.combo_recent_file_advanced.bind("<<ComboboxSelected>>", self._select_recent_file)

        self.lbl_prolif = ttk.Label(frame, text="ProLIF JSON override (optional)")
        self.lbl_prolif.grid(row=5, column=0, columnspan=3, sticky=tk.W)
        self.entry_prolif = ttk.Entry(frame, textvariable=self.var_prolif_path)
        self.entry_prolif.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(2, 6))
        self._bind_field_update(self.entry_prolif)
        self.btn_browse_prolif = ttk.Button(frame, text="Browse...", width=9, command=self.browse_prolif)
        self.btn_browse_prolif.grid(row=6, column=2, padx=(4, 0), pady=(2, 6))
        self.lbl_prolif_hint = ttk.Label(
            frame,
            text=(
                "Leave blank to generate ProLIF annotations automatically. "
                "If generation fails, the error dialog suggests the next step."
            ),
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 86,
        )
        self.lbl_prolif_hint.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))

        self.lbl_output_dir = ttk.Label(frame, text="Default save directory")
        self.lbl_output_dir.grid(row=9, column=0, columnspan=3, sticky=tk.W)
        self.entry_output_dir = ttk.Entry(frame, textvariable=self.var_output_dir)
        self.entry_output_dir.grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
        self._bind_field_update(self.entry_output_dir)
        ttk.Button(frame, text="Browse...", width=9, command=self.browse_output_dir).grid(
            row=10, column=2, padx=(4, 0), pady=(2, 0)
        )

        ttk.Label(frame, text="Recent save directory").grid(row=12, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        self.combo_recent_output_dir = ttk.Combobox(
            frame,
            values=self.recent_output_dirs,
            textvariable=self.var_recent_output_dir,
            state="readonly",
        )
        self.combo_recent_output_dir.grid(row=13, column=0, columnspan=3, sticky=tk.EW, pady=(2, 0))
        self.combo_recent_output_dir.bind("<<ComboboxSelected>>", self._select_recent_output_dir)

        self.var_auto_save = tk.BooleanVar(value=self.config.auto_save_single_run)
        self.chk_auto_save = ttk.Checkbutton(
            frame, text="Auto-save figure and manifest after single run", variable=self.var_auto_save
        )
        self.chk_auto_save.grid(row=14, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        self.benchmark_run_mode_row = ttk.Frame(frame)
        self.benchmark_run_mode_row.grid(row=15, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        ttk.Label(self.benchmark_run_mode_row, text="Benchmark output").pack(side=tk.LEFT)
        self.var_benchmark_run_mode = tk.StringVar(value=self.config.default_benchmark_run_mode)
        self.combo_benchmark_run_mode = ttk.Combobox(
            self.benchmark_run_mode_row,
            values=["resume", "new", "overwrite"],
            width=10,
            state="readonly",
            textvariable=self.var_benchmark_run_mode,
        )
        self.combo_benchmark_run_mode.pack(side=tk.RIGHT)
        self.combo_benchmark_run_mode.bind("<<ComboboxSelected>>", lambda _event: self._update_run_summary())
        self.lbl_benchmark_mode_note = ttk.Label(
            frame,
            text="Benchmark mode processes top-level .pdb, .cif, and .mmcif files.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_benchmark_mode_note.grid(row=16, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    def _init_core_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Core Analysis", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)

        self.entry_chain_a = self._chain_row(
            frame, 0, "Surface chain", DEFAULT_RUN_CONFIG.chain_a, variable=self.var_chain_a
        )
        self.entry_chain_b = self._chain_row(
            frame, 1, "Partner chain", DEFAULT_RUN_CONFIG.chain_b, variable=self.var_chain_b
        )
        self.btn_swap_chains_advanced = ttk.Button(
            frame, text="Swap A/B", style="Tool.TButton", command=self.swap_chains
        )
        self.btn_swap_chains_advanced.grid(row=0, column=2, rowspan=2, sticky=tk.NS, padx=(6, 0), pady=3)
        self.entry_cutoff = self._entry_row(
            frame,
            3,
            "Interface cutoff",
            self.config.default_patch_cutoff,
            unit="Å",
        )
        self.entry_res = self._entry_row(
            frame, 4, "Grid resolution", DEFAULT_RUN_CONFIG.surface.grid_resolution, unit="Å"
        )
        self.entry_sigma = self._entry_row(frame, 5, "Surface sigma", DEFAULT_RUN_CONFIG.surface.sigma, unit="Å")
        self.entry_surface_level = self._entry_row(frame, 6, "Surface isovalue", DEFAULT_RUN_CONFIG.surface.level)
        self.entry_surface_padding = self._entry_row(
            frame, 7, "Surface padding", DEFAULT_RUN_CONFIG.surface.padding, unit="Å"
        )
        self.entry_max_voxels = self._entry_row(frame, 8, "Maximum voxels", DEFAULT_RUN_CONFIG.surface.max_voxels)
        self.entry_max_adaptive_resolution = self._entry_row(
            frame,
            9,
            "Maximum adaptive spacing",
            DEFAULT_RUN_CONFIG.surface.max_adaptive_resolution,
            unit="Å",
        )
        self.var_adaptive_resolution = tk.BooleanVar(value=DEFAULT_RUN_CONFIG.surface.adaptive_resolution)
        ttk.Checkbutton(
            frame,
            text="Adapt grid spacing to the voxel budget",
            variable=self.var_adaptive_resolution,
        ).grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        ttk.Label(frame, text="Initial parameterization").grid(row=11, column=0, sticky=tk.W, pady=3)
        self.var_parameterization_method = tk.StringVar(value=DEFAULT_RUN_CONFIG.parameterization.method)
        self.combo_parameterization_method = ttk.Combobox(
            frame,
            values=["auto", "lscm", "harmonic", "slim", "spherical", "cylindrical"],
            textvariable=self.var_parameterization_method,
            state="readonly",
            width=14,
        )
        self.combo_parameterization_method.grid(row=11, column=1, columnspan=2, sticky=tk.EW, pady=3)
        self.entry_slim_iterations = self._entry_row(
            frame,
            12,
            "SLIM iterations",
            DEFAULT_RUN_CONFIG.parameterization.slim_iterations,
        )
        self.entry_slim_boundary_constraint_weight = self._entry_row(
            frame,
            13,
            "SLIM boundary weight",
            DEFAULT_RUN_CONFIG.parameterization.slim_boundary_constraint_weight,
        )
        self.lbl_min_points = ttk.Label(frame, text="Minimum interaction residues")
        self.lbl_min_points.grid(row=14, column=0, sticky=tk.W, pady=3)
        self.entry_min_points = ttk.Entry(frame, width=10)
        self.entry_min_points.insert(0, str(self.config.default_min_points))
        self.entry_min_points.grid(row=14, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(self.entry_min_points)
        self.lbl_min_points_hint = ttk.Label(
            frame,
            text=(
                "This display threshold counts Chain A residues present in ProLIF interactions; "
                "every parameterized patch is sent to OptCuts."
            ),
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_min_points_hint.grid(row=15, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

    def _init_benchmark_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Benchmark Design", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)
        self.benchmark_controls_frame = frame

        ttk.Label(frame, text="Purpose").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.var_benchmark_purpose = tk.StringVar(value=DEFAULT_BENCHMARK_CONFIG.benchmark_purpose)
        self.combo_benchmark_purpose = ttk.Combobox(
            frame,
            values=["quality", "performance"],
            textvariable=self.var_benchmark_purpose,
            state="readonly",
            width=18,
        )
        self.combo_benchmark_purpose.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=3)
        self.combo_benchmark_purpose.bind("<<ComboboxSelected>>", lambda _event: self._benchmark_mode_changed())

        ttk.Label(frame, text="Execution profile").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.var_execution_profile = tk.StringVar(value=DEFAULT_BENCHMARK_CONFIG.execution_profile)
        self.combo_execution_profile = ttk.Combobox(
            frame,
            values=["comparative", "operational_optcuts"],
            textvariable=self.var_execution_profile,
            state="readonly",
            width=18,
        )
        self.combo_execution_profile.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=3)
        self.combo_execution_profile.bind("<<ComboboxSelected>>", lambda _event: self._benchmark_profile_changed())

        self.lbl_chain_selection_mode = ttk.Label(frame, text="Chain selection")
        self.lbl_chain_selection_mode.grid(row=2, column=0, sticky=tk.W, pady=3)
        self.var_chain_selection_mode = tk.StringVar(value="configured")
        self.combo_chain_selection_mode = ttk.Combobox(
            frame,
            values=["configured", "auto_contact", "manifest"],
            textvariable=self.var_chain_selection_mode,
            state="readonly",
            width=14,
        )
        self.combo_chain_selection_mode.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=3)
        self._bind_field_update(self.combo_chain_selection_mode)

        self.lbl_manifest_path = ttk.Label(frame, text="Dataset manifest")
        self.lbl_manifest_path.grid(row=3, column=0, sticky=tk.W, pady=3)
        self.entry_manifest_path = ttk.Entry(frame)
        self.entry_manifest_path.grid(row=3, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(self.entry_manifest_path)
        self.btn_manifest_path = ttk.Button(
            frame,
            text="Browse...",
            width=9,
            command=self.browse_benchmark_manifest,
        )
        self.btn_manifest_path.grid(row=3, column=2, padx=(4, 0), pady=3)

        ttk.Label(frame, text="Method arms").grid(row=5, column=0, sticky=tk.NW, pady=(6, 2))
        method_frame = ttk.Frame(frame)
        method_frame.grid(row=5, column=1, columnspan=2, sticky=tk.EW, pady=(6, 2))
        self.var_method_topoppi = tk.BooleanVar(value=True)
        self.var_method_optcuts_automatic = tk.BooleanVar(value=True)
        self.var_method_optcuts_lscm = tk.BooleanVar(value=True)
        self.chk_method_topoppi = ttk.Checkbutton(
            method_frame,
            text="TopoPPI (complete)",
            variable=self.var_method_topoppi,
            command=lambda: self._benchmark_method_changed("topoppi"),
        )
        self.chk_method_topoppi.pack(anchor=tk.W)
        self.chk_method_optcuts_automatic = ttk.Checkbutton(
            method_frame,
            text="Geometry-only OptCuts (automatic)",
            variable=self.var_method_optcuts_automatic,
            command=lambda: self._benchmark_method_changed("automatic"),
        )
        self.chk_method_optcuts_automatic.pack(anchor=tk.W)
        self.chk_method_optcuts_lscm = ttk.Checkbutton(
            method_frame,
            text="Geometry-only OptCuts (LSCM initialized)",
            variable=self.var_method_optcuts_lscm,
            command=lambda: self._benchmark_method_changed("lscm"),
        )
        self.chk_method_optcuts_lscm.pack(anchor=tk.W)

        self.var_include_topology_ablation = tk.BooleanVar(value=DEFAULT_BENCHMARK_CONFIG.include_topology_ablation)
        self.chk_include_topology_ablation = ttk.Checkbutton(
            frame,
            text="Include topology-gate ablation",
            variable=self.var_include_topology_ablation,
            command=self._form_changed,
        )
        self.chk_include_topology_ablation.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        self.entry_max_workers = self._entry_row(
            frame,
            8,
            "Worker processes",
            DEFAULT_BENCHMARK_CONFIG.max_workers or "",
        )
        self.entry_threads_per_worker = self._entry_row(
            frame,
            9,
            "Threads per worker",
            DEFAULT_BENCHMARK_CONFIG.threads_per_worker,
        )
        self.entry_repetitions = self._entry_row(
            frame,
            10,
            "Measured repetitions",
            DEFAULT_BENCHMARK_CONFIG.repetitions,
        )
        self.entry_warmup_runs = self._entry_row(
            frame,
            11,
            "Warm-up runs",
            DEFAULT_BENCHMARK_CONFIG.warmup_runs,
        )
        self.entry_worker_timeout = self._entry_row(
            frame,
            12,
            "Worker timeout",
            DEFAULT_BENCHMARK_CONFIG.worker_timeout_sec,
            unit="s",
        )
        self.entry_worker_memory_limit = self._entry_row(
            frame,
            13,
            "Worker memory limit",
            "",
            unit="MB",
        )
        ttk.Label(
            frame,
            text="Leave memory blank for no explicit RSS limit.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        ).grid(row=14, column=0, columnspan=3, sticky=tk.W, pady=(0, 3))

        self.entry_raster_size = self._entry_row(frame, 15, "Audit raster size", DEFAULT_BENCHMARK_CONFIG.raster_size)
        self.entry_min_chain_residues = self._entry_row(
            frame,
            16,
            "Minimum chain residues",
            DEFAULT_BENCHMARK_CONFIG.min_chain_residues,
        )
        self.entry_per_face_sample_size = self._entry_row(
            frame,
            17,
            "Face samples per patch",
            DEFAULT_BENCHMARK_CONFIG.per_face_sample_size_per_patch,
        )
        self.entry_bootstrap_iterations = self._entry_row(
            frame,
            18,
            "Bootstrap iterations",
            DEFAULT_BENCHMARK_CONFIG.bootstrap_iterations,
        )
        self.entry_random_seed = self._entry_row(frame, 19, "Random seed", DEFAULT_BENCHMARK_CONFIG.random_seed)

        self.entry_expected_git_commit = self._entry_row(frame, 20, "Expected Git commit", "", width=16)
        ttk.Label(frame, text="Coordinate audit").grid(row=21, column=0, sticky=tk.W, pady=3)
        self.entry_coordinate_audit_path = ttk.Entry(frame)
        self.entry_coordinate_audit_path.grid(row=21, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(self.entry_coordinate_audit_path)
        self.btn_coordinate_audit_path = ttk.Button(
            frame, text="Browse...", width=9, command=self.browse_coordinate_audit
        )
        self.btn_coordinate_audit_path.grid(row=21, column=2, padx=(4, 0), pady=3)
        self.entry_coordinate_audit_sha256 = self._entry_row(
            frame,
            23,
            "Coordinate-audit SHA-256",
            "",
            width=16,
        )

        self.var_formal_benchmark = tk.BooleanVar(value=False)
        self.chk_formal_benchmark = ttk.Checkbutton(
            frame,
            text="Formal benchmark safeguards",
            variable=self.var_formal_benchmark,
            command=self._apply_formal_benchmark_defaults,
        )
        self.chk_formal_benchmark.grid(row=24, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        self.lbl_benchmark_resources_hint = ttk.Label(
            frame,
            text="Performance safeguards use one uncontended worker; quality runs may use several isolated workers.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_benchmark_resources_hint.grid(row=25, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    def _init_optcuts_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="OptCuts and Export", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)
        self.optcuts_controls_frame = frame

        self.entry_optcuts_bin = self._entry_row(
            frame, 0, "OptCuts binary", DEFAULT_RUN_CONFIG.optcuts.optcuts_bin, width=16
        )
        self.btn_optcuts_bin = ttk.Button(frame, text="Browse...", width=9, command=self.browse_optcuts_binary)
        self.btn_optcuts_bin.grid(row=0, column=2, padx=(4, 0), pady=3)
        self.entry_patch_gap = self._entry_row(frame, 1, "Atlas chart gap", DEFAULT_RUN_CONFIG.optcuts.patch_gap)
        self.entry_optcuts_lambda = self._entry_row(
            frame,
            2,
            "OptCuts lambda",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_lambda_init,
        )
        self.entry_optcuts_distortion_bound = self._entry_row(
            frame,
            3,
            "Distortion bound",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_distortion_bound,
        )
        self.entry_optcuts_initial_cut_option = self._entry_row(
            frame,
            4,
            "Initial cut option",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_initial_cut_option,
        )
        self.var_optcuts_use_bijectivity = tk.BooleanVar(value=DEFAULT_RUN_CONFIG.optcuts.optcuts_use_bijectivity)
        ttk.Checkbutton(
            frame,
            text="Enforce OptCuts bijectivity",
            variable=self.var_optcuts_use_bijectivity,
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        self.lbl_optcuts_initialization = ttk.Label(frame, text="OptCuts initialization")
        self.lbl_optcuts_initialization.grid(row=6, column=0, sticky=tk.W, pady=3)
        self.var_optcuts_initialization = tk.StringVar(
            value="provided" if DEFAULT_RUN_CONFIG.optcuts.use_input_uv else "automatic"
        )
        self.combo_optcuts_initialization = ttk.Combobox(
            frame,
            values=["provided", "automatic"],
            textvariable=self.var_optcuts_initialization,
            state="readonly",
            width=14,
        )
        self.combo_optcuts_initialization.grid(row=6, column=1, columnspan=2, sticky=tk.EW, pady=3)
        self.entry_optcuts_timeout = self._entry_row(
            frame,
            7,
            "Timeout per patch",
            DEFAULT_RUN_CONFIG.optcuts.timeout_sec,
            unit="s",
        )
        self.entry_expected_optcuts_sha256 = self._entry_row(
            frame,
            8,
            "Frozen binary SHA-256",
            DEFAULT_RUN_CONFIG.optcuts.expected_binary_sha256,
            width=16,
        )
        self.entry_residue_fragmentation_weight = self._entry_row(
            frame,
            9,
            "TopoPPI objective weight",
            DEFAULT_RUN_CONFIG.optcuts.residue_fragmentation_weight,
        )
        self.entry_contact_distance = self._entry_row(
            frame,
            10,
            "Geometric fallback distance",
            DEFAULT_RUN_CONFIG.contact_distance_angstrom,
            unit="Å",
        )

        self.var_save_optcuts_frames = tk.BooleanVar(value=False)
        self.chk_save_optcuts_frames = ttk.Checkbutton(
            frame,
            text="Export OptCuts frames",
            variable=self.var_save_optcuts_frames,
            command=self._sync_optcuts_frame_controls,
        )
        self.chk_save_optcuts_frames.grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))

        self.entry_optcuts_frame_stride = self._entry_row(
            frame,
            12,
            "Frame stride",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_frame_stride,
        )
        self.entry_optcuts_min_frame_long_edge = self._entry_row(
            frame,
            13,
            "Minimum frame size",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_min_frame_long_edge,
            unit="px",
        )

        self.lbl_optcuts_frames_dir = ttk.Label(frame, text="Frame output directory")
        self.lbl_optcuts_frames_dir.grid(row=14, column=0, sticky=tk.W, pady=(4, 2))
        self.entry_optcuts_frames_dir = ttk.Entry(frame)
        self.entry_optcuts_frames_dir.grid(row=14, column=1, sticky=tk.EW, pady=(4, 2))
        self._bind_field_update(self.entry_optcuts_frames_dir)
        self.btn_optcuts_frames_dir = ttk.Button(
            frame, text="Browse...", width=9, command=self.browse_optcuts_frames_dir
        )
        self.btn_optcuts_frames_dir.grid(
            row=14,
            column=2,
            padx=(4, 0),
            pady=(4, 2),
        )
        ttk.Label(
            frame,
            text="The default weight 20 runs complete TopoPPI; 0 selects the geometry-only ablation.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        ).grid(row=15, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        self._sync_optcuts_frame_controls()

    def _init_interaction_controls(self, parent):
        color_frame = ttk.LabelFrame(parent, text="Interactions and Colors", padding=10)
        color_frame.pack(fill=tk.X, pady=5)
        color_frame.columnconfigure(0, weight=1)

        ttk.Label(color_frame, text="Style preset").grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        self.var_style_preset = tk.StringVar(value="Exploration")
        self.combo_style_preset = ttk.Combobox(
            color_frame,
            values=["Exploration", "Publication", "High contrast", "Custom"],
            textvariable=self.var_style_preset,
            state="readonly",
            width=18,
        )
        self.combo_style_preset.grid(row=0, column=1, sticky=tk.EW, pady=(0, 2))
        self.combo_style_preset.bind("<<ComboboxSelected>>", lambda _event: self.apply_style_preset())

        self.var_color_type = tk.BooleanVar(value=True)
        self.chk_type = ttk.Checkbutton(
            color_frame,
            text="Color by interaction type",
            variable=self.var_color_type,
            command=self.toggle_color_mode,
        )
        self.chk_type.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 6))

        swatch_row = ttk.Frame(color_frame)
        swatch_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        ttk.Label(swatch_row, text="Default residue color").pack(side=tk.LEFT)
        self.residue_color = self.config.default_residue_color
        self.color_swatch = tk.Canvas(
            swatch_row, width=26, height=16, highlightthickness=1, highlightbackground="#9ca3af"
        )
        self.color_swatch.pack(side=tk.LEFT, padx=(8, 4))
        self.btn_color = ttk.Button(swatch_row, text="Choose...", style="Tool.TButton", command=self.choose_color)
        self.btn_color.pack(side=tk.LEFT)
        self._update_color_swatch()

        filter_actions = ttk.Frame(color_frame)
        filter_actions.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        ttk.Label(filter_actions, text="Interaction filters").pack(side=tk.LEFT)
        self.btn_filter_all = ttk.Button(
            filter_actions, text="All", width=5, style="Tool.TButton", command=lambda: self.set_all_interactions(True)
        )
        self.btn_filter_all.pack(
            side=tk.RIGHT,
            padx=(4, 0),
        )
        self.btn_filter_none = ttk.Button(
            filter_actions, text="None", width=6, style="Tool.TButton", command=lambda: self.set_all_interactions(False)
        )
        self.btn_filter_none.pack(
            side=tk.RIGHT,
        )

        self.filter_frame = ttk.Frame(color_frame)
        self.filter_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW)
        self.filter_controls = []
        for i, itype in enumerate(self.interaction_types_list):
            cell = ttk.Frame(self.filter_frame)
            cell.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=(0, 10), pady=2)
            swatch = tk.Canvas(
                cell,
                width=14,
                height=14,
                cursor="hand2",
                highlightthickness=1,
                highlightbackground="#9ca3af",
            )
            swatch.bind(
                "<Button-1>", lambda _event, interaction_type=itype: self.choose_interaction_color(interaction_type)
            )
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            self.interaction_color_swatches[itype] = swatch
            self._update_interaction_color_swatch(itype)
            var = tk.BooleanVar(value=(itype in self.default_active))
            self.interaction_vars[itype] = var
            chk = ttk.Checkbutton(cell, text=itype, variable=var, command=self.redraw_plot)
            chk.pack(side=tk.LEFT)
            self.filter_controls.append(chk)

    def _init_label_layout_controls(self, parent):
        label_frame = ttk.LabelFrame(parent, text="Labels and Layout", padding=10)
        label_frame.pack(fill=tk.X, pady=5)
        label_frame.columnconfigure(1, weight=1)
        self.label_layout_frame = label_frame

        self.var_show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(label_frame, text="Show labels", variable=self.var_show_labels, command=self.redraw_plot).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky=tk.W,
            pady=(0, 2),
        )

        ttk.Label(label_frame, text="Label mode").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.label_mode_options = {
            "Chain A residue": "chain_a",
            "Chain B residue": "chain_b",
            "A-B pair": "pair",
        }
        self.combo_label_mode = ttk.Combobox(
            label_frame, values=list(self.label_mode_options.keys()), width=18, state="readonly"
        )
        self.combo_label_mode.set("Chain A residue")
        self.combo_label_mode.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.combo_label_mode.bind("<<ComboboxSelected>>", lambda _event: self.redraw_plot())

        ttk.Label(label_frame, text="Residue scope").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.residue_scope_options = {
            "ProLIF interactions": "interaction",
            "Full patch context": "patch",
        }
        self.combo_residue_scope = ttk.Combobox(
            label_frame,
            values=list(self.residue_scope_options.keys()),
            width=18,
            state="readonly",
        )
        self.combo_residue_scope.set("ProLIF interactions")
        self.combo_residue_scope.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.combo_residue_scope.bind("<<ComboboxSelected>>", lambda _event: self.redraw_plot())

        ttk.Label(label_frame, text="Font").grid(row=3, column=0, sticky=tk.W, pady=2)
        font_row = ttk.Frame(label_frame)
        font_row.grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.combo_font = ttk.Combobox(
            font_row,
            values=["Arial", "Times New Roman", "Courier New", "sans-serif"],
            width=13,
            state="readonly",
        )
        self.combo_font.set("sans-serif")
        self.combo_font.pack(side=tk.LEFT)
        self.spin_size = ttk.Spinbox(
            font_row,
            from_=self.config.label_font_min_size,
            to=self.config.label_font_max_size,
            width=4,
            command=self.redraw_plot,
        )
        self.spin_size.set(self.config.label_font_size)
        self.spin_size.pack(side=tk.LEFT, padx=(6, 0))

        self.var_avoid_overlap = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            label_frame,
            text="Reduce label overlap",
            variable=self.var_avoid_overlap,
            command=self.redraw_plot,
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        ttk.Label(label_frame, text="Patch layout").grid(row=5, column=0, sticky=tk.W, pady=(6, 0))
        self.var_patch_layout = tk.StringVar(value="atlas")
        ttk.Radiobutton(
            label_frame, text="Atlas", value="atlas", variable=self.var_patch_layout, command=self.redraw_plot
        ).grid(
            row=5,
            column=1,
            sticky=tk.W,
            pady=(6, 0),
        )
        ttk.Radiobutton(
            label_frame, text="Per patch", value="per_patch", variable=self.var_patch_layout, command=self.redraw_plot
        ).grid(
            row=5,
            column=2,
            sticky=tk.W,
            pady=(6, 0),
        )

    def _init_run_controls(self):
        frame = self.sidebar_action_frame
        frame.pack(fill=tk.X, pady=(8, 5))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.btn_run = ttk.Button(
            frame, text="Create Interface Map", style="Primary.TButton", command=self.start_selected_run
        )
        self.btn_run.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6), padx=(0, 4))
        self.btn_cancel = ttk.Button(frame, text="Cancel", command=self.request_cancel, state=tk.DISABLED)
        self.btn_cancel.grid(row=0, column=1, sticky=tk.EW, pady=(0, 6), padx=(4, 0))

        self.stage_status_var = tk.StringVar(value="")
        self.lbl_stage_status = ttk.Label(
            frame,
            textvariable=self.stage_status_var,
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_stage_status.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self.lbl_stage_status.grid_remove()

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        self.progress.grid_remove()

    def _init_run_summary(self):
        frame = ttk.LabelFrame(self.left_frame, text="Run Summary", padding=10)
        frame.pack(fill=tk.X, pady=5)
        self.run_summary_var = tk.StringVar(value="Select an input to preview the run setup.")
        ttk.Label(
            frame,
            textvariable=self.run_summary_var,
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        ).pack(anchor=tk.W)

    def _init_log_panel(self):
        log_frame = ttk.LabelFrame(self.left_frame, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=self.config.log_visible_lines,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=1,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _init_plot_area(self):
        toolbar = ttk.Frame(self.right_frame, padding=(10, 8), style="Workspace.TFrame")
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Interface Map", style="Workspace.TLabel").pack(side=tk.LEFT)
        self.btn_save = ttk.Button(
            toolbar, text="Save Figure...", style="Tool.TButton", command=self.save_figure, state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.RIGHT)
        self.btn_redraw = ttk.Button(
            toolbar, text="Apply Style", style="Tool.TButton", command=self.redraw_plot, state=tk.DISABLED
        )
        self.btn_redraw.pack(side=tk.RIGHT, padx=(0, 6))

        self.canvas_frame = ttk.Frame(self.right_frame, style="Workspace.TFrame")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.current_canvas = None
        self._show_empty_plot_state()

    def _show_empty_plot_state(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        holder = ttk.Frame(self.canvas_frame, style="Workspace.TFrame")
        holder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        ttk.Label(holder, text="No interface map yet", style="Workspace.TLabel").pack(pady=(0, 4))
        ttk.Label(
            holder, text="Choose a structure for a single map or a folder for benchmarking.", style="Workspace.TLabel"
        ).pack(pady=(0, 10))
        action_row = ttk.Frame(holder, style="Workspace.TFrame")
        action_row.pack()
        ttk.Button(action_row, text="Browse Structure...", command=self.browse_file).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(action_row, text="Browse Benchmark Folder...", command=self.browse_folder).pack(side=tk.LEFT)

    def _init_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        self.status = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel"
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _entry_row(self, parent, row, label, default, width=10, unit=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(entry)
        if unit:
            ttk.Label(parent, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=3)
        return entry

    def _chain_row(self, parent, row, label, default, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        combo = ttk.Combobox(parent, width=10, textvariable=variable)
        combo.set(str(default))
        combo.grid(row=row, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(combo)
        return combo

    def _bind_field_update(self, widget):
        widget.bind("<KeyRelease>", lambda _event: self._form_changed())
        widget.bind("<FocusOut>", lambda _event: self._form_changed())
        widget.bind("<<ComboboxSelected>>", lambda _event: self._form_changed())

    def _form_changed(self):
        self._update_run_summary()
        self._update_chain_preview()
        self._refresh_action_states()

    def _recent_store_path(self):
        return Path.home() / ".topoppi" / "gui_recent.json"

    def _load_recent_items(self):
        path = self._recent_store_path()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = {}
        self.recent_files = self._filter_recent_paths(payload.get("files", []))
        self.recent_output_dirs = self._filter_recent_paths(payload.get("output_dirs", []), directories_only=True)

    def _save_recent_items(self):
        path = self._recent_store_path()
        try:
            dump_json_atomic({"files": self.recent_files, "output_dirs": self.recent_output_dirs}, path)
        except OSError as exc:
            self.log(f"Recent list save skipped: {exc}")

    def _filter_recent_paths(self, paths, directories_only=False):
        unique = []
        for item in paths:
            text = str(item).strip()
            if text in unique:
                continue
            if directories_only and not os.path.isdir(text):
                continue
            if not directories_only and not os.path.exists(text):
                continue
            unique.append(text)
            if len(unique) >= 8:
                break
        return unique

    def _remember_recent_file(self, path):
        text = str(path)
        self.recent_files = [text] + [item for item in self.recent_files if item != text]
        self.recent_files = self.recent_files[:8]
        self._refresh_recent_controls()
        self._save_recent_items()

    def _remember_recent_output_dir(self, path):
        text = str(path)
        self.recent_output_dirs = [text] + [item for item in self.recent_output_dirs if item != text]
        self.recent_output_dirs = self.recent_output_dirs[:8]
        self._refresh_recent_controls()
        self._save_recent_items()

    def _refresh_recent_controls(self):
        self.combo_recent_file.configure(values=self.recent_files)
        self.combo_recent_file_advanced.configure(values=self.recent_files)
        self.combo_recent_output_dir.configure(values=self.recent_output_dirs)

    def _select_recent_file(self, _event):
        path = self.var_recent_file.get().strip()
        self.var_input_path.set(path)
        if os.path.isdir(path):
            self.var_run_mode.set("benchmark")
            self.var_settings_page.set("advanced")
            self._sync_settings_page()
        else:
            self.var_run_mode.set("single")
            self._populate_chain_choices(path)
        self._sync_mode_controls()
        self._form_changed()

    def _select_recent_output_dir(self, _event):
        path = self.var_recent_output_dir.get().strip()
        self.entry_output_dir.delete(0, tk.END)
        self.entry_output_dir.insert(0, path)
        self._form_changed()

    def _sync_mode_controls(self):
        is_benchmark = self.var_run_mode.get() == "benchmark"
        self.lbl_input_path.config(text="Benchmark folder" if is_benchmark else "Structure file")
        self.lbl_output_dir.config(text="Benchmark output folder" if is_benchmark else "Default save directory")
        self.btn_run.config(text="Run Benchmark" if is_benchmark else "Create Interface Map")

        prolif_state = tk.DISABLED if is_benchmark else tk.NORMAL
        self.entry_prolif.config(state=prolif_state)
        self.btn_browse_prolif.config(state=prolif_state)
        self.lbl_prolif.config(state=prolif_state)
        self.lbl_prolif_hint.config(state=prolif_state)
        self.chk_auto_save.config(state=tk.DISABLED if is_benchmark else tk.NORMAL)
        self.combo_benchmark_run_mode.config(state="readonly" if is_benchmark else tk.DISABLED)
        if is_benchmark:
            self.benchmark_run_mode_row.grid()
            self.lbl_benchmark_mode_note.grid()
            self.benchmark_controls_frame.pack(fill=tk.X, pady=5, before=self.optcuts_controls_frame)
            self.lbl_min_points.grid_remove()
            self.entry_min_points.grid_remove()
            self.lbl_min_points_hint.grid_remove()
            self.label_layout_frame.pack_forget()
        else:
            self.benchmark_run_mode_row.grid_remove()
            self.lbl_benchmark_mode_note.grid_remove()
            self.benchmark_controls_frame.pack_forget()
            self.lbl_min_points.grid()
            self.entry_min_points.grid()
            self.lbl_min_points_hint.grid()
            self.label_layout_frame.pack(fill=tk.X, pady=5)
        initialization_state = tk.DISABLED if is_benchmark else "readonly"
        self.combo_optcuts_initialization.config(state=initialization_state)
        self.lbl_optcuts_initialization.config(state=tk.DISABLED if is_benchmark else tk.NORMAL)
        self.chk_save_optcuts_frames.config(state=tk.DISABLED if is_benchmark else tk.NORMAL)
        self._sync_optcuts_frame_controls()
        self._benchmark_profile_changed(update_summary=False)
        self._update_run_summary()
        self._refresh_action_states()

    def _update_run_summary(self):
        mode = self.var_run_mode.get()
        path = self.entry_file.get().strip()
        chain_a = self.entry_chain_a.get().strip()
        chain_b = self.entry_chain_b.get().strip()
        optcuts_bin = self.entry_optcuts_bin.get().strip()
        if mode == "benchmark":
            output = self.entry_output_dir.get().strip()
            resume = self.var_benchmark_run_mode.get()
            purpose = self.var_benchmark_purpose.get()
            profile = self.var_execution_profile.get()
            workers = self.entry_max_workers.get().strip() or "auto"
            threads = self.entry_threads_per_worker.get().strip() or "?"
            self.run_summary_var.set(
                f"Benchmark: {path or '(no folder)'} | {purpose}/{profile} | "
                f"chains {chain_a or '?'} / {chain_b or '?'} | output {output or '(default)'} | "
                f"mode {resume} | {workers} worker(s) × {threads} thread(s) | "
                f"OptCuts {optcuts_bin or '(default)'}"
            )
        else:
            output = self.entry_output_dir.get().strip()
            prolif = "provided" if self.entry_prolif.get().strip() else "auto-generate"
            try:
                objective_weight = float(self.entry_residue_fragmentation_weight.get())
            except ValueError:
                objective_weight = 0.0
            method = "TopoPPI complete" if objective_weight > 0.0 else "geometry-only ablation"
            self.run_summary_var.set(
                f"Single: {path or '(no structure)'} | chains {chain_a or '?'} / {chain_b or '?'} | "
                f"save {output or '(input folder)'} | {method} | ProLIF {prolif} | "
                f"OptCuts {optcuts_bin or '(default)'}"
            )

    def _sync_optcuts_frame_controls(self):
        enabled = bool(self.var_save_optcuts_frames.get()) and self.var_run_mode.get() == "single"
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (
            self.entry_optcuts_frame_stride,
            self.entry_optcuts_min_frame_long_edge,
            self.entry_optcuts_frames_dir,
            self.btn_optcuts_frames_dir,
        ):
            widget.config(state=state)

    def _benchmark_mode_changed(self):
        self._apply_formal_benchmark_defaults()
        self._update_run_summary()

    def _benchmark_profile_changed(self, update_summary=True):
        operational = self.var_execution_profile.get() == "operational_optcuts"
        if operational:
            self.var_benchmark_purpose.set("performance")
            self.var_method_optcuts_lscm.set(False)
            self.var_include_topology_ablation.set(False)
            if self.var_method_topoppi.get() == self.var_method_optcuts_automatic.get():
                self.var_method_topoppi.set(True)
                self.var_method_optcuts_automatic.set(False)
            if self.var_formal_benchmark.get() and self.var_run_mode.get() == "benchmark":
                self._apply_formal_benchmark_defaults()
        elif self.var_method_topoppi.get():
            self.var_method_optcuts_automatic.set(True)
        self.chk_method_optcuts_lscm.config(state=tk.DISABLED if operational else tk.NORMAL)
        self.chk_include_topology_ablation.config(state=tk.DISABLED if operational else tk.NORMAL)
        self.combo_benchmark_purpose.config(state=tk.DISABLED if operational else "readonly")
        if update_summary:
            self._form_changed()

    def _benchmark_method_changed(self, source):
        if self.var_execution_profile.get() == "operational_optcuts":
            if source == "topoppi" and self.var_method_topoppi.get():
                self.var_method_optcuts_automatic.set(False)
            elif source == "automatic" and self.var_method_optcuts_automatic.get():
                self.var_method_topoppi.set(False)
        self._form_changed()

    def _selected_benchmark_variants(self):
        variants = []
        if self.var_method_optcuts_automatic.get():
            variants.append("optcuts_automatic")
        if self.var_method_optcuts_lscm.get():
            variants.append("optcuts_lscm_initialized")
        if self.var_method_topoppi.get():
            variants.append("residue_aware_optcuts")
        return tuple(variants)

    def _apply_formal_benchmark_defaults(self):
        if self.var_formal_benchmark.get():
            self.var_chain_selection_mode.set("manifest")
            if self.var_benchmark_purpose.get() == "performance":
                values = (
                    (self.entry_max_workers, "1"),
                    (self.entry_repetitions, "3"),
                    (self.entry_warmup_runs, "1"),
                )
            else:
                values = (
                    (self.entry_repetitions, "1"),
                    (self.entry_warmup_runs, "0"),
                )
            for entry, value in values:
                entry.delete(0, tk.END)
                entry.insert(0, value)
        self._form_changed()

    def start_selected_run(self):
        if self.var_run_mode.get() == "benchmark":
            self.start_benchmark()
        else:
            self.start_analysis()

    def post_to_ui(self, callback, *args, **kwargs):
        if self._closed:
            return
        if threading.get_ident() == self._ui_thread_id:
            callback(*args, **kwargs)
        else:
            self._ui_queue.put((callback, args, kwargs))

    def _drain_ui_queue(self):
        if self._closed:
            return
        while not self._ui_queue.empty():
            callback, args, kwargs = self._ui_queue.get()
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                self._append_log_message(f"UI callback failed: {exc}")
                if self._busy:
                    self._finish_task()
                try:
                    messagebox.showerror("GUI Error", f"An internal GUI update failed:\n{exc}")
                except tk.TclError:
                    pass
        try:
            self.root.after(self.config.ui_poll_interval_ms, self._drain_ui_queue)
        except tk.TclError:
            pass

    def log(self, message):
        text = str(message)
        print(f"[GUI Log] {text}")
        self.post_to_ui(self._append_log_message, text)

    def _append_log_message(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.log_history.append(line)
        self.log_history = self.log_history[-500:]
        self.current_run_log.append(line)
        self.status_var.set(message)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _reset_run_log(self):
        self.current_run_log = []
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _begin_task(self, message, progress_mode="indeterminate"):
        self._reset_run_log()
        self._cancel_event.clear()
        self._busy = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL, text="Cancel")
        self.btn_redraw.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.progress.stop()
        self.progress.grid()
        self.progress.configure(mode=progress_mode)
        if progress_mode == "determinate":
            self.progress.configure(maximum=100, value=0)
        else:
            self.progress.start(10)
        self.stage_status_var.set("")
        self.lbl_stage_status.grid()
        self.log(message)

    def _finish_task(self):
        self._busy = False
        self.btn_cancel.config(state=tk.DISABLED, text="Cancel")
        self.progress.stop()
        self.progress.grid_remove()
        self.lbl_stage_status.grid_remove()
        self._refresh_action_states()

    def request_cancel(self):
        self._cancel_event.set()
        self.btn_cancel.config(state=tk.DISABLED, text="Cancelling")
        self.stage_status_var.set("Cancellation requested. The current stage will stop as soon as it can.")
        self.log("Cancellation requested.")

    def finish_cancelled(self, message):
        self._finish_task()
        self.log(message)

    def set_stage_progress(self, stage, value, message=None):
        self.post_to_ui(self._set_stage_progress_ui, stage, value, message)

    def _set_stage_progress_ui(self, stage, value, message=None):
        self.progress.configure(mode="determinate", maximum=100)
        self.progress.stop()
        self.progress.grid()
        percent = max(0, min(int(value), 100))
        self.progress.configure(value=percent)
        label = f"{stage}: {percent}%"
        if message:
            label = f"{label} - {message}"
        self.stage_status_var.set(label)

    def _refresh_action_states(self):
        ready = self._required_inputs_ready(
            self.var_run_mode.get(),
            self.var_input_path.get(),
            self.var_chain_a.get(),
            self.var_chain_b.get(),
        )
        run_state = tk.NORMAL if ready and not self._busy else tk.DISABLED
        self.btn_run.config(state=run_state)
        successful_run = self._successful_single_run
        has_plot_data = successful_run is not None
        has_figure = successful_run is not None and successful_run["figure"] is self.current_fig
        plot_state = tk.NORMAL if (has_plot_data and not self._busy) else tk.DISABLED
        save_state = tk.NORMAL if (has_figure and not self._busy) else tk.DISABLED
        self.btn_redraw.config(state=plot_state)
        self.btn_save.config(state=save_state)

    @staticmethod
    def _required_inputs_ready(mode, path, chain_a, chain_b):
        if not str(path).strip():
            return False
        if mode == "benchmark":
            return True
        surface = str(chain_a).strip()
        partner = str(chain_b).strip()
        return bool(surface and partner and surface != partner)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Structure Files", "*.pdb *.cif *.mmcif"),
                ("PDB Files", "*.pdb"),
                ("mmCIF Files", "*.cif *.mmcif"),
                ("All Files", "*.*"),
            ]
        )
        if filename:
            self.var_input_path.set(filename)
            self.var_run_mode.set("single")
            self.var_prolif_path.set("")
            self._remember_recent_file(filename)
            if not self.entry_output_dir.get().strip():
                self.entry_output_dir.insert(0, os.path.dirname(filename))
                self._remember_recent_output_dir(os.path.dirname(filename))
            self._populate_chain_choices(filename)
            self._sync_mode_controls()
            self._update_run_summary()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.var_input_path.set(folder)
            self.var_run_mode.set("benchmark")
            self._remember_recent_file(folder)
            self.var_settings_page.set("advanced")
            self._sync_settings_page()
            self.entry_output_dir.delete(0, tk.END)
            self.entry_output_dir.insert(0, os.path.join(folder, self.config.benchmark_output_folder))
            self._sync_mode_controls()
            self._update_run_summary()
            self.log(f"Selected folder: {os.path.basename(folder)}")

    def _populate_chain_choices(self, filename):
        try:
            from topoppi.io.io_loader import PDBLoader

            loader = PDBLoader(filename)
            chains = loader.get_protein_chain_ids()
        except Exception as exc:
            self.chain_residue_counts = {}
            self._update_chain_preview()
            self.log(f"Chain scan skipped: {exc}")
            return
        if not chains:
            self.chain_residue_counts = {}
            self._update_chain_preview()
            self.log("No protein chains detected in selected structure.")
            return
        self.chain_residue_counts = {chain_id: loader.get_chain_residue_count(chain_id) for chain_id in chains}
        counts = [f"{chain_id}:{count}" for chain_id, count in self.chain_residue_counts.items()]
        for combo in (
            self.combo_basic_chain_a,
            self.combo_basic_chain_b,
            self.entry_chain_a,
            self.entry_chain_b,
        ):
            combo.configure(values=chains)
        if self.var_chain_a.get().strip() not in chains:
            self.var_chain_a.set(chains[0])
        if self.var_chain_b.get().strip() not in chains and len(chains) > 1:
            self.var_chain_b.set(chains[1])
        self._update_chain_preview()
        self._form_changed()
        self.log("Detected protein chains (residues): " + ", ".join(counts))

    def _update_chain_preview(self):
        if not self.chain_residue_counts:
            self.chain_preview_var.set("Select a structure to detect chains.")
            return
        parts = []
        for chain_id, count in self.chain_residue_counts.items():
            marker = ""
            if chain_id == self.var_chain_a.get().strip():
                marker = " surface"
            elif chain_id == self.var_chain_b.get().strip():
                marker = " partner"
            parts.append(f"{chain_id}: {count} residues{marker}")
        self.chain_preview_var.set(" | ".join(parts))

    def swap_chains(self):
        chain_a = self.var_chain_a.get()
        self.var_chain_a.set(self.var_chain_b.get())
        self.var_chain_b.set(chain_a)
        self._update_chain_preview()
        self._form_changed()

    def browse_prolif(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filename:
            self.var_prolif_path.set(filename)
            self._update_run_summary()

    def browse_benchmark_manifest(self):
        filename = filedialog.askopenfilename(
            title="Select benchmark manifest",
            filetypes=[("Manifest files", "*.csv *.json"), ("All files", "*.*")],
        )
        if filename:
            self.entry_manifest_path.delete(0, tk.END)
            self.entry_manifest_path.insert(0, filename)
            self.var_chain_selection_mode.set("manifest")
            self._form_changed()

    def browse_coordinate_audit(self):
        filename = filedialog.askopenfilename(
            title="Select coordinate-audit report",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.entry_coordinate_audit_path.delete(0, tk.END)
            self.entry_coordinate_audit_path.insert(0, filename)
            self.entry_coordinate_audit_sha256.delete(0, tk.END)
            self.entry_coordinate_audit_sha256.insert(0, sha256_file(filename))
            self._form_changed()

    def browse_output_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_output_dir.delete(0, tk.END)
            self.entry_output_dir.insert(0, folder)
            self._remember_recent_output_dir(folder)
            self._update_run_summary()

    def browse_optcuts_frames_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_optcuts_frames_dir.delete(0, tk.END)
            self.entry_optcuts_frames_dir.insert(0, folder)

    def browse_optcuts_binary(self):
        filename = filedialog.askopenfilename(
            title="Select OptCuts executable",
            filetypes=[("Executable files", "*.exe *.bin"), ("All files", "*.*")],
        )
        if filename:
            self.entry_optcuts_bin.delete(0, tk.END)
            self.entry_optcuts_bin.insert(0, filename)
            self.entry_expected_optcuts_sha256.delete(0, tk.END)
            self.entry_expected_optcuts_sha256.insert(0, sha256_file(filename))
            self._form_changed()

    def choose_color(self):
        color = colorchooser.askcolor(color=self.residue_color, title="Select Residue Color")[1]
        if color:
            self._mark_style_custom()
            self.residue_color = color
            self._update_color_swatch()
            self.redraw_plot()

    def _update_color_swatch(self):
        self.color_swatch.delete("all")
        self.color_swatch.create_rectangle(0, 0, 26, 16, fill=self.residue_color, outline="")

    def choose_interaction_color(self, interaction_type):
        current = self.interaction_colors[interaction_type]
        color = colorchooser.askcolor(color=current, title=f"Color for {interaction_type}")[1]
        if color:
            self._mark_style_custom()
            self.interaction_colors[interaction_type] = color
            self._update_interaction_color_swatch(interaction_type)
            self.redraw_plot()

    def _update_interaction_color_swatch(self, interaction_type):
        swatch = self.interaction_color_swatches[interaction_type]
        swatch.delete("all")
        swatch.create_rectangle(
            1,
            1,
            13,
            13,
            fill=self.interaction_colors[interaction_type],
            outline="",
        )

    def set_all_interactions(self, enabled):
        self._mark_style_custom()
        for var in self.interaction_vars.values():
            var.set(bool(enabled))
        self.redraw_plot()

    def toggle_color_mode(self, mark_custom=True):
        if mark_custom:
            self._mark_style_custom()
        color_by_type = self.var_color_type.get()
        self.btn_color.config(state=tk.DISABLED if color_by_type else tk.NORMAL)
        for child in self.filter_controls:
            child.configure(state=tk.NORMAL if color_by_type else tk.DISABLED)
        self.btn_filter_all.config(state=tk.NORMAL if color_by_type else tk.DISABLED)
        self.btn_filter_none.config(state=tk.NORMAL if color_by_type else tk.DISABLED)
        if self._successful_single_run:
            self.redraw_plot()

    def _style_presets(self):
        return {
            "Exploration": {
                "color_by_type": True,
                "show_labels": True,
                "avoid_overlap": True,
                "label_mode": "Chain A residue",
                "residue_scope": "ProLIF interactions",
                "font_size": self.config.label_font_size,
                "patch_layout": "atlas",
                "active_types": set(self.interaction_types_list),
                "residue_color": self.config.default_residue_color,
                "interaction_colors": dict(INTERACTION_COLORS),
            },
            "Publication": {
                "color_by_type": True,
                "show_labels": True,
                "avoid_overlap": True,
                "label_mode": "A-B pair",
                "residue_scope": "ProLIF interactions",
                "font_size": 8,
                "patch_layout": "atlas",
                "active_types": set(self.interaction_types_list),
                "residue_color": "#2f3640",
                "interaction_colors": {
                    "HydrogenBond": "#009E73",
                    "Ionic": "#E69F00",
                    "Hydrophobic": "#6b7280",
                    "PiStacking": "#CC79A7",
                    "PiCation": "#D55E00",
                    "HalogenBond": "#00A6D6",
                    "MetalCoordination": "#8B6F47",
                    "PolarContact": "#56B4E9",
                    "VdWContact": "#0072B2",
                    "Other": "#9ca3af",
                },
            },
            "High contrast": {
                "color_by_type": True,
                "show_labels": True,
                "avoid_overlap": True,
                "label_mode": "Chain A residue",
                "residue_scope": "ProLIF interactions",
                "font_size": 10,
                "patch_layout": "atlas",
                "active_types": set(self.interaction_types_list),
                "residue_color": "#111827",
                "interaction_colors": {
                    "HydrogenBond": "#0057ff",
                    "Ionic": "#ffd400",
                    "Hydrophobic": "#666666",
                    "PiStacking": "#a100ff",
                    "PiCation": "#ff8c00",
                    "HalogenBond": "#00d5ff",
                    "MetalCoordination": "#7a3f00",
                    "PolarContact": "#00a2ff",
                    "VdWContact": "#000000",
                    "Other": "#9ca3af",
                },
            },
        }

    def apply_style_preset(self):
        name = self.var_style_preset.get()
        preset = self._style_presets().get(name)
        if not preset:
            return
        self.var_color_type.set(bool(preset["color_by_type"]))
        self.residue_color = preset["residue_color"]
        self.interaction_colors = dict(INTERACTION_COLORS)
        self.interaction_colors.update(preset["interaction_colors"])
        active_types = preset["active_types"]
        for itype, var in self.interaction_vars.items():
            var.set(itype in active_types)
        self._update_color_swatch()
        for itype in self.interaction_types_list:
            self._update_interaction_color_swatch(itype)
        self.var_show_labels.set(bool(preset["show_labels"]))
        self.var_avoid_overlap.set(bool(preset["avoid_overlap"]))
        self.combo_label_mode.set(preset["label_mode"])
        self.combo_residue_scope.set(preset["residue_scope"])
        self.spin_size.set(int(preset["font_size"]))
        self.var_patch_layout.set(preset["patch_layout"])
        self.marker_color_overrides = {}
        self.toggle_color_mode(mark_custom=False)

    def _mark_style_custom(self):
        if self.var_style_preset.get() != "Custom":
            self.var_style_preset.set("Custom")

    def get_style_config(self):
        active_types = [t for t, var in self.interaction_vars.items() if var.get()]
        try:
            font_size = int(self.spin_size.get())
        except (TypeError, ValueError):
            font_size = int(self.config.label_font_size)
            self.spin_size.set(font_size)
        font_size = max(self.config.label_font_min_size, min(self.config.label_font_max_size, font_size))
        return {
            "color": self.residue_color,
            "font_family": self.combo_font.get(),
            "font_size": font_size,
            "color_by_type": self.var_color_type.get(),
            "active_types": active_types,
            "interaction_colors": dict(self.interaction_colors),
            "show_labels": bool(self.var_show_labels.get()),
            "label_mode": self.label_mode_options[self.combo_label_mode.get()],
            "residue_scope": self.residue_scope_options[self.combo_residue_scope.get()],
            "avoid_label_overlap": bool(self.var_avoid_overlap.get()),
            "use_uv_atlas": self.var_patch_layout.get() == "atlas",
            "label_offsets": dict(self.label_offsets),
            "marker_color_overrides": dict(self.marker_color_overrides),
        }

    def read_single_form(self):
        return parse_single_run_form(
            {
                "path": self.entry_file.get(),
                "chain_a": self.entry_chain_a.get(),
                "chain_b": self.entry_chain_b.get(),
                "prolif": self.entry_prolif.get(),
                "cutoff": self.entry_cutoff.get(),
                "res": self.entry_res.get(),
                "sigma": self.entry_sigma.get(),
                "surface_level": self.entry_surface_level.get(),
                "surface_padding": self.entry_surface_padding.get(),
                "max_voxels": self.entry_max_voxels.get(),
                "adaptive_resolution": self.var_adaptive_resolution.get(),
                "max_adaptive_resolution": self.entry_max_adaptive_resolution.get(),
                "parameterization_method": self.var_parameterization_method.get(),
                "slim_iterations": self.entry_slim_iterations.get(),
                "slim_boundary_constraint_weight": self.entry_slim_boundary_constraint_weight.get(),
                "min_points": self.entry_min_points.get(),
                "optcuts_bin": self.entry_optcuts_bin.get(),
                "expected_optcuts_sha256": self.entry_expected_optcuts_sha256.get(),
                "patch_gap": self.entry_patch_gap.get(),
                "optcuts_lambda": self.entry_optcuts_lambda.get(),
                "optcuts_distortion_bound": self.entry_optcuts_distortion_bound.get(),
                "optcuts_initial_cut_option": self.entry_optcuts_initial_cut_option.get(),
                "optcuts_use_bijectivity": self.var_optcuts_use_bijectivity.get(),
                "optcuts_initialization": self.var_optcuts_initialization.get(),
                "optcuts_timeout": self.entry_optcuts_timeout.get(),
                "residue_fragmentation_weight": self.entry_residue_fragmentation_weight.get(),
                "contact_distance_angstrom": self.entry_contact_distance.get(),
                "save_optcuts_frames": self.var_save_optcuts_frames.get(),
                "optcuts_frame_stride": self.entry_optcuts_frame_stride.get(),
                "optcuts_min_frame_long_edge": self.entry_optcuts_min_frame_long_edge.get(),
                "optcuts_frames_dir": self.entry_optcuts_frames_dir.get(),
                "output_dir": self.entry_output_dir.get(),
                "auto_save": self.var_auto_save.get(),
            }
        )

    def read_benchmark_form(self):
        return parse_benchmark_form(
            {
                "folder": self.entry_file.get(),
                "chain_a": self.entry_chain_a.get(),
                "chain_b": self.entry_chain_b.get(),
                "cutoff": self.entry_cutoff.get(),
                "res": self.entry_res.get(),
                "sigma": self.entry_sigma.get(),
                "surface_level": self.entry_surface_level.get(),
                "surface_padding": self.entry_surface_padding.get(),
                "max_voxels": self.entry_max_voxels.get(),
                "adaptive_resolution": self.var_adaptive_resolution.get(),
                "max_adaptive_resolution": self.entry_max_adaptive_resolution.get(),
                "parameterization_method": self.var_parameterization_method.get(),
                "slim_iterations": self.entry_slim_iterations.get(),
                "slim_boundary_constraint_weight": self.entry_slim_boundary_constraint_weight.get(),
                "patch_gap": self.entry_patch_gap.get(),
                "optcuts_bin": self.entry_optcuts_bin.get(),
                "expected_optcuts_sha256": self.entry_expected_optcuts_sha256.get(),
                "optcuts_lambda": self.entry_optcuts_lambda.get(),
                "optcuts_distortion_bound": self.entry_optcuts_distortion_bound.get(),
                "optcuts_initial_cut_option": self.entry_optcuts_initial_cut_option.get(),
                "optcuts_use_bijectivity": self.var_optcuts_use_bijectivity.get(),
                "optcuts_timeout": self.entry_optcuts_timeout.get(),
                "residue_fragmentation_weight": self.entry_residue_fragmentation_weight.get(),
                "contact_distance_angstrom": self.entry_contact_distance.get(),
                "output_root": self.entry_output_dir.get(),
                "run_mode": self.var_benchmark_run_mode.get(),
                "chain_selection_mode": self.var_chain_selection_mode.get(),
                "manifest_path": self.entry_manifest_path.get(),
                "repetitions": self.entry_repetitions.get(),
                "warmup_runs": self.entry_warmup_runs.get(),
                "formal_mode": self.var_formal_benchmark.get(),
                "max_workers": self.entry_max_workers.get(),
                "benchmark_purpose": self.var_benchmark_purpose.get(),
                "execution_profile": self.var_execution_profile.get(),
                "optcuts_variants": self._selected_benchmark_variants(),
                "include_topology_ablation": self.var_include_topology_ablation.get(),
                "threads_per_worker": self.entry_threads_per_worker.get(),
                "worker_timeout_sec": self.entry_worker_timeout.get(),
                "worker_memory_limit_mb": self.entry_worker_memory_limit.get(),
                "raster_size": self.entry_raster_size.get(),
                "min_chain_residues": self.entry_min_chain_residues.get(),
                "per_face_sample_size_per_patch": self.entry_per_face_sample_size.get(),
                "bootstrap_iterations": self.entry_bootstrap_iterations.get(),
                "random_seed": self.entry_random_seed.get(),
                "expected_git_commit": self.entry_expected_git_commit.get(),
                "coordinate_audit_path": self.entry_coordinate_audit_path.get(),
                "expected_coordinate_audit_sha256": self.entry_coordinate_audit_sha256.get(),
            }
        )

    def save_figure(self):
        successful_run = self._successful_single_run
        if successful_run is None:
            return
        initialdir = self._default_output_dir(successful_run)
        file_path = filedialog.asksaveasfilename(
            initialdir=initialdir,
            initialfile=self._default_figure_filename(successful_run),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("TIFF Image", "*.tif *.tiff"), ("All Files", "*.*")],
            title="Save Figure As",
        )
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                save_kwargs = {"dpi": self.config.figure_dpi, "bbox_inches": "tight", "facecolor": "white"}
                if ext in {".tif", ".tiff"}:
                    successful_run["figure"].savefig(
                        file_path, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs
                    )
                else:
                    successful_run["figure"].savefig(file_path, **save_kwargs)
                manifest_path = self._write_figure_manifest(file_path, successful_run)
                self._remember_recent_output_dir(os.path.dirname(file_path))
                self.log(f"Figure saved to {file_path}")
                self.log(f"Run manifest saved to {manifest_path}")
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{file_path}\n\nManifest:\n{manifest_path}")
            except Exception as e:
                self.log(f"Error saving image: {e}")
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def _default_output_dir(self, successful_run):
        params = successful_run["params"]
        if params.get("output_dir"):
            return str(params["output_dir"])
        return os.path.dirname(str(params["path"])) or os.getcwd()

    def _default_figure_filename(self, successful_run):
        params = successful_run["params"]
        manifest = successful_run["manifest"]
        source = Path(str(params["path"]))
        chain_a = str(params["chain_a"])
        chain_b = str(params["chain_b"])
        cutoff = self._format_number(params["cutoff"])
        res = self._format_number(params["res"])
        sigma = self._format_number(params["sigma"])
        min_points = str(params["min_points"])
        prolif_source = str(manifest["prolif_source"]).replace("_", "-")
        run_id = str(manifest["run_id"])
        return f"{source.stem}_{chain_a}-{chain_b}_cutoff{cutoff}_res{res}_sigma{sigma}_min{min_points}_{prolif_source}_{run_id}.png"

    def _format_number(self, value):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number).replace(".", "p")

    def _write_figure_manifest(self, figure_path, successful_run):
        manifest_path = os.path.splitext(figure_path)[0] + ".topoppi.json"
        run_payload = deepcopy(successful_run["manifest"])
        absolute_figure = os.path.abspath(figure_path)
        run_payload["output_file"] = absolute_figure
        run_payload["config"]["output_file"] = absolute_figure
        payload = {
            "topoppi_version": __version__,
            "schema_name": "topoppi_gui_figure",
            "schema_version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "figure_file": absolute_figure,
            "run": run_payload,
            "style": deepcopy(successful_run["style"]),
            "log": list(successful_run["log"]),
        }
        dump_json_atomic(payload, manifest_path)
        return manifest_path

    def _auto_save_current_figure(self, successful_run):
        output_dir = self._default_output_dir(successful_run)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, self._default_figure_filename(successful_run))
        ext = os.path.splitext(file_path)[1].lower()
        save_kwargs = {"dpi": self.config.figure_dpi, "bbox_inches": "tight", "facecolor": "white"}
        if ext in {".tif", ".tiff"}:
            successful_run["figure"].savefig(
                file_path, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs
            )
        else:
            successful_run["figure"].savefig(file_path, **save_kwargs)
        manifest_path = self._write_figure_manifest(file_path, successful_run)
        self._remember_recent_output_dir(output_dir)
        self.log(f"Auto-saved figure to {file_path}")
        self.log(f"Auto-saved manifest to {manifest_path}")

    def start_analysis(self):
        try:
            form = self.read_single_form()
        except ConfigurationError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        if form.save_optcuts_frames and not form.optcuts_frames_dir:
            form = replace(form, optcuts_frames_dir=self._default_optcuts_frame_dir(form.path))
        params = form.to_params()
        config = form.to_config()
        params["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._remember_recent_file(params["path"])
        self._remember_recent_output_dir(params.get("output_dir") or os.path.dirname(params["path"]))
        self.label_offsets = {}
        self.marker_color_overrides = {}
        self._update_run_summary()
        self._begin_task("Creating interface map...", progress_mode="determinate")
        threading.Thread(target=self.run_pipeline, args=(params, config), daemon=True).start()
