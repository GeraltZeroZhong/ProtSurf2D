import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, scrolledtext, ttk

from topoppi import __version__
from topoppi.config import DEFAULT_RUN_CONFIG
from topoppi.errors import ConfigurationError

from .constants import INTERACTION_COLORS
from .forms import parse_single_run_form


class UIMixin:
    def _init_controls(self):
        ttk.Label(self.left_frame, text="TopoPPI", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(
            self.left_frame,
            text="Protein interface mapping for reproducible UV atlas figures",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 48,
        ).pack(anchor=tk.W, pady=(0, 10))

        self._init_settings_tabs()
        self._init_basic_input_controls(self.basic_tab)
        self._init_interaction_controls(self.basic_tab)
        self._init_mode_controls(self.advanced_tab)
        self._init_advanced_input_controls(self.advanced_tab)
        self._init_core_controls(self.advanced_tab)
        self._init_optcuts_controls(self.advanced_tab)
        self._init_label_layout_controls(self.advanced_tab)
        self.apply_style_preset(redraw=False)
        self._init_run_summary()
        self._init_run_controls()
        self._init_log_panel()
        self.toggle_color_mode(mark_custom=False)
        self._sync_mode_controls()

    def _init_settings_tabs(self):
        self.var_input_path = tk.StringVar(value="")
        self.var_prolif_path = tk.StringVar(value="")
        self.var_chain_a = tk.StringVar(value=str(DEFAULT_RUN_CONFIG.chain_a))
        self.var_chain_b = tk.StringVar(value=str(DEFAULT_RUN_CONFIG.chain_b))
        self.var_recent_file = tk.StringVar(value="")
        self.var_recent_output_dir = tk.StringVar(value="")
        self.validation_labels = {}
        self.validation_fields = {}
        self._validation_after_id = None
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

    def _sync_settings_page(self):
        if not hasattr(self, "basic_tab") or not hasattr(self, "advanced_tab"):
            return
        self.basic_tab.pack_forget()
        self.advanced_tab.pack_forget()
        if self.var_settings_page.get() == "advanced":
            self.advanced_tab.pack(fill=tk.X)
        else:
            self.basic_tab.pack(fill=tk.X)
        if (
            self.var_settings_page.get() == "basic"
            and hasattr(self, "var_run_mode")
            and self.var_run_mode.get() == "benchmark"
        ):
            self.var_run_mode.set("single")
            self._sync_mode_controls()

    def _init_mode_controls(self, parent=None):
        parent = parent or self.left_frame
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

    def _init_basic_input_controls(self, parent=None):
        parent = parent or self.left_frame
        frame = ttk.LabelFrame(parent, text="Input", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(0, weight=1)

        self.lbl_basic_input_path = ttk.Label(frame, text="Structure file")
        self.lbl_basic_input_path.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.entry_file = ttk.Entry(frame, textvariable=self.var_input_path)
        self.entry_file.grid(row=1, column=0, sticky=tk.EW, pady=(2, 0))
        self._bind_field_update(self.entry_file)
        ttk.Button(frame, text="File...", width=9, command=self.browse_file).grid(row=1, column=1, padx=(6, 0), pady=(2, 0))
        self._validation_label(frame, "path").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

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
        self.combo_basic_chain_a = self._chain_row(chain_frame, 0, "Surface chain", DEFAULT_RUN_CONFIG.chain_a, variable=self.var_chain_a)
        self.combo_basic_chain_b = self._chain_row(chain_frame, 1, "Partner chain", DEFAULT_RUN_CONFIG.chain_b, variable=self.var_chain_b)
        self.btn_swap_chains_basic = ttk.Button(chain_frame, text="Swap A/B", style="Tool.TButton", command=self.swap_chains)
        self.btn_swap_chains_basic.grid(row=0, column=2, rowspan=2, sticky=tk.NS, padx=(6, 0), pady=3)
        self.chain_preview_var = tk.StringVar(value="Select a structure to detect chains.")
        ttk.Label(
            chain_frame,
            textvariable=self.chain_preview_var,
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 86,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        self._validation_label(chain_frame, "chains").grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

    def _init_advanced_input_controls(self, parent=None):
        parent = parent or self.left_frame
        frame = ttk.LabelFrame(parent, text="Files and Output", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(0, weight=1)

        self.lbl_input_path = ttk.Label(frame, text="Structure file")
        self.lbl_input_path.grid(row=0, column=0, columnspan=3, sticky=tk.W)
        self.entry_file_advanced = ttk.Entry(frame, textvariable=self.var_input_path)
        self.entry_file_advanced.grid(row=1, column=0, sticky=tk.EW, pady=(2, 6))
        self._bind_field_update(self.entry_file_advanced)
        ttk.Button(frame, text="File...", width=9, command=self.browse_file).grid(row=1, column=1, padx=(6, 0), pady=(2, 6))
        ttk.Button(frame, text="Folder...", width=9, command=self.browse_folder).grid(row=1, column=2, padx=(4, 0), pady=(2, 6))
        self._validation_label(frame, "path").grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))

        ttk.Label(frame, text="Recent file/folder").grid(row=3, column=0, columnspan=3, sticky=tk.W)
        self.combo_recent_file_advanced = ttk.Combobox(
            frame,
            values=self.recent_files,
            textvariable=self.var_recent_file,
            state="readonly",
        )
        self.combo_recent_file_advanced.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(2, 6))
        self.combo_recent_file_advanced.bind("<<ComboboxSelected>>", self._select_recent_file)

        self.lbl_prolif = ttk.Label(frame, text="Optional ProLIF JSON")
        self.lbl_prolif.grid(row=5, column=0, columnspan=3, sticky=tk.W)
        self.entry_prolif = ttk.Entry(frame, textvariable=self.var_prolif_path)
        self.entry_prolif.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(2, 6))
        self._bind_field_update(self.entry_prolif)
        self.btn_browse_prolif = ttk.Button(frame, text="Browse...", width=9, command=self.browse_prolif)
        self.btn_browse_prolif.grid(row=6, column=2, padx=(4, 0), pady=(2, 6))
        self.lbl_prolif_hint = ttk.Label(
            frame,
            text="Leave blank to auto-generate interactions; if unavailable, geometric fallback is used.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 86,
        )
        self.lbl_prolif_hint.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        self._validation_label(frame, "prolif").grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))

        self.lbl_output_dir = ttk.Label(frame, text="Default save directory")
        self.lbl_output_dir.grid(row=9, column=0, columnspan=3, sticky=tk.W)
        self.entry_output_dir = ttk.Entry(frame)
        self.entry_output_dir.grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
        self._bind_field_update(self.entry_output_dir)
        ttk.Button(frame, text="Browse...", width=9, command=self.browse_output_dir).grid(row=10, column=2, padx=(4, 0), pady=(2, 0))
        self._validation_label(frame, "output_dir").grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

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
        self.chk_auto_save = ttk.Checkbutton(frame, text="Auto-save figure and manifest after single run", variable=self.var_auto_save)
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
            text="Benchmark mode currently processes .pdb files in the selected folder.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_benchmark_mode_note.grid(row=16, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    def _init_core_controls(self, parent=None):
        parent = parent or self.left_frame
        frame = ttk.LabelFrame(parent, text="Core Analysis", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)

        self.entry_chain_a = self._chain_row(frame, 0, "Surface chain", DEFAULT_RUN_CONFIG.chain_a, variable=self.var_chain_a)
        self.entry_chain_b = self._chain_row(frame, 1, "Partner chain", DEFAULT_RUN_CONFIG.chain_b, variable=self.var_chain_b)
        self.btn_swap_chains_advanced = ttk.Button(frame, text="Swap A/B", style="Tool.TButton", command=self.swap_chains)
        self.btn_swap_chains_advanced.grid(row=0, column=2, rowspan=2, sticky=tk.NS, padx=(6, 0), pady=3)
        self._validation_label(frame, "chains").grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(2, 4))
        self.entry_cutoff = self._entry_row(frame, 3, "Interface cutoff", DEFAULT_RUN_CONFIG.topology.distance_cutoff, unit="Å", field="cutoff")
        self.entry_res = self._entry_row(frame, 4, "Grid resolution", DEFAULT_RUN_CONFIG.surface.grid_resolution, unit="Å", field="res")
        self.entry_sigma = self._entry_row(frame, 5, "Surface sigma", DEFAULT_RUN_CONFIG.surface.sigma, field="sigma")
        self.entry_min_points = self._entry_row(frame, 6, "Minimum interacting residues", self.config.default_min_points, field="min_points")
        ttk.Label(
            frame,
            text="Interfaces below this residue threshold are excluded before OptCuts.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        ).grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        self.lbl_max_workers = ttk.Label(frame, text="Benchmark workers")
        self.lbl_max_workers.grid(row=8, column=0, sticky=tk.W, pady=3)
        self.entry_max_workers = ttk.Entry(frame, width=10)
        self.entry_max_workers.grid(row=8, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(self.entry_max_workers)
        self.lbl_max_workers_hint = ttk.Label(
            frame,
            text="Leave blank to use the CPU count.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_max_workers_hint.grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))
        self.lbl_max_workers_error = self._validation_label(frame, "max_workers")
        self.lbl_max_workers_error.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

    def _init_optcuts_controls(self, parent=None):
        parent = parent or self.left_frame
        frame = ttk.LabelFrame(parent, text="OptCuts and Export", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)

        self.entry_optcuts_bin = self._entry_row(frame, 0, "OptCuts binary", DEFAULT_RUN_CONFIG.optcuts.optcuts_bin, width=16, field="optcuts_bin")

        self.var_save_optcuts_frames = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Export OptCuts frames",
            variable=self.var_save_optcuts_frames,
            command=self._sync_optcuts_frame_controls,
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))

        self.entry_optcuts_frame_stride = self._entry_row(
            frame,
            2,
            "Frame stride",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_frame_stride,
            field="optcuts_frame_stride",
        )
        self.entry_optcuts_min_frame_long_edge = self._entry_row(
            frame,
            3,
            "Minimum frame size",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_min_frame_long_edge,
            unit="px",
            field="optcuts_min_frame_long_edge",
        )

        ttk.Label(frame, text="Frame output directory").grid(row=4, column=0, sticky=tk.W, pady=(4, 2))
        self.entry_optcuts_frames_dir = ttk.Entry(frame)
        self.entry_optcuts_frames_dir.grid(row=4, column=1, sticky=tk.EW, pady=(4, 2))
        self._bind_field_update(self.entry_optcuts_frames_dir)
        self.btn_optcuts_frames_dir = ttk.Button(frame, text="Browse...", width=9, command=self.browse_optcuts_frames_dir)
        self.btn_optcuts_frames_dir.grid(
            row=4,
            column=2,
            padx=(4, 0),
            pady=(4, 2),
        )
        self._sync_optcuts_frame_controls()

    def _init_interaction_controls(self, parent=None):
        parent = parent or self.left_frame
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
        self.color_swatch = tk.Canvas(swatch_row, width=26, height=16, highlightthickness=1, highlightbackground="#9ca3af")
        self.color_swatch.pack(side=tk.LEFT, padx=(8, 4))
        self.btn_color = ttk.Button(swatch_row, text="Choose...", style="Tool.TButton", command=self.choose_color)
        self.btn_color.pack(side=tk.LEFT)
        self._update_color_swatch()

        filter_actions = ttk.Frame(color_frame)
        filter_actions.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        ttk.Label(filter_actions, text="Interaction filters").pack(side=tk.LEFT)
        self.btn_filter_all = ttk.Button(filter_actions, text="All", width=5, style="Tool.TButton", command=lambda: self.set_all_interactions(True))
        self.btn_filter_all.pack(
            side=tk.RIGHT,
            padx=(4, 0),
        )
        self.btn_filter_none = ttk.Button(filter_actions, text="None", width=6, style="Tool.TButton", command=lambda: self.set_all_interactions(False))
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
            swatch.bind("<Button-1>", lambda _event, interaction_type=itype: self.choose_interaction_color(interaction_type))
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            self.interaction_color_swatches[itype] = swatch
            self._update_interaction_color_swatch(itype)
            var = tk.BooleanVar(value=(itype in self.default_active))
            self.interaction_vars[itype] = var
            chk = ttk.Checkbutton(cell, text=itype, variable=var, command=self.redraw_plot)
            chk.pack(side=tk.LEFT)
            self.filter_controls.append(chk)

    def _init_label_layout_controls(self, parent=None):
        parent = parent or self.left_frame
        label_frame = ttk.LabelFrame(parent, text="Labels and Layout", padding=10)
        label_frame.pack(fill=tk.X, pady=5)
        label_frame.columnconfigure(1, weight=1)

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
        self.combo_label_mode = ttk.Combobox(label_frame, values=list(self.label_mode_options.keys()), width=18, state="readonly")
        self.combo_label_mode.set("Chain A residue")
        self.combo_label_mode.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=2)
        self.combo_label_mode.bind("<<ComboboxSelected>>", lambda _event: self.redraw_plot())

        ttk.Label(label_frame, text="Font").grid(row=2, column=0, sticky=tk.W, pady=2)
        font_row = ttk.Frame(label_frame)
        font_row.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=2)
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
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        ttk.Label(label_frame, text="Patch layout").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
        self.var_patch_layout = tk.StringVar(value="atlas")
        ttk.Radiobutton(label_frame, text="Atlas", value="atlas", variable=self.var_patch_layout, command=self.redraw_plot).grid(
            row=4,
            column=1,
            sticky=tk.W,
            pady=(6, 0),
        )
        ttk.Radiobutton(label_frame, text="Per patch", value="per_patch", variable=self.var_patch_layout, command=self.redraw_plot).grid(
            row=4,
            column=2,
            sticky=tk.W,
            pady=(6, 0),
        )

    def _init_run_controls(self):
        frame = self.sidebar_action_frame
        frame.pack(fill=tk.X, pady=(8, 5))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.btn_run = ttk.Button(frame, text="Run Single Analysis", style="Primary.TButton", command=self.start_selected_run)
        self.btn_run.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6), padx=(0, 4))
        self.btn_cancel = ttk.Button(frame, text="Cancel", command=self.request_cancel, state=tk.DISABLED)
        self.btn_cancel.grid(row=0, column=1, sticky=tk.EW, pady=(0, 6), padx=(4, 0))
        self.btn_bench = None

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
        self.btn_save = ttk.Button(toolbar, text="Save Figure...", style="Tool.TButton", command=self.save_figure, state=tk.DISABLED)
        self.btn_save.pack(side=tk.RIGHT)
        self.btn_redraw = ttk.Button(toolbar, text="Apply Style", style="Tool.TButton", command=self.redraw_plot, state=tk.DISABLED)
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
        ttk.Label(holder, text="Choose a structure for a single map or a folder for benchmarking.", style="Workspace.TLabel").pack(pady=(0, 10))
        action_row = ttk.Frame(holder, style="Workspace.TFrame")
        action_row.pack()
        ttk.Button(action_row, text="Browse Structure...", command=self.browse_file).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(action_row, text="Browse Benchmark Folder...", command=self.browse_folder).pack(side=tk.LEFT)

    def _init_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        self.status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _entry_row(self, parent, row, label, default, width=10, unit=None, field=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(entry)
        if unit:
            ttk.Label(parent, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=3)
        if field:
            self.validation_fields.setdefault(field, []).append(entry)
            self._validation_label(parent, field).grid(row=row, column=3, sticky=tk.W, padx=(6, 0), pady=3)
        return entry

    def _chain_row(self, parent, row, label, default, variable=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        combo = ttk.Combobox(parent, width=10, textvariable=variable)
        combo.set(str(default))
        combo.grid(row=row, column=1, sticky=tk.EW, pady=3)
        self._bind_field_update(combo)
        return combo

    def _validation_label(self, parent, field):
        label = ttk.Label(parent, text="", style="Error.TLabel", wraplength=self.config.sidebar_width - 86)
        self.validation_labels.setdefault(field, []).append(label)
        return label

    def _bind_field_update(self, widget):
        widget.bind("<KeyRelease>", lambda _event: self._schedule_inline_validation())
        widget.bind("<FocusOut>", lambda _event: self._schedule_inline_validation())
        widget.bind("<<ComboboxSelected>>", lambda _event: self._schedule_inline_validation())

    def _schedule_inline_validation(self):
        self._update_run_summary()
        self._update_chain_preview()
        if self._validation_after_id:
            try:
                self.root.after_cancel(self._validation_after_id)
            except tk.TclError:
                pass
        self._validation_after_id = self.root.after(120, self._validate_inputs)

    def _set_validation_errors(self, errors):
        all_fields = set(self.validation_labels) | set(self.validation_fields)
        for field in all_fields:
            message = errors.get(field, "")
            for label in self.validation_labels.get(field, []):
                label.config(text=message)
            for widget in self.validation_fields.get(field, []):
                try:
                    widget.config(style="Invalid.TEntry" if message else "TEntry")
                except tk.TclError:
                    pass

    def _validate_inputs(self):
        if not hasattr(self, "var_run_mode"):
            return True
        mode = self.var_run_mode.get()
        errors = {}
        path = self.entry_file.get().strip() if hasattr(self, "entry_file") else ""
        if not path:
            errors["path"] = "Required."
        elif mode == "benchmark":
            if not os.path.isdir(path):
                errors["path"] = "Choose an existing benchmark folder."
        else:
            if os.path.isdir(path):
                errors["path"] = "Choose a structure file, not a folder."
            elif not os.path.isfile(path):
                errors["path"] = "Choose an existing .pdb or .cif file."

        chain_a = self.var_chain_a.get().strip() if hasattr(self, "var_chain_a") else ""
        chain_b = self.var_chain_b.get().strip() if hasattr(self, "var_chain_b") else ""
        if not chain_a or not chain_b:
            errors["chains"] = "Both chains are required."
        elif chain_a == chain_b:
            errors["chains"] = "Surface and partner chains must differ."
        elif self.chain_residue_counts:
            missing = [chain for chain in (chain_a, chain_b) if chain not in self.chain_residue_counts]
            if missing:
                errors["chains"] = "Selected chain not detected in this structure."

        for field, label in (("cutoff", "Cutoff"), ("res", "Grid"), ("sigma", "Sigma")):
            widget = getattr(self, f"entry_{field}", None)
            if widget is not None and not self._is_positive_float(widget.get()):
                errors[field] = f"{label} must be > 0."
        if hasattr(self, "entry_min_points") and not self._is_positive_int(self.entry_min_points.get()):
            errors["min_points"] = "Must be > 0."

        if mode == "single":
            prolif = self.entry_prolif.get().strip() if hasattr(self, "entry_prolif") else ""
            if prolif and not os.path.isfile(prolif):
                errors["prolif"] = "Choose an existing JSON file or leave blank."
            output_dir = self.entry_output_dir.get().strip() if hasattr(self, "entry_output_dir") else ""
            if output_dir and not os.path.isdir(output_dir):
                errors["output_dir"] = "Choose an existing folder or leave blank."
        else:
            output_root = self.entry_output_dir.get().strip() if hasattr(self, "entry_output_dir") else ""
            if output_root and os.path.exists(output_root) and not os.path.isdir(output_root):
                errors["output_dir"] = "Benchmark output must be a folder."
            elif output_root:
                parent = os.path.dirname(os.path.abspath(output_root)) or os.getcwd()
                if not os.path.isdir(parent):
                    errors["output_dir"] = "Output parent folder does not exist."
            max_workers = self.entry_max_workers.get().strip() if hasattr(self, "entry_max_workers") else ""
            if max_workers and not self._is_positive_int(max_workers):
                errors["max_workers"] = "Use a positive integer or leave blank."

        if hasattr(self, "entry_optcuts_bin") and not self.entry_optcuts_bin.get().strip():
            errors["optcuts_bin"] = "Required."
        if hasattr(self, "entry_optcuts_frame_stride") and not self._is_positive_int(self.entry_optcuts_frame_stride.get()):
            errors["optcuts_frame_stride"] = "Must be > 0."
        if hasattr(self, "entry_optcuts_min_frame_long_edge") and not self._is_non_negative_int(self.entry_optcuts_min_frame_long_edge.get()):
            errors["optcuts_min_frame_long_edge"] = "Must be >= 0."

        self._set_validation_errors(errors)
        return not errors

    @staticmethod
    def _is_positive_float(value):
        try:
            return float(str(value).strip()) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_positive_int(value):
        try:
            return int(str(value).strip()) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_non_negative_int(value):
        try:
            return int(str(value).strip()) >= 0
        except (TypeError, ValueError):
            return False

    def _recent_store_path(self):
        return Path.home() / ".topoppi" / "gui_recent.json"

    def _load_recent_items(self):
        path = self._recent_store_path()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}
        self.recent_files = self._filter_recent_paths(payload.get("files", []))
        self.recent_output_dirs = self._filter_recent_paths(payload.get("output_dirs", []), directories_only=True)

    def _save_recent_items(self):
        path = self._recent_store_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"files": self.recent_files, "output_dirs": self.recent_output_dirs}, handle, indent=2)
        except Exception as exc:
            self.log(f"Recent list save skipped: {exc}")

    def _filter_recent_paths(self, paths, directories_only=False):
        unique = []
        for item in paths or []:
            text = str(item or "").strip()
            if not text or text in unique:
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
        text = str(path or "").strip()
        if not text or not os.path.exists(text):
            return
        self.recent_files = [text] + [item for item in self.recent_files if item != text]
        self.recent_files = self.recent_files[:8]
        self._refresh_recent_controls()
        self._save_recent_items()

    def _remember_recent_output_dir(self, path):
        text = str(path or "").strip()
        if not text or not os.path.isdir(text):
            return
        self.recent_output_dirs = [text] + [item for item in self.recent_output_dirs if item != text]
        self.recent_output_dirs = self.recent_output_dirs[:8]
        self._refresh_recent_controls()
        self._save_recent_items()

    def _refresh_recent_controls(self):
        for combo in (
            getattr(self, "combo_recent_file", None),
            getattr(self, "combo_recent_file_advanced", None),
        ):
            if combo is not None:
                combo.configure(values=self.recent_files)
        combo = getattr(self, "combo_recent_output_dir", None)
        if combo is not None:
            combo.configure(values=self.recent_output_dirs)

    def _select_recent_file(self, _event=None):
        path = self.var_recent_file.get().strip()
        if not path:
            return
        self.var_input_path.set(path)
        if os.path.isdir(path):
            self.var_run_mode.set("benchmark")
            if hasattr(self, "var_settings_page"):
                self.var_settings_page.set("advanced")
                self._sync_settings_page()
        else:
            self.var_run_mode.set("single")
            self._populate_chain_choices(path)
        self._sync_mode_controls()
        self._schedule_inline_validation()

    def _select_recent_output_dir(self, _event=None):
        path = self.var_recent_output_dir.get().strip()
        if not path:
            return
        self.entry_output_dir.delete(0, tk.END)
        self.entry_output_dir.insert(0, path)
        self._schedule_inline_validation()

    def _sync_mode_controls(self):
        if not hasattr(self, "var_run_mode"):
            return
        is_benchmark = self.var_run_mode.get() == "benchmark"
        self.lbl_input_path.config(text="Benchmark folder" if is_benchmark else "Structure file")
        self.lbl_output_dir.config(text="Benchmark output folder" if is_benchmark else "Default save directory")
        self.btn_run.config(text="Run Benchmark" if is_benchmark else "Run Single Analysis")

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
            self.lbl_max_workers.grid()
            self.entry_max_workers.grid()
            self.lbl_max_workers_hint.grid()
            self.lbl_max_workers_error.grid()
        else:
            self.benchmark_run_mode_row.grid_remove()
            self.lbl_benchmark_mode_note.grid_remove()
            self.lbl_max_workers.grid_remove()
            self.entry_max_workers.grid_remove()
            self.lbl_max_workers_hint.grid_remove()
            self.lbl_max_workers_error.grid_remove()
        self._update_run_summary()

    def _update_run_summary(self):
        if not hasattr(self, "run_summary_var"):
            return
        mode = self.var_run_mode.get() if hasattr(self, "var_run_mode") else "single"
        path = self.entry_file.get().strip() if hasattr(self, "entry_file") else ""
        chain_a = self.entry_chain_a.get().strip() if hasattr(self, "entry_chain_a") else ""
        chain_b = self.entry_chain_b.get().strip() if hasattr(self, "entry_chain_b") else ""
        optcuts_bin = self.entry_optcuts_bin.get().strip() if hasattr(self, "entry_optcuts_bin") else ""
        if mode == "benchmark":
            output = self.entry_output_dir.get().strip() if hasattr(self, "entry_output_dir") else ""
            resume = self.var_benchmark_run_mode.get() if hasattr(self, "var_benchmark_run_mode") else "resume"
            self.run_summary_var.set(
                f"Benchmark: {path or '(no folder)'} | chains {chain_a or '?'} / {chain_b or '?'} | "
                f"output {output or '(default)'} | mode {resume} | OptCuts {optcuts_bin or '(default)'}"
            )
        else:
            output = self.entry_output_dir.get().strip() if hasattr(self, "entry_output_dir") else ""
            prolif = "provided" if self.entry_prolif.get().strip() else "auto/geometric"
            self.run_summary_var.set(
                f"Single: {path or '(no structure)'} | chains {chain_a or '?'} / {chain_b or '?'} | "
                f"save {output or '(input folder)'} | ProLIF {prolif} | OptCuts {optcuts_bin or '(default)'}"
            )

    def _sync_optcuts_frame_controls(self):
        enabled = bool(self.var_save_optcuts_frames.get())
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (
            getattr(self, "entry_optcuts_frame_stride", None),
            getattr(self, "entry_optcuts_min_frame_long_edge", None),
            getattr(self, "entry_optcuts_frames_dir", None),
            getattr(self, "btn_optcuts_frames_dir", None),
        ):
            if widget is not None:
                widget.config(state=state)

    def start_selected_run(self):
        if self.var_run_mode.get() == "benchmark":
            self.start_benchmark()
        else:
            self.start_analysis()

    def post_to_ui(self, callback, *args, **kwargs):
        if getattr(self, "_closed", False):
            return
        if threading.get_ident() == self._ui_thread_id:
            callback(*args, **kwargs)
        else:
            self._ui_queue.put((callback, args, kwargs))

    def _drain_ui_queue(self):
        if getattr(self, "_closed", False):
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
        if hasattr(self, "log_text"):
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

    def _reset_run_log(self):
        self.current_run_log = []
        if hasattr(self, "log_text"):
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.configure(state=tk.DISABLED)

    def _begin_task(self, message, progress_mode="indeterminate"):
        self._reset_run_log()
        self._cancel_event.clear()
        self._busy = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL, text="Cancel")
        if getattr(self, "btn_bench", None) is not None:
            self.btn_bench.config(state=tk.DISABLED)
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
        if not self._busy:
            return
        self._cancel_event.set()
        self.btn_cancel.config(state=tk.DISABLED, text="Cancelling")
        self.stage_status_var.set("Cancellation requested. The current stage will stop as soon as it can.")
        self.log("Cancellation requested.")

    def finish_cancelled(self, message="Run cancelled."):
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
        run_state = tk.DISABLED if self._busy else tk.NORMAL
        if hasattr(self, "btn_run"):
            self.btn_run.config(state=run_state)
        if hasattr(self, "btn_bench") and self.btn_bench is not None:
            self.btn_bench.config(state=run_state)
        has_plot_data = self.cached_viz is not None and self.cached_patches is not None
        has_figure = self.current_fig is not None
        plot_state = tk.NORMAL if (has_plot_data and not self._busy) else tk.DISABLED
        save_state = tk.NORMAL if (has_figure and not self._busy) else tk.DISABLED
        if hasattr(self, "btn_redraw"):
            self.btn_redraw.config(state=plot_state)
        if hasattr(self, "btn_save"):
            self.btn_save.config(state=save_state)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Structure Files", "*.pdb *.cif"), ("PDB Files", "*.pdb"), ("CIF Files", "*.cif"), ("All Files", "*.*")])
        if filename:
            self.var_input_path.set(filename)
            self.var_run_mode.set("single")
            self.var_prolif_path.set("")
            self._remember_recent_file(filename)
            if hasattr(self, "entry_output_dir") and not self.entry_output_dir.get().strip():
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
            if hasattr(self, "var_settings_page"):
                self.var_settings_page.set("advanced")
                self._sync_settings_page()
            if hasattr(self, "entry_output_dir"):
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
        counts = []
        self.chain_residue_counts = {}
        for chain_id in chains:
            try:
                count = loader.get_chain_residue_count(chain_id)
                self.chain_residue_counts[chain_id] = count
                counts.append(f"{chain_id}:{count}")
            except Exception:
                self.chain_residue_counts[chain_id] = None
                counts.append(f"{chain_id}:?")
        for combo in (
            getattr(self, "combo_basic_chain_a", None),
            getattr(self, "combo_basic_chain_b", None),
            getattr(self, "entry_chain_a", None),
            getattr(self, "entry_chain_b", None),
        ):
            if combo is not None:
                combo.configure(values=chains)
        if self.var_chain_a.get().strip() not in chains:
            self.var_chain_a.set(chains[0])
        if self.var_chain_b.get().strip() not in chains and len(chains) > 1:
            self.var_chain_b.set(chains[1])
        self._update_chain_preview()
        self._schedule_inline_validation()
        self.log("Detected protein chains (residues): " + ", ".join(counts))

    def _update_chain_preview(self):
        if not hasattr(self, "chain_preview_var"):
            return
        if not self.chain_residue_counts:
            self.chain_preview_var.set("Select a structure to detect chains.")
            return
        parts = []
        for chain_id, count in self.chain_residue_counts.items():
            suffix = "?" if count is None else str(count)
            marker = ""
            if chain_id == self.var_chain_a.get().strip():
                marker = " surface"
            elif chain_id == self.var_chain_b.get().strip():
                marker = " partner"
            parts.append(f"{chain_id}: {suffix} residues{marker}")
        self.chain_preview_var.set(" | ".join(parts))

    def swap_chains(self):
        chain_a = self.var_chain_a.get()
        self.var_chain_a.set(self.var_chain_b.get())
        self.var_chain_b.set(chain_a)
        self._update_chain_preview()
        self._schedule_inline_validation()

    def browse_prolif(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filename:
            self.var_prolif_path.set(filename)
            self._update_run_summary()

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
        current = self.interaction_colors.get(interaction_type, INTERACTION_COLORS.get(interaction_type, "#9ca3af"))
        color = colorchooser.askcolor(color=current, title=f"Color for {interaction_type}")[1]
        if color:
            self._mark_style_custom()
            self.interaction_colors[interaction_type] = color
            self._update_interaction_color_swatch(interaction_type)
            self.redraw_plot()

    def _update_interaction_color_swatch(self, interaction_type):
        swatch = self.interaction_color_swatches.get(interaction_type)
        if swatch is None:
            return
        swatch.delete("all")
        swatch.create_rectangle(
            1,
            1,
            13,
            13,
            fill=self.interaction_colors.get(interaction_type, INTERACTION_COLORS.get(interaction_type, "#9ca3af")),
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
        if self.cached_viz:
            self.redraw_plot()

    def _style_presets(self):
        return {
            "Exploration": {
                "color_by_type": True,
                "show_labels": True,
                "avoid_overlap": True,
                "label_mode": "Chain A residue",
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
                "font_size": 8,
                "patch_layout": "atlas",
                "active_types": set(self.interaction_types_list),
                "residue_color": "#2f3640",
                "interaction_colors": {
                    "VdWContact": "#0072B2",
                    "HydrogenBond": "#009E73",
                    "Hydrophobic": "#6b7280",
                    "PiStacking": "#CC79A7",
                    "PiCation": "#E69F00",
                    "CationPi": "#D55E00",
                    "Cationic": "#F0E442",
                    "Anionic": "#56B4E9",
                    "HalogenBond": "#00A6D6",
                    "MetalCoordination": "#8B6F47",
                },
            },
            "High contrast": {
                "color_by_type": True,
                "show_labels": True,
                "avoid_overlap": True,
                "label_mode": "Chain A residue",
                "font_size": 10,
                "patch_layout": "atlas",
                "active_types": set(self.interaction_types_list),
                "residue_color": "#111827",
                "interaction_colors": {
                    "VdWContact": "#000000",
                    "HydrogenBond": "#0057ff",
                    "Hydrophobic": "#666666",
                    "PiStacking": "#a100ff",
                    "PiCation": "#ff8c00",
                    "CationPi": "#ff0000",
                    "Cationic": "#ffd400",
                    "Anionic": "#00a2ff",
                    "HalogenBond": "#00d5ff",
                    "MetalCoordination": "#7a3f00",
                },
            },
        }

    def apply_style_preset(self, redraw=True):
        name = self.var_style_preset.get() if hasattr(self, "var_style_preset") else "Exploration"
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
        if hasattr(self, "var_show_labels"):
            self.var_show_labels.set(bool(preset["show_labels"]))
        if hasattr(self, "var_avoid_overlap"):
            self.var_avoid_overlap.set(bool(preset["avoid_overlap"]))
        if hasattr(self, "combo_label_mode"):
            self.combo_label_mode.set(preset["label_mode"])
        if hasattr(self, "spin_size"):
            self.spin_size.set(int(preset["font_size"]))
        if hasattr(self, "var_patch_layout"):
            self.var_patch_layout.set(preset["patch_layout"])
        self.toggle_color_mode(mark_custom=False)

    def _mark_style_custom(self):
        if hasattr(self, "var_style_preset") and self.var_style_preset.get() != "Custom":
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
            "label_mode": self.label_mode_options.get(self.combo_label_mode.get(), "chain_a"),
            "avoid_label_overlap": bool(self.var_avoid_overlap.get()),
            "use_uv_atlas": self.var_patch_layout.get() == "atlas",
            "label_offsets": dict(self.label_offsets),
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
                "min_points": self.entry_min_points.get(),
                "optcuts_bin": self.entry_optcuts_bin.get(),
                "save_optcuts_frames": self.var_save_optcuts_frames.get(),
                "optcuts_frame_stride": self.entry_optcuts_frame_stride.get(),
                "optcuts_min_frame_long_edge": self.entry_optcuts_min_frame_long_edge.get(),
                "optcuts_frames_dir": self.entry_optcuts_frames_dir.get(),
                "output_dir": self.entry_output_dir.get(),
                "auto_save": self.var_auto_save.get(),
            }
        )

    def save_figure(self):
        if not self.current_fig:
            return
        initialdir = self._default_output_dir()
        file_path = filedialog.asksaveasfilename(
            initialdir=initialdir,
            initialfile=self._default_figure_filename(),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("TIFF Image", "*.tif *.tiff"), ("All Files", "*.*")],
            title="Save Figure As",
        )
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                save_kwargs = {"dpi": self.config.figure_dpi, "bbox_inches": "tight", "facecolor": "white"}
                if ext in {".tif", ".tiff"}:
                    self.current_fig.savefig(file_path, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs)
                else:
                    self.current_fig.savefig(file_path, **save_kwargs)
                manifest_path = self._write_figure_manifest(file_path)
                self._remember_recent_output_dir(os.path.dirname(file_path))
                self.log(f"Figure saved to {file_path}")
                self.log(f"Run manifest saved to {manifest_path}")
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{file_path}\n\nManifest:\n{manifest_path}")
            except Exception as e:
                self.log(f"Error saving image: {e}")
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def _default_output_dir(self):
        if self.last_run_params.get("output_dir"):
            return str(self.last_run_params["output_dir"])
        if self.last_run_params.get("path"):
            return os.path.dirname(str(self.last_run_params["path"])) or os.getcwd()
        text = self.entry_output_dir.get().strip() if hasattr(self, "entry_output_dir") else ""
        return text or os.getcwd()

    def _default_figure_filename(self):
        params = self.last_run_params or {}
        source = Path(str(params.get("path") or self.entry_file.get() or "topoppi"))
        chain_a = str(params.get("chain_a") or self.entry_chain_a.get() or DEFAULT_RUN_CONFIG.chain_a)
        chain_b = str(params.get("chain_b") or self.entry_chain_b.get() or DEFAULT_RUN_CONFIG.chain_b)
        cutoff = self._format_number(params.get("cutoff") or self.entry_cutoff.get())
        res = self._format_number(params.get("res") or self.entry_res.get())
        sigma = self._format_number(params.get("sigma") or self.entry_sigma.get())
        min_points = str(params.get("min_points") or self.entry_min_points.get() or self.config.default_min_points)
        prolif_source = str((self.last_run_manifest or {}).get("prolif_source") or "pending").replace("_", "-")
        run_id = str((self.last_run_manifest or {}).get("run_id") or params.get("run_id") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        return f"{source.stem}_{chain_a}-{chain_b}_cutoff{cutoff}_res{res}_sigma{sigma}_min{min_points}_{prolif_source}_{run_id}.png"

    def _format_number(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).replace(".", "p")
        if number.is_integer():
            return str(int(number))
        return str(number).replace(".", "p")

    def _write_figure_manifest(self, figure_path):
        manifest_path = os.path.splitext(figure_path)[0] + ".topoppi.json"
        payload = {
            "topoppi_version": __version__,
            "schema_version": "1.1",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "figure_file": os.path.abspath(figure_path),
            "run": self.last_run_manifest,
            "style": self.get_style_config(),
            "log": list(self.current_run_log),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return manifest_path

    def _auto_save_current_figure(self):
        if not self.current_fig:
            return
        output_dir = self._default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, self._default_figure_filename())
        ext = os.path.splitext(file_path)[1].lower()
        save_kwargs = {"dpi": self.config.figure_dpi, "bbox_inches": "tight", "facecolor": "white"}
        if ext in {".tif", ".tiff"}:
            self.current_fig.savefig(file_path, format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs)
        else:
            self.current_fig.savefig(file_path, **save_kwargs)
        manifest_path = self._write_figure_manifest(file_path)
        self._remember_recent_output_dir(output_dir)
        self.log(f"Auto-saved figure to {file_path}")
        self.log(f"Auto-saved manifest to {manifest_path}")

    def start_analysis(self):
        if not self._validate_inputs():
            self.log("Run blocked by invalid input. Review the highlighted fields.")
            return
        try:
            form = self.read_single_form()
        except ConfigurationError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        params = form.to_params()
        try:
            self._preflight_optcuts(params["optcuts_bin"])
        except ConfigurationError as exc:
            messagebox.showerror("Invalid OptCuts Configuration", str(exc))
            return
        params["run_id"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self._remember_recent_file(params["path"])
        self._remember_recent_output_dir(params.get("output_dir") or os.path.dirname(params["path"]))
        self.label_offsets = {}
        self.last_run_params = dict(params)
        self._update_run_summary()
        self._begin_task("Starting single analysis pipeline...", progress_mode="determinate")
        threading.Thread(target=self.run_pipeline, args=(params,), daemon=True).start()
