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

        self._init_mode_controls()
        self._init_input_controls()
        self._init_core_controls()
        self._init_optcuts_controls()
        self._init_visual_controls()
        self._init_run_summary()
        self._init_run_controls()
        self._init_log_panel()
        self._sync_mode_controls()

    def _init_mode_controls(self):
        frame = ttk.LabelFrame(self.left_frame, text="Run Mode", padding=10)
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

    def _init_input_controls(self):
        frame = ttk.LabelFrame(self.left_frame, text="Input and Output", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(0, weight=1)

        self.lbl_input_path = ttk.Label(frame, text="Structure file")
        self.lbl_input_path.grid(row=0, column=0, columnspan=3, sticky=tk.W)
        self.entry_file = ttk.Entry(frame)
        self.entry_file.grid(row=1, column=0, sticky=tk.EW, pady=(2, 6))
        ttk.Button(frame, text="File...", width=9, command=self.browse_file).grid(row=1, column=1, padx=(6, 0), pady=(2, 6))
        ttk.Button(frame, text="Folder...", width=9, command=self.browse_folder).grid(row=1, column=2, padx=(4, 0), pady=(2, 6))

        self.lbl_prolif = ttk.Label(frame, text="ProLIF interactions JSON")
        self.lbl_prolif.grid(row=2, column=0, columnspan=3, sticky=tk.W)
        self.entry_prolif = ttk.Entry(frame)
        self.entry_prolif.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(2, 6))
        self.btn_browse_prolif = ttk.Button(frame, text="Browse...", width=9, command=self.browse_prolif)
        self.btn_browse_prolif.grid(row=3, column=2, padx=(4, 0), pady=(2, 6))

        self.lbl_output_dir = ttk.Label(frame, text="Default save directory")
        self.lbl_output_dir.grid(row=4, column=0, columnspan=3, sticky=tk.W)
        self.entry_output_dir = ttk.Entry(frame)
        self.entry_output_dir.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(2, 0))
        ttk.Button(frame, text="Browse...", width=9, command=self.browse_output_dir).grid(row=5, column=2, padx=(4, 0), pady=(2, 0))

        self.var_auto_save = tk.BooleanVar(value=self.config.auto_save_single_run)
        self.chk_auto_save = ttk.Checkbutton(frame, text="Auto-save figure and manifest after single run", variable=self.var_auto_save)
        self.chk_auto_save.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        self.benchmark_run_mode_row = ttk.Frame(frame)
        self.benchmark_run_mode_row.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
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
        self.lbl_benchmark_mode_note.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    def _init_core_controls(self):
        frame = ttk.LabelFrame(self.left_frame, text="Core Analysis", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)

        self.entry_chain_a = self._chain_row(frame, 0, "Surface chain", DEFAULT_RUN_CONFIG.chain_a)
        self.entry_chain_b = self._chain_row(frame, 1, "Partner chain", DEFAULT_RUN_CONFIG.chain_b)
        self.entry_cutoff = self._entry_row(frame, 2, "Interface cutoff", DEFAULT_RUN_CONFIG.topology.distance_cutoff, unit="Å")
        self.entry_res = self._entry_row(frame, 3, "Grid resolution", DEFAULT_RUN_CONFIG.surface.grid_resolution, unit="Å")
        self.entry_sigma = self._entry_row(frame, 4, "Surface sigma", DEFAULT_RUN_CONFIG.surface.sigma)
        self.entry_min_points = self._entry_row(frame, 5, "Minimum interacting residues", self.config.default_min_points)
        ttk.Label(
            frame,
            text="Interfaces below this residue threshold are excluded before OptCuts.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        ).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        self.lbl_max_workers = ttk.Label(frame, text="Benchmark workers")
        self.lbl_max_workers.grid(row=7, column=0, sticky=tk.W, pady=3)
        self.entry_max_workers = ttk.Entry(frame, width=10)
        self.entry_max_workers.grid(row=7, column=1, sticky=tk.EW, pady=3)
        self.lbl_max_workers_hint = ttk.Label(
            frame,
            text="Leave blank to use the CPU count.",
            style="Muted.TLabel",
            wraplength=self.config.sidebar_width - 70,
        )
        self.lbl_max_workers_hint.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

    def _init_optcuts_controls(self):
        frame = ttk.LabelFrame(self.left_frame, text="OptCuts and Export", padding=10)
        frame.pack(fill=tk.X, pady=5)
        frame.columnconfigure(1, weight=1)

        self.entry_optcuts_bin = self._entry_row(frame, 0, "OptCuts binary", DEFAULT_RUN_CONFIG.optcuts.optcuts_bin, width=16)

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
        )
        self.entry_optcuts_min_frame_long_edge = self._entry_row(
            frame,
            3,
            "Minimum frame size",
            DEFAULT_RUN_CONFIG.optcuts.optcuts_min_frame_long_edge,
            unit="px",
        )

        ttk.Label(frame, text="Frame output directory").grid(row=4, column=0, sticky=tk.W, pady=(4, 2))
        self.entry_optcuts_frames_dir = ttk.Entry(frame)
        self.entry_optcuts_frames_dir.grid(row=4, column=1, sticky=tk.EW, pady=(4, 2))
        self.btn_optcuts_frames_dir = ttk.Button(frame, text="Browse...", width=9, command=self.browse_optcuts_frames_dir)
        self.btn_optcuts_frames_dir.grid(
            row=4,
            column=2,
            padx=(4, 0),
            pady=(4, 2),
        )
        self._sync_optcuts_frame_controls()

    def _init_visual_controls(self):
        color_frame = ttk.LabelFrame(self.left_frame, text="Color and Interactions", padding=10)
        color_frame.pack(fill=tk.X, pady=5)
        color_frame.columnconfigure(0, weight=1)

        self.var_color_type = tk.BooleanVar(value=True)
        self.chk_type = ttk.Checkbutton(
            color_frame,
            text="Color by interaction type",
            variable=self.var_color_type,
            command=self.toggle_color_mode,
        )
        self.chk_type.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        swatch_row = ttk.Frame(color_frame)
        swatch_row.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        ttk.Label(swatch_row, text="Default residue color").pack(side=tk.LEFT)
        self.residue_color = self.config.default_residue_color
        self.color_swatch = tk.Canvas(swatch_row, width=26, height=16, highlightthickness=1, highlightbackground="#9ca3af")
        self.color_swatch.pack(side=tk.LEFT, padx=(8, 4))
        self.btn_color = ttk.Button(swatch_row, text="Choose...", style="Tool.TButton", command=self.choose_color)
        self.btn_color.pack(side=tk.LEFT)
        self._update_color_swatch()

        filter_actions = ttk.Frame(color_frame)
        filter_actions.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
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
        self.filter_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW)
        self.filter_controls = []
        for i, itype in enumerate(self.interaction_types_list):
            cell = ttk.Frame(self.filter_frame)
            cell.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=(0, 10), pady=2)
            swatch = tk.Canvas(cell, width=12, height=12, highlightthickness=0)
            swatch.create_rectangle(0, 0, 12, 12, fill=INTERACTION_COLORS.get(itype, "#9ca3af"), outline="")
            swatch.pack(side=tk.LEFT, padx=(0, 4))
            var = tk.BooleanVar(value=(itype in self.default_active))
            self.interaction_vars[itype] = var
            chk = ttk.Checkbutton(cell, text=itype, variable=var, command=self.redraw_plot)
            chk.pack(side=tk.LEFT)
            self.filter_controls.append(chk)

        label_frame = ttk.LabelFrame(self.left_frame, text="Labels and Layout", padding=10)
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

        self.toggle_color_mode()

    def _init_run_controls(self):
        frame = self.sidebar_action_frame
        frame.pack(fill=tk.X, pady=(8, 5))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.btn_run = ttk.Button(frame, text="Run Single Analysis", style="Primary.TButton", command=self.start_selected_run)
        self.btn_run.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 6))
        self.btn_bench = None

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
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

    def _entry_row(self, parent, row, label, default, width=10, unit=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
        if unit:
            ttk.Label(parent, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=3)
        return entry

    def _chain_row(self, parent, row, label, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        combo = ttk.Combobox(parent, width=10)
        combo.set(str(default))
        combo.grid(row=row, column=1, sticky=tk.EW, pady=3)
        return combo

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
        self.chk_auto_save.config(state=tk.DISABLED if is_benchmark else tk.NORMAL)
        self.combo_benchmark_run_mode.config(state="readonly" if is_benchmark else tk.DISABLED)
        if is_benchmark:
            self.benchmark_run_mode_row.grid()
            self.lbl_benchmark_mode_note.grid()
            self.lbl_max_workers.grid()
            self.entry_max_workers.grid()
            self.lbl_max_workers_hint.grid()
        else:
            self.benchmark_run_mode_row.grid_remove()
            self.lbl_benchmark_mode_note.grid_remove()
            self.lbl_max_workers.grid_remove()
            self.entry_max_workers.grid_remove()
            self.lbl_max_workers_hint.grid_remove()
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
        self._busy = True
        self.btn_run.config(state=tk.DISABLED)
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
        self.log(message)

    def _finish_task(self):
        self._busy = False
        self.progress.stop()
        self.progress.grid_remove()
        self._refresh_action_states()

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
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, filename)
            self.var_run_mode.set("single")
            self.entry_prolif.delete(0, tk.END)
            if hasattr(self, "entry_output_dir") and not self.entry_output_dir.get().strip():
                self.entry_output_dir.insert(0, os.path.dirname(filename))
            self._populate_chain_choices(filename)
            self._sync_mode_controls()
            self._update_run_summary()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, folder)
            self.var_run_mode.set("benchmark")
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
            self.log(f"Chain scan skipped: {exc}")
            return
        if not chains:
            self.log("No protein chains detected in selected structure.")
            return
        self.entry_chain_a.configure(values=chains)
        self.entry_chain_b.configure(values=chains)
        if self.entry_chain_a.get().strip() not in chains:
            self.entry_chain_a.set(chains[0])
        if self.entry_chain_b.get().strip() not in chains and len(chains) > 1:
            self.entry_chain_b.set(chains[1])
        counts = []
        for chain_id in chains:
            try:
                counts.append(f"{chain_id}:{loader.get_chain_residue_count(chain_id)}")
            except Exception:
                counts.append(f"{chain_id}:?")
        self.log("Detected protein chains (residues): " + ", ".join(counts))

    def browse_prolif(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filename:
            self.entry_prolif.delete(0, tk.END)
            self.entry_prolif.insert(0, filename)
            self._update_run_summary()

    def browse_output_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_output_dir.delete(0, tk.END)
            self.entry_output_dir.insert(0, folder)
            self._update_run_summary()

    def browse_optcuts_frames_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_optcuts_frames_dir.delete(0, tk.END)
            self.entry_optcuts_frames_dir.insert(0, folder)

    def choose_color(self):
        color = colorchooser.askcolor(color=self.residue_color, title="Select Residue Color")[1]
        if color:
            self.residue_color = color
            self._update_color_swatch()
            self.redraw_plot()

    def _update_color_swatch(self):
        self.color_swatch.delete("all")
        self.color_swatch.create_rectangle(0, 0, 26, 16, fill=self.residue_color, outline="")

    def set_all_interactions(self, enabled):
        for var in self.interaction_vars.values():
            var.set(bool(enabled))
        self.redraw_plot()

    def toggle_color_mode(self):
        color_by_type = self.var_color_type.get()
        self.btn_color.config(state=tk.DISABLED if color_by_type else tk.NORMAL)
        for child in self.filter_controls:
            child.configure(state=tk.NORMAL if color_by_type else tk.DISABLED)
        self.btn_filter_all.config(state=tk.NORMAL if color_by_type else tk.DISABLED)
        self.btn_filter_none.config(state=tk.NORMAL if color_by_type else tk.DISABLED)
        if self.cached_viz:
            self.redraw_plot()

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
        self.log(f"Auto-saved figure to {file_path}")
        self.log(f"Auto-saved manifest to {manifest_path}")

    def start_analysis(self):
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
        self.label_offsets = {}
        self.last_run_params = dict(params)
        self._update_run_summary()
        self._begin_task("Starting single analysis pipeline...", progress_mode="indeterminate")
        threading.Thread(target=self.run_pipeline, args=(params,), daemon=True).start()
