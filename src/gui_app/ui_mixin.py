import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser

class UIMixin:
    def _init_controls(self):
        lbl_frame = ttk.LabelFrame(self.left_frame, text="1. Input Data", padding=10)
        lbl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(lbl_frame, text="Input (PDB File or Batch Folder):").pack(anchor=tk.W)
        self.entry_file = ttk.Entry(lbl_frame)
        self.entry_file.pack(fill=tk.X, pady=2)

        btn_box = ttk.Frame(lbl_frame)
        btn_box.pack(fill=tk.X)
        ttk.Button(btn_box, text="File...", width=8, command=self.browse_file).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_box, text="Folder...", width=8, command=self.browse_folder).pack(side=tk.RIGHT, padx=2)

        ttk.Label(lbl_frame, text="ProLIF JSON (Optional):").pack(anchor=tk.W, pady=(5, 0))
        self.entry_prolif = ttk.Entry(lbl_frame)
        self.entry_prolif.pack(fill=tk.X, pady=2)
        ttk.Button(lbl_frame, text="Browse JSON...", command=self.browse_prolif).pack(anchor=tk.E)

        param_frame = ttk.LabelFrame(self.left_frame, text="2. Analysis Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=1)

        ttk.Label(param_frame, text="Chain A (Surf):").grid(row=0, column=0, sticky='w')
        self.entry_chain_a = ttk.Entry(param_frame, width=10)
        self.entry_chain_a.insert(0, "A")
        self.entry_chain_a.grid(row=0, column=1, pady=2)

        ttk.Label(param_frame, text="Chain B (Lig):").grid(row=1, column=0, sticky='w')
        self.entry_chain_b = ttk.Entry(param_frame, width=10)
        self.entry_chain_b.insert(0, "B")
        self.entry_chain_b.grid(row=1, column=1, pady=2)

        ttk.Label(param_frame, text="Cutoff (Å):").grid(row=2, column=0, sticky='w')
        self.entry_cutoff = ttk.Entry(param_frame, width=10)
        self.entry_cutoff.insert(0, "5.0")
        self.entry_cutoff.grid(row=2, column=1, pady=2)

        ttk.Label(param_frame, text="Grid Res (Å):").grid(row=3, column=0, sticky='w')
        self.entry_res = ttk.Entry(param_frame, width=10)
        self.entry_res.insert(0, "1.0")
        self.entry_res.grid(row=3, column=1, pady=2)

        ttk.Label(param_frame, text="Sigma:").grid(row=4, column=0, sticky='w')
        self.entry_sigma = ttk.Entry(param_frame, width=10)
        self.entry_sigma.insert(0, "1.0")
        self.entry_sigma.grid(row=4, column=1, pady=2)

        ttk.Label(param_frame, text="Min Points/Interface:").grid(row=5, column=0, sticky='w')
        self.entry_min_points = ttk.Entry(param_frame, width=10)
        self.entry_min_points.insert(0, "10")
        self.entry_min_points.grid(row=5, column=1, pady=2)

        self.var_filter_valid_only = tk.BooleanVar(value=True)
        self.chk_filter_valid_only = ttk.Checkbutton(param_frame, text="Show Valid Interfaces Only", variable=self.var_filter_valid_only)
        self.chk_filter_valid_only.grid(row=6, column=0, columnspan=2, sticky='w', pady=(4, 0))

        ttk.Label(param_frame, text="OptCuts Bin:").grid(row=8, column=0, sticky='w')
        self.entry_optcuts_bin = ttk.Entry(param_frame, width=14)
        self.entry_optcuts_bin.insert(0, "OptCuts_bin")
        self.entry_optcuts_bin.grid(row=8, column=1, pady=2)

        self.var_save_optcuts_frames = tk.BooleanVar(value=False)
        self.chk_save_optcuts_frames = ttk.Checkbutton(
            param_frame,
            text="Export OptCuts Frames",
            variable=self.var_save_optcuts_frames,
        )
        self.chk_save_optcuts_frames.grid(row=9, column=0, columnspan=2, sticky='w', pady=(4, 0))

        ttk.Label(param_frame, text="Frame Stride:").grid(row=10, column=0, sticky='w')
        self.entry_optcuts_frame_stride = ttk.Entry(param_frame, width=10)
        self.entry_optcuts_frame_stride.insert(0, "5")
        self.entry_optcuts_frame_stride.grid(row=10, column=1, pady=2)

        ttk.Label(param_frame, text="Frame Dir (optional):").grid(row=11, column=0, sticky='w')
        self.entry_optcuts_frames_dir = ttk.Entry(param_frame, width=14)
        self.entry_optcuts_frames_dir.grid(row=11, column=1, pady=2)

        self.var_auto_cutoff = tk.BooleanVar(value=False)
        self.chk_auto_cutoff = ttk.Checkbutton(param_frame, text="Auto Search Best Cutoff", variable=self.var_auto_cutoff)
        self.chk_auto_cutoff.grid(row=12, column=0, columnspan=2, sticky='w', pady=(4, 0))

        ttk.Label(param_frame, text="Cutoff Start/End:").grid(row=13, column=0, sticky='w')
        cutoff_range_frame = ttk.Frame(param_frame)
        cutoff_range_frame.grid(row=13, column=1, pady=2, sticky='w')
        self.entry_cutoff_start = ttk.Entry(cutoff_range_frame, width=4)
        self.entry_cutoff_start.insert(0, "3.0")
        self.entry_cutoff_start.pack(side=tk.LEFT)
        ttk.Label(cutoff_range_frame, text="~").pack(side=tk.LEFT, padx=2)
        self.entry_cutoff_end = ttk.Entry(cutoff_range_frame, width=4)
        self.entry_cutoff_end.insert(0, "10.0")
        self.entry_cutoff_end.pack(side=tk.LEFT)

        ttk.Label(param_frame, text="Cutoff Step:").grid(row=14, column=0, sticky='w')
        self.entry_cutoff_step = ttk.Entry(param_frame, width=10)
        self.entry_cutoff_step.insert(0, "0.5")
        self.entry_cutoff_step.grid(row=14, column=1, pady=2)

        style_frame = ttk.LabelFrame(self.left_frame, text="3. Visualization Style", padding=10)
        style_frame.pack(fill=tk.X, padx=5, pady=5)

        self.var_color_type = tk.BooleanVar(value=True)
        self.chk_type = ttk.Checkbutton(style_frame, text="Color by Interaction Type", variable=self.var_color_type, command=self.toggle_color_mode)
        self.chk_type.pack(anchor=tk.W, pady=2)

        self.filter_frame = ttk.LabelFrame(style_frame, text="Show ProLIF Interactions", padding=5)
        self.filter_frame.pack(fill=tk.X, pady=5)

        for i, itype in enumerate(self.interaction_types_list):
            var = tk.BooleanVar(value=(itype in self.default_active))
            self.interaction_vars[itype] = var
            ttk.Checkbutton(
                self.filter_frame,
                text=itype,
                variable=var,
                command=self.redraw_plot
            ).grid(row=i // 3, column=i % 3, sticky='w', padx=4, pady=1)

        f_frame = ttk.Frame(style_frame)
        f_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f_frame, text="Font:").pack(side=tk.LEFT)
        self.combo_font = ttk.Combobox(f_frame, values=["Arial", "Times New Roman", "Courier New", "sans-serif"], width=10)
        self.combo_font.current(3)
        self.combo_font.pack(side=tk.LEFT, padx=2)
        self.spin_size = ttk.Spinbox(f_frame, from_=5, to=20, width=4)
        self.spin_size.set(9)
        self.spin_size.pack(side=tk.LEFT)

        self.residue_color = "#ff0000"
        self.btn_color = tk.Button(f_frame, text="Def. Color", bg=self.residue_color, command=self.choose_color, relief=tk.RAISED, state=tk.DISABLED, width=10)
        self.btn_color.pack(side=tk.RIGHT, padx=5)

        self.var_show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(style_frame, text="Show Labels", variable=self.var_show_labels, command=self.redraw_plot).pack(anchor=tk.W, pady=2)

        label_mode_frame = ttk.Frame(style_frame)
        label_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(label_mode_frame, text="Label Mode:").pack(side=tk.LEFT)
        self.label_mode_options = {
            "Chain A Residue": "chain_a",
            "Chain B Residue": "chain_b",
            "A-B Pair": "pair"
        }
        self.combo_label_mode = ttk.Combobox(label_mode_frame, values=list(self.label_mode_options.keys()), width=16, state="readonly")
        self.combo_label_mode.set("Chain A Residue")
        self.combo_label_mode.pack(side=tk.LEFT, padx=4)
        self.combo_label_mode.bind("<<ComboboxSelected>>", lambda _e: self.redraw_plot())

        self.var_avoid_overlap = tk.BooleanVar(value=True)
        ttk.Checkbutton(style_frame, text="Reduce Label Overlap", variable=self.var_avoid_overlap, command=self.redraw_plot).pack(anchor=tk.W, pady=2)

        self.var_use_uv_atlas = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            style_frame,
            text="Merge Patches into UV Atlas View",
            variable=self.var_use_uv_atlas,
            command=self.redraw_plot
        ).pack(anchor=tk.W, pady=2)

        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.btn_run = ttk.Button(btn_frame, text="Run Single Analysis", command=self.start_analysis)
        self.btn_run.grid(row=0, column=0, columnspan=2, sticky='ew', pady=4)

        self.btn_bench = ttk.Button(btn_frame, text="Run Benchmark", command=self.start_benchmark)
        self.btn_bench.grid(row=1, column=0, columnspan=2, sticky='ew', pady=4)

        self.btn_redraw = ttk.Button(btn_frame, text="Update Style Only", command=self.redraw_plot, state=tk.DISABLED)
        self.btn_redraw.grid(row=2, column=0, sticky='ew', pady=4, padx=(0, 4))

        self.btn_save = ttk.Button(btn_frame, text="Save Figure...", command=self.save_figure, state=tk.DISABLED)
        self.btn_save.grid(row=2, column=1, sticky='ew', pady=4, padx=(4, 0))

        self.progress = ttk.Progressbar(self.left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def _init_plot_area(self):
        self.canvas_frame = ttk.Frame(self.right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.current_canvas = None
        lbl = ttk.Label(self.canvas_frame, text="Load a PDB then click Run.", font=("Arial", 12))
        lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _init_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, message):
        self.status_var.set(message)
        print(f"[GUI Log] {message}")
        self.root.update_idletasks()

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("PDB Files", "*.pdb"), ("CIF Files", "*.cif"), ("All Files", "*.*")])
        if filename:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, filename)
            self.entry_prolif.delete(0, tk.END)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, folder)
            self.log(f"Selected folder: {os.path.basename(folder)}")

    def browse_prolif(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filename:
            self.entry_prolif.delete(0, tk.END)
            self.entry_prolif.insert(0, filename)

    def choose_color(self):
        color = colorchooser.askcolor(color=self.residue_color, title="Select Residue Color")[1]
        if color:
            self.residue_color = color
            self.btn_color.config(bg=color)

    def toggle_color_mode(self):
        if self.var_color_type.get():
            self.btn_color.config(state=tk.DISABLED)
            for child in self.filter_frame.winfo_children():
                child.configure(state='normal')
        else:
            self.btn_color.config(state=tk.NORMAL)
            for child in self.filter_frame.winfo_children():
                child.configure(state='disabled')
        if self.cached_viz:
            self.redraw_plot()

    def get_style_config(self):
        active_types = [t for t, var in self.interaction_vars.items() if var.get()]
        return {
            'color': self.residue_color,
            'font_family': self.combo_font.get(),
            'font_size': int(self.spin_size.get()),
            'color_by_type': self.var_color_type.get(),
            'active_types': active_types,
            'show_labels': bool(self.var_show_labels.get()),
            'label_mode': self.label_mode_options.get(self.combo_label_mode.get(), "chain_a"),
            'avoid_label_overlap': bool(self.var_avoid_overlap.get()),
            'use_uv_atlas': bool(self.var_use_uv_atlas.get()),
            'label_offsets': dict(self.label_offsets)
        }

    def save_figure(self):
        if not self.current_fig:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("TIFF Image", "*.tif *.tiff"), ("All Files", "*.*")],
            title="Save Figure As"
        )
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'facecolor': 'white'}
                if ext in {'.tif', '.tiff'}:
                    self.current_fig.savefig(file_path, format='tiff', pil_kwargs={'compression': 'tiff_lzw'}, **save_kwargs)
                else:
                    self.current_fig.savefig(file_path, **save_kwargs)
                self.log(f"Figure saved to {file_path}")
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            except Exception as e:
                self.log(f"Error saving image: {e}")
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def start_analysis(self):
        path = self.entry_file.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid PDB file.")
            return
        if os.path.isdir(path):
            messagebox.showerror("Error", "Single Analysis requires a .pdb file, not a folder.\nUse 'Run Benchmark' for folders.")
            return

        params = {
            'path': path,
            'chain_a': self.entry_chain_a.get().strip(),
            'chain_b': self.entry_chain_b.get().strip(),
            'prolif': self.entry_prolif.get().strip(),
            'cutoff': float(self.entry_cutoff.get()),
            'res': float(self.entry_res.get()),
            'sigma': float(self.entry_sigma.get()),
            'min_points': int(self.entry_min_points.get()),
            'filter_valid_only': bool(self.var_filter_valid_only.get()),
            'optcuts_bin': self.entry_optcuts_bin.get().strip() or "OptCuts_bin",
            'save_optcuts_frames': bool(self.var_save_optcuts_frames.get()),
            'optcuts_frame_stride': int(self.entry_optcuts_frame_stride.get() or "5"),
            'optcuts_frames_dir': self.entry_optcuts_frames_dir.get().strip(),
            'auto_cutoff': bool(self.var_auto_cutoff.get()),
            'cutoff_start': float(self.entry_cutoff_start.get()),
            'cutoff_end': float(self.entry_cutoff_end.get()),
            'cutoff_step': float(self.entry_cutoff_step.get())
        }
        self.label_offsets = {}
        self.btn_run.config(state=tk.DISABLED)
        self.btn_bench.config(state=tk.DISABLED)
        self.btn_redraw.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.progress.start(10)
        self.log("Starting analysis pipeline...")
        threading.Thread(target=self.run_pipeline, args=(params,), daemon=True).start()
