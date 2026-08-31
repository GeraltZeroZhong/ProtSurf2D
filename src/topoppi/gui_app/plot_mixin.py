import tkinter as tk
from tkinter import colorchooser, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotMixin:
    def redraw_plot(self):
        if self._busy:
            self.log("Style updates are disabled while a run is active.")
            return
        successful_run = self._successful_single_run
        if successful_run is None:
            return
        style = self.get_style_config()
        self.log("Updating plot style...")
        self.update_plot(
            successful_run["viz"],
            successful_run["patches"],
            style,
            complete_task=False,
        )

    def on_pick(self, event):
        artist = event.artist
        if artist.__class__.__name__ != "PathCollection":
            return
        gid = artist.get_gid()
        successful_run = self._successful_single_run
        viz = successful_run["viz"] if successful_run else None
        if gid and viz and gid in viz.artist_map:
            color = colorchooser.askcolor(title=f"Color for {gid}")[1]
            if color:
                self._mark_style_custom()
                self.marker_color_overrides[gid] = color
                target_objs = viz.artist_map[gid]
                target_objs["scatter"].set_facecolor(color)
                self.current_canvas.draw()
                self._successful_single_run["style"] = self.get_style_config()

    def on_mouse_press(self, event):
        successful_run = self._successful_single_run
        if successful_run is None or not event.inaxes:
            return
        for gid, objs in successful_run["viz"].artist_map.items():
            txt = objs.get("text")
            if txt is None:
                continue
            contains, _ = txt.contains(event)
            if contains:
                self._drag_state = {"gid": gid}
                break

    def on_mouse_move(self, event):
        if not self._drag_state or not event.inaxes or event.xdata is None or event.ydata is None:
            return
        gid = self._drag_state["gid"]
        objs = self._successful_single_run["viz"].artist_map.get(gid, {})
        txt = objs.get("text")
        if txt is None:
            return
        txt.set_position((event.xdata, event.ydata))
        connector = objs.get("connector")
        scatter = objs.get("scatter")
        if connector is not None and scatter is not None:
            pt = scatter.get_offsets()[0]
            connector.set_data([pt[0], event.xdata], [pt[1], event.ydata])
        self.current_canvas.draw_idle()

    def on_mouse_release(self, event):
        if not self._drag_state:
            return
        gid = self._drag_state["gid"]
        objs = self._successful_single_run["viz"].artist_map.get(gid, {})
        txt = objs.get("text")
        scatter = objs.get("scatter")
        if txt is not None and scatter is not None:
            pt = scatter.get_offsets()[0]
            tx, ty = txt.get_position()
            self.label_offsets[gid] = (float(tx - pt[0]), float(ty - pt[1]))
            self._successful_single_run["style"] = self.get_style_config()
        self._drag_state = None

    def update_plot(
        self,
        viz,
        patches,
        style,
        complete_task=False,
        *,
        run_params=None,
        run_manifest=None,
    ):
        try:
            fig = viz.plot_patches(patches, show=False, style_config=style)
        except Exception as exc:
            if complete_task:
                self.show_error(self._previous_result_message(f"Failed to generate plot: {exc}"))
            else:
                self.log(f"Failed to update plot style: {exc}")
            return

        annotation = dict(viz.last_report)

        old_fig = self.current_fig
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        if old_fig is not None and old_fig is not fig:
            plt.close(old_fig)

        self.current_fig = fig
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.current_canvas.draw()
        toolbar_frame = tk.Frame(self.canvas_frame, bg="#ffffff")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))
        hidden_toolbar = tk.Frame(self.canvas_frame)
        self.current_toolbar = NavigationToolbar2Tk(self.current_canvas, hidden_toolbar, pack_toolbar=False)
        self.current_toolbar.update()
        self._build_plot_toolbar(toolbar_frame)
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas.mpl_connect("pick_event", self.on_pick)
        self.current_canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.current_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.current_canvas.mpl_connect("button_release_event", self.on_mouse_release)
        if complete_task:
            manifest = dict(run_manifest)
            manifest["visualization_annotation"] = annotation
            successful_run = {
                "figure": fig,
                "manifest": manifest,
                "params": dict(run_params),
                "patches": patches,
                "log": list(self.current_run_log),
                "style": dict(style),
                "viz": viz,
            }
            self._successful_single_run = successful_run
            self._finish_task()
            if successful_run["params"].get("auto_save"):
                try:
                    self._auto_save_current_figure(successful_run)
                except Exception as exc:
                    self.log(f"Auto-save failed: {exc}")
        else:
            successful_run = self._successful_single_run
            successful_run["figure"] = fig
            successful_run["manifest"]["visualization_annotation"] = annotation
            successful_run["style"] = dict(style)
            self._refresh_action_states()
        annotation = viz.last_report
        interaction_source = str(annotation.get("interaction_residue_source", "interaction"))
        self.log(
            "Success! Displaying {} patches and {} annotated residues "
            "({} {} interaction residues on the patch domain).".format(
                len(patches),
                int(annotation.get("displayed_residue_count", 0)),
                int(annotation.get("patch_interaction_residue_count", 0)),
                interaction_source,
            )
        )

    def _build_plot_toolbar(self, parent):
        actions = [
            ("Home", self.current_toolbar.home),
            ("Back", self.current_toolbar.back),
            ("Forward", self.current_toolbar.forward),
            ("Pan", self.current_toolbar.pan),
            ("Zoom", self.current_toolbar.zoom),
        ]
        for label, command in actions:
            tk.Button(parent, text=label, command=command, relief=tk.FLAT, padx=8, pady=2).pack(
                side=tk.LEFT, padx=(0, 4)
            )
        tk.Label(
            parent,
            text="Drag labels to reposition; click residue markers to recolor.",
            bg="#ffffff",
            fg="#6b7280",
        ).pack(side=tk.RIGHT)

    def _close_current_figure(self):
        if self.current_fig is not None:
            plt.close(self.current_fig)
            self.current_fig = None
        self.current_canvas = None
        self.current_toolbar = None

    def show_error(self, msg):
        self._finish_task()
        self.log(f"Error: {msg}")
        messagebox.showerror("Pipeline Error", msg)
