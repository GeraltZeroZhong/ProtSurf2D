import tkinter as tk
from tkinter import colorchooser, messagebox

import matplotlib

try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotMixin:
    def redraw_plot(self):
        if self._busy:
            self.log("Style updates are disabled while a run is active.")
            return
        if not self.cached_viz or not self.cached_patches:
            return
        style = self.get_style_config()
        self.log("Updating plot style...")
        self.update_plot(self.cached_viz, self.cached_patches, style, complete_task=False)

    def finish_success(self, plot_result=True):
        if plot_result and self.cached_viz is not None and self.cached_patches is not None:
            style = self.get_style_config()
            self.update_plot(self.cached_viz, self.cached_patches, style, complete_task=True)
        else:
            self._finish_task()

    def on_pick(self, event):
        if self._picking:
            return
        artist = event.artist
        if artist.__class__.__name__ != "PathCollection":
            return
        gid = artist.get_gid()
        if gid and self.cached_viz and gid in self.cached_viz.artist_map:
            self._picking = True
            try:
                color = colorchooser.askcolor(title=f"Color for {gid}")[1]
                if color:
                    target_objs = self.cached_viz.artist_map[gid]
                    target_objs['scatter'].set_facecolor(color)
                    target_objs['scatter'].set_edgecolor(color)
                    self.current_canvas.draw()
            finally:
                self._picking = False

    def on_mouse_press(self, event):
        if not self.cached_viz or not event.inaxes:
            return
        for gid, objs in self.cached_viz.artist_map.items():
            txt = objs.get('text')
            if txt is None:
                continue
            contains, _ = txt.contains(event)
            if contains:
                self._drag_state = {'gid': gid}
                break

    def on_mouse_move(self, event):
        if not self._drag_state or not event.inaxes or event.xdata is None or event.ydata is None:
            return
        gid = self._drag_state['gid']
        objs = self.cached_viz.artist_map.get(gid, {})
        txt = objs.get('text')
        if txt is None:
            return
        txt.set_position((event.xdata, event.ydata))
        connector = objs.get('connector')
        scatter = objs.get('scatter')
        if connector is not None and scatter is not None:
            pt = scatter.get_offsets()[0]
            connector.set_data([pt[0], event.xdata], [pt[1], event.ydata])
        self.current_canvas.draw_idle()

    def on_mouse_release(self, event):
        if not self._drag_state:
            return
        gid = self._drag_state['gid']
        objs = self.cached_viz.artist_map.get(gid, {})
        txt = objs.get('text')
        scatter = objs.get('scatter')
        if txt is not None and scatter is not None:
            pt = scatter.get_offsets()[0]
            tx, ty = txt.get_position()
            self.label_offsets[gid] = (float(tx - pt[0]), float(ty - pt[1]))
        self._drag_state = None

    def update_plot(self, viz, patches, style, complete_task=False):
        try:
            fig = viz.plot_patches(patches, show=False, style_config=style)
        except Exception as exc:
            if complete_task:
                self.show_error(f"Failed to generate plot: {exc}")
            else:
                self.log(f"Failed to update plot style: {exc}")
            return

        if fig is None:
            if complete_task:
                self.show_error("Failed to generate plot.")
            else:
                self.log("Failed to update plot style.")
            return

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
        self.current_canvas.mpl_connect('pick_event', self.on_pick)
        self.current_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.current_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.current_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        if complete_task:
            self._finish_task()
            if self.last_run_params.get("auto_save"):
                try:
                    self._auto_save_current_figure()
                except Exception as exc:
                    self.log(f"Auto-save failed: {exc}")
        else:
            self._refresh_action_states()
        self.log(f"Success! Displaying {len(patches)} patches.")

    def _build_plot_toolbar(self, parent):
        actions = [
            ("Home", self.current_toolbar.home),
            ("Back", self.current_toolbar.back),
            ("Forward", self.current_toolbar.forward),
            ("Pan", self.current_toolbar.pan),
            ("Zoom", self.current_toolbar.zoom),
        ]
        for label, command in actions:
            tk.Button(parent, text=label, command=command, relief=tk.FLAT, padx=8, pady=2).pack(side=tk.LEFT, padx=(0, 4))
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
