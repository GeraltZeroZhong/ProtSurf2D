import tkinter as tk
from tkinter import colorchooser, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PlotMixin:
    def redraw_plot(self):
        if not self.cached_viz or not self.cached_patches:
            return
        style = self.get_style_config()
        self.log("Updating plot style...")
        self.update_plot(self.cached_viz, self.cached_patches, style)

    def finish_success(self):
        if self.cached_viz is not None and self.cached_patches is not None:
            style = self.get_style_config()
            self.update_plot(self.cached_viz, self.cached_patches, style)
        else:
            self.progress.stop()
            self.btn_run.config(state=tk.NORMAL)
            self.btn_bench.config(state=tk.NORMAL)
        self.btn_redraw.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

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

    def update_plot(self, viz, patches, style):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        fig = viz.plot_patches(patches, show=False, style_config=style)
        self.current_fig = fig

        if fig is None:
            self.show_error("Failed to generate plot.")
            return

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas.mpl_connect('pick_event', self.on_pick)
        self.current_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.current_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.current_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.progress.stop()
        self.btn_run.config(state=tk.NORMAL)
        self.btn_bench.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.log(f"Success! Displaying {len(patches)} patches.")

    def show_error(self, msg):
        self.progress.stop()
        self.btn_run.config(state=tk.NORMAL)
        self.btn_bench.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.log("Error occurred.")
        messagebox.showerror("Pipeline Error", msg)
