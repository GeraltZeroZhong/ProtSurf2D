import os
import threading
from dataclasses import replace
from datetime import datetime
from tkinter import messagebox

from topoppi.config import BenchmarkConfig, DEFAULT_RUN_CONFIG
from topoppi.io.io_loader import PDBLoader
from topoppi.mesh.surface import SurfaceGenerator
from topoppi.mesh.topology import TopologyManager
from topoppi.mesh.parameterization import Parameterizer
from topoppi.optimization.optcuts import OptCutsUVOptimizer
from topoppi.visualization.visualizer import InterfaceVisualizer
from topoppi.interactions.interaction_engine import generate_prolif_interactions
from topoppi.benchmarking import BenchmarkRunner


class WorkflowMixin:
    @staticmethod
    def _default_optcuts_frame_dir(input_path: str) -> str:
        base_dir = os.path.dirname(input_path) or os.getcwd()
        stem = os.path.splitext(os.path.basename(input_path))[0]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{stem}_optcuts_frames_{ts}")

    def start_benchmark(self):
        folder = self.entry_file.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder containing .pdb files.")
            return

        params = {
            'folder': folder,
            'chain_a': self.entry_chain_a.get().strip(),
            'chain_b': self.entry_chain_b.get().strip(),
            'cutoff': float(self.entry_cutoff.get()),
            'res': float(self.entry_res.get()),
            'sigma': float(self.entry_sigma.get()),
            'patch_gap': DEFAULT_RUN_CONFIG.optcuts.patch_gap,
            'optcuts_bin': self.entry_optcuts_bin.get().strip() or DEFAULT_RUN_CONFIG.optcuts.optcuts_bin,
        }
        self.btn_run.config(state="disabled")
        self.btn_bench.config(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.log("Starting benchmark pipeline...")
        threading.Thread(target=self.run_benchmark_pipeline, args=(params,), daemon=True).start()

    def run_benchmark_pipeline(self, params):
        try:
            output_root = os.path.join(params["folder"], "benchmark_results_resume")
            config = BenchmarkConfig(
                input_folder=params["folder"],
                output_root=output_root,
                chain_a=params["chain_a"],
                chain_b=params["chain_b"],
                surface=replace(DEFAULT_RUN_CONFIG.surface, grid_resolution=params["res"], sigma=params["sigma"]),
                topology=replace(DEFAULT_RUN_CONFIG.topology, distance_cutoff=params["cutoff"]),
                parameterization=DEFAULT_RUN_CONFIG.parameterization,
                optcuts=replace(
                    DEFAULT_RUN_CONFIG.optcuts,
                    patch_gap=params["patch_gap"],
                    optcuts_bin=params["optcuts_bin"],
                ).for_headless(),
                show_tqdm=False,
                resume=True,
            )
            runner = BenchmarkRunner(config=config, log_fn=self.log, progress_fn=self._on_benchmark_progress)
            self.log("Benchmark OptCuts runs in headless mode (viewer disabled).")
            output = runner.run()
            self.root.after(0, lambda: self.progress.configure(value=100))
            summary = output.get("summary", {})
            self.log(
                "Benchmark done. valid_structures={}, lscm_mean={:.4f}, lscm_optcuts_mean={:.4f}, harmonic_mean={:.4f}, spherical_mean={:.4f}, cylindrical_mean={:.4f}".format(
                    int(summary.get("valid_structure_count", 0)),
                    float(summary.get("distortion_lscm_mean", float("inf"))),
                    float(summary.get("distortion_lscm_optcuts_mean", float("inf"))),
                    float(summary.get("distortion_harmonic_mean", float("inf"))),
                    float(summary.get("distortion_spherical_mean", float("inf"))),
                    float(summary.get("distortion_cylindrical_mean", float("inf"))),
                )
            )
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Benchmark Completed",
                    f"Results saved to:\n{output_root}\n\n"
                    "Generated files:\n- benchmark_report.json\n- benchmark_summary.csv",
                ),
            )
            self.root.after(0, lambda: self.finish_success())
        except Exception as e:
            self.root.after(0, lambda msg=str(e): self.show_error(f"Benchmark failed: {msg}"))

    def _on_benchmark_progress(self, completed: int, total: int, message: str):
        self.root.after(0, lambda: self._set_benchmark_progress_ui(completed, total, message))

    def _set_benchmark_progress_ui(self, completed: int, total: int, message: str):
        total_safe = max(1, int(total))
        completed_safe = max(0, min(int(completed), total_safe))
        percent = int((completed_safe / total_safe) * 100.0)
        self.progress.configure(mode="determinate", maximum=100, value=percent)
        self.log(f"[Benchmark][Progress] {completed_safe}/{total_safe} ({percent}%) - {message}")

    def generate_prolif_interactions(self, pdb_path, chain_a, chain_b):
        self.log("Checking ProLIF requirements...")
        output_json = generate_prolif_interactions(pdb_path, chain_a, chain_b)
        if output_json:
            self.root.after(0, lambda: self.entry_prolif.delete(0, "end"))
            self.root.after(0, lambda: self.entry_prolif.insert(0, output_json))
            return output_json
        self.log("ProLIF interaction generation skipped/failed. Falling back to geometric heuristics.")
        return None

    def run_pipeline(self, params):
        try:
            prolif_file = params.get('prolif')
            if not prolif_file or not os.path.exists(prolif_file):
                generated_json = self.generate_prolif_interactions(params['path'], params['chain_a'], params['chain_b'])
                prolif_file = generated_json if generated_json else None

            self.log("Loading PDB structure...")
            loader = PDBLoader(params['path'])
            coords_A, atoms_A = loader.get_chain_data(params['chain_a'])
            coords_B, atoms_B = loader.get_chain_data(params['chain_b'])

            self.log("Generating molecular surface...")
            surface_config = replace(DEFAULT_RUN_CONFIG.surface, grid_resolution=params['res'], sigma=params['sigma'])
            topology_config = replace(DEFAULT_RUN_CONFIG.topology, distance_cutoff=params['cutoff'])
            optcuts_config = replace(
                DEFAULT_RUN_CONFIG.optcuts,
                optcuts_bin=params.get('optcuts_bin', DEFAULT_RUN_CONFIG.optcuts.optcuts_bin),
                patch_gap=params.get('patch_gap', DEFAULT_RUN_CONFIG.optcuts.patch_gap),
                save_optcuts_frames=params.get('save_optcuts_frames', False),
                optcuts_frame_stride=max(1, int(params.get('optcuts_frame_stride', DEFAULT_RUN_CONFIG.optcuts.optcuts_frame_stride))),
                optcuts_min_frame_long_edge=max(0, int(params.get('optcuts_min_frame_long_edge', DEFAULT_RUN_CONFIG.optcuts.optcuts_min_frame_long_edge))),
                optcuts_frames_dir=params.get('optcuts_frames_dir') or self._default_optcuts_frame_dir(params['path']),
            )

            surf_gen = SurfaceGenerator(coords_A, config=surface_config)
            mesh_A = surf_gen.generate_mesh()
            if mesh_A is None:
                raise ValueError("Surface generation failed.")

            self.log("Extracting interface patches...")
            param = Parameterizer(config=DEFAULT_RUN_CONFIG.parameterization)
            optimizer = OptCutsUVOptimizer(optcuts_config)
            if params.get('save_optcuts_frames', False):
                self.log(
                    "OptCuts frame export enabled: stride={}, min_size={}px, output={}".format(
                        optcuts_config.optcuts_frame_stride,
                        optcuts_config.optcuts_min_frame_long_edge,
                        optcuts_config.optcuts_frames_dir,
                    )
                )
            viz = InterfaceVisualizer(
                chain_A_atoms=atoms_A,
                chain_A_coords=coords_A,
                chain_B_coords=coords_B,
                chain_B_atoms=atoms_B,
                chain_a_id=params['chain_a'],
                chain_b_id=params['chain_b'],
                prolif_file=prolif_file,
                config=DEFAULT_RUN_CONFIG.visualization,
            )

            topo = TopologyManager(mesh_A, coords_B, config=topology_config)
            patches = topo.get_interface_patches()
            if not patches:
                raise ValueError(f"No interface found with cutoff {params['cutoff']:.2f}.")

            self.log(f"Flattening {len(patches)} patches...")
            parameterized_patches = self._parameterize_patches(patches, param)
            if not parameterized_patches:
                raise ValueError("LSCM Parameterization failed for all patches.")

            valid_patches, valid_count, invalid_count = self._split_interfaces_by_point_count(
                parameterized_patches,
                viz,
                params['min_points'],
            )
            self.log(f"Interface count summary (min points = {params['min_points']}): valid={valid_count}, invalid={invalid_count}")
            if not valid_patches:
                raise ValueError(f"All interfaces are invalid (point count < {params['min_points']}).")

            self.log(f"Running OptCuts only on valid interfaces ({len(valid_patches)} patches)...")
            optimized_valid_patches = self._optimize_patches(valid_patches, optimizer)

            if params.get('filter_valid_only', True):
                selected_patches = optimized_valid_patches
                self.log("Display mode: valid interfaces only.")
            else:
                selected_patches = optimized_valid_patches
                self.log("Display mode: all interfaces request ignored because invalid interfaces are now excluded before OptCuts.")

            self.log("Rendering visualization...")
            self.cached_viz = viz
            self.cached_patches = selected_patches
            self.root.after(0, lambda: self.finish_success())
        except Exception as e:
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.show_error(msg))

    def _parameterize_patches(self, patches, parameterizer):
        valid_patches = []
        for p in patches:
            uv = parameterizer.flatten_patch(p)
            if uv is not None:
                p.metadata['uv'] = uv
                valid_patches.append(p)
        return valid_patches

    def _optimize_patches(self, patches, optimizer):
        if not patches:
            return []
        result = optimizer.optimize_patches(patches)
        self._log_joint_report(optimizer)
        return result

    def _log_joint_report(self, optimizer):
        report = getattr(optimizer, "get_last_report", lambda: {})()
        if not report:
            self.log("Joint optimization report unavailable.")
            return
        pq = report.get("parameterization_quality", {})
        tc = report.get("topology_complexity", {})
        au = report.get("atlas_usability", {})
        se = report.get("stability_efficiency", {})
        self.log(
            "[JointReport] flip={:.4f}, dist(mean/max/p95)=({:.4f}/{:.4f}/{:.4f})".format(
                float(pq.get("flip_rate_mean", 1.0)),
                float(pq.get("distortion", {}).get("mean", float("inf"))),
                float(pq.get("distortion", {}).get("max", float("inf"))),
                float(pq.get("distortion", {}).get("p95", float("inf"))),
            )
        )
        self.log(
            "[JointReport] seam_len={:.3f}, charts={}, overlap={:.4f}, padding_viol={}, util={:.4f}".format(
                float(tc.get("seam_total_length", 0.0)),
                int(tc.get("chart_count", 0)),
                float(au.get("overlap_area", 0.0)),
                int(au.get("padding_violations", 0)),
                float(au.get("utilization", 0.0)),
            )
        )
        self.log(
            "[JointReport] obj_drop={:.4f}, total_time={:.3f}s, failure_rate={:.3f}".format(
                float(se.get("objective_drop", 0.0)),
                float(se.get("total_time_sec", 0.0)),
                float(se.get("failure_rate", 0.0)),
            )
        )

    def _split_interfaces_by_point_count(self, patches, viz, min_points):
        valid = []
        invalid = 0
        for p in patches:
            point_count = viz.count_patch_points(p)
            p.metadata['point_count'] = point_count
            if point_count >= min_points:
                valid.append(p)
            else:
                invalid += 1
        return valid, len(valid), invalid
