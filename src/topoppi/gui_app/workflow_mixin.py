import os
import threading
import logging
import hashlib
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from tkinter import messagebox

from topoppi.config import BenchmarkConfig, DEFAULT_RUN_CONFIG
from topoppi.errors import ConfigurationError
from topoppi.io.io_loader import PDBLoader
from topoppi.mesh.surface import SurfaceGenerator
from topoppi.mesh.topology import TopologyManager
from topoppi.mesh.parameterization import Parameterizer
from topoppi.optimization.optcuts import OptCutsUVOptimizer
from topoppi.visualization.visualizer import InterfaceVisualizer
from topoppi.interactions.interaction_engine import generate_prolif_interactions
from topoppi.benchmarking import BenchmarkRunner
from topoppi.optimization.optcuts import resolve_optcuts_binary
from .forms import parse_benchmark_form


logger = logging.getLogger("topoppi.gui")


class PipelineCancelled(Exception):
    """Raised when a GUI run is cancelled cooperatively."""


class _GuiLogAdapter:
    def __init__(self, log_fn):
        self._log_fn = log_fn

    def info(self, message, *args):
        self._emit(message, *args)

    def warning(self, message, *args):
        self._emit("Warning: " + str(message), *args)

    def error(self, message, *args):
        self._emit("Error: " + str(message), *args)

    def _emit(self, message, *args):
        if args:
            try:
                message = str(message) % args
            except TypeError:
                message = " ".join([str(message), *(str(arg) for arg in args)])
        self._log_fn(str(message))


class WorkflowMixin:
    def _check_cancelled(self):
        if getattr(self, "_cancel_event", None) is not None and self._cancel_event.is_set():
            raise PipelineCancelled("Run cancelled by user.")

    @staticmethod
    def _default_optcuts_frame_dir(input_path: str) -> str:
        base_dir = os.path.dirname(input_path) or os.getcwd()
        stem = os.path.splitext(os.path.basename(input_path))[0]
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{stem}_optcuts_frames_{ts}")

    def start_benchmark(self):
        if not self._validate_inputs():
            self.log("Benchmark blocked by invalid input. Review the highlighted fields.")
            return
        try:
            form = parse_benchmark_form(
                {
                    "folder": self.entry_file.get(),
                    "chain_a": self.entry_chain_a.get(),
                    "chain_b": self.entry_chain_b.get(),
                    "cutoff": self.entry_cutoff.get(),
                    "res": self.entry_res.get(),
                    "sigma": self.entry_sigma.get(),
                    "optcuts_bin": self.entry_optcuts_bin.get(),
                    "output_root": self.entry_output_dir.get(),
                    "run_mode": self.var_benchmark_run_mode.get(),
                    "max_workers": self.entry_max_workers.get(),
                }
            )
        except ConfigurationError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        params = form.to_params()
        try:
            self._preflight_optcuts(params["optcuts_bin"])
        except ConfigurationError as exc:
            messagebox.showerror("Invalid OptCuts Configuration", str(exc))
            return
        params["run_id"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if params["run_mode"] == "new":
            params["output_root"] = self._timestamped_output_root(params["output_root"], params["run_id"])
        self._remember_recent_file(params["folder"])
        self._remember_recent_output_dir(params["output_root"] if os.path.isdir(params["output_root"]) else os.path.dirname(params["output_root"]))
        self.last_run_params = dict(params)
        self._update_run_summary()
        self._begin_task("Starting benchmark pipeline...", progress_mode="determinate")
        threading.Thread(target=self.run_benchmark_pipeline, args=(params,), daemon=True).start()

    def run_benchmark_pipeline(self, params):
        try:
            self.set_stage_progress("Benchmark", 0, "Preparing jobs")
            self._check_cancelled()
            config = BenchmarkConfig(
                input_folder=params["folder"],
                output_root=params["output_root"],
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
                max_workers=params.get("max_workers"),
                show_tqdm=False,
                resume=bool(params.get("resume", True)),
            )
            if params.get("run_mode") == "overwrite":
                self._clear_benchmark_outputs(config)
            runner = BenchmarkRunner(config=config, log_fn=self.log, progress_fn=self._on_benchmark_progress, cancel_event=self._cancel_event)
            self.log("Benchmark OptCuts runs in headless mode (viewer disabled).")
            output = runner.run()
            self._check_cancelled()
            self.post_to_ui(lambda: self.progress.configure(value=100))
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
            self.post_to_ui(
                lambda: messagebox.showinfo(
                    "Benchmark Completed",
                    f"Results saved to:\n{params['output_root']}\n\n"
                    "Generated files:\n- benchmark_report.json\n- benchmark_summary.csv",
                ),
            )
            self.post_to_ui(self.finish_success, False)
        except PipelineCancelled as exc:
            self.post_to_ui(self.finish_cancelled, str(exc))
        except Exception as e:
            logger.exception("GUI benchmark failed")
            if getattr(self, "_cancel_event", None) is not None and self._cancel_event.is_set():
                self.post_to_ui(self.finish_cancelled, "Benchmark cancellation requested.")
            else:
                self.post_to_ui(self.show_error, f"Benchmark failed: {e}")

    def _on_benchmark_progress(self, completed: int, total: int, message: str):
        self.post_to_ui(self._set_benchmark_progress_ui, completed, total, message)

    def _set_benchmark_progress_ui(self, completed: int, total: int, message: str):
        total_safe = max(1, int(total))
        completed_safe = max(0, min(int(completed), total_safe))
        percent = int((completed_safe / total_safe) * 100.0)
        self.progress.configure(mode="determinate", maximum=100, value=percent)
        self.stage_status_var.set(f"Benchmark: {percent}% - {message}")
        self.log(f"[Benchmark][Progress] {completed_safe}/{total_safe} ({percent}%) - {message}")

    def generate_prolif_interactions(self, pdb_path, chain_a, chain_b):
        self.log("Checking ProLIF requirements...")
        output_json = generate_prolif_interactions(pdb_path, chain_a, chain_b, log=_GuiLogAdapter(self.log))
        if output_json:
            self.post_to_ui(self._set_prolif_entry, output_json)
            return output_json
        self.log("ProLIF interaction generation skipped/failed. Falling back to geometric heuristics.")
        return None

    def _set_prolif_entry(self, output_json):
        if hasattr(self, "var_prolif_path"):
            self.var_prolif_path.set(output_json)
        else:
            self.entry_prolif.delete(0, "end")
            self.entry_prolif.insert(0, output_json)

    def run_pipeline(self, params):
        try:
            self.set_stage_progress("Load", 0, "Preparing inputs")
            self._check_cancelled()
            self._preflight_optcuts(params.get("optcuts_bin", DEFAULT_RUN_CONFIG.optcuts.optcuts_bin))
            stage_timings = {}
            run_start = time.perf_counter()
            provided_prolif = params.get('prolif')
            prolif_file = provided_prolif
            if not prolif_file or not os.path.exists(prolif_file):
                if not provided_prolif:
                    self.log("No ProLIF JSON selected. Trying automatic ProLIF generation; geometric fallback will be used if it is unavailable.")
                self.set_stage_progress("Load", 8, "Resolving interactions")
                generated_json = self.generate_prolif_interactions(params['path'], params['chain_a'], params['chain_b'])
                prolif_file = generated_json if generated_json else None
                self._check_cancelled()
            prolif_source = "provided" if provided_prolif and prolif_file == provided_prolif else "generated_or_existing" if prolif_file else "geometric_fallback"

            self.log("Loading PDB structure...")
            self.set_stage_progress("Load", 12, "Reading structure")
            t0 = time.perf_counter()
            loader = PDBLoader(params['path'])
            coords_A, atoms_A = loader.get_chain_data(params['chain_a'])
            coords_B, atoms_B = loader.get_chain_data(params['chain_b'])
            stage_timings["load_structure_sec"] = time.perf_counter() - t0
            self.set_stage_progress("Load", 20, "Structure loaded")
            self._check_cancelled()

            self.log("Generating molecular surface...")
            self.set_stage_progress("Surface", 25, "Generating molecular surface")
            t0 = time.perf_counter()
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
            stage_timings["surface_sec"] = time.perf_counter() - t0
            self.set_stage_progress("Surface", 40, "Surface ready")
            self._check_cancelled()

            self.log("Extracting interface patches...")
            self.set_stage_progress("Patch", 44, "Extracting interface patches")
            t0 = time.perf_counter()
            param = Parameterizer(config=DEFAULT_RUN_CONFIG.parameterization)
            optimizer = OptCutsUVOptimizer(optcuts_config, cancel_event=self._cancel_event)
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
            stage_timings["patch_extraction_sec"] = time.perf_counter() - t0
            self.set_stage_progress("Patch", 50, f"{len(patches)} patch(es) found")
            self._check_cancelled()

            self.log(f"Flattening {len(patches)} patches...")
            self.set_stage_progress("Patch", 54, "Flattening patches")
            t0 = time.perf_counter()
            parameterized_patches = self._parameterize_patches(patches, param)
            if not parameterized_patches:
                raise ValueError("LSCM Parameterization failed for all patches.")
            stage_timings["parameterization_sec"] = time.perf_counter() - t0
            self._check_cancelled()

            t0 = time.perf_counter()
            valid_patches, valid_count, invalid_count = self._split_interfaces_by_point_count(
                parameterized_patches,
                viz,
                params['min_points'],
            )
            self.log(f"Interface count summary (min points = {params['min_points']}): valid={valid_count}, invalid={invalid_count}")
            if not valid_patches:
                raise ValueError(f"All interfaces are invalid (point count < {params['min_points']}).")
            self.set_stage_progress("Patch", 60, f"{valid_count} valid interface(s)")
            self._check_cancelled()

            self.log(f"Running OptCuts only on valid interfaces ({len(valid_patches)} patches)...")
            self.set_stage_progress("OptCuts", 64, "Optimizing UVs")
            t0 = time.perf_counter()
            optimized_valid_patches = self._optimize_patches(valid_patches, optimizer)
            stage_timings["optcuts_sec"] = time.perf_counter() - t0
            optimizer_report = getattr(optimizer, "get_last_report", lambda: {})()
            self.set_stage_progress("OptCuts", 85, "Optimization complete")
            self._check_cancelled()

            selected_patches = optimized_valid_patches
            self.log("Display mode: valid interfaces only.")

            self.log("Rendering visualization...")
            self.set_stage_progress("Render", 92, "Preparing figure")
            stage_timings["total_pipeline_sec"] = time.perf_counter() - run_start
            resolved_optcuts = self._optcuts_artifact_block(optcuts_config)
            manifest = {
                "schema_version": "1.1",
                "run_id": params.get("run_id"),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "input_file": os.path.abspath(params["path"]),
                "input_sha256": self._sha256_file(params["path"]),
                "output_dir": os.path.abspath(params.get("output_dir") or os.path.dirname(params["path"]) or os.getcwd()),
                "chain_a": params["chain_a"],
                "chain_b": params["chain_b"],
                "prolif_file": os.path.abspath(prolif_file) if prolif_file else None,
                "prolif_sha256": self._sha256_file(prolif_file) if prolif_file else None,
                "prolif_source": prolif_source,
                "environment": self._environment_block(),
                "git_commit": self._git_commit(),
                "optcuts_resolved": resolved_optcuts,
                "surface": asdict(surface_config),
                "topology": asdict(topology_config),
                "parameterization": asdict(DEFAULT_RUN_CONFIG.parameterization),
                "optcuts": asdict(optcuts_config),
                "visualization": asdict(DEFAULT_RUN_CONFIG.visualization),
                "stage_timings": stage_timings,
                "optimizer_report": optimizer_report,
                "interface_counts": {
                    "raw_patches": len(patches),
                    "parameterized_patches": len(parameterized_patches),
                    "valid_patches": valid_count,
                    "invalid_patches": invalid_count,
                    "displayed_patches": len(selected_patches),
                },
            }
            self.set_stage_progress("Render", 100, "Rendering")
            self.post_to_ui(self.accept_pipeline_result, viz, selected_patches, manifest)
        except PipelineCancelled as exc:
            self.post_to_ui(self.finish_cancelled, str(exc))
        except Exception as e:
            logger.exception("GUI single-run pipeline failed")
            if getattr(self, "_cancel_event", None) is not None and self._cancel_event.is_set():
                self.post_to_ui(self.finish_cancelled, "Run cancelled by user.")
            else:
                self.post_to_ui(self.show_error, str(e))

    def accept_pipeline_result(self, viz, patches, manifest):
        self.cached_viz = viz
        self.cached_patches = patches
        self.last_run_manifest = manifest
        self.finish_success(True)

    def _parameterize_patches(self, patches, parameterizer):
        valid_patches = []
        for p in patches:
            self._check_cancelled()
            uv = parameterizer.flatten_patch(p)
            if uv is not None:
                p.metadata['uv'] = uv
                valid_patches.append(p)
        return valid_patches

    def _optimize_patches(self, patches, optimizer):
        if not patches:
            return []
        self._check_cancelled()
        result = optimizer.optimize_patches(patches)
        self._check_cancelled()
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
            self._check_cancelled()
            point_count = viz.count_patch_points(p)
            p.metadata['point_count'] = point_count
            if point_count >= min_points:
                valid.append(p)
            else:
                invalid += 1
        return valid, len(valid), invalid

    def _preflight_optcuts(self, optcuts_bin: str) -> None:
        config = replace(DEFAULT_RUN_CONFIG.optcuts, optcuts_bin=optcuts_bin)
        resolved = resolve_optcuts_binary(config)
        if not resolved:
            raise ConfigurationError(
                f"OptCuts binary not found: {optcuts_bin}. Set an absolute path or {config.optcuts_env_var}."
            )

    def _timestamped_output_root(self, output_root: str, run_id: str) -> str:
        base = output_root.rstrip(os.sep)
        return f"{base}_{run_id}"

    def _clear_benchmark_outputs(self, config: BenchmarkConfig) -> None:
        os.makedirs(config.output_root, exist_ok=True)
        for filename in (config.checkpoint_filename, config.report_filename, config.summary_filename):
            path = os.path.join(config.output_root, filename)
            if os.path.exists(path):
                os.remove(path)
                self.log(f"Removed previous benchmark output: {path}")

    def _sha256_file(self, path: str | None) -> str | None:
        if not path or not os.path.exists(path):
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _git_commit(self) -> str | None:
        repo_root = Path(__file__).resolve().parents[3]
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        return proc.stdout.strip() or None

    def _environment_block(self) -> dict[str, object]:
        names = ["numpy", "scipy", "biopython", "matplotlib", "trimesh", "scikit-image", "topoppi"]
        versions = {}
        for name in names:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": versions,
        }

    def _optcuts_artifact_block(self, config) -> dict[str, object]:
        resolved = resolve_optcuts_binary(config)
        return {
            "requested": os.environ.get(config.optcuts_env_var, config.optcuts_bin),
            "resolved": resolved,
            "sha256": self._sha256_file(resolved) if resolved else None,
            "env_var": config.optcuts_env_var,
        }
