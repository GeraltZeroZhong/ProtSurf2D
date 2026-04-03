from __future__ import annotations

import json
import os
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # optional dependency
    tqdm = None

try:
    import psutil
except Exception:  # optional
    psutil = None

from src.benchmarking.config import BenchmarkConfig
from src.benchmarking.metrics_utils import (
    atlas_trainability_metrics,
    avg_energy,
    avg_seam_length,
    improvement_rate,
    quality_block,
    rasterize_feature_maps,
)
from src.benchmarking.reporting import aggregate_results, write_csv
from src.io.io_loader import PDBLoader
from src.mesh.parameterization import Parameterizer
from src.mesh.surface import SurfaceGenerator
from src.mesh.topology import TopologyManager
from src.optimization.uv_optimizer import OptCutsUVOptimizer, UVOptimizerConfig


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        log_fn: Optional[Callable[[str], None]] = None,
        progress_fn: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.config = config
        self.log = log_fn or (lambda msg: None)
        self.progress = progress_fn or (lambda *_: None)
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._checkpoint_path = os.path.join(self.config.output_root, "benchmark_checkpoint.json")

    def run(self) -> Dict[str, object]:
        os.makedirs(self.config.output_root, exist_ok=True)
        pdb_files = sorted([f for f in os.listdir(self.config.input_folder) if f.lower().endswith(".pdb")])
        if not pdb_files:
            raise ValueError("No .pdb files found for benchmark.")

        prepared_jobs, preprocessing_log = self._prepare_benchmark_jobs(pdb_files)
        if not prepared_jobs:
            raise ValueError("No valid .pdb files after preprocessing: each file must contain at least two protein chains.")
        completed_results, prepared_jobs = self._load_resume_state(prepared_jobs)

        worker_count = self._resolve_worker_count(len(prepared_jobs))
        self.log(f"[Benchmark] Running with {worker_count} worker thread(s).")
        self._safe_progress(len(completed_results), len(completed_results) + len(prepared_jobs), "Benchmark started")
        new_results = self._run_files_concurrently(
            prepared_jobs,
            worker_count,
            completed_results=completed_results,
            total_jobs=len(completed_results) + len(prepared_jobs),
        )
        all_results = completed_results + new_results

        output = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": asdict(self.config),
            "preprocessing": preprocessing_log,
            "files": all_results,
            "summary": aggregate_results(all_results),
        }

        with open(os.path.join(self.config.output_root, "benchmark_report.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        self._save_checkpoint(all_results)
        write_csv(all_results, self.config.output_root)
        return output

    def _run_files_concurrently(
        self,
        jobs: List[Dict[str, object]],
        worker_count: int,
        completed_results: Optional[List[Dict[str, object]]] = None,
        total_jobs: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        completed_results = completed_results or []
        total_jobs = int(total_jobs if total_jobs is not None else len(jobs))
        all_results: List[Optional[Dict[str, object]]] = [None] * len(jobs)
        completed = len(completed_results)
        if tqdm is not None:
            progress_ctx = tqdm(
                total=total_jobs,
                desc="Benchmark",
                unit="file",
                disable=not bool(self.config.show_tqdm),
                initial=completed,
            )
        else:
            progress_ctx = nullcontext(None)
        with ThreadPoolExecutor(max_workers=worker_count) as executor, progress_ctx as pbar:
            future_to_job = {}
            for idx, job in enumerate(jobs, start=1):
                pdb_name = str(job["pdb"])
                pdb_path = os.path.join(self.config.input_folder, pdb_name)
                chain_a = str(job["chain_a"])
                chain_b = str(job["chain_b"])
                self.log(f"[Benchmark] ({idx}/{len(jobs)}) Queued {pdb_name} with chains {chain_a}/{chain_b}")
                future = executor.submit(self._run_single, pdb_path, chain_a, chain_b)
                future_to_job[future] = (idx - 1, pdb_name)

            for future in as_completed(future_to_job):
                out_idx, pdb_name = future_to_job[future]
                completed += 1
                self._safe_progress(completed, total_jobs, f"Finished {pdb_name}")
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(pdb_name)
                try:
                    all_results[out_idx] = future.result()
                    self.log(f"[Benchmark] Finished {pdb_name}")
                except Exception as exc:
                    self.log(f"[Benchmark] Failed {pdb_name}: {exc}")
                    all_results[out_idx] = {
                        "pdb": pdb_name,
                        "patch_count": 0,
                        "error": f"Benchmark file execution failed: {exc}",
                    }
                checkpoint_results = completed_results + [r for r in all_results if r is not None]
                self._save_checkpoint(checkpoint_results)

        return [r for r in all_results if r is not None]

    def _resolve_worker_count(self, file_count: int) -> int:
        configured_workers = self.config.max_workers
        if configured_workers is None:
            configured_workers = os.cpu_count() or 1
        return max(1, min(int(configured_workers), int(file_count)))

    def _run_single(self, pdb_path: str, chain_a: str, chain_b: str) -> Dict[str, object]:
        self._log_thread(f"Start processing {os.path.basename(pdb_path)} ({chain_a}/{chain_b})")
        mem_peak = self._memory_rss_mb()

        stage = {}
        t0, c0 = time.perf_counter(), time.process_time()
        loader = PDBLoader(pdb_path)
        coords_a, _ = loader.get_chain_data(chain_a)
        coords_b, _ = loader.get_chain_data(chain_b)
        surf_gen = SurfaceGenerator(coords_a)
        mesh_a = surf_gen.generate_mesh(grid_resolution=self.config.res, sigma=self.config.sigma)
        if mesh_a is None or len(mesh_a.vertices) == 0:
            raise ValueError(f"Surface generation failed for {os.path.basename(pdb_path)}")
        topo_mgr = TopologyManager(mesh_a, coords_b)
        patches = topo_mgr.get_interface_patches(distance_cutoff=self.config.cutoff)
        stage["mesh_and_patch"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())
        if not patches:
            return {
                "pdb": os.path.basename(pdb_path),
                "chain_selection": {"chain_a": chain_a, "chain_b": chain_b},
                "patch_count": 0,
                "error": "No interface patches found",
            }

        patch_results = {}
        self._log_thread("Patch parameterization runs sequentially per file; file-level parallelism is handled by worker threads.")
        for method in ("lscm", "harmonic", "spherical", "cylindrical"):
            p, wt, ct, diag = self._parameterize_patches(patches, method=method)
            patch_results[method] = {"patches": p, "wall": wt, "cpu": ct, "diag": diag}
            stage[f"{method}_parameterization"] = self._from_timing_list(wt, ct)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        t0, c0 = time.perf_counter(), time.process_time()
        lscm_optcuts, optcuts_diag = self._run_optcuts(patch_results["lscm"]["patches"])
        stage["optcuts_optimization"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        t0, c0 = time.perf_counter(), time.process_time()
        atlas_map, patch_coverages = rasterize_feature_maps(lscm_optcuts, size=self.config.raster_size, return_patch_coverage=True)
        atlas_trainability = atlas_trainability_metrics(atlas_map, patch_coverages)
        stage["feature_rasterization"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        energy_raw = avg_energy(patch_results["lscm"]["patches"])
        energy_opt = avg_energy(lscm_optcuts)
        seam_raw = avg_seam_length(patch_results["lscm"]["patches"])
        seam_opt = avg_seam_length(lscm_optcuts)

        lscm_raw_quality = quality_block(patch_results["lscm"]["patches"], patch_gap=self.config.patch_gap)
        lscm_optcuts_quality = quality_block(lscm_optcuts, patch_gap=self.config.patch_gap)

        result = {
            "pdb": os.path.basename(pdb_path),
            "chain_selection": {"chain_a": chain_a, "chain_b": chain_b},
            "patch_count": len(patches),
            "mesh_stats": {"vertex_count": int(len(mesh_a.vertices)), "face_count": int(len(mesh_a.faces))},
            "lscm_raw": lscm_raw_quality,
            "lscm_optcuts": lscm_optcuts_quality,
            "harmonic_raw": quality_block(patch_results["harmonic"]["patches"], patch_gap=self.config.patch_gap),
            "spherical_raw": quality_block(patch_results["spherical"]["patches"], patch_gap=self.config.patch_gap),
            "cylindrical_raw": quality_block(patch_results["cylindrical"]["patches"], patch_gap=self.config.patch_gap),
            "topology_repair": {
                "lscm": patch_results["lscm"]["diag"],
                "harmonic": patch_results["harmonic"]["diag"],
                "spherical": patch_results["spherical"]["diag"],
                "cylindrical": patch_results["cylindrical"]["diag"],
            },
            "timing": {
                "stages": stage,
                "parameterization": self._parameterization_timing_block(patch_results),
                "gpu": {"available": False, "note": "No GPU timing backend integrated in current benchmark environment."},
                "scalability": {
                    "wall_sec_per_patch": float(stage["mesh_and_patch"]["wall_sec"] / max(1, len(patches))),
                    "wall_sec_per_vertex": float(stage["mesh_and_patch"]["wall_sec"] / max(1, len(mesh_a.vertices))),
                },
            },
            "memory": {"peak_rss_mb": mem_peak},
            "topology_optimization": {
                "energy": {
                    "lscm_raw": energy_raw,
                    "lscm_optcuts": energy_opt,
                    "improvement_rate": improvement_rate(energy_raw, energy_opt),
                },
                "seam_length": {
                    "lscm_raw": seam_raw,
                    "lscm_optcuts": seam_opt,
                    "improvement_rate": improvement_rate(seam_raw, seam_opt),
                },
            },
            "optcuts_ablation": {
                "enabled": bool(optcuts_diag.get("enabled", False)),
                "baseline": "lscm_raw",
                "treatment": "lscm_optcuts",
                "distortion_mean_before": float(lscm_raw_quality["distortion"]["mean"]),
                "distortion_mean_after": float(lscm_optcuts_quality["distortion"]["mean"]),
                "flip_rate_before": float(lscm_raw_quality["flip_rate"]),
                "flip_rate_after": float(lscm_optcuts_quality["flip_rate"]),
                "energy_before": energy_raw,
                "energy_after": energy_opt,
                "seam_before": seam_raw,
                "seam_after": seam_opt,
                "energy_improvement_rate": improvement_rate(energy_raw, energy_opt),
                "seam_improvement_rate": improvement_rate(seam_raw, seam_opt),
                "diag": optcuts_diag,
            },
            "atlas_trainability": atlas_trainability,
        }
        self._log_thread(f"Finished processing {os.path.basename(pdb_path)}")
        return result

    def _log_thread(self, message: str) -> None:
        tid = threading.get_ident()
        tname = threading.current_thread().name
        self.log(f"[Benchmark][Thread {tname}:{tid}] {message}")

    def _safe_progress(self, completed: int, total: int, message: str) -> None:
        try:
            self.progress(int(completed), int(total), str(message))
        except Exception:
            pass

    def _prepare_benchmark_jobs(self, pdb_files: List[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        min_chain_residues = 11
        accepted_jobs: List[Dict[str, object]] = []
        skipped_files: List[Dict[str, str]] = []

        for pdb_name in pdb_files:
            pdb_path = os.path.join(self.config.input_folder, pdb_name)
            try:
                loader = PDBLoader(pdb_path)
                chain_ids = loader.get_protein_chain_ids()
            except Exception as exc:
                skipped_files.append({"pdb": pdb_name, "reason": f"Failed to parse PDB: {exc}"})
                self.log(f"[Benchmark][Preprocess] Skipped {pdb_name}: failed to parse ({exc})")
                continue

            if len(chain_ids) < 2:
                skipped_files.append({"pdb": pdb_name, "reason": f"Need >=2 protein chains, found {len(chain_ids)}"})
                self.log(f"[Benchmark][Preprocess] Skipped {pdb_name}: found {len(chain_ids)} protein chain(s)")
                continue

            if "A" in chain_ids and "B" in chain_ids:
                chain_a, chain_b = "A", "B"
                selection_mode = "prefer_AB"
            else:
                chain_a, chain_b = chain_ids[0], chain_ids[1]
                selection_mode = "first_two_chains"

            try:
                coords_a, _ = loader.get_chain_data(chain_a)
                coords_b, _ = loader.get_chain_data(chain_b)
                if len(coords_a) == 0 or len(coords_b) == 0:
                    skipped_files.append(
                        {
                            "pdb": pdb_name,
                            "reason": f"Selected chains contain no standard atoms: {chain_a}({len(coords_a)}), {chain_b}({len(coords_b)})",
                        }
                    )
                    self.log(
                        f"[Benchmark][Preprocess] Skipped {pdb_name}: selected chains empty "
                        f"{chain_a}({len(coords_a)}), {chain_b}({len(coords_b)})"
                    )
                    continue

                residue_count_a = loader.get_chain_residue_count(chain_a)
                residue_count_b = loader.get_chain_residue_count(chain_b)
                if residue_count_a < min_chain_residues or residue_count_b < min_chain_residues:
                    skipped_files.append(
                        {
                            "pdb": pdb_name,
                            "reason": (
                                "Selected chain length too short: "
                                f"{chain_a}({residue_count_a}), {chain_b}({residue_count_b}); "
                                "each chain must be >10 amino acids"
                            ),
                        }
                    )
                    self.log(
                        f"[Benchmark][Preprocess] Skipped {pdb_name}: selected chains too short "
                        f"{chain_a}({residue_count_a}), {chain_b}({residue_count_b}); each must be >10 aa"
                    )
                    continue
            except Exception as exc:
                skipped_files.append({"pdb": pdb_name, "reason": f"Selected chain extraction failed: {exc}"})
                self.log(f"[Benchmark][Preprocess] Skipped {pdb_name}: selected chain extraction failed ({exc})")
                continue

            accepted_jobs.append(
                {
                    "pdb": pdb_name,
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "selection_mode": selection_mode,
                    "available_chains": chain_ids,
                }
            )
            self.log(
                f"[Benchmark][Preprocess] Accepted {pdb_name}: using chains {chain_a}/{chain_b} "
                f"({selection_mode})"
            )

        summary = {
            "total_files": len(pdb_files),
            "accepted_files": len(accepted_jobs),
            "skipped_files": len(skipped_files),
            "accepted": accepted_jobs,
            "skipped": skipped_files,
            "rules": [
                "File must contain at least two protein chains.",
                "Selected chains must each have >10 amino acids.",
                "If both A and B chains exist, use A/B.",
                "Otherwise use the first two protein chains discovered in structure order.",
            ],
        }
        return accepted_jobs, summary

    def _parameterize_patches(self, patches, method: str):
        parameterizer = Parameterizer()
        out, times, cpu_times = [], [], []
        diag = {
            "attempted": 0,
            "success": 0,
            "diskification_triggered": 0,
            "diskification_success": 0,
            "failure_reasons": {},
            "topology_before_boundary_loops": [],
            "topology_after_boundary_loops": [],
            "topology_gate_failed_count": 0,
            "face_retention_ratios": [],
            "vertex_retention_ratios": [],
            "area_retention_ratios": [],
        }
        for p in patches:
            diag["attempted"] += 1
            patch_copy = p.copy()
            if method == "lscm" and self._is_patch_too_small_for_lscm(patch_copy):
                diag["failure_reasons"]["too_small_for_lscm"] = int(diag["failure_reasons"].get("too_small_for_lscm", 0)) + 1
                continue
            t0 = time.perf_counter()
            c0 = time.process_time()
            uv, info = parameterizer.flatten_patch(patch_copy, method=method, return_info=True)
            dt = time.perf_counter() - t0
            cpu_dt = time.process_time() - c0
            if info.get("diskification_triggered"):
                diag["diskification_triggered"] += 1
            if info.get("diskification_success"):
                diag["diskification_success"] += 1
            if info.get("topology_before", {}).get("boundary_loops") is not None:
                diag["topology_before_boundary_loops"].append(int(info["topology_before"]["boundary_loops"]))
            if info.get("topology_after", {}).get("boundary_loops") is not None:
                diag["topology_after_boundary_loops"].append(int(info["topology_after"]["boundary_loops"]))
            if uv is None:
                reason = info.get("failure_reason", "unknown_failure")
                diag["failure_reasons"][reason] = int(diag["failure_reasons"].get(reason, 0)) + 1
                if reason == "topology_gate_failed":
                    diag["topology_gate_failed_count"] += 1
                continue
            self._collect_retention_stats(diag, info)
            diag["success"] += 1
            patch_copy.metadata["uv"] = uv
            out.append(patch_copy)
            times.append(dt)
            cpu_times.append(cpu_dt)

        self._finalize_diag(diag)
        return out, times, cpu_times, diag

    def _run_optcuts(self, patches):
        if not patches:
            return [], {"enabled": False, "status": "skipped_no_lscm_patches", "error_count": 0, "errors": []}
        optimizer = OptCutsUVOptimizer(
            UVOptimizerConfig(
                optcuts_bin=self.config.optcuts_bin,
                patch_gap=self.config.patch_gap,
                # OptCuts README:
                #   mode=100 => headless (no viewer window)
                #   mode=10  => offline mode with visualization
                optcuts_mode=100 if self.config.optcuts_headless else 10,
                # testID (initial homotopy parameter), keep default 1 for benchmark consistency.
                optcuts_prog_mode=1,
                optcuts_quick_mode=bool(self.config.optcuts_quick_mode),
            )
        )
        try:
            optimized = optimizer.optimize_patches([p.copy() for p in patches])
            return optimized, {"enabled": True, "status": "ok", "error_count": 0, "errors": []}
        except Exception as exc:
            msg = f"OptCuts error: {exc}"
            self.log(f"[Benchmark] {msg}")
            return [p.copy() for p in patches], {"enabled": False, "status": "failed", "error_count": 1, "errors": [msg]}

    def _load_resume_state(self, jobs: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not bool(self.config.resume):
            return [], jobs
        if not os.path.exists(self._checkpoint_path):
            return [], jobs
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            file_results = checkpoint.get("files", [])
            if not isinstance(file_results, list):
                return [], jobs
            completed_names = {str(item.get("pdb")) for item in file_results if isinstance(item, dict) and item.get("pdb")}
            remaining_jobs = [j for j in jobs if str(j["pdb"]) not in completed_names]
            if completed_names:
                self.log(f"[Benchmark] Resume enabled: {len(completed_names)} file(s) already completed, {len(remaining_jobs)} pending.")
            return [r for r in file_results if isinstance(r, dict)], remaining_jobs
        except Exception as exc:
            self.log(f"[Benchmark] Failed to load checkpoint, running full benchmark: {exc}")
            return [], jobs

    def _save_checkpoint(self, results: List[Dict[str, object]]) -> None:
        payload = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "files": results,
        }
        try:
            with open(self._checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            self.log(f"[Benchmark] Failed to save checkpoint: {exc}")

    def _is_patch_too_small_for_lscm(self, patch) -> bool:
        return bool(
            len(getattr(patch, "vertices", [])) < int(self.config.min_lscm_patch_vertices)
            or len(getattr(patch, "faces", [])) < int(self.config.min_lscm_patch_faces)
        )

    def _memory_rss_mb(self) -> float:
        if self._proc is None:
            return 0.0
        try:
            return float(self._proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    @staticmethod
    def _stage_stats(start_wall: float, start_cpu: float) -> Dict[str, float]:
        return {"wall_sec": float(time.perf_counter() - start_wall), "cpu_sec": float(time.process_time() - start_cpu)}

    @staticmethod
    def _from_timing_list(times: List[float], cpu_times: List[float]) -> Dict[str, float]:
        if not times:
            return {"wall_sec": float("inf"), "cpu_sec": float("inf")}
        return {"wall_sec": float(np.sum(times)), "cpu_sec": float(np.sum(cpu_times)) if cpu_times else float("inf")}

    @staticmethod
    def _collect_retention_stats(diag: Dict[str, object], info: Dict[str, object]) -> None:
        f_before = float(info.get("face_count_before_topology_gate", 0))
        f_after = float(info.get("face_count_after_topology_gate", 0))
        if f_before > 0:
            diag["face_retention_ratios"].append(f_after / f_before)

        v_before = float(info.get("vertex_count_before_topology_gate", 0))
        v_after = float(info.get("vertex_count_after_topology_gate", 0))
        if v_before > 0:
            diag["vertex_retention_ratios"].append(v_after / v_before)

        a_before = float(info.get("area_before_topology_gate", 0.0))
        a_after = float(info.get("area_after_topology_gate", 0.0))
        if a_before > 1e-12:
            diag["area_retention_ratios"].append(a_after / a_before)

    @staticmethod
    def _finalize_diag(diag: Dict[str, object]) -> None:
        before_vals = diag["topology_before_boundary_loops"]
        after_vals = diag["topology_after_boundary_loops"]
        diag["topology_before_boundary_loops_mean"] = float(np.mean(before_vals)) if before_vals else float("nan")
        diag["topology_after_boundary_loops_mean"] = float(np.mean(after_vals)) if after_vals else float("nan")
        diag["face_retention_ratio_mean"] = float(np.mean(diag["face_retention_ratios"])) if diag["face_retention_ratios"] else float("nan")
        diag["vertex_retention_ratio_mean"] = float(np.mean(diag["vertex_retention_ratios"])) if diag["vertex_retention_ratios"] else float("nan")
        diag["area_retention_ratio_mean"] = float(np.mean(diag["area_retention_ratios"])) if diag["area_retention_ratios"] else float("nan")
        diag["topology_gate_fail_rate"] = float(diag["topology_gate_failed_count"] / max(1, diag["attempted"]))

    @staticmethod
    def _parameterization_timing_block(patch_results: Dict[str, Dict[str, object]]) -> Dict[str, float]:
        out = {}
        for method in ("lscm", "harmonic", "spherical", "cylindrical"):
            wall = patch_results[method]["wall"]
            cpu = patch_results[method]["cpu"]
            out[f"{method}_mean_wall_sec"] = float(np.mean(wall)) if wall else float("inf")
            out[f"{method}_total_wall_sec"] = float(np.sum(wall)) if wall else float("inf")
            out[f"{method}_total_cpu_sec"] = float(np.sum(cpu)) if cpu else float("inf")
        return out
