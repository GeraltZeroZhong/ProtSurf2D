from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

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
from src.io_loader import PDBLoader
from src.parameterization import Parameterizer
from src.surface import SurfaceGenerator
from src.topology import TopologyManager
from src.uv_optimizer import OptCutsUVOptimizer, UVOptimizerConfig


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, log_fn: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log = log_fn or (lambda msg: None)
        self._proc = psutil.Process(os.getpid()) if psutil else None

    def run(self) -> Dict[str, object]:
        os.makedirs(self.config.output_root, exist_ok=True)
        pdb_files = sorted([f for f in os.listdir(self.config.input_folder) if f.lower().endswith(".pdb")])
        if not pdb_files:
            raise ValueError("No .pdb files found for benchmark.")

        all_results = []
        for idx, pdb_name in enumerate(pdb_files, start=1):
            pdb_path = os.path.join(self.config.input_folder, pdb_name)
            self.log(f"[Benchmark] ({idx}/{len(pdb_files)}) {pdb_name}")
            all_results.append(self._run_single(pdb_path))

        output = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": asdict(self.config),
            "files": all_results,
            "summary": aggregate_results(all_results),
            "sensitivity": self._run_sensitivity(pdb_files),
        }

        with open(os.path.join(self.config.output_root, "benchmark_report.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        write_csv(all_results, self.config.output_root)
        return output

    def _run_single(self, pdb_path: str) -> Dict[str, object]:
        mem_peak = self._memory_rss_mb()

        stage = {}
        t0, c0 = time.perf_counter(), time.process_time()
        loader = PDBLoader(pdb_path)
        coords_a, _ = loader.get_chain_data(self.config.chain_a)
        coords_b, _ = loader.get_chain_data(self.config.chain_b)
        surf_gen = SurfaceGenerator(coords_a)
        mesh_a = surf_gen.generate_mesh(grid_resolution=self.config.res, sigma=self.config.sigma)
        if mesh_a is None or len(mesh_a.vertices) == 0:
            raise ValueError(f"Surface generation failed for {os.path.basename(pdb_path)}")
        topo_mgr = TopologyManager(mesh_a, coords_b)
        patches = topo_mgr.get_interface_patches(distance_cutoff=self.config.cutoff)
        stage["mesh_and_patch"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())
        if not patches:
            return {"pdb": os.path.basename(pdb_path), "patch_count": 0, "error": "No interface patches found"}

        patch_results = {}
        for method in ("lscm", "harmonic", "spherical", "cylindrical"):
            p, wt, ct, diag = self._parameterize_patches(patches, method=method)
            patch_results[method] = {"patches": p, "wall": wt, "cpu": ct, "diag": diag}
            stage[f"{method}_parameterization"] = self._from_timing_list(wt, ct)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        t0, c0 = time.perf_counter(), time.process_time()
        lscm_optcuts = self._run_optcuts(patch_results["lscm"]["patches"])
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
                "enabled": True,
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
            },
            "atlas_trainability": atlas_trainability,
        }
        return result

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
            return []
        optimizer = OptCutsUVOptimizer(
            UVOptimizerConfig(
                optcuts_bin=self.config.optcuts_bin,
                patch_gap=self.config.patch_gap,
                optcuts_prog_mode=2 if self.config.optcuts_headless else 1,
            )
        )
        return optimizer.optimize_patches([p.copy() for p in patches])

    def _run_sensitivity(self, pdb_files: List[str]) -> Dict[str, object]:
        sweep = {"cutoff": self.config.cutoff_sweep, "sigma": self.config.sigma_sweep, "res": self.config.res_sweep}
        if all(v is None for v in sweep.values()):
            return {"enabled": False}
        results = {"enabled": True, "items": []}
        for name, values in sweep.items():
            if not values:
                continue
            for val in values:
                cfg = BenchmarkConfig(**{**asdict(self.config), name: float(val), "cutoff_sweep": None, "sigma_sweep": None, "res_sweep": None})
                runner = BenchmarkRunner(cfg, log_fn=lambda *_: None)
                subset = pdb_files[: min(3, len(pdb_files))]
                rows = []
                for fn in subset:
                    try:
                        rows.append(runner._run_single(os.path.join(cfg.input_folder, fn)))
                    except Exception:
                        continue
                valid = [r for r in rows if "error" not in r]
                results["items"].append(
                    {
                        "param": name,
                        "value": float(val),
                        "valid_count": len(valid),
                        "distortion_lscm_optcuts_mean": float(np.mean([r["lscm_optcuts"]["distortion"]["mean"] for r in valid])) if valid else float("inf"),
                        "flip_lscm_optcuts_mean": float(np.mean([r["lscm_optcuts"]["flip_rate"] for r in valid])) if valid else float("inf"),
                        "optcuts_energy_gain_mean": float(np.mean([r["topology_optimization"]["energy"]["improvement_rate"] for r in valid])) if valid else float("nan"),
                        "atlas_nonzero_ratio_mean": float(np.mean([r["atlas_trainability"]["nonzero_ratio"] for r in valid])) if valid else float("nan"),
                    }
                )
        return results

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
