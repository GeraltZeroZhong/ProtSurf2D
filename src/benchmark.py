from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil
except Exception:  # optional
    psutil = None

from src.io_loader import PDBLoader
from src.metrics import UVAtlasMetrics
from src.parameterization import Parameterizer
from src.surface import SurfaceGenerator
from src.topology import TopologyManager
from src.uv_optimizer import OptCutsUVOptimizer, UVOptimizerConfig


@dataclass
class BenchmarkConfig:
    input_folder: str
    output_root: str
    chain_a: str
    chain_b: str
    cutoff: float
    res: float
    sigma: float
    patch_gap: float = 0.08
    optcuts_bin: str = "OptCuts_bin"
    raster_size: int = 256
    cutoff_sweep: Optional[List[float]] = None
    sigma_sweep: Optional[List[float]] = None
    res_sweep: Optional[List[float]] = None


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

        summary = self._aggregate(all_results)
        output = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": asdict(self.config),
            "files": all_results,
            "summary": summary,
        }
        output["sensitivity"] = self._run_sensitivity(pdb_files)

        with open(os.path.join(self.config.output_root, "benchmark_report.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        self._write_csv(all_results)
        return output

    def _run_single(self, pdb_path: str) -> Dict[str, object]:
        mem_peak = self._memory_rss_mb()

        stage = {}
        t0, c0 = time.perf_counter(), time.process_time()
        loader = PDBLoader(pdb_path)
        coords_A, _ = loader.get_chain_data(self.config.chain_a)
        coords_B, _ = loader.get_chain_data(self.config.chain_b)
        surf_gen = SurfaceGenerator(coords_A)
        mesh_A = surf_gen.generate_mesh(grid_resolution=self.config.res, sigma=self.config.sigma)
        if mesh_A is None or len(mesh_A.vertices) == 0:
            raise ValueError(f"Surface generation failed for {os.path.basename(pdb_path)}")
        topo_mgr = TopologyManager(mesh_A, coords_B)
        patches = topo_mgr.get_interface_patches(distance_cutoff=self.config.cutoff)
        stage["mesh_and_patch"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())
        if not patches:
            return {"pdb": os.path.basename(pdb_path), "patch_count": 0, "error": "No interface patches found"}

        lscm_patches, lscm_times, lscm_cpu_times, lscm_diag = self._parameterize_patches(patches, method="lscm")
        stage["lscm_parameterization"] = self._from_timing_list(lscm_times, lscm_cpu_times)
        harmonic_patches, harmonic_times, harmonic_cpu_times, harmonic_diag = self._parameterize_patches(patches, method="harmonic")
        stage["harmonic_parameterization"] = self._from_timing_list(harmonic_times, harmonic_cpu_times)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        t0, c0 = time.perf_counter(), time.process_time()
        lscm_optcuts = self._run_optcuts(lscm_patches)
        stage["optcuts_optimization"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        t0, c0 = time.perf_counter(), time.process_time()
        atlas_map, patch_coverages = self._rasterize_feature_maps(lscm_optcuts, size=self.config.raster_size, return_patch_coverage=True)
        atlas_trainability = self._atlas_trainability_metrics(atlas_map, patch_coverages)
        stage["feature_rasterization"] = self._stage_stats(t0, c0)
        mem_peak = max(mem_peak, self._memory_rss_mb())

        energy_raw = self._avg_energy(lscm_patches)
        energy_opt = self._avg_energy(lscm_optcuts)
        seam_raw = self._avg_seam_length(lscm_patches)
        seam_opt = self._avg_seam_length(lscm_optcuts)
        result = {
            "pdb": os.path.basename(pdb_path),
            "patch_count": len(patches),
            "mesh_stats": {
                "vertex_count": int(len(mesh_A.vertices)),
                "face_count": int(len(mesh_A.faces)),
            },
            "lscm_raw": self._quality_block(lscm_patches, patch_gap=self.config.patch_gap),
            "lscm_optcuts": self._quality_block(lscm_optcuts, patch_gap=self.config.patch_gap),
            "harmonic_raw": self._quality_block(harmonic_patches, patch_gap=self.config.patch_gap),
            "topology_repair": {
                "lscm": lscm_diag,
                "harmonic": harmonic_diag,
            },
            "timing": {
                "stages": stage,
                "parameterization": {
                    "lscm_mean_wall_sec": float(np.mean(lscm_times)) if lscm_times else float("inf"),
                    "harmonic_mean_wall_sec": float(np.mean(harmonic_times)) if harmonic_times else float("inf"),
                    "lscm_total_wall_sec": float(np.sum(lscm_times)) if lscm_times else float("inf"),
                    "harmonic_total_wall_sec": float(np.sum(harmonic_times)) if harmonic_times else float("inf"),
                    "lscm_total_cpu_sec": float(np.sum(lscm_cpu_times)) if lscm_cpu_times else float("inf"),
                    "harmonic_total_cpu_sec": float(np.sum(harmonic_cpu_times)) if harmonic_cpu_times else float("inf"),
                },
                "gpu": {
                    "available": False,
                    "note": "No GPU timing backend integrated in current benchmark environment.",
                },
                "scalability": {
                    "wall_sec_per_patch": float(stage["mesh_and_patch"]["wall_sec"] / max(1, len(patches))),
                    "wall_sec_per_vertex": float(stage["mesh_and_patch"]["wall_sec"] / max(1, len(mesh_A.vertices))),
                },
            },
            "memory": {
                "peak_rss_mb": mem_peak,
            },
            "topology_optimization": {
                "energy": {
                    "lscm_raw": energy_raw,
                    "lscm_optcuts": energy_opt,
                    "improvement_rate": self._improvement_rate(energy_raw, energy_opt),
                },
                "seam_length": {
                    "lscm_raw": seam_raw,
                    "lscm_optcuts": seam_opt,
                    "improvement_rate": self._improvement_rate(seam_raw, seam_opt),
                },
            },
            "atlas_trainability": atlas_trainability,
        }
        return result

    def _parameterize_patches(self, patches, method: str):
        parameterizer = Parameterizer()
        out = []
        times = []
        cpu_times = []
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
            diag["success"] += 1
            patch_copy.metadata["uv"] = uv
            out.append(patch_copy)
            times.append(dt)
            cpu_times.append(cpu_dt)
        before_vals = diag["topology_before_boundary_loops"]
        after_vals = diag["topology_after_boundary_loops"]
        diag["topology_before_boundary_loops_mean"] = float(np.mean(before_vals)) if before_vals else float("nan")
        diag["topology_after_boundary_loops_mean"] = float(np.mean(after_vals)) if after_vals else float("nan")
        diag["face_retention_ratio_mean"] = float(np.mean(diag["face_retention_ratios"])) if diag["face_retention_ratios"] else float("nan")
        diag["vertex_retention_ratio_mean"] = float(np.mean(diag["vertex_retention_ratios"])) if diag["vertex_retention_ratios"] else float("nan")
        diag["area_retention_ratio_mean"] = float(np.mean(diag["area_retention_ratios"])) if diag["area_retention_ratios"] else float("nan")
        diag["topology_gate_fail_rate"] = float(diag["topology_gate_failed_count"] / max(1, diag["attempted"]))
        return out, times, cpu_times, diag

    def _run_optcuts(self, patches):
        if not patches:
            return []
        optimizer = OptCutsUVOptimizer(UVOptimizerConfig(optcuts_bin=self.config.optcuts_bin, patch_gap=self.config.patch_gap))
        return optimizer.optimize_patches([p.copy() for p in patches])

    def _quality_block(self, patches, patch_gap: float) -> Dict[str, object]:
        if not patches:
            inf_stats = {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
            return {
                "valid_patch_count": 0,
                "distortion": inf_stats,
                "angle_distortion": inf_stats,
                "area_distortion": inf_stats,
                "flip_rate": 1.0,
                "atlas_utilization": 0.0,
                "atlas_overlap_ratio": 0.0,
                "atlas_min_gap_stats": {"mean": 0.0, "min": 0.0, "p05": 0.0},
                "padding_violations": 0,
                "geometric_stability": {
                    "condition_mean": float("inf"),
                    "condition_p95": float("inf"),
                    "condition_p99": float("inf"),
                    "condition_max": float("inf"),
                    "det_mean": float("nan"),
                    "det_p05": float("nan"),
                    "det_p95": float("nan"),
                    "negative_jacobian_ratio": 1.0,
                },
            }

        dist, ang, area, flips, uv_list = [], [], [], [], []
        for p in patches:
            uv = p.metadata.get("uv")
            if uv is None:
                continue
            dist.append(UVAtlasMetrics.distortion_stats(p, uv))
            ang.append(UVAtlasMetrics.angle_distortion_stats(p, uv))
            area.append(UVAtlasMetrics.area_distortion_stats(p, uv))
            flips.append(UVAtlasMetrics.flip_rate(p, uv))
            uv_list.append(uv)

        return {
            "valid_patch_count": len(uv_list),
            "distortion": self._agg_stats(dist),
            "angle_distortion": self._agg_stats(ang),
            "area_distortion": self._agg_stats(area),
            "flip_rate": float(np.mean(flips)) if flips else 1.0,
            "atlas_utilization": UVAtlasMetrics.atlas_utilization(uv_list),
            "atlas_overlap_ratio": self._atlas_overlap_ratio(uv_list),
            "atlas_min_gap_stats": self._atlas_min_gap_stats(uv_list),
            "padding_violations": UVAtlasMetrics.padding_violations(uv_list, padding=patch_gap),
            "geometric_stability": self._agg_geo_stability(patches),
        }

    @staticmethod
    def _agg_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        return {
            "mean": float(np.mean([x["mean"] for x in stats_list])),
            "max": float(np.mean([x["max"] for x in stats_list])),
            "p95": float(np.mean([x["p95"] for x in stats_list])),
        }

    def _aggregate(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        valid = [r for r in rows if "error" not in r]
        if not valid:
            return {"valid_structure_count": 0}

        def _avg(path: List[str], default=float("inf")):
            vals = []
            for r in valid:
                cur = r
                for k in path:
                    cur = cur[k]
                vals.append(float(cur))
            return float(np.mean(vals)) if vals else default

        disk_attempts = sum(int(r["topology_repair"]["lscm"].get("attempted", 0)) for r in valid)
        disk_trigger = sum(int(r["topology_repair"]["lscm"].get("diskification_triggered", 0)) for r in valid)
        disk_success = sum(int(r["topology_repair"]["lscm"].get("diskification_success", 0)) for r in valid)
        total_attempt = sum(int(r["topology_repair"]["lscm"].get("attempted", 0)) for r in valid)
        total_success = sum(int(r["topology_repair"]["lscm"].get("success", 0)) for r in valid)
        fail_reason_hist = {}
        for r in valid:
            for k, v in r["topology_repair"]["lscm"].get("failure_reasons", {}).items():
                fail_reason_hist[k] = int(fail_reason_hist.get(k, 0)) + int(v)

        patch_counts = np.asarray([int(r.get("patch_count", 0)) for r in valid], dtype=np.float64)
        mesh_times = np.asarray([float(r["timing"]["stages"]["mesh_and_patch"]["wall_sec"]) for r in valid], dtype=np.float64)
        slope_patch_to_time = float(np.polyfit(patch_counts, mesh_times, 1)[0]) if len(valid) >= 2 else float("nan")
        energy_gains = [float(r["topology_optimization"]["energy"]["improvement_rate"]) for r in valid]
        seam_gains = [float(r["topology_optimization"]["seam_length"]["improvement_rate"]) for r in valid]

        def _dist(vals):
            arr = np.asarray(vals, dtype=np.float64)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50.0)),
                "p95": float(np.percentile(arr, 95.0)),
            }

        return {
            "valid_structure_count": len(valid),
            "distortion_lscm_mean": _avg(["lscm_raw", "distortion", "mean"]),
            "distortion_lscm_optcuts_mean": _avg(["lscm_optcuts", "distortion", "mean"]),
            "distortion_harmonic_mean": _avg(["harmonic_raw", "distortion", "mean"]),
            "flip_lscm_mean": _avg(["lscm_raw", "flip_rate"]),
            "flip_lscm_optcuts_mean": _avg(["lscm_optcuts", "flip_rate"]),
            "flip_harmonic_mean": _avg(["harmonic_raw", "flip_rate"]),
            "atlas_util_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_utilization"]),
            "atlas_overlap_ratio_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_overlap_ratio"]),
            "atlas_min_gap_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_min_gap_stats", "mean"]),
            "padding_violation_lscm_optcuts_mean": _avg(["lscm_optcuts", "padding_violations"]),
            "timing_lscm_total_wall_sec_mean": _avg(["timing", "parameterization", "lscm_total_wall_sec"]),
            "timing_harmonic_total_wall_sec_mean": _avg(["timing", "parameterization", "harmonic_total_wall_sec"]),
            "timing_mesh_and_patch_wall_sec_mean": _avg(["timing", "stages", "mesh_and_patch", "wall_sec"]),
            "timing_optcuts_wall_sec_mean": _avg(["timing", "stages", "optcuts_optimization", "wall_sec"]),
            "timing_rasterization_wall_sec_mean": _avg(["timing", "stages", "feature_rasterization", "wall_sec"]),
            "peak_rss_mb_mean": _avg(["memory", "peak_rss_mb"]),
            "energy_lscm_raw_mean": _avg(["topology_optimization", "energy", "lscm_raw"]),
            "energy_lscm_optcuts_mean": _avg(["topology_optimization", "energy", "lscm_optcuts"]),
            "seam_length_lscm_raw_mean": _avg(["topology_optimization", "seam_length", "lscm_raw"]),
            "seam_length_lscm_optcuts_mean": _avg(["topology_optimization", "seam_length", "lscm_optcuts"]),
            "parameterization_validity_rate": float(total_success / max(1, total_attempt)),
            "geometric_stability_condition_mean": _avg(["lscm_optcuts", "geometric_stability", "condition_mean"]),
            "geometric_stability_condition_p95": _avg(["lscm_optcuts", "geometric_stability", "condition_p95"]),
            "geometric_stability_condition_p99": _avg(["lscm_optcuts", "geometric_stability", "condition_p99"]),
            "geometric_stability_det_mean": _avg(["lscm_optcuts", "geometric_stability", "det_mean"], default=float("nan")),
            "geometric_stability_det_p05": _avg(["lscm_optcuts", "geometric_stability", "det_p05"], default=float("nan")),
            "geometric_stability_det_p95": _avg(["lscm_optcuts", "geometric_stability", "det_p95"], default=float("nan")),
            "geometric_stability_negative_jacobian_ratio": _avg(["lscm_optcuts", "geometric_stability", "negative_jacobian_ratio"]),
            "topology_boundary_loops_before_mean": _avg(["topology_repair", "lscm", "topology_before_boundary_loops_mean"]),
            "topology_boundary_loops_after_mean": _avg(["topology_repair", "lscm", "topology_after_boundary_loops_mean"]),
            "topology_face_retention_ratio_mean": _avg(["topology_repair", "lscm", "face_retention_ratio_mean"]),
            "topology_vertex_retention_ratio_mean": _avg(["topology_repair", "lscm", "vertex_retention_ratio_mean"], default=float("nan")),
            "topology_area_retention_ratio_mean": _avg(["topology_repair", "lscm", "area_retention_ratio_mean"], default=float("nan")),
            "topology_gate_fail_rate_mean": _avg(["topology_repair", "lscm", "topology_gate_fail_rate"]),
            "diskification_trigger_frequency": (float(disk_trigger) / float(disk_attempts)) if disk_attempts else 0.0,
            "diskification_success_rate_when_triggered": (float(disk_success) / float(disk_trigger)) if disk_trigger else 0.0,
            "diskification_attempts": int(disk_attempts),
            "diskification_triggered": int(disk_trigger),
            "diskification_success": int(disk_success),
            "optcuts_energy_gain_distribution": _dist(energy_gains) if energy_gains else {},
            "optcuts_seam_gain_distribution": _dist(seam_gains) if seam_gains else {},
            "atlas_trainability_nonzero_ratio_mean": _avg(["atlas_trainability", "nonzero_ratio"]),
            "atlas_trainability_entropy_mean": _avg(["atlas_trainability", "density_entropy"]),
            "atlas_trainability_patch_coverage_mean": _avg(["atlas_trainability", "patch_coverage_mean"]),
            "atlas_trainability_patch_coverage_p05": _avg(["atlas_trainability", "patch_coverage_p05"]),
            "atlas_trainability_patch_coverage_p95": _avg(["atlas_trainability", "patch_coverage_p95"]),
            "failure_reason_histogram": fail_reason_hist,
            "scalability_patch_to_mesh_stage_slope": slope_patch_to_time,
        }

    def _write_csv(self, rows: List[Dict[str, object]]) -> None:
        csv_path = os.path.join(self.config.output_root, "benchmark_summary.csv")
        header = [
            "pdb", "patch_count", "lscm_dist_mean", "lscm_dist_max", "lscm_dist_p95",
            "lscm_angle_mean", "lscm_area_mean", "lscm_flip", "lscm_optcuts_dist_mean", "lscm_optcuts_flip",
            "harmonic_dist_mean", "harmonic_flip", "atlas_util_lscm_optcuts", "padding_viol_lscm_optcuts",
            "mesh_stage_wall_sec", "lscm_wall_sec", "harmonic_wall_sec", "optcuts_wall_sec", "raster_wall_sec",
            "peak_rss_mb", "energy_lscm_raw", "energy_lscm_optcuts", "seam_lscm_raw", "seam_lscm_optcuts",
            "diskify_triggered", "diskify_success", "validity_rate_lscm", "geo_cond_mean_lscm_optcuts",
            "geo_neg_jac_ratio_lscm_optcuts", "atlas_overlap_ratio_lscm_optcuts", "atlas_min_gap_lscm_optcuts",
            "topology_face_retention_ratio", "topology_gate_fail_rate", "optcuts_energy_gain", "optcuts_seam_gain",
            "topology_vertex_retention_ratio", "topology_area_retention_ratio", "atlas_nonzero_ratio", "atlas_density_entropy",
            "atlas_patch_coverage_mean", "atlas_patch_coverage_p05", "atlas_patch_coverage_p95", "error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                if "error" in r:
                    writer.writerow({"pdb": r["pdb"], "patch_count": r.get("patch_count", 0), "error": r["error"]})
                    continue
                writer.writerow(
                    {
                        "pdb": r["pdb"],
                        "patch_count": r["patch_count"],
                        "lscm_dist_mean": r["lscm_raw"]["distortion"]["mean"],
                        "lscm_dist_max": r["lscm_raw"]["distortion"]["max"],
                        "lscm_dist_p95": r["lscm_raw"]["distortion"]["p95"],
                        "lscm_angle_mean": r["lscm_raw"]["angle_distortion"]["mean"],
                        "lscm_area_mean": r["lscm_raw"]["area_distortion"]["mean"],
                        "lscm_flip": r["lscm_raw"]["flip_rate"],
                        "lscm_optcuts_dist_mean": r["lscm_optcuts"]["distortion"]["mean"],
                        "lscm_optcuts_flip": r["lscm_optcuts"]["flip_rate"],
                        "harmonic_dist_mean": r["harmonic_raw"]["distortion"]["mean"],
                        "harmonic_flip": r["harmonic_raw"]["flip_rate"],
                        "atlas_util_lscm_optcuts": r["lscm_optcuts"]["atlas_utilization"],
                        "padding_viol_lscm_optcuts": r["lscm_optcuts"]["padding_violations"],
                        "mesh_stage_wall_sec": r["timing"]["stages"]["mesh_and_patch"]["wall_sec"],
                        "lscm_wall_sec": r["timing"]["parameterization"]["lscm_total_wall_sec"],
                        "harmonic_wall_sec": r["timing"]["parameterization"]["harmonic_total_wall_sec"],
                        "optcuts_wall_sec": r["timing"]["stages"]["optcuts_optimization"]["wall_sec"],
                        "raster_wall_sec": r["timing"]["stages"]["feature_rasterization"]["wall_sec"],
                        "peak_rss_mb": r["memory"]["peak_rss_mb"],
                        "energy_lscm_raw": r["topology_optimization"]["energy"]["lscm_raw"],
                        "energy_lscm_optcuts": r["topology_optimization"]["energy"]["lscm_optcuts"],
                        "seam_lscm_raw": r["topology_optimization"]["seam_length"]["lscm_raw"],
                        "seam_lscm_optcuts": r["topology_optimization"]["seam_length"]["lscm_optcuts"],
                        "diskify_triggered": r["topology_repair"]["lscm"].get("diskification_triggered", 0),
                        "diskify_success": r["topology_repair"]["lscm"].get("diskification_success", 0),
                        "validity_rate_lscm": float(r["topology_repair"]["lscm"].get("success", 0) / max(1, r["topology_repair"]["lscm"].get("attempted", 0))),
                        "geo_cond_mean_lscm_optcuts": r["lscm_optcuts"]["geometric_stability"]["condition_mean"],
                        "geo_neg_jac_ratio_lscm_optcuts": r["lscm_optcuts"]["geometric_stability"]["negative_jacobian_ratio"],
                        "atlas_overlap_ratio_lscm_optcuts": r["lscm_optcuts"]["atlas_overlap_ratio"],
                        "atlas_min_gap_lscm_optcuts": r["lscm_optcuts"]["atlas_min_gap_stats"]["mean"],
                        "topology_face_retention_ratio": r["topology_repair"]["lscm"].get("face_retention_ratio_mean", float("nan")),
                        "topology_vertex_retention_ratio": r["topology_repair"]["lscm"].get("vertex_retention_ratio_mean", float("nan")),
                        "topology_area_retention_ratio": r["topology_repair"]["lscm"].get("area_retention_ratio_mean", float("nan")),
                        "topology_gate_fail_rate": r["topology_repair"]["lscm"].get("topology_gate_fail_rate", float("nan")),
                        "optcuts_energy_gain": r["topology_optimization"]["energy"].get("improvement_rate", float("nan")),
                        "optcuts_seam_gain": r["topology_optimization"]["seam_length"].get("improvement_rate", float("nan")),
                        "atlas_nonzero_ratio": r["atlas_trainability"]["nonzero_ratio"],
                        "atlas_density_entropy": r["atlas_trainability"]["density_entropy"],
                        "atlas_patch_coverage_mean": r["atlas_trainability"]["patch_coverage_mean"],
                        "atlas_patch_coverage_p05": r["atlas_trainability"]["patch_coverage_p05"],
                        "atlas_patch_coverage_p95": r["atlas_trainability"]["patch_coverage_p95"],
                        "error": "",
                    }
                )

    def _run_sensitivity(self, pdb_files: List[str]) -> Dict[str, object]:
        sweep = {
            "cutoff": self.config.cutoff_sweep,
            "sigma": self.config.sigma_sweep,
            "res": self.config.res_sweep,
        }
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

    @staticmethod
    def _triangle_local_coords(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        e1 = v1 - v0
        e2 = v2 - v0
        l1 = np.linalg.norm(e1)
        if l1 < 1e-12:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        x2 = np.dot(e2, e1 / l1)
        y2 = np.sqrt(max(np.dot(e2, e2) - x2 * x2, 0.0))
        return np.array([l1, 0.0], dtype=np.float64), np.array([x2, y2], dtype=np.float64)

    def _avg_energy(self, patches) -> float:
        vals = []
        for p in patches:
            uv = p.metadata.get("uv")
            if uv is None:
                continue
            vals.append(self._symmetric_dirichlet_energy(p, uv))
        return float(np.mean(vals)) if vals else float("inf")

    def _symmetric_dirichlet_energy(self, mesh, uv: np.ndarray) -> float:
        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        v2 = np.asarray(uv, dtype=np.float64)
        per_face = []
        for tri in f:
            a3, b3, c3 = v3[tri[0]], v3[tri[1]], v3[tri[2]]
            a2, b2, c2 = v2[tri[0]], v2[tri[1]], v2[tri[2]]
            q1, q2 = self._triangle_local_coords(a3, b3, c3)
            P = np.column_stack([q1, q2])
            detP = np.linalg.det(P)
            if abs(detP) < 1e-12:
                continue
            U = np.column_stack([b2 - a2, c2 - a2])
            J = U @ np.linalg.inv(P)
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-12:
                per_face.append(1e6)
                continue
            jn = np.sum(J * J)
            invjn = np.sum(np.linalg.inv(J) ** 2)
            per_face.append(0.5 * (jn + invjn))
        return float(np.mean(per_face)) if per_face else float("inf")

    def _avg_seam_length(self, patches) -> float:
        vals = []
        for p in patches:
            uv = p.metadata.get("uv")
            if uv is None:
                continue
            vals.append(self._boundary_length_uv(p, uv))
        return float(np.mean(vals)) if vals else float("inf")

    @staticmethod
    def _boundary_length_uv(mesh, uv: np.ndarray) -> float:
        f = np.asarray(mesh.faces, dtype=np.int64)
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        uniq, cnt = np.unique(edges, axis=0, return_counts=True)
        bnd = uniq[cnt == 1]
        if len(bnd) == 0:
            return 0.0
        v2 = np.asarray(uv, dtype=np.float64)
        seg = v2[bnd[:, 1]] - v2[bnd[:, 0]]
        return float(np.sum(np.linalg.norm(seg, axis=1)))

    @staticmethod
    def _rasterize_feature_maps(patches, size: int = 256, return_patch_coverage: bool = False):
        atlas = np.zeros((size, size), dtype=np.float32)
        patch_coverages = []
        for p in patches:
            uv = p.metadata.get("uv")
            if uv is None or len(uv) == 0:
                continue
            uvn = np.clip(np.asarray(uv, dtype=np.float64), 0.0, 1.0)
            xi = np.minimum((uvn[:, 0] * (size - 1)).astype(np.int32), size - 1)
            yi = np.minimum((uvn[:, 1] * (size - 1)).astype(np.int32), size - 1)
            np.add.at(atlas, (yi, xi), 1.0)
            if return_patch_coverage:
                patch_mask = np.zeros((size, size), dtype=np.uint8)
                patch_mask[yi, xi] = 1
                patch_coverages.append(float(np.count_nonzero(patch_mask) / max(1, patch_mask.size)))
        if return_patch_coverage:
            return atlas, patch_coverages
        return atlas

    @staticmethod
    def _atlas_trainability_metrics(atlas: np.ndarray, patch_coverages: Optional[List[float]] = None) -> Dict[str, float]:
        total = float(atlas.size)
        nonzero = float(np.count_nonzero(atlas))
        nonzero_ratio = nonzero / max(1.0, total)
        mass = np.sum(atlas)
        if mass <= 0:
            return {
                "nonzero_ratio": 0.0,
                "density_entropy": 0.0,
                "patch_coverage_mean": 0.0,
                "patch_coverage_p05": 0.0,
                "patch_coverage_p95": 0.0,
            }
        prob = (atlas.flatten() / mass).astype(np.float64)
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log(prob)))
        cov = np.asarray(patch_coverages or [], dtype=np.float64)
        return {
            "nonzero_ratio": float(nonzero_ratio),
            "density_entropy": entropy,
            "patch_coverage_mean": float(np.mean(cov)) if len(cov) else 0.0,
            "patch_coverage_p05": float(np.percentile(cov, 5.0)) if len(cov) else 0.0,
            "patch_coverage_p95": float(np.percentile(cov, 95.0)) if len(cov) else 0.0,
        }

    @staticmethod
    def _improvement_rate(before: float, after: float) -> float:
        if not np.isfinite(before) or abs(before) < 1e-12:
            return float("nan")
        return float((before - after) / before)

    def _agg_geo_stability(self, patches) -> Dict[str, float]:
        all_cond = []
        all_neg = []
        all_det = []
        for p in patches:
            uv = p.metadata.get("uv")
            if uv is None:
                continue
            st = self._jacobian_stability_stats(p, uv)
            all_cond.extend(st["condition_values"])
            all_neg.append(st["negative_jacobian_ratio"])
            all_det.extend(st.get("determinant_values", []))
        if not all_cond:
            return {
                "condition_mean": float("inf"),
                "condition_p95": float("inf"),
                "condition_p99": float("inf"),
                "condition_max": float("inf"),
                "det_mean": float("nan"),
                "det_p05": float("nan"),
                "det_p95": float("nan"),
                "negative_jacobian_ratio": 1.0,
            }
        det_vals = np.asarray(all_det, dtype=np.float64)
        return {
            "condition_mean": float(np.mean(all_cond)),
            "condition_p95": float(np.percentile(all_cond, 95.0)),
            "condition_p99": float(np.percentile(all_cond, 99.0)),
            "condition_max": float(np.max(all_cond)),
            "det_mean": float(np.mean(det_vals)) if len(det_vals) else float("nan"),
            "det_p05": float(np.percentile(det_vals, 5.0)) if len(det_vals) else float("nan"),
            "det_p95": float(np.percentile(det_vals, 95.0)) if len(det_vals) else float("nan"),
            "negative_jacobian_ratio": float(np.mean(all_neg)) if all_neg else 1.0,
        }

    def _jacobian_stability_stats(self, mesh, uv: np.ndarray) -> Dict[str, object]:
        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        v2 = np.asarray(uv, dtype=np.float64)
        cond_vals = []
        det_vals = []
        neg = 0
        total = 0
        for tri in f:
            a3, b3, c3 = v3[tri[0]], v3[tri[1]], v3[tri[2]]
            a2, b2, c2 = v2[tri[0]], v2[tri[1]], v2[tri[2]]
            q1, q2 = self._triangle_local_coords(a3, b3, c3)
            P = np.column_stack([q1, q2])
            if abs(np.linalg.det(P)) < 1e-12:
                continue
            J = np.column_stack([b2 - a2, c2 - a2]) @ np.linalg.inv(P)
            total += 1
            det_j = float(np.linalg.det(J))
            det_vals.append(det_j)
            if det_j <= 0:
                neg += 1
                continue
            s = np.linalg.svd(J, compute_uv=False)
            smin = max(float(np.min(s)), 1e-12)
            smax = float(np.max(s))
            cond_vals.append(smax / smin)
        return {
            "condition_values": cond_vals,
            "determinant_values": det_vals,
            "negative_jacobian_ratio": float(neg / max(1, total)),
        }

    @staticmethod
    def _atlas_overlap_ratio(uv_list: List[np.ndarray]) -> float:
        if not uv_list:
            return 0.0
        overlap = UVAtlasMetrics.atlas_bbox_overlap_area(uv_list)
        mins = np.min(np.stack([uv.min(axis=0) for uv in uv_list], axis=0), axis=0)
        maxs = np.max(np.stack([uv.max(axis=0) for uv in uv_list], axis=0), axis=0)
        atlas_area = max(1e-12, float((maxs[0] - mins[0]) * (maxs[1] - mins[1])))
        return float(overlap / atlas_area)

    @staticmethod
    def _atlas_min_gap_stats(uv_list: List[np.ndarray]) -> Dict[str, float]:
        if len(uv_list) < 2:
            return {"mean": 0.0, "min": 0.0, "p05": 0.0}
        gaps = []
        for i in range(len(uv_list)):
            a_min, a_max = uv_list[i].min(axis=0), uv_list[i].max(axis=0)
            for j in range(i + 1, len(uv_list)):
                b_min, b_max = uv_list[j].min(axis=0), uv_list[j].max(axis=0)
                dx = max(a_min[0] - b_max[0], b_min[0] - a_max[0], 0.0)
                dy = max(a_min[1] - b_max[1], b_min[1] - a_max[1], 0.0)
                gaps.append(float(np.hypot(dx, dy)))
        if not gaps:
            return {"mean": 0.0, "min": 0.0, "p05": 0.0}
        arr = np.asarray(gaps, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "p05": float(np.percentile(arr, 5.0))}

    def _memory_rss_mb(self) -> float:
        if self._proc is None:
            return 0.0
        try:
            return float(self._proc.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    @staticmethod
    def _stage_stats(start_wall: float, start_cpu: float) -> Dict[str, float]:
        return {
            "wall_sec": float(time.perf_counter() - start_wall),
            "cpu_sec": float(time.process_time() - start_cpu),
        }

    @staticmethod
    def _from_timing_list(times: List[float], cpu_times: List[float]) -> Dict[str, float]:
        if not times:
            return {"wall_sec": float("inf"), "cpu_sec": float("inf")}
        return {
            "wall_sec": float(np.sum(times)),
            "cpu_sec": float(np.sum(cpu_times)) if cpu_times else float("inf"),
        }
