from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np


def aggregate_results(rows: List[Dict[str, object]]) -> Dict[str, object]:
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
        "distortion_spherical_mean": _avg(["spherical_raw", "distortion", "mean"]),
        "distortion_cylindrical_mean": _avg(["cylindrical_raw", "distortion", "mean"]),
        "flip_lscm_mean": _avg(["lscm_raw", "flip_rate"]),
        "flip_lscm_optcuts_mean": _avg(["lscm_optcuts", "flip_rate"]),
        "flip_harmonic_mean": _avg(["harmonic_raw", "flip_rate"]),
        "flip_spherical_mean": _avg(["spherical_raw", "flip_rate"]),
        "flip_cylindrical_mean": _avg(["cylindrical_raw", "flip_rate"]),
        "atlas_util_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_utilization"]),
        "atlas_overlap_ratio_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_overlap_ratio"]),
        "atlas_min_gap_lscm_optcuts_mean": _avg(["lscm_optcuts", "atlas_min_gap_stats", "mean"]),
        "padding_violation_lscm_optcuts_mean": _avg(["lscm_optcuts", "padding_violations"]),
        "timing_lscm_total_wall_sec_mean": _avg(["timing", "parameterization", "lscm_total_wall_sec"]),
        "timing_harmonic_total_wall_sec_mean": _avg(["timing", "parameterization", "harmonic_total_wall_sec"]),
        "timing_spherical_total_wall_sec_mean": _avg(["timing", "parameterization", "spherical_total_wall_sec"]),
        "timing_cylindrical_total_wall_sec_mean": _avg(["timing", "parameterization", "cylindrical_total_wall_sec"]),
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
        "ablation_energy_improvement_mean": _avg(["optcuts_ablation", "energy_improvement_rate"], default=float("nan")),
        "ablation_seam_improvement_mean": _avg(["optcuts_ablation", "seam_improvement_rate"], default=float("nan")),
        "atlas_trainability_nonzero_ratio_mean": _avg(["atlas_trainability", "nonzero_ratio"]),
        "atlas_trainability_entropy_mean": _avg(["atlas_trainability", "density_entropy"]),
        "atlas_trainability_patch_coverage_mean": _avg(["atlas_trainability", "patch_coverage_mean"]),
        "atlas_trainability_patch_coverage_p05": _avg(["atlas_trainability", "patch_coverage_p05"]),
        "atlas_trainability_patch_coverage_p95": _avg(["atlas_trainability", "patch_coverage_p95"]),
        "failure_reason_histogram": fail_reason_hist,
        "scalability_patch_to_mesh_stage_slope": slope_patch_to_time,
    }


def write_csv(rows: List[Dict[str, object]], output_root: str, filename: str = "benchmark_summary.csv") -> None:
    csv_path = os.path.join(output_root, filename)
    header = [
        "pdb", "patch_count", "lscm_dist_mean", "lscm_dist_max", "lscm_dist_p95",
        "lscm_angle_mean", "lscm_area_mean", "lscm_flip", "lscm_optcuts_dist_mean", "lscm_optcuts_flip",
        "harmonic_dist_mean", "harmonic_flip", "spherical_dist_mean", "spherical_flip",
        "cylindrical_dist_mean", "cylindrical_flip", "atlas_util_lscm_optcuts", "padding_viol_lscm_optcuts",
        "mesh_stage_wall_sec", "lscm_wall_sec", "harmonic_wall_sec", "spherical_wall_sec", "cylindrical_wall_sec",
        "optcuts_wall_sec", "raster_wall_sec",
        "peak_rss_mb", "energy_lscm_raw", "energy_lscm_optcuts", "seam_lscm_raw", "seam_lscm_optcuts",
        "ablation_energy_improvement", "ablation_seam_improvement",
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
                    "spherical_dist_mean": r["spherical_raw"]["distortion"]["mean"],
                    "spherical_flip": r["spherical_raw"]["flip_rate"],
                    "cylindrical_dist_mean": r["cylindrical_raw"]["distortion"]["mean"],
                    "cylindrical_flip": r["cylindrical_raw"]["flip_rate"],
                    "atlas_util_lscm_optcuts": r["lscm_optcuts"]["atlas_utilization"],
                    "padding_viol_lscm_optcuts": r["lscm_optcuts"]["padding_violations"],
                    "mesh_stage_wall_sec": r["timing"]["stages"]["mesh_and_patch"]["wall_sec"],
                    "lscm_wall_sec": r["timing"]["parameterization"]["lscm_total_wall_sec"],
                    "harmonic_wall_sec": r["timing"]["parameterization"]["harmonic_total_wall_sec"],
                    "spherical_wall_sec": r["timing"]["parameterization"]["spherical_total_wall_sec"],
                    "cylindrical_wall_sec": r["timing"]["parameterization"]["cylindrical_total_wall_sec"],
                    "optcuts_wall_sec": r["timing"]["stages"]["optcuts_optimization"]["wall_sec"],
                    "raster_wall_sec": r["timing"]["stages"]["feature_rasterization"]["wall_sec"],
                    "peak_rss_mb": r["memory"]["peak_rss_mb"],
                    "energy_lscm_raw": r["topology_optimization"]["energy"]["lscm_raw"],
                    "energy_lscm_optcuts": r["topology_optimization"]["energy"]["lscm_optcuts"],
                    "seam_lscm_raw": r["topology_optimization"]["seam_length"]["lscm_raw"],
                    "seam_lscm_optcuts": r["topology_optimization"]["seam_length"]["lscm_optcuts"],
                    "ablation_energy_improvement": r["optcuts_ablation"]["energy_improvement_rate"],
                    "ablation_seam_improvement": r["optcuts_ablation"]["seam_improvement_rate"],
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
