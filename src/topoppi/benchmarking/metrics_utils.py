"""Benchmark metrics with explicit domains, units, and aggregation rules."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.draw import polygon

from topoppi.atlas.metrics import UVAtlasMetrics, face_jacobians, weighted_percentile, weighted_stats
from topoppi.atlas.packing import pack_mesh_charts, resolved_chart_gap
from topoppi.atlas.uv import as_corner_uv, face_domain_hash
from topoppi.config import DEFAULT_BENCHMARK_CONFIG


def _stats(values: List[np.ndarray], weights: List[np.ndarray]) -> Dict[str, float | bool]:
    if not values:
        return {
            "mean": float("inf"),
            "max": float("inf"),
            "p95": float("inf"),
            "area_weighted": True,
            "invalid_area_ratio": 1.0,
        }
    return {
        **weighted_stats(np.concatenate(values), np.concatenate(weights)),
        "area_weighted": True,
    }


def symmetric_dirichlet_energy(mesh, uv: np.ndarray) -> float:
    values, weights = UVAtlasMetrics.symmetric_dirichlet_samples(mesh, uv)
    if np.any((weights > 0.0) & ~np.isfinite(values)):
        return float("inf")
    finite = np.isfinite(values) & (weights > 0.0)
    return float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else float("inf")


def avg_energy(patches, uv_key: str = "uv") -> float:
    values, weights = [], []
    for patch in patches:
        uv = as_corner_uv(patch, key=uv_key)
        values.append(symmetric_dirichlet_energy(patch, uv))
        weights.append(float(patch.area))
    return float(np.average(values, weights=weights)) if values else float("inf")


def avg_seam_length(patches, uv_key: str = "uv") -> float:
    """Total three-dimensional internal seam length divided by total-area scale."""

    seam_length = 0.0
    area = 0.0
    for patch in patches:
        seam = UVAtlasMetrics.seam_stats(patch, as_corner_uv(patch, key=uv_key))
        seam_length += float(seam["seam_length_3d"])
        area += float(patch.area)
    return float(seam_length / np.sqrt(area)) if area > 0.0 else float("inf")


def improvement_rate(before: float, after: float, *, reference: float = 0.0) -> float:
    denominator = before - reference
    if not np.isfinite(before) or not np.isfinite(after) or abs(denominator) < 1e-12:
        return float("nan")
    return float((before - after) / denominator)


def jacobian_stability_stats(mesh, uv: np.ndarray) -> Dict[str, object]:
    jacobians, weights, valid = face_jacobians(mesh, uv)
    determinants = np.full(len(jacobians), np.nan, dtype=np.float64)
    condition = np.full(len(jacobians), np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices):
        singular = np.linalg.svd(jacobians[indices], compute_uv=False)
        nonsingular = np.min(singular, axis=1) > 1e-12
        stable_indices = indices[nonsingular]
        stable_singular = singular[nonsingular]
        determinants[stable_indices] = np.linalg.det(jacobians[stable_indices])
        condition[stable_indices] = np.max(stable_singular, axis=1) / np.min(stable_singular, axis=1)
    finite_det = np.isfinite(determinants)
    orientation_score = (
        float(np.sum(np.sign(determinants[finite_det]) * weights[finite_det])) if np.any(finite_det) else 1.0
    )
    global_orientation = 1.0 if orientation_score >= 0.0 else -1.0
    corrected_det = global_orientation * determinants
    positive_area = np.isfinite(weights) & (weights > 0.0)
    invalid_area = positive_area & ~np.isfinite(determinants)
    return {
        "condition_values": condition,
        "determinant_values": corrected_det,
        "face_area_weights": weights,
        "negative_jacobian_ratio": float(
            np.sum(weights[np.isfinite(corrected_det) & (corrected_det <= 0.0)]) / max(float(np.sum(weights)), 1e-12)
        ),
        "invalid_jacobian_area_ratio": float(
            np.sum(weights[invalid_area]) / max(float(np.sum(weights[positive_area])), 1e-12)
        ),
        "global_reflection_corrected": global_orientation < 0.0,
    }


def agg_geo_stability(patches, uv_key: str = "uv") -> Dict[str, float | bool]:
    conditions, determinants, weights = [], [], []
    reflection_count = 0
    for patch in patches:
        stats = jacobian_stability_stats(patch, as_corner_uv(patch, key=uv_key))
        conditions.append(np.asarray(stats["condition_values"], dtype=np.float64))
        determinants.append(np.asarray(stats["determinant_values"], dtype=np.float64))
        weights.append(np.asarray(stats["face_area_weights"], dtype=np.float64))
        reflection_count += int(bool(stats["global_reflection_corrected"]))
    if not conditions:
        return {
            "condition_mean": float("inf"),
            "condition_p95": float("inf"),
            "condition_p99": float("inf"),
            "condition_max": float("inf"),
            "det_mean": float("nan"),
            "det_p05": float("nan"),
            "det_p95": float("nan"),
            "negative_jacobian_ratio": 1.0,
            "invalid_jacobian_area_ratio": 1.0,
            "global_reflection_corrected_patch_count": 0,
        }
    condition = np.concatenate(conditions)
    determinant = np.concatenate(determinants)
    area_weights = np.concatenate(weights)
    valid_condition = np.isfinite(condition) & (area_weights > 0.0)
    valid_det = np.isfinite(determinant) & (area_weights > 0.0)
    negative = valid_det & (determinant <= 0.0)
    positive_area = np.isfinite(area_weights) & (area_weights > 0.0)
    invalid = positive_area & ~valid_det
    total_positive_area = max(float(np.sum(area_weights[positive_area])), 1e-12)
    return {
        "condition_mean": float(np.average(condition[valid_condition], weights=area_weights[valid_condition]))
        if np.any(valid_condition)
        else float("inf"),
        "condition_p95": weighted_percentile(condition, area_weights, 95.0),
        "condition_p99": weighted_percentile(condition, area_weights, 99.0),
        "condition_max": float(np.max(condition[valid_condition])) if np.any(valid_condition) else float("inf"),
        "det_mean": float(np.average(determinant[valid_det], weights=area_weights[valid_det]))
        if np.any(valid_det)
        else float("nan"),
        "det_p05": weighted_percentile(determinant, area_weights, 5.0) if np.any(valid_det) else float("nan"),
        "det_p95": weighted_percentile(determinant, area_weights, 95.0) if np.any(valid_det) else float("nan"),
        "negative_jacobian_ratio": float(np.sum(area_weights[negative]) / total_positive_area),
        "invalid_jacobian_area_ratio": float(np.sum(area_weights[invalid]) / total_positive_area),
        "global_reflection_corrected_patch_count": int(reflection_count),
    }


def quality_block(
    patches,
    patch_gap: float,
    *,
    uv_key: str = "uv",
) -> Dict[str, object]:
    if not patches:
        inf_stats = {
            "mean": float("inf"),
            "max": float("inf"),
            "p95": float("inf"),
            "area_weighted": True,
            "invalid_area_ratio": 1.0,
        }
        return {
            "valid_patch_count": 0,
            "scored_face_count": 0,
            "scored_area_3d": 0.0,
            "domain_hashes": [],
            "distortion": inf_stats,
            "symmetric_dirichlet": inf_stats,
            "angle_distortion": inf_stats,
            "area_distortion": inf_stats,
            "flip_rate": 1.0,
            "seam": {},
            "injectivity": {
                "all_patches_globally_injective": False,
                "globally_injective_patch_count": 0,
                "globally_injective_patch_rate": float("nan"),
            },
            "atlas": {"status": "empty", "chart_count": 0},
            "geometric_stability": agg_geo_stability([], uv_key=uv_key),
        }

    distortion_values: List[np.ndarray] = []
    symmetric_dirichlet_values: List[np.ndarray] = []
    angle_values: List[np.ndarray] = []
    area_values: List[np.ndarray] = []
    face_weights: List[np.ndarray] = []
    flips, patch_areas, seam_records, injectivity_records = [], [], [], []
    for patch in patches:
        uv = as_corner_uv(patch, key=uv_key)
        dist, weights = UVAtlasMetrics.distortion_samples(patch, uv)
        symmetric_dirichlet, _ = UVAtlasMetrics.symmetric_dirichlet_samples(patch, uv)
        angle, _ = UVAtlasMetrics.angle_distortion_samples(patch, uv)
        area, _ = UVAtlasMetrics.area_distortion_samples(patch, uv)
        distortion_values.append(dist)
        symmetric_dirichlet_values.append(symmetric_dirichlet)
        angle_values.append(angle)
        area_values.append(area)
        face_weights.append(weights)
        patch_area = float(np.sum(weights))
        patch_areas.append(patch_area)
        flips.append(UVAtlasMetrics.flip_rate(patch, uv))
        seam_records.append(UVAtlasMetrics.seam_stats(patch, uv))
        injectivity_records.append(UVAtlasMetrics.parameterization_injectivity_stats(patch, uv))

    total_area = float(np.sum(patch_areas))
    seam = {
        "seam_edge_count": int(sum(int(item["seam_edge_count"]) for item in seam_records)),
        "seam_length_3d": float(sum(float(item["seam_length_3d"]) for item in seam_records)),
        "seam_length_3d_normalized": float(
            sum(float(item["seam_length_3d"]) for item in seam_records) / np.sqrt(total_area)
        ),
        "boundary_edge_count": int(sum(int(item["boundary_edge_count"]) for item in seam_records)),
        "boundary_length_3d": float(sum(float(item["boundary_length_3d"]) for item in seam_records)),
    }
    try:
        packed_uv, _transforms, packing = pack_mesh_charts(
            patches,
            key=uv_key,
            gap=patch_gap,
        )
    except ValueError as exc:
        atlas = {
            "status": "not_packable_degenerate_chart",
            "chart_count": int(len(patches)),
            "reason": str(exc),
        }
    else:
        atlas = {
            "status": "evaluated_polygonal_triangle_geometry_with_numeric_tolerance",
            "packing": packing,
            **UVAtlasMetrics.atlas_geometry_stats(
                patches,
                padding=resolved_chart_gap(patches, patch_gap),
                uv_arrays=packed_uv,
            ),
        }
    globally_injective_count = int(sum(bool(record["globally_injective"]) for record in injectivity_records))
    injectivity = {
        "all_patches_globally_injective": globally_injective_count == len(patches),
        "globally_injective_patch_count": globally_injective_count,
        "globally_injective_patch_rate": float(globally_injective_count / len(patches)),
        "locally_injective_patch_count": int(
            sum(int(record["flip_face_count"]) == 0 for record in injectivity_records)
        ),
        "continuous_on_input_mesh_patch_count": int(
            sum(bool(record["continuous_on_input_mesh"]) for record in injectivity_records)
        ),
        "invalid_triangle_polygon_count": int(
            sum(int(record["invalid_triangle_polygon_count"]) for record in injectivity_records)
        ),
        "max_patch_overdraw_ratio": float(max(float(record["overdraw_ratio"]) for record in injectivity_records)),
    }

    return {
        "valid_patch_count": int(len(patches)),
        "scored_face_count": int(sum(len(patch.faces) for patch in patches)),
        "scored_area_3d": float(np.sum(patch_areas)),
        "domain_hashes": [face_domain_hash(patch) for patch in patches],
        "distortion": _stats(distortion_values, face_weights),
        "symmetric_dirichlet": {
            **_stats(symmetric_dirichlet_values, face_weights),
            "identity_value": 2.0,
            "scale_alignment": "analytic_global_symmetric_dirichlet_minimum",
        },
        "angle_distortion": {**_stats(angle_values, face_weights), "unit": "radian"},
        "area_distortion": _stats(area_values, face_weights),
        "flip_rate": float(np.average(flips, weights=patch_areas)) if flips else 1.0,
        "seam": seam,
        "injectivity": injectivity,
        "atlas": atlas,
        "geometric_stability": agg_geo_stability(patches, uv_key=uv_key),
    }


def _atlas_pixel_transform(corner_sets: List[np.ndarray], size: int) -> Tuple[np.ndarray, float]:
    points = np.concatenate([corners.reshape(-1, 2) for corners in corner_sets], axis=0)
    minimum = np.min(points, axis=0)
    extent = np.ptp(points, axis=0)
    longest = max(float(np.max(extent)), 1e-12)
    margin = 1.0
    scale = max(float(size - 1) - 2.0 * margin, 1.0) / longest
    return minimum - margin / scale, scale


def rasterize_feature_maps(
    patches,
    size: int = DEFAULT_BENCHMARK_CONFIG.raster_size,
    return_patch_coverage: bool = False,
    uv_arrays: Optional[List[np.ndarray]] = None,
):
    """Rasterize filled triangles after one global aspect-preserving transform."""

    atlas = np.zeros((size, size), dtype=np.float32)
    if uv_arrays is not None and len(uv_arrays) != len(patches):
        raise ValueError("uv_arrays must contain one UV array per patch.")
    layouts = [
        (
            patch,
            as_corner_uv(patch, uv_arrays[index])
            if uv_arrays is not None
            else as_corner_uv(patch, key="uv_global" if "uv_global" in patch.metadata else "uv"),
        )
        for index, patch in enumerate(patches)
    ]
    if not layouts:
        return (atlas, []) if return_patch_coverage else atlas

    origin, scale = _atlas_pixel_transform([corners for _, corners in layouts], size)
    patch_coverages = []
    for _patch, corners in layouts:
        patch_mask = np.zeros((size, size), dtype=bool) if return_patch_coverage else None
        pixels = (corners - origin) * scale
        for triangle in pixels:
            rows, cols = polygon(triangle[:, 1], triangle[:, 0], shape=atlas.shape)
            atlas[rows, cols] += 1.0
            if patch_mask is not None:
                patch_mask[rows, cols] = True
        if patch_mask is not None:
            patch_coverages.append(float(np.count_nonzero(patch_mask) / patch_mask.size))
    return (atlas, patch_coverages) if return_patch_coverage else atlas


def atlas_trainability_metrics(atlas: np.ndarray, patch_coverages: Optional[List[float]] = None) -> Dict[str, float]:
    total = float(atlas.size)
    nonzero = float(np.count_nonzero(atlas))
    nonzero_ratio = nonzero / max(1.0, total)
    mass = float(np.sum(atlas))
    if mass <= 0.0:
        return {
            "nonzero_ratio": 0.0,
            "density_entropy": 0.0,
            "patch_coverage_mean": 0.0,
            "patch_coverage_p05": 0.0,
            "patch_coverage_p95": 0.0,
            "rasterization": "filled_triangles_global_similarity",
        }
    probability = atlas[atlas > 0.0].astype(np.float64) / mass
    coverage = np.asarray(patch_coverages or [], dtype=np.float64)
    return {
        "nonzero_ratio": float(nonzero_ratio),
        "density_entropy": float(-np.sum(probability * np.log(probability))),
        "patch_coverage_mean": float(np.mean(coverage)) if len(coverage) else 0.0,
        "patch_coverage_p05": float(np.percentile(coverage, 5.0)) if len(coverage) else 0.0,
        "patch_coverage_p95": float(np.percentile(coverage, 95.0)) if len(coverage) else 0.0,
        "rasterization": "filled_triangles_global_similarity",
    }
