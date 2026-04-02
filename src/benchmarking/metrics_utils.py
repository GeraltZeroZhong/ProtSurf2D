from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.atlas.metrics import UVAtlasMetrics


def triangle_local_coords(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e1 = v1 - v0
    e2 = v2 - v0
    l1 = np.linalg.norm(e1)
    if l1 < 1e-12:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    x2 = np.dot(e2, e1 / l1)
    y2 = np.sqrt(max(np.dot(e2, e2) - x2 * x2, 0.0))
    return np.array([l1, 0.0], dtype=np.float64), np.array([x2, y2], dtype=np.float64)


def symmetric_dirichlet_energy(mesh, uv: np.ndarray) -> float:
    v3 = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    v2 = np.asarray(uv, dtype=np.float64)
    per_face = []
    for tri in f:
        a3, b3, c3 = v3[tri[0]], v3[tri[1]], v3[tri[2]]
        a2, b2, c2 = v2[tri[0]], v2[tri[1]], v2[tri[2]]
        q1, q2 = triangle_local_coords(a3, b3, c3)
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


def avg_energy(patches) -> float:
    vals = []
    for p in patches:
        uv = p.metadata.get("uv")
        if uv is not None:
            vals.append(symmetric_dirichlet_energy(p, uv))
    return float(np.mean(vals)) if vals else float("inf")


def boundary_length_uv(mesh, uv: np.ndarray) -> float:
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


def avg_seam_length(patches) -> float:
    vals = []
    for p in patches:
        uv = p.metadata.get("uv")
        if uv is not None:
            vals.append(boundary_length_uv(p, uv))
    return float(np.mean(vals)) if vals else float("inf")


def improvement_rate(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1e-12:
        return float("nan")
    return float((before - after) / before)


def jacobian_stability_stats(mesh, uv: np.ndarray) -> Dict[str, object]:
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
        q1, q2 = triangle_local_coords(a3, b3, c3)
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


def agg_geo_stability(patches) -> Dict[str, float]:
    all_cond, all_neg, all_det = [], [], []
    for p in patches:
        uv = p.metadata.get("uv")
        if uv is None:
            continue
        st = jacobian_stability_stats(p, uv)
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


def atlas_overlap_ratio(uv_list: List[np.ndarray]) -> float:
    if not uv_list:
        return 0.0
    overlap = UVAtlasMetrics.atlas_bbox_overlap_area(uv_list)
    mins = np.min(np.stack([uv.min(axis=0) for uv in uv_list], axis=0), axis=0)
    maxs = np.max(np.stack([uv.max(axis=0) for uv in uv_list], axis=0), axis=0)
    atlas_area = max(1e-12, float((maxs[0] - mins[0]) * (maxs[1] - mins[1])))
    return float(overlap / atlas_area)


def atlas_min_gap_stats(uv_list: List[np.ndarray]) -> Dict[str, float]:
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


def agg_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
    return {
        "mean": float(np.mean([x["mean"] for x in stats_list])),
        "max": float(np.mean([x["max"] for x in stats_list])),
        "p95": float(np.mean([x["p95"] for x in stats_list])),
    }


def quality_block(patches, patch_gap: float) -> Dict[str, object]:
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
        "distortion": agg_stats(dist),
        "angle_distortion": agg_stats(ang),
        "area_distortion": agg_stats(area),
        "flip_rate": float(np.mean(flips)) if flips else 1.0,
        "atlas_utilization": UVAtlasMetrics.atlas_utilization(uv_list),
        "atlas_overlap_ratio": atlas_overlap_ratio(uv_list),
        "atlas_min_gap_stats": atlas_min_gap_stats(uv_list),
        "padding_violations": UVAtlasMetrics.padding_violations(uv_list, padding=patch_gap),
        "geometric_stability": agg_geo_stability(patches),
    }


def rasterize_feature_maps(patches, size: int = 256, return_patch_coverage: bool = False):
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


def atlas_trainability_metrics(atlas: np.ndarray, patch_coverages: Optional[List[float]] = None) -> Dict[str, float]:
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
