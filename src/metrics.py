from __future__ import annotations

from typing import Dict, List

import numpy as np
import trimesh


class UVAtlasMetrics:
    @staticmethod
    def distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}

        f = mesh.faces
        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        v2 = np.asarray(uv, dtype=np.float64)

        e3 = np.stack([
            np.linalg.norm(v3[f[:, 1]] - v3[f[:, 0]], axis=1),
            np.linalg.norm(v3[f[:, 2]] - v3[f[:, 1]], axis=1),
            np.linalg.norm(v3[f[:, 0]] - v3[f[:, 2]], axis=1),
        ], axis=1)
        e2 = np.stack([
            np.linalg.norm(v2[f[:, 1]] - v2[f[:, 0]], axis=1),
            np.linalg.norm(v2[f[:, 2]] - v2[f[:, 1]], axis=1),
            np.linalg.norm(v2[f[:, 0]] - v2[f[:, 2]], axis=1),
        ], axis=1)

        ratio = e2 / np.maximum(e3, 1e-8)
        per_face = np.abs(np.log(np.maximum(ratio, 1e-8))).mean(axis=1)
        return {
            "mean": float(np.mean(per_face)),
            "max": float(np.max(per_face)),
            "p95": float(np.percentile(per_face, 95.0)),
        }

    @staticmethod
    def flip_rate(mesh: trimesh.Trimesh, uv: np.ndarray) -> float:
        if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
            return 1.0
        f = np.asarray(mesh.faces, dtype=np.int64)
        tri = np.asarray(uv, dtype=np.float64)[f]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        signed2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        flips = np.count_nonzero(signed2 <= 1e-14)
        return float(flips / max(1, len(f)))

    @staticmethod
    def angle_distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        f = np.asarray(mesh.faces, dtype=np.int64)
        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        v2 = np.asarray(uv, dtype=np.float64)

        tri3 = v3[f]
        tri2 = v2[f]

        def _angles(tri):
            a = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
            b = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
            c = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
            cosA = np.clip((b * b + c * c - a * a) / np.maximum(2 * b * c, 1e-12), -1.0, 1.0)
            cosB = np.clip((a * a + c * c - b * b) / np.maximum(2 * a * c, 1e-12), -1.0, 1.0)
            A = np.arccos(cosA)
            B = np.arccos(cosB)
            C = np.maximum(np.pi - A - B, 0.0)
            return np.stack([A, B, C], axis=1)

        ang3 = _angles(tri3)
        ang2 = _angles(tri2)
        err = np.abs(ang2 - ang3).mean(axis=1)
        return {"mean": float(np.mean(err)), "max": float(np.max(err)), "p95": float(np.percentile(err, 95.0))}

    @staticmethod
    def area_distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        f = np.asarray(mesh.faces, dtype=np.int64)
        tri3 = np.asarray(mesh.vertices, dtype=np.float64)[f]
        tri2 = np.asarray(uv, dtype=np.float64)[f]
        a3 = 0.5 * np.linalg.norm(np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1)
        s2 = (tri2[:, 1, 0] - tri2[:, 0, 0]) * (tri2[:, 2, 1] - tri2[:, 0, 1]) - (tri2[:, 1, 1] - tri2[:, 0, 1]) * (tri2[:, 2, 0] - tri2[:, 0, 0])
        a2 = 0.5 * np.abs(s2)
        ratio = a2 / np.maximum(a3, 1e-12)
        err = np.abs(np.log(np.maximum(ratio, 1e-12)))
        return {"mean": float(np.mean(err)), "max": float(np.max(err)), "p95": float(np.percentile(err, 95.0))}

    @staticmethod
    def atlas_bbox_overlap_area(uv_list: List[np.ndarray]) -> float:
        total = 0.0
        for i in range(len(uv_list)):
            for j in range(i + 1, len(uv_list)):
                a_min, a_max = uv_list[i].min(axis=0), uv_list[i].max(axis=0)
                b_min, b_max = uv_list[j].min(axis=0), uv_list[j].max(axis=0)
                ox = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
                oy = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
                if ox > 0 and oy > 0:
                    total += float(ox * oy)
        return total

    @staticmethod
    def padding_violations(uv_list: List[np.ndarray], padding: float) -> int:
        violations = 0
        for i in range(len(uv_list)):
            for j in range(i + 1, len(uv_list)):
                a_min, a_max = uv_list[i].min(axis=0), uv_list[i].max(axis=0)
                b_min, b_max = uv_list[j].min(axis=0), uv_list[j].max(axis=0)
                dx = max(a_min[0] - b_max[0], b_min[0] - a_max[0], 0.0)
                dy = max(a_min[1] - b_max[1], b_min[1] - a_max[1], 0.0)
                sep = np.hypot(dx, dy)
                if sep < padding:
                    violations += 1
        return violations

    @staticmethod
    def atlas_utilization(uv_list: List[np.ndarray]) -> float:
        if not uv_list:
            return 0.0
        box_areas = []
        mins = []
        maxs = []
        for uv in uv_list:
            uv_min, uv_max = uv.min(axis=0), uv.max(axis=0)
            mins.append(uv_min)
            maxs.append(uv_max)
            box_areas.append(max(0.0, float((uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1]))))
        atlas_min = np.min(np.stack(mins, axis=0), axis=0)
        atlas_max = np.max(np.stack(maxs, axis=0), axis=0)
        atlas_area = max(1e-12, float((atlas_max[0] - atlas_min[0]) * (atlas_max[1] - atlas_min[1])))
        overlap = UVAtlasMetrics.atlas_bbox_overlap_area(uv_list)
        effective = max(0.0, float(np.sum(box_areas) - overlap))
        return float(min(1.0, effective / atlas_area))
