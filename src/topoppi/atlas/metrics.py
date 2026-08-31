"""Scale-fair, area-weighted UV and polygonal atlas geometry metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh
from shapely.errors import ShapelyError
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from topoppi.atlas.seams import uv_seam_topology
from topoppi.atlas.uv import (
    as_corner_uv,
    canonical_geometry_corner_uv,
    corner_to_vertex_uv,
)
from topoppi.mesh.provenance import OPTCUTS_GEOMETRY_VERTEX_IDS, SOURCE_VERTEX_IDS

_EPS = 1e-12


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("inf")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    threshold = float(percentile) / 100.0 * float(np.sum(weights))
    index = min(int(np.searchsorted(np.cumsum(weights), threshold, side="left")), len(values) - 1)
    return float(values[index])


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive_weight = np.isfinite(weights) & (weights > 0.0)
    if np.any(positive_weight & ~np.isfinite(values)):
        invalid_area = float(np.sum(weights[positive_weight & ~np.isfinite(values)]))
        total_area = max(float(np.sum(weights[positive_weight])), _EPS)
        return {
            "mean": float("inf"),
            "max": float("inf"),
            "p95": float("inf"),
            "invalid_area_ratio": float(invalid_area / total_area),
        }
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return {
            "mean": float("inf"),
            "max": float("inf"),
            "p95": float("inf"),
            "invalid_area_ratio": 1.0,
        }
    values = values[valid]
    weights = weights[valid]
    return {
        "mean": float(np.average(values, weights=weights)),
        "max": float(np.max(values)),
        "p95": weighted_percentile(values, weights, 95.0),
        "invalid_area_ratio": 0.0,
    }


def _triangle_areas_3d(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(mesh.area_faces, dtype=np.float64)


def _signed_twice_area_2d(corners: np.ndarray) -> np.ndarray:
    return (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1]) - (
        corners[:, 1, 1] - corners[:, 0, 1]
    ) * (corners[:, 2, 0] - corners[:, 0, 0])


def similarity_aligned_corner_uv(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply one uniform scale so total 2-D area equals total 3-D area."""

    raw_corners = as_corner_uv(mesh, uv).copy()
    corners, coordinate_scale = canonical_geometry_corner_uv(mesh, raw_corners)
    area3 = float(np.sum(_triangle_areas_3d(mesh)))
    area2 = float(np.sum(0.5 * np.abs(_signed_twice_area_2d(corners))))
    if area3 <= _EPS or area2 <= _EPS or not np.isfinite(coordinate_scale):
        return raw_corners, float("nan")
    normalized_scale = float(np.sqrt(area3 / area2))
    center = np.mean(corners.reshape(-1, 2), axis=0)
    return (
        (corners - center) * normalized_scale,
        normalized_scale / coordinate_scale,
    )


def face_jacobians(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    *,
    align_total_area: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-face Jacobians, 3-D area weights, and validity.

    Reported atlas metrics use one global total-area similarity alignment.
    Setting ``align_total_area=False`` retains the raw UV scale and is used to
    reproduce OptCuts' native constrained symmetric-Dirichlet objective.
    """

    corners = similarity_aligned_corner_uv(mesh, uv)[0] if align_total_area else as_corner_uv(mesh, uv).copy()
    tri3 = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    areas = _triangle_areas_3d(mesh)
    edge1 = tri3[:, 1] - tri3[:, 0]
    edge2 = tri3[:, 2] - tri3[:, 0]
    length1 = np.linalg.norm(edge1, axis=1)
    local_x = np.zeros(len(tri3), dtype=np.float64)
    nonzero_edge = length1 > _EPS
    local_x[nonzero_edge] = np.einsum(
        "ij,ij->i",
        edge2[nonzero_edge],
        edge1[nonzero_edge] / length1[nonzero_edge, None],
    )
    local_y_sq = np.einsum("ij,ij->i", edge2, edge2) - local_x * local_x
    valid = nonzero_edge & (local_y_sq > _EPS)

    jacobians = np.full((len(tri3), 2, 2), np.nan, dtype=np.float64)
    indices = np.flatnonzero(valid)
    if len(indices):
        local_y = np.sqrt(local_y_sq[indices])
        inverse_source = np.zeros((len(indices), 2, 2), dtype=np.float64)
        inverse_source[:, 0, 0] = 1.0 / length1[indices]
        inverse_source[:, 0, 1] = -local_x[indices] / (length1[indices] * local_y)
        inverse_source[:, 1, 1] = 1.0 / local_y
        target = np.stack(
            [corners[indices, 1] - corners[indices, 0], corners[indices, 2] - corners[indices, 0]],
            axis=2,
        )
        jacobians[indices] = target @ inverse_source
        valid[indices] = np.isfinite(jacobians[indices]).all(axis=(1, 2))
    return jacobians, areas, valid


class UVAtlasMetrics:
    """Metrics whose means are weighted by original 3-D face area."""

    @staticmethod
    def symmetric_dirichlet_samples(
        mesh: trimesh.Trimesh,
        uv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-face energy at the optimal global similarity scale."""

        jacobians, weights, valid = face_jacobians(mesh, uv)
        values = np.full(len(jacobians), float("inf"), dtype=np.float64)
        indices = np.flatnonzero(valid)
        if len(indices):
            singular = np.linalg.svd(jacobians[indices], compute_uv=False)
            nonsingular = np.min(singular, axis=1) > _EPS
            stable_indices = indices[nonsingular]
            stable = singular[nonsingular]
            stable_weights = weights[stable_indices]
            forward = float(np.sum(stable_weights[:, None] * stable**2))
            inverse = float(np.sum(stable_weights[:, None] * stable**-2))
            if forward > _EPS and inverse > _EPS:
                scale = float((inverse / forward) ** 0.25)
                scaled = scale * stable
                values[stable_indices] = 0.5 * np.sum(scaled**2 + scaled**-2, axis=1)
        return values, weights

    @staticmethod
    def symmetric_dirichlet_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        values, weights = UVAtlasMetrics.symmetric_dirichlet_samples(mesh, uv)
        result = weighted_stats(values, weights)
        result.update(
            {
                "area_weighted": True,
                "scale_alignment": "analytic_global_symmetric_dirichlet_minimum",
                "identity_value": 2.0,
            }
        )
        return result

    @staticmethod
    def optcuts_constraint_energy(mesh: trimesh.Trimesh, uv: np.ndarray) -> float:
        """Recompute OptCuts' raw-scale symmetric-Dirichlet constraint.

        This follows the pinned upstream implementation: each face contributes
        ``||J||_F^2 + ||J^-1||_F^2``, weighted by its original 3-D area and
        normalized by total area.  Its identity value is 4, whereas TopoPPI's
        scale-fair reporting convention divides the same expression by two and
        analytically optimizes one global similarity scale.
        """

        jacobians, weights, valid = face_jacobians(
            mesh,
            uv,
            align_total_area=False,
        )
        positive_area = np.isfinite(weights) & (weights > 0.0)
        if not np.any(positive_area) or np.any(positive_area & ~valid):
            return float("inf")
        indices = np.flatnonzero(positive_area)
        singular = np.linalg.svd(jacobians[indices], compute_uv=False)
        if not np.isfinite(singular).all() or np.any(np.min(singular, axis=1) <= _EPS):
            return float("inf")
        per_face = np.sum(singular**2 + singular**-2, axis=1)
        return float(np.average(per_face, weights=weights[indices]))

    @staticmethod
    def distortion_samples(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jacobians, weights, valid = face_jacobians(mesh, uv)
        values = np.full(len(jacobians), float("inf"), dtype=np.float64)
        indices = np.flatnonzero(valid)
        if len(indices):
            singular = np.linalg.svd(jacobians[indices], compute_uv=False)
            nonsingular = np.min(singular, axis=1) > _EPS
            values[indices[nonsingular]] = np.mean(np.abs(np.log(singular[nonsingular])), axis=1)
        return values, weights

    @staticmethod
    def distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        values, weights = UVAtlasMetrics.distortion_samples(mesh, uv)
        result = weighted_stats(values, weights)
        result.update({"area_weighted": True, "scale_alignment": "global_total_area_similarity"})
        return result

    @staticmethod
    def flip_rate(mesh: trimesh.Trimesh, uv: np.ndarray) -> float:
        if uv is None or len(mesh.faces) == 0:
            return 1.0
        flipped, weights = UVAtlasMetrics.flip_samples(mesh, uv)
        return float(np.sum(weights[flipped]) / max(float(np.sum(weights)), _EPS))

    @staticmethod
    def flip_samples(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return reflection-corrected per-face flip flags and 3-D weights."""

        if uv is None or len(mesh.faces) == 0:
            return np.ones(len(mesh.faces), dtype=bool), _triangle_areas_3d(mesh)
        corners, predicate_scale = canonical_geometry_corner_uv(mesh, uv)
        if not np.isfinite(predicate_scale):
            return np.ones(len(mesh.faces), dtype=bool), _triangle_areas_3d(mesh)
        signed = _signed_twice_area_2d(corners)
        weights = _triangle_areas_3d(mesh)
        nondegenerate = np.abs(signed) > 1e-14
        if not np.any(nondegenerate):
            return np.ones(len(mesh.faces), dtype=bool), weights
        orientation_score = float(np.sum(np.sign(signed[nondegenerate]) * weights[nondegenerate]))
        global_orientation = 1.0 if orientation_score >= 0.0 else -1.0
        flipped = (global_orientation * signed) <= 1e-14
        return flipped, weights

    @staticmethod
    def angle_distortion_samples(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        tri3 = np.asarray(mesh.vertices, dtype=np.float64)[faces]
        tri2, uv_coordinate_scale = canonical_geometry_corner_uv(mesh, uv)

        def _angles(tri):
            a = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
            b = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
            c = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
            cos_a = np.clip((b * b + c * c - a * a) / np.maximum(2.0 * b * c, _EPS), -1.0, 1.0)
            cos_b = np.clip((a * a + c * c - b * b) / np.maximum(2.0 * a * c, _EPS), -1.0, 1.0)
            angle_a = np.arccos(cos_a)
            angle_b = np.arccos(cos_b)
            angle_c = np.maximum(np.pi - angle_a - angle_b, 0.0)
            return np.stack([angle_a, angle_b, angle_c], axis=1)

        source_edges = np.stack(
            [
                np.linalg.norm(tri3[:, 1] - tri3[:, 2], axis=1),
                np.linalg.norm(tri3[:, 2] - tri3[:, 0], axis=1),
                np.linalg.norm(tri3[:, 0] - tri3[:, 1], axis=1),
            ],
            axis=1,
        )
        target_edges = np.stack(
            [
                np.linalg.norm(tri2[:, 1] - tri2[:, 2], axis=1),
                np.linalg.norm(tri2[:, 2] - tri2[:, 0], axis=1),
                np.linalg.norm(tri2[:, 0] - tri2[:, 1], axis=1),
            ],
            axis=1,
        )
        source_twice_area = np.linalg.norm(
            np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]),
            axis=1,
        )
        target_twice_area = np.abs(_signed_twice_area_2d(tri2))
        valid = (
            (np.min(source_edges, axis=1) > _EPS)
            & (np.min(target_edges, axis=1) > _EPS)
            & (source_twice_area > _EPS)
            & (target_twice_area > _EPS)
            & np.isfinite(source_edges).all(axis=1)
            & np.isfinite(target_edges).all(axis=1)
            & np.isfinite(uv_coordinate_scale)
        )
        error = np.full(len(tri3), float("inf"), dtype=np.float64)
        angle_error = np.mean(np.abs(_angles(tri2) - _angles(tri3)), axis=1)
        error[valid] = angle_error[valid]
        return error, _triangle_areas_3d(mesh)

    @staticmethod
    def angle_distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        values, weights = UVAtlasMetrics.angle_distortion_samples(mesh, uv)
        result = weighted_stats(values, weights)
        result.update({"unit": "radian", "area_weighted": True})
        return result

    @staticmethod
    def area_distortion_samples(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        corners, _scale = similarity_aligned_corner_uv(mesh, uv)
        area3 = _triangle_areas_3d(mesh)
        area2 = 0.5 * np.abs(_signed_twice_area_2d(corners))
        values = np.full(len(area3), float("inf"), dtype=np.float64)
        valid = (area2 > _EPS) & (area3 > _EPS) & np.isfinite(area2) & np.isfinite(area3)
        values[valid] = np.abs(np.log(area2[valid] / area3[valid]))
        return values, area3

    @staticmethod
    def area_distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, float]:
        if uv is None or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf"), "p95": float("inf")}
        values, weights = UVAtlasMetrics.area_distortion_samples(mesh, uv)
        result = weighted_stats(values, weights)
        result.update({"area_weighted": True, "scale_alignment": "global_total_area_similarity"})
        return result

    @staticmethod
    def seam_stats(mesh: trimesh.Trimesh, uv: np.ndarray, atol: float = 1e-9) -> Dict[str, float | int]:
        """Measure actual discontinuities on internal edges, separately from boundary."""

        topology = uv_seam_topology(mesh, uv, atol=atol)
        seam_count = int(np.count_nonzero(topology.seam_mask))
        seam_length_3d = float(np.sum(topology.edge_lengths_3d[topology.seam_mask]))
        seam_length_uv_two_sided = float(np.sum(topology.edge_uv_length_sum[topology.seam_mask]))
        boundary_count = int(np.count_nonzero(topology.boundary_mask))
        boundary_length_3d = float(np.sum(topology.edge_lengths_3d[topology.boundary_mask]))

        area_scale = np.sqrt(max(float(mesh.area), _EPS))
        return {
            "seam_edge_count": int(seam_count),
            "seam_length_3d": float(seam_length_3d),
            "seam_length_3d_normalized": float(seam_length_3d / area_scale),
            "seam_length_uv_two_sided": float(seam_length_uv_two_sided),
            "boundary_edge_count": int(boundary_count),
            "boundary_length_3d": float(boundary_length_3d),
            "boundary_length_3d_normalized": float(boundary_length_3d / area_scale),
        }

    @staticmethod
    def _triangle_polygons(mesh: trimesh.Trimesh, uv: np.ndarray) -> List[Polygon]:
        polygons = []
        for triangle in as_corner_uv(mesh, uv):
            if not np.isfinite(triangle).all():
                continue
            polygon = Polygon(triangle)
            if polygon.is_valid and polygon.area > 0.0:
                polygons.append(polygon)
        return polygons

    @staticmethod
    def _pairwise_overlap_geometry(geometries):
        """Return the union of positive-area pairwise intersections."""

        if len(geometries) < 2:
            return GeometryCollection()
        tree = STRtree(geometries)
        indices_by_identity = {id(geometry): index for index, geometry in enumerate(geometries)}
        intersections = []
        for left, geometry in enumerate(geometries):
            candidates = tree.query(geometry)
            for candidate in candidates:
                if isinstance(candidate, (int, np.integer)):
                    right = int(candidate)
                    other = geometries[right]
                else:  # Shapely 1.8 returns geometry objects.
                    right = indices_by_identity[id(candidate)]
                    other = candidate
                if right <= left:
                    continue
                intersection = geometry.intersection(other)
                if not intersection.is_empty and float(intersection.area) > _EPS:
                    intersections.append(intersection)
        return unary_union(intersections) if intersections else GeometryCollection()

    @staticmethod
    def _source_distinct_zero_measure_contact_count(
        mesh: trimesh.Trimesh,
        corners: np.ndarray,
        polygons: Sequence[Polygon],
        *,
        coordinate_tolerance: float = 1e-10,
    ) -> int:
        """Count point/line contacts that do not identify the same source simplex.

        Positive-area intersections are handled by the overdraw certificate.  A
        strict piecewise-linear injectivity claim must additionally reject an
        edge crossing or point contact between different source locations.  The
        same source vertex or source edge remains a legal contact, including at
        a topology-cut endpoint represented by duplicated mesh vertices.
        """

        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(polygons) != len(faces):
            raise ValueError("Zero-measure contact audit requires one valid polygon per face.")
        identity_key = (
            OPTCUTS_GEOMETRY_VERTEX_IDS if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata else SOURCE_VERTEX_IDS
        )
        source_vertices = np.asarray(
            mesh.metadata.get(identity_key, np.arange(len(mesh.vertices))),
            dtype=np.int64,
        )
        if source_vertices.shape != (len(mesh.vertices),):
            raise ValueError(f"{identity_key} must contain one ID per mesh vertex.")
        face_sources = source_vertices[faces]
        tree = STRtree(polygons)
        indices_by_identity = {id(geometry): index for index, geometry in enumerate(polygons)}
        forbidden = 0
        for left, geometry in enumerate(polygons):
            for candidate in tree.query(geometry):
                if isinstance(candidate, (int, np.integer)):
                    right = int(candidate)
                    other = polygons[right]
                else:  # Shapely 1.8 returns geometry objects.
                    right = indices_by_identity[id(candidate)]
                    other = candidate
                if right <= left:
                    continue
                intersection = geometry.intersection(other)
                if intersection.is_empty or float(intersection.area) > 0.0:
                    continue

                left_uv_by_source = {
                    int(source): corners[left, corner] for corner, source in enumerate(face_sources[left])
                }
                right_uv_by_source = {
                    int(source): corners[right, corner] for corner, source in enumerate(face_sources[right])
                }
                shared = sorted(set(left_uv_by_source) & set(right_uv_by_source))
                matched = [
                    source
                    for source in shared
                    if np.linalg.norm(left_uv_by_source[source] - right_uv_by_source[source]) <= coordinate_tolerance
                ]
                allowed_parts = [
                    Point(0.5 * (left_uv_by_source[source] + right_uv_by_source[source])) for source in matched
                ]
                if len(matched) >= 2:
                    for first_index, first in enumerate(matched):
                        for second in matched[first_index + 1 :]:
                            allowed_parts.append(
                                LineString(
                                    [
                                        0.5 * (left_uv_by_source[first] + right_uv_by_source[first]),
                                        0.5 * (left_uv_by_source[second] + right_uv_by_source[second]),
                                    ]
                                )
                            )
                if not allowed_parts:
                    forbidden += 1
                    continue
                allowed = unary_union(allowed_parts)
                residual = intersection.difference(allowed.buffer(coordinate_tolerance))
                forbidden += int(not residual.is_empty)
        return int(forbidden)

    @staticmethod
    def _continuous_on_pre_cut_geometry_domain(
        mesh: trimesh.Trimesh,
        corners: np.ndarray,
        *,
        atol: float = 1e-9,
    ) -> bool:
        """Test continuity after identifying topology-cut vertex copies.

        ``OPTCUTS_GEOMETRY_VERTEX_IDS`` names vertices on the repaired 3-D
        domain before diskification or later seam edits duplicate them. A map
        is continuous on that domain exactly when every occurrence of one such
        vertex has the same normalized UV coordinate. Copies made earlier to
        separate disconnected vertex fans intentionally retain distinct IDs.
        """

        normalized, scale = canonical_geometry_corner_uv(mesh, corners)
        if not np.isfinite(scale):
            return False
        geometry_vertices = np.asarray(
            mesh.metadata.get(
                OPTCUTS_GEOMETRY_VERTEX_IDS,
                np.arange(len(mesh.vertices), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        if geometry_vertices.shape != (len(mesh.vertices),):
            raise ValueError("optcuts_geometry_vertex_ids must contain one ID per mesh vertex.")
        occurrence_ids = geometry_vertices[np.asarray(mesh.faces, dtype=np.int64)].reshape(-1)
        occurrence_uv = normalized.reshape(-1, 2)
        unique_ids, inverse = np.unique(occurrence_ids, return_inverse=True)
        minima = np.full((len(unique_ids), 2), np.inf, dtype=np.float64)
        maxima = np.full((len(unique_ids), 2), -np.inf, dtype=np.float64)
        np.minimum.at(minima, inverse, occurrence_uv)
        np.maximum.at(maxima, inverse, occurrence_uv)
        return bool(np.all(maxima - minima <= float(atol)))

    @staticmethod
    def atlas_geometry_stats(
        meshes: Sequence[trimesh.Trimesh],
        *,
        key: str = "uv_global",
        padding: float = 0.0,
        uv_arrays: Sequence[np.ndarray] | None = None,
    ) -> Dict[str, float | int | bool]:
        if uv_arrays is not None and len(uv_arrays) != len(meshes):
            raise ValueError("uv_arrays must contain one UV array per mesh.")
        chart_geometries = []
        triangle_area_sum = 0.0
        within_chart_overlap_geometries = []
        within_chart_overdraw_area = 0.0
        invalid_triangle_polygon_count = 0
        try:
            for index, mesh in enumerate(meshes):
                corners = as_corner_uv(mesh, uv_arrays[index]) if uv_arrays is not None else as_corner_uv(mesh, key=key)
                polygons = UVAtlasMetrics._triangle_polygons(mesh, corners)
                invalid_triangle_polygon_count += int(len(corners) - len(polygons))
                chart_triangle_area = float(sum(polygon.area for polygon in polygons))
                chart_geometry = unary_union(polygons) if polygons else Polygon()
                chart_geometries.append(chart_geometry)
                triangle_area_sum += chart_triangle_area
                chart_overdraw = max(0.0, chart_triangle_area - float(chart_geometry.area))
                within_chart_overdraw_area += chart_overdraw
                chart_tolerance = 1e-10 * max(chart_triangle_area, 1.0)
                if chart_overdraw >= chart_tolerance:
                    overlap_geometry = UVAtlasMetrics._pairwise_overlap_geometry(polygons)
                    if not overlap_geometry.is_empty:
                        within_chart_overlap_geometries.append(overlap_geometry)
        except ShapelyError as exc:
            return UVAtlasMetrics._geometry_failure_stats(
                len(meshes),
                invalid_triangle_polygon_count,
                exc,
            )

        nonempty = [geometry for geometry in chart_geometries if not geometry.is_empty]
        if not nonempty:
            return {
                "chart_count": int(len(meshes)),
                "covered_area": 0.0,
                "atlas_bbox_area": 0.0,
                "utilization": 0.0,
                "waste_ratio": 1.0,
                "overlap_area": 0.0,
                "within_chart_overlap_area": 0.0,
                "between_chart_overlap_area": 0.0,
                "overlap_ratio": 0.0,
                "overdraw_area": 0.0,
                "within_chart_overdraw_area": 0.0,
                "between_chart_overdraw_area": 0.0,
                "overdraw_ratio": 0.0,
                "padding_violations": 0,
                "min_chart_gap": float("nan"),
                "single_chart": len(meshes) == 1,
                "invalid_triangle_polygon_count": int(invalid_triangle_polygon_count),
                "numeric_area_tolerance": 1e-10,
                "geometry_evaluation_status": "ok",
            }

        numeric_tolerance = 1e-10 * max(triangle_area_sum, 1.0)
        try:
            combined = unary_union(nonempty)
            covered_area = float(combined.area)
            min_x, min_y, max_x, max_y = combined.bounds
            bbox_area = max(0.0, float((max_x - min_x) * (max_y - min_y)))
            between_chart_overdraw_area = max(
                0.0,
                float(sum(float(geometry.area) for geometry in nonempty) - covered_area),
            )
            between_chart_overlap_geometry = (
                UVAtlasMetrics._pairwise_overlap_geometry(nonempty)
                if between_chart_overdraw_area >= numeric_tolerance
                else GeometryCollection()
            )
            within_chart_overlap_geometry = (
                unary_union(within_chart_overlap_geometries)
                if within_chart_overlap_geometries
                else GeometryCollection()
            )
            overlap_parts = [
                geometry
                for geometry in (within_chart_overlap_geometry, between_chart_overlap_geometry)
                if not geometry.is_empty
            ]
            overlap_geometry = unary_union(overlap_parts) if overlap_parts else GeometryCollection()
            overlap_area = float(overlap_geometry.area)
            within_chart_overlap_area = float(within_chart_overlap_geometry.area)
            between_chart_overlap_area = float(between_chart_overlap_geometry.area)
            overdraw_area = max(0.0, triangle_area_sum - covered_area)
        except ShapelyError as exc:
            return UVAtlasMetrics._geometry_failure_stats(
                len(meshes),
                invalid_triangle_polygon_count,
                exc,
            )
        if overlap_area < numeric_tolerance:
            overlap_area = 0.0
        if within_chart_overlap_area < numeric_tolerance:
            within_chart_overlap_area = 0.0
        if between_chart_overlap_area < numeric_tolerance:
            between_chart_overlap_area = 0.0
        if overdraw_area < numeric_tolerance:
            overdraw_area = 0.0
        if within_chart_overdraw_area < numeric_tolerance:
            within_chart_overdraw_area = 0.0
        if between_chart_overdraw_area < numeric_tolerance:
            between_chart_overdraw_area = 0.0
        pair_distances = []
        padding_violations = 0
        try:
            for left in range(len(chart_geometries)):
                for right in range(left + 1, len(chart_geometries)):
                    a, b = chart_geometries[left], chart_geometries[right]
                    if a.is_empty or b.is_empty:
                        continue
                    distance = float(a.distance(b))
                    pair_distances.append(distance)
                    if a.intersects(b) or distance + 1e-12 < padding:
                        padding_violations += 1
        except ShapelyError as exc:
            return UVAtlasMetrics._geometry_failure_stats(
                len(meshes),
                invalid_triangle_polygon_count,
                exc,
            )
        return {
            "chart_count": int(len(meshes)),
            "covered_area": covered_area,
            "atlas_bbox_area": bbox_area,
            "utilization": float(covered_area / bbox_area) if bbox_area > _EPS else 0.0,
            "waste_ratio": float(1.0 - covered_area / bbox_area) if bbox_area > _EPS else 1.0,
            "overlap_area": overlap_area,
            "within_chart_overlap_area": float(within_chart_overlap_area),
            "between_chart_overlap_area": float(between_chart_overlap_area),
            "overlap_ratio": float(overlap_area / triangle_area_sum) if triangle_area_sum > _EPS else 0.0,
            "overdraw_area": float(overdraw_area),
            "within_chart_overdraw_area": float(within_chart_overdraw_area),
            "between_chart_overdraw_area": float(between_chart_overdraw_area),
            "overdraw_ratio": float(overdraw_area / triangle_area_sum) if triangle_area_sum > _EPS else 0.0,
            "padding_violations": int(padding_violations),
            "min_chart_gap": float(min(pair_distances)) if pair_distances else float("nan"),
            "single_chart": len(meshes) == 1,
            "invalid_triangle_polygon_count": int(invalid_triangle_polygon_count),
            "numeric_area_tolerance": float(numeric_tolerance),
            "geometry_evaluation_status": "ok",
        }

    @staticmethod
    def _geometry_failure_stats(
        chart_count: int,
        invalid_triangle_polygon_count: int,
        error: ShapelyError,
    ) -> Dict[str, float | int | bool | str]:
        """Represent a polygonal-geometry failure conservatively, without data loss."""

        return {
            "status": "geometry_evaluation_failed",
            "geometry_evaluation_status": "failed",
            "geometry_error_type": type(error).__name__,
            "chart_count": int(chart_count),
            "covered_area": float("nan"),
            "atlas_bbox_area": float("nan"),
            "utilization": float("nan"),
            "waste_ratio": float("nan"),
            "overlap_area": float("inf"),
            "within_chart_overlap_area": float("inf"),
            "between_chart_overlap_area": float("inf"),
            "overlap_ratio": float("inf"),
            "overdraw_area": float("inf"),
            "within_chart_overdraw_area": float("inf"),
            "between_chart_overdraw_area": float("inf"),
            "overdraw_ratio": float("inf"),
            "padding_violations": -1,
            "min_chart_gap": float("nan"),
            "single_chart": int(chart_count) == 1,
            "invalid_triangle_polygon_count": int(invalid_triangle_polygon_count),
            "numeric_area_tolerance": float("nan"),
        }

    @staticmethod
    def parameterization_injectivity_stats(
        mesh: trimesh.Trimesh,
        uv: np.ndarray,
    ) -> Dict[str, float | int | bool]:
        """Check local orientation and polygonal positive-area atlas overdraw.

        A locally orientation-preserving map can still overlap itself.  The
        triangle union, evaluated in normalized double precision with the
        reported area tolerance, complements the Jacobian-sign test before a
        map is used with a global bijectivity constraint.
        """

        source_corners = as_corner_uv(mesh, uv)
        corners, predicate_scale = canonical_geometry_corner_uv(mesh, source_corners)
        finite = bool(np.isfinite(source_corners).all() and np.isfinite(predicate_scale))
        global_reflection_required = False
        if finite:
            signed_area = _signed_twice_area_2d(corners)
            nondegenerate = np.abs(signed_area) > 1e-14
            if np.any(nondegenerate):
                orientation_score = float(
                    np.sum(np.sign(signed_area[nondegenerate]) * _triangle_areas_3d(mesh)[nondegenerate])
                )
                global_reflection_required = orientation_score < 0.0
            flipped, weights = UVAtlasMetrics.flip_samples(mesh, corners)
            flip_face_count = int(np.count_nonzero(flipped))
            flip_rate = float(np.sum(weights[flipped]) / max(float(np.sum(weights)), _EPS))
            geometry_evaluation_status = "ok"
            geometry_error_type = None
            try:
                polygons = UVAtlasMetrics._triangle_polygons(mesh, corners)
                invalid_polygon_count = int(len(corners) - len(polygons))
                triangle_area_sum = float(sum(polygon.area for polygon in polygons))
                covered = unary_union(polygons) if polygons else Polygon()
                numeric_tolerance = max(
                    1e-10 * triangle_area_sum,
                    256.0 * np.finfo(np.float64).eps,
                )
                overdraw_area = max(0.0, triangle_area_sum - float(covered.area))
                if overdraw_area < numeric_tolerance:
                    overdraw_area = 0.0
                    overlap_area = 0.0
                else:
                    overlap_area = float(UVAtlasMetrics._pairwise_overlap_geometry(polygons).area)
                    if overlap_area < numeric_tolerance:
                        overlap_area = 0.0
                zero_measure_contact_count = (
                    UVAtlasMetrics._source_distinct_zero_measure_contact_count(
                        mesh,
                        corners,
                        polygons,
                    )
                    if invalid_polygon_count == 0 and overlap_area == 0.0
                    else 0
                )
            except ShapelyError as exc:
                geometry_evaluation_status = "failed"
                geometry_error_type = type(exc).__name__
                invalid_polygon_count = int(len(corners))
                triangle_area_sum = 0.0
                overlap_area = float("inf")
                overdraw_area = float("inf")
                numeric_tolerance = float("nan")
                zero_measure_contact_count = -1
        else:
            flip_face_count = int(len(mesh.faces))
            flip_rate = 1.0
            invalid_polygon_count = int(len(mesh.faces))
            triangle_area_sum = 0.0
            overlap_area = float("inf")
            overdraw_area = float("inf")
            numeric_tolerance = 1e-10
            geometry_evaluation_status = "not_evaluated_nonfinite_uv"
            geometry_error_type = None
            zero_measure_contact_count = -1
        overlap_ratio = (
            float(overlap_area / triangle_area_sum)
            if triangle_area_sum > _EPS
            else (0.0 if overlap_area == 0.0 else float("inf"))
        )
        overdraw_ratio = (
            float(overdraw_area / triangle_area_sum)
            if triangle_area_sum > _EPS
            else (0.0 if overdraw_area == 0.0 else float("inf"))
        )
        globally_injective = bool(
            finite
            and flip_face_count == 0
            and invalid_polygon_count == 0
            and overlap_area == 0.0
            and overdraw_area == 0.0
            and zero_measure_contact_count == 0
        )
        continuous_on_materialized_cut_mesh = bool(
            corner_to_vertex_uv(mesh, source_corners) is not None if finite else False
        )
        continuous_on_pre_cut_geometry_domain = bool(
            UVAtlasMetrics._continuous_on_pre_cut_geometry_domain(mesh, source_corners) if finite else False
        )
        return {
            "globally_injective": globally_injective,
            "finite": finite,
            "global_reflection_required_for_positive_orientation": global_reflection_required,
            "continuous_on_input_mesh": continuous_on_pre_cut_geometry_domain,
            "continuous_on_pre_cut_geometry_domain": continuous_on_pre_cut_geometry_domain,
            "continuous_on_materialized_cut_mesh": continuous_on_materialized_cut_mesh,
            "continuity_vertex_identity": (
                "optcuts_geometry_vertex_ids"
                if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata
                else "materialized_mesh_vertex_indices"
            ),
            "flip_face_count": flip_face_count,
            "flip_rate_3d_area_weighted": flip_rate,
            "invalid_triangle_polygon_count": invalid_polygon_count,
            "within_chart_overlap_area": float(overlap_area),
            "within_chart_overdraw_area": float(overdraw_area),
            "source_distinct_zero_measure_contact_pair_count": int(zero_measure_contact_count),
            "overlap_ratio": overlap_ratio,
            "overdraw_ratio": overdraw_ratio,
            "numeric_area_tolerance": float(numeric_tolerance),
            "geometry_coordinate_normalization": "translated_uniform_unit_longest_extent",
            "geometry_coordinate_scale": float(predicate_scale),
            "geometry_evaluation_status": geometry_evaluation_status,
            "geometry_error_type": geometry_error_type,
        }


__all__ = [
    "UVAtlasMetrics",
    "face_jacobians",
    "similarity_aligned_corner_uv",
    "weighted_percentile",
    "weighted_stats",
]
