import logging

import igl
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from topoppi.atlas.uv import face_domain_hash, unwrap_periodic_corner_uv
from topoppi.config import ParameterizationConfig
from topoppi.mesh.provenance import (
    OPTCUTS_GEOMETRY_VERTEX_IDS,
    connected_face_components,
    filter_faces,
    initialize_provenance,
    merge_duplicate_vertices,
    provenance_summary,
    remove_duplicate_faces,
    replace_mesh,
    tracked_vertex_duplication,
)

logger = logging.getLogger("LSCM")


class Parameterizer:
    """
    Handles the flattening of 3D meshes into 2D UV coordinates using LSCM
    (Least Squares Conformal Maps), with optional configured fallback methods.
    """

    def __init__(self, config: ParameterizationConfig | None = None):
        self.config = config or ParameterizationConfig()

    def prepare_patch(self, mesh: trimesh.Trimesh, return_info: bool = False):
        """Create one deterministic disk-like domain shared by every method."""

        if mesh.metadata.get("topoppi_parameterization_prepared"):
            info = dict(mesh.metadata.get("parameterization_preparation", {}))
            return (mesh, info) if return_info else mesh

        config = self.config
        info = {
            "diskification_triggered": False,
            "diskification_success": False,
            "topology_before": {},
            "topology_after": {},
            "face_quality_thresholds": {
                "min_area": float(config.min_face_area),
                "min_angle_deg": float(config.min_angle_deg),
                "max_aspect_ratio": float(config.max_aspect_ratio),
            },
            "failure_reason": None,
        }
        try:
            initialize_provenance(mesh, stage="parameterization_input")
            input_summary = provenance_summary(mesh)
            source_vertices = np.asarray(
                mesh.metadata.get("source_vertex_ids", np.arange(len(mesh.vertices))),
                dtype=np.int64,
            )
            preserve_topology_copies = bool(
                mesh.metadata.get("topoppi_topology_sanitized")
                or len(np.unique(source_vertices)) < len(source_vertices)
            )
            working = (
                mesh.copy()
                if preserve_topology_copies
                else merge_duplicate_vertices(
                    mesh,
                    stage="parameterization_merge_duplicate_vertices",
                )
            )
            info["topological_vertex_copies_preserved"] = preserve_topology_copies
            valid_faces = self._face_quality_mask(
                np.ascontiguousarray(working.vertices, dtype=np.float64),
                np.ascontiguousarray(working.faces, dtype=np.int64),
                working.area_faces,
                min_area=config.min_face_area,
                min_angle_deg=config.min_angle_deg,
                max_aspect_ratio=config.max_aspect_ratio,
            )
            info["low_quality_face_count"] = int(np.count_nonzero(~valid_faces))
            if not np.all(valid_faces):
                working = filter_faces(working, valid_faces, stage="parameterization_remove_low_quality_faces")
            working = remove_duplicate_faces(working, stage="parameterization_remove_duplicate_faces")
            component_count = len(connected_face_components(working))
            info["connected_component_count_after_cleanup"] = int(component_count)
            if component_count != 1:
                info["failure_reason"] = "mesh_disconnected_after_cleanup"
                return (None, info) if return_info else None
            if len(working.vertices) < config.min_vertices or len(working.faces) == 0:
                info["failure_reason"] = "mesh_degenerate_after_cleanup"
                return (None, info) if return_info else None
            if OPTCUTS_GEOMETRY_VERTEX_IDS not in working.metadata:
                # These IDs describe the repaired 3-D topology before any
                # diskification seam is materialized.  OptCuts must recover
                # that topology while retaining vertex-fan copies introduced
                # to repair a genuinely non-manifold input vertex.
                working.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.arange(len(working.vertices), dtype=np.int64)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Mesh preparation failed: %s", exc)
            info["failure_reason"] = "mesh_sanitation_failed"
            info["error"] = str(exc)
            return (None, info) if return_info else None

        v = np.ascontiguousarray(working.vertices, dtype=np.float64)
        f = np.ascontiguousarray(working.faces, dtype=np.int32)
        topology = self._topology_report(f, len(v))
        info.update(
            {
                "face_count_before_topology_gate": int(len(f)),
                "vertex_count_before_topology_gate": int(len(v)),
                "area_before_topology_gate": float(working.area),
                "topology_before": topology,
            }
        )
        if not self._is_expected_disk(topology, config):
            info["diskification_triggered"] = True
            if not bool(topology["is_connected_two_manifold"]):
                disk_mesh, cut_info = None, {"diskification_failure": "input_is_not_a_connected_two_manifold"}
            else:
                disk_mesh, cut_info = self._cut_mesh_to_disk(working)
            info.update(cut_info)
            if disk_mesh is not None and len(disk_mesh.faces):
                working = disk_mesh
                v = np.ascontiguousarray(working.vertices, dtype=np.float64)
                f = np.ascontiguousarray(working.faces, dtype=np.int32)
                topology = self._topology_report(f, len(v))
                info["diskification_success"] = self._is_expected_disk(topology, config)

        info.update(
            {
                "face_count_after_topology_gate": int(len(f)),
                "vertex_count_after_topology_gate": int(len(v)),
                "area_after_topology_gate": float(working.area),
                "topology_after": topology,
            }
        )
        if not self._is_expected_disk(topology, config):
            info["failure_reason"] = "topology_gate_failed"
            return (None, info) if return_info else None

        output_summary = provenance_summary(working)
        for name in ("face", "area"):
            before = float(input_summary[f"{name}_count"] if name != "area" else input_summary["area"])
            after = float(output_summary[f"{name}_count"] if name != "area" else output_summary["area"])
            info[f"{name}_retention_ratio"] = float(after / before) if before > 0.0 else float("nan")
        input_source_vertices = set(input_summary.get("source_vertex_ids", []))
        output_source_vertices = set(output_summary.get("source_vertex_ids", []))
        info["materialized_vertex_count_before"] = int(input_summary["vertex_count"])
        info["materialized_vertex_count_after"] = int(output_summary["vertex_count"])
        info["materialized_vertex_count_ratio"] = (
            float(output_summary["vertex_count"] / input_summary["vertex_count"])
            if input_summary["vertex_count"]
            else float("nan")
        )
        info["source_vertex_count_before"] = int(len(input_source_vertices))
        info["source_vertex_count_after"] = int(len(output_source_vertices))
        info["source_vertex_retention_ratio"] = (
            float(len(output_source_vertices) / len(input_source_vertices)) if input_source_vertices else float("nan")
        )
        info["removed_source_vertex_ids"] = sorted(
            int(value) for value in input_source_vertices - output_source_vertices
        )
        before_atoms = set(input_summary.get("source_atom_indices", []))
        after_atoms = set(output_summary.get("source_atom_indices", []))
        info["source_atom_count_before"] = int(len(before_atoms))
        info["source_atom_count_after"] = int(len(after_atoms))
        info["source_atom_indices_before"] = sorted(int(x) for x in before_atoms)
        info["source_atom_indices_after"] = sorted(int(x) for x in after_atoms)
        info["source_atom_retention_ratio"] = (
            float(len(after_atoms) / len(before_atoms)) if before_atoms else float("nan")
        )
        info["removed_source_atom_indices"] = sorted(int(x) for x in before_atoms - after_atoms)
        info["source_face_hash"] = face_domain_hash(working)
        info["provenance"] = output_summary

        replace_mesh(mesh, working)
        mesh.metadata["topoppi_parameterization_prepared"] = True
        mesh.metadata["parameterization_preparation"] = info
        mesh.metadata["source_face_hash"] = info["source_face_hash"]
        return (mesh, info) if return_info else mesh

    def flatten_patch(self, mesh: trimesh.Trimesh, method: str | None = None, return_info: bool = False):
        """Flatten a shared prepared domain without method-specific face removal."""

        config = self.config
        selected_method = method or config.method
        mode = selected_method.strip().lower()
        if mode not in {"auto", "lscm", "harmonic", "slim", "spherical", "cylindrical"}:
            raise ValueError(f"Unsupported parameterization method: {selected_method}")

        prepared, preparation = self.prepare_patch(mesh, return_info=True)
        diag = dict(preparation)
        diag["method"] = mode
        if prepared is None:
            return (None, diag) if return_info else None

        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        diag["source_face_hash"] = face_domain_hash(mesh)

        if mode == "spherical":
            vertex_uv = self._flatten_spherical(v)
            uv = None if vertex_uv is None else unwrap_periodic_corner_uv(mesh, vertex_uv)
            uv = self._normalize_uv(uv, epsilon=config.uv_epsilon)
            if uv is None:
                diag["failure_reason"] = "spherical_projection_failed"
            diag["output_layout"] = "face_corner"
            return (uv, diag) if return_info else uv
        if mode == "cylindrical":
            vertex_uv = self._flatten_cylindrical(v)
            uv = None if vertex_uv is None else unwrap_periodic_corner_uv(mesh, vertex_uv)
            uv = self._normalize_uv(uv, epsilon=config.uv_epsilon)
            if uv is None:
                diag["failure_reason"] = "cylindrical_projection_failed"
            diag["output_layout"] = "face_corner"
            return (uv, diag) if return_info else uv

        try:
            bnd = self._ordered_boundary_loop(v, f)
        except Exception as exc:
            logger.error("Failed to detect boundary: %s", exc)
            diag["failure_reason"] = "boundary_detection_failed"
            return (None, diag) if return_info else None
        if len(bnd) < 3:
            diag["failure_reason"] = "invalid_boundary"
            return (None, diag) if return_info else None

        b = self._boundary_antipodal_pins(v, bnd)
        if b is None:
            diag["failure_reason"] = "invalid_boundary_geometry"
            return (None, diag) if return_info else None
        bc = np.array([config.lscm_pin_a, config.lscm_pin_b], dtype=np.float64)
        uv = None
        if mode in {"auto", "lscm"}:
            uv = self._flatten_lscm(v, f, b, bc)
        if mode == "slim":
            initial_uv = self._flatten_harmonic(v, f, bnd)
            if initial_uv is not None:
                initial_uv = self._normalize_uv(initial_uv, epsilon=config.uv_epsilon)
                uv = self._flatten_slim(
                    v,
                    f,
                    initial_uv,
                    boundary_vertices=bnd,
                    iterations=config.slim_iterations,
                    boundary_constraint_weight=config.slim_boundary_constraint_weight,
                )
            diag["initializer"] = "uniform_weight_tutte_harmonic"
            diag["boundary_condition"] = "soft_fixed_convex_circle"
            diag["boundary_constraint_weight"] = float(config.slim_boundary_constraint_weight)
        if uv is None and mode in {"auto", "harmonic"}:
            uv = self._flatten_harmonic(v, f, bnd)
            if mode == "auto" and uv is not None:
                diag["fallback_method"] = "harmonic"
        if uv is None:
            diag["failure_reason"] = "parameterization_failed"
        else:
            uv = self._normalize_uv(uv, epsilon=config.uv_epsilon)
            diag["failure_reason"] = None
            diag["output_layout"] = "vertex"
        return (uv, diag) if return_info else uv

    @staticmethod
    def _flatten_lscm(v, f, b, bc):
        try:
            uv, _q = igl.lscm(v, f, b, bc)
            if isinstance(uv, np.ndarray) and uv.shape == (len(v), 2):
                return uv
        except Exception as exc:
            logger.warning("LSCM solver failed: %s", exc)
        return None

    @staticmethod
    def _flatten_slim(
        v,
        f,
        initial_uv,
        *,
        boundary_vertices,
        iterations: int,
        boundary_constraint_weight: float,
    ):
        """Run boundary-constrained libigl SLIM from a valid Tutte embedding."""

        try:
            boundary = np.asarray(boundary_vertices, dtype=np.int32).reshape(-1)
            boundary_uv = np.asarray(initial_uv, dtype=np.float64)[boundary]
            data = igl.slim_precompute(
                np.asarray(v, dtype=np.float64, order="F"),
                np.asarray(f, dtype=np.int32, order="F"),
                np.asarray(initial_uv, dtype=np.float64, order="F"),
                igl.SYMMETRIC_DIRICHLET,
                boundary,
                np.asarray(boundary_uv, dtype=np.float64, order="F"),
                float(boundary_constraint_weight),
            )
            uv = igl.slim_solve(data, int(iterations))
            if isinstance(uv, np.ndarray) and uv.shape == (len(v), 2):
                return uv
        except Exception as exc:
            logger.warning("SLIM parameterization failed: %s", exc)
        return None

    @staticmethod
    def _flatten_harmonic(v, f, bnd):
        """Uniform-weight Tutte embedding with an arc-length circular boundary.

        Positive graph-Laplacian weights are used deliberately.  Cotangent
        weights can be negative on obtuse surface triangles and therefore do
        not provide the valid disk embedding required to initialize SLIM.  The
        boundary angles follow cumulative 3-D chord length, so an irregularly
        sampled boundary does not receive an artificial parameter-space stretch.
        """
        try:
            import warnings

            bnd = np.asarray(bnd, dtype=np.int64).reshape(-1)
            if len(bnd) < 3 or len(np.unique(bnd)) != len(bnd):
                return None
            boundary_points = np.asarray(v, dtype=np.float64)[bnd]
            edge_lengths = np.linalg.norm(
                np.roll(boundary_points, -1, axis=0) - boundary_points,
                axis=1,
            )
            perimeter = float(np.sum(edge_lengths))
            if (
                not np.isfinite(edge_lengths).all()
                or np.any(edge_lengths <= 1e-12)
                or not np.isfinite(perimeter)
                or perimeter <= 1e-12
            ):
                return None
            cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths[:-1])))
            angles = 2.0 * np.pi * cumulative / perimeter
            bnd_uv = np.column_stack((np.cos(angles), np.sin(angles)))

            vertex_count = int(len(v))
            boundary_mask = np.zeros(vertex_count, dtype=bool)
            boundary_mask[bnd] = True
            interior = np.flatnonzero(~boundary_mask)
            uv = np.zeros((vertex_count, 2), dtype=np.float64)
            uv[bnd] = np.asarray(bnd_uv, dtype=np.float64)
            if len(interior) == 0:
                return uv

            faces = np.asarray(f, dtype=np.int64)
            edge_rows = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
            edge_cols = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
            rows = np.concatenate((edge_rows, edge_cols))
            cols = np.concatenate((edge_cols, edge_rows))
            adjacency = sparse.coo_matrix(
                (np.ones(len(rows), dtype=np.float64), (rows, cols)),
                shape=(vertex_count, vertex_count),
            ).tocsr()
            adjacency.data[:] = 1.0
            adjacency.eliminate_zeros()
            degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
            if np.any(degrees[interior] <= 0.0):
                return None
            laplacian = sparse.diags(degrees, format="csr") - adjacency
            system = laplacian[interior][:, interior]
            rhs = -(laplacian[interior][:, bnd] @ uv[bnd])
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                uv[interior] = spsolve(system, rhs)
            return uv if np.isfinite(uv).all() else None
        except Exception as exc:
            logger.error("Harmonic parameterization failed: %s", exc)
            return None

    @staticmethod
    def _boundary_antipodal_pins(vertices: np.ndarray, boundary: np.ndarray) -> np.ndarray | None:
        """Choose LSCM pins closest to half a 3-D boundary perimeter apart."""

        bnd = np.asarray(boundary, dtype=np.int64).reshape(-1)
        if len(bnd) < 3 or len(np.unique(bnd)) != len(bnd):
            return None
        points = np.asarray(vertices, dtype=np.float64)[bnd]
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        perimeter = float(np.sum(edge_lengths))
        if (
            not np.isfinite(edge_lengths).all()
            or np.any(edge_lengths <= 1e-12)
            or not np.isfinite(perimeter)
            or perimeter <= 1e-12
        ):
            return None
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths[:-1])))
        opposite = int(np.argmin(np.abs(cumulative - 0.5 * perimeter)))
        if opposite == 0:
            return None
        return np.asarray([bnd[0], bnd[opposite]], dtype=np.int32)

    @staticmethod
    def _flatten_spherical(v):
        """
        Spherical parameterization in a data-oriented principal frame.
        """
        pts = np.asarray(v, dtype=np.float64)
        if len(pts) < 3:
            return None
        c = pts.mean(axis=0, keepdims=True)
        d = pts - c
        r = np.linalg.norm(d, axis=1)
        if np.all(r < 1e-12):
            return None
        frame = Parameterizer._principal_frame(d)
        if frame is None:
            return None
        u, w, axis = frame
        radial_u = d @ u
        radial_w = d @ w
        height = d @ axis
        theta = np.arctan2(radial_w, radial_u)
        rr = np.maximum(r, 1e-12)
        phi = np.arccos(np.clip(height / rr, -1.0, 1.0))
        return np.column_stack([(theta + np.pi) / (2.0 * np.pi), phi / np.pi])

    @staticmethod
    def _flatten_cylindrical(v):
        """
        Cylindrical parameterization using PCA major axis as cylinder axis.
        """
        pts = np.asarray(v, dtype=np.float64)
        if len(pts) < 3:
            return None
        c = pts.mean(axis=0, keepdims=True)
        d = pts - c
        frame = Parameterizer._principal_frame(d)
        if frame is None:
            return None
        u, w, axis = frame

        radial_u = d @ u
        radial_w = d @ w
        angle = np.arctan2(radial_w, radial_u)
        h = d @ axis
        return np.column_stack([(angle + np.pi) / (2.0 * np.pi), h])

    @staticmethod
    def _principal_frame(centered_points: np.ndarray):
        """Return a reproducibly oriented PCA frame for angular comparators."""

        points = np.asarray(centered_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
            return None
        covariance = points.T @ points / float(len(points))
        if not np.isfinite(covariance).all():
            return None
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        largest = float(eigenvalues[order[0]])
        second = float(eigenvalues[order[1]])
        third = float(eigenvalues[order[2]])
        if largest <= 1e-24:
            return None
        eigengap_tolerance = max(largest, 1e-24) * np.sqrt(np.finfo(np.float64).eps)
        # Both axes used by the angular map must be intrinsic.  Equal leading
        # moments leave the polar axis undefined; equal transverse moments
        # leave the longitude origin undefined.  The latter is not a harmless
        # UV translation because periodic unwrapping turns that arbitrary
        # origin into an actual seam. Reject either near-degeneracy so a
        # numerical eigenspace basis cannot choose the scored cut location.
        if largest - second <= eigengap_tolerance or second - third <= eigengap_tolerance:
            return None

        def orient(vector: np.ndarray) -> np.ndarray | None:
            vector = np.asarray(vector, dtype=np.float64)
            projection = points @ vector
            scale = max(float(np.max(np.abs(projection))), 1.0)
            third_moment = float(np.sum(projection**3))
            tolerance = np.finfo(np.float64).eps * len(projection) * scale**3 * 32.0
            if abs(third_moment) > tolerance:
                sign = 1.0 if third_moment > 0.0 else -1.0
            else:
                # A row-index tie break is not a geometric convention: for a
                # centrally symmetric point cloud it can move the longitude
                # seam when the same vertices are merely permuted.  Compare
                # the two sorted projected distributions instead.  This is
                # permutation invariant and changes sign with the eigenvector.
                # If the distributions are indistinguishable, the axis sign is
                # genuinely not encoded by the surface and the angular map has
                # no reproducible seam origin.
                positive = np.sort(projection)
                negative = np.sort(-projection)
                distribution_tolerance = 256.0 * np.finfo(np.float64).eps * scale
                differences = positive - negative
                decisive = np.flatnonzero(np.abs(differences) > distribution_tolerance)
                if len(decisive) == 0:
                    return None
                sign = 1.0 if differences[int(decisive[0])] > 0.0 else -1.0
            return vector * sign

        axis = orient(eigenvectors[:, int(order[0])])
        u = orient(eigenvectors[:, int(order[1])])
        if axis is None or u is None:
            return None
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        u -= axis * float(np.dot(axis, u))
        norm_u = float(np.linalg.norm(u))
        if norm_u <= 1e-12:
            return None
        u /= norm_u
        w = np.cross(axis, u)
        norm_w = float(np.linalg.norm(w))
        if norm_w <= 1e-12:
            return None
        w /= norm_w
        return u, w, axis

    @staticmethod
    def _normalize_uv(uv, *, epsilon: float = 1e-6):
        """Translate and uniformly scale UV without changing aspect ratio."""
        if uv is None or len(uv) == 0:
            return None
        arr = np.asarray(uv, dtype=np.float64)
        if arr.shape[-1] != 2 or not np.isfinite(arr).all():
            return None
        points = arr.reshape(-1, 2)
        uv_min = np.min(points, axis=0)
        extent = np.ptp(points, axis=0)
        scale = max(float(np.max(extent)), float(epsilon))
        return (arr - uv_min) / scale

    @staticmethod
    def _face_quality_mask(
        vertices,
        faces,
        area_faces,
        min_area,
        min_angle_deg,
        max_aspect_ratio,
    ):
        """
        Build a robust per-face validity mask using:
        1) area threshold
        2) minimal angle threshold
        3) maximal edge aspect-ratio threshold
        """
        if len(faces) == 0:
            return np.array([], dtype=bool)

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)

        edges = np.stack([e01, e12, e20], axis=1)
        max_edge = edges.max(axis=1)
        min_edge = edges.min(axis=1)
        aspect_ratio = max_edge / np.maximum(min_edge, 1e-12)

        # Triangle angles from law of cosines.
        a2 = e12 * e12  # opposite v0
        b2 = e20 * e20  # opposite v1
        c2 = e01 * e01  # opposite v2

        cos0 = (b2 + c2 - a2) / np.maximum(2.0 * e20 * e01, 1e-12)
        cos1 = (a2 + c2 - b2) / np.maximum(2.0 * e12 * e01, 1e-12)
        cos2 = (a2 + b2 - c2) / np.maximum(2.0 * e12 * e20, 1e-12)
        cos_stack = np.clip(np.stack([cos0, cos1, cos2], axis=1), -1.0, 1.0)
        min_angle = np.degrees(np.arccos(cos_stack)).min(axis=1)

        area_ok = area_faces > min_area
        aspect_ok = aspect_ratio < max_aspect_ratio
        angle_ok = min_angle >= min_angle_deg
        return area_ok & aspect_ok & angle_ok

    @staticmethod
    def _topology_report(faces, n_vertices) -> dict[str, int | bool]:
        """Return a combinatorial 2-manifold and boundary audit.

        For every vertex, its link must be exactly one cycle (interior) or one
        path (boundary).  This catches pinched vertices and branched boundaries
        that can share the Euler characteristic of a disk.  Edge incidence,
        face connectivity, and unused vertices are checked independently.
        """

        triangles = np.asarray(faces, dtype=np.int64)
        vertex_count = int(n_vertices)
        if len(triangles) == 0:
            return {
                "chi": 0,
                "edges": 0,
                "boundary_loops": 0,
                "boundary_edge_count": 0,
                "boundary_branch_vertex_count": 0,
                "nonmanifold_edge_count": 0,
                "nonmanifold_vertex_count": 0,
                "face_component_count": 0,
                "unused_vertex_count": vertex_count,
                "is_connected_two_manifold": False,
                "boundary_is_disjoint_cycles": False,
            }

        edge_occurrences = np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
            axis=0,
        )
        edge_occurrences = np.sort(edge_occurrences, axis=1)
        unique_edges, inverse, counts = np.unique(
            edge_occurrences,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        edge_count = int(len(unique_edges))
        referenced = np.unique(triangles)
        unused_vertex_count = int(vertex_count - len(referenced))
        chi = int(vertex_count - edge_count + len(triangles))
        boundary_edges = unique_edges[counts == 1]
        boundary_loops = Parameterizer._count_boundary_loops(boundary_edges) if len(boundary_edges) else 0
        boundary_degree = np.zeros(vertex_count, dtype=np.int64)
        if len(boundary_edges):
            np.add.at(boundary_degree, boundary_edges.reshape(-1), 1)
        boundary_branch_vertex_count = int(np.count_nonzero((boundary_degree != 0) & (boundary_degree != 2)))
        nonmanifold_edge_count = int(np.count_nonzero(counts > 2))

        # Face components under shared-edge adjacency.
        parent = np.arange(len(triangles), dtype=np.int64)

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = int(parent[value])
            return value

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        occurrence_faces = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
        order = np.argsort(inverse, kind="stable")
        starts = np.cumsum(np.r_[0, counts[:-1]])
        for start, count in zip(starts, counts, strict=True):
            incident = occurrence_faces[order[int(start) : int(start + count)]]
            for other in incident[1:]:
                union(int(incident[0]), int(other))
        face_component_count = int(len({find(index) for index in range(len(triangles))}))

        # A manifold vertex link is one cycle (interior) or one path (boundary).
        link_edges: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
        for a, b, c in triangles:
            link_edges[int(a)].append((int(b), int(c)))
            link_edges[int(b)].append((int(a), int(c)))
            link_edges[int(c)].append((int(a), int(b)))

        nonmanifold_vertex_count = 0
        for vertex in referenced:
            pairs = link_edges[int(vertex)]
            adjacency: dict[int, list[int]] = {}
            for left, right in pairs:
                adjacency.setdefault(left, []).append(right)
                adjacency.setdefault(right, []).append(left)
            link_components = Parameterizer._adjacency_component_count(adjacency)
            degrees = np.asarray([len(neighbors) for neighbors in adjacency.values()], dtype=np.int64)
            if boundary_degree[int(vertex)] == 0:
                valid_link = link_components == 1 and len(degrees) > 0 and bool(np.all(degrees == 2))
            elif boundary_degree[int(vertex)] == 2:
                valid_link = bool(
                    link_components == 1
                    and np.count_nonzero(degrees == 1) == 2
                    and np.all((degrees == 1) | (degrees == 2))
                )
            else:
                valid_link = False
            nonmanifold_vertex_count += int(not valid_link)

        connected_two_manifold = bool(
            face_component_count == 1
            and unused_vertex_count == 0
            and nonmanifold_edge_count == 0
            and nonmanifold_vertex_count == 0
        )
        return {
            "chi": chi,
            "edges": edge_count,
            "boundary_loops": int(boundary_loops),
            "boundary_edge_count": int(len(boundary_edges)),
            "boundary_branch_vertex_count": boundary_branch_vertex_count,
            "nonmanifold_edge_count": nonmanifold_edge_count,
            "nonmanifold_vertex_count": int(nonmanifold_vertex_count),
            "face_component_count": face_component_count,
            "unused_vertex_count": unused_vertex_count,
            "is_connected_two_manifold": connected_two_manifold,
            "boundary_is_disjoint_cycles": bool(len(boundary_edges) > 0 and boundary_branch_vertex_count == 0),
        }

    @staticmethod
    def _is_expected_disk(topology: dict[str, int | bool], config: ParameterizationConfig) -> bool:
        return bool(
            topology["is_connected_two_manifold"]
            and topology["boundary_is_disjoint_cycles"]
            and int(topology["chi"]) == int(config.expected_euler_characteristic)
            and int(topology["boundary_loops"]) == int(config.expected_boundary_loops)
        )

    @staticmethod
    def _adjacency_component_count(adjacency: dict[int, list[int]]) -> int:
        seen: set[int] = set()
        components = 0
        for start in adjacency:
            if start in seen:
                continue
            components += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        return components

    @staticmethod
    def _count_boundary_loops(boundary_edges):
        """
        Count connected boundary edge components (loops/chains).
        """
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))

        seen = set()
        components = 0
        for v in adj:
            if v in seen:
                continue
            components += 1
            stack = [v]
            seen.add(v)
            while stack:
                cur = stack.pop()
                for nei in adj.get(cur, []):
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
        return components

    @staticmethod
    def _cut_mesh_to_disk(mesh: trimesh.Trimesh):
        """Cut topology by duplicating seam vertices without deleting any face."""

        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            return None, {"diskification_failure": "empty_mesh"}

        # Canonical vertex and face numbering makes libigl's combinatorial cut
        # independent of incidental OBJ/PDB row order.
        source_vertices = np.asarray(
            mesh.metadata.get("source_vertex_ids", np.arange(len(vertices))),
            dtype=np.int64,
        )
        order = np.lexsort((source_vertices, vertices[:, 2], vertices[:, 1], vertices[:, 0]))
        old_to_canonical = np.empty(len(order), dtype=np.int64)
        old_to_canonical[order] = np.arange(len(order), dtype=np.int64)
        canonical_faces = old_to_canonical[faces]
        starts = np.argmin(canonical_faces, axis=1)
        canonical_faces = np.asarray(
            [np.roll(tri, -int(start)) for tri, start in zip(canonical_faces, starts, strict=True)],
            dtype=np.int64,
        )
        face_order = np.lexsort((canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0]))
        paths = igl.cut_to_disk(np.ascontiguousarray(canonical_faces[face_order], dtype=np.int64))
        closed_surface_opening = False
        if not paths:
            incidence: dict[tuple[int, int], int] = {}
            adjacency: dict[int, set[int]] = {}
            for tri in canonical_faces:
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edge = tuple(sorted((int(a), int(b))))
                    incidence[edge] = incidence.get(edge, 0) + 1
                    adjacency.setdefault(edge[0], set()).add(edge[1])
                    adjacency.setdefault(edge[1], set()).add(edge[0])
            if incidence and all(count == 2 for count in incidence.values()):
                center = min(vertex for vertex, neighbors in adjacency.items() if len(neighbors) >= 2)
                neighbors = sorted(adjacency[center])[:2]
                paths = [[neighbors[0], center, neighbors[1]]]
                closed_surface_opening = True
        cut_edges = {
            tuple(sorted((int(order[a]), int(order[b]))))
            for path in paths
            for a, b in zip(path, path[1:], strict=False)
        }

        edge_incidence: dict[tuple[int, int], int] = {}
        cut_mask = np.zeros((len(faces), 3), dtype=bool)
        for face_index, tri in enumerate(faces):
            for corner in range(3):
                # Python libigl's cut_mesh binding indexes the directed face
                # edge F(f, corner) -> F(f, corner + 1).
                edge = tuple(sorted((int(tri[corner]), int(tri[(corner + 1) % 3]))))
                edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
                cut_mask[face_index, corner] = edge in cut_edges

        introduced_edges = sorted(edge for edge in cut_edges if edge_incidence.get(edge) == 2)
        if not introduced_edges:
            return None, {"diskification_failure": "no_topology_cut_returned"}
        try:
            cut_vertices, cut_faces, vertex_to_input = igl.cut_mesh(vertices, faces, cut_mask)
        except Exception as exc:
            logger.warning("Topology-preserving disk cut failed: %s", exc)
            return None, {"diskification_failure": "cut_mesh_failed", "diskification_error": str(exc)}

        result = tracked_vertex_duplication(
            mesh,
            cut_vertices,
            cut_faces,
            vertex_to_input,
            stage="parameterization_topology_cut",
        )
        source_pairs = [sorted((int(source_vertices[a]), int(source_vertices[b]))) for a, b in introduced_edges]
        return result, {
            "diskification_method": "libigl_cut_to_disk_vertex_duplication",
            "diskification_cut_edge_count": int(len(introduced_edges)),
            "diskification_added_vertex_count": int(len(result.vertices) - len(mesh.vertices)),
            "diskification_cut_edges_source_vertex_ids": source_pairs,
            "diskification_faces_removed": 0,
            "diskification_closed_surface_opening": closed_surface_opening,
        }

    @staticmethod
    def _ordered_boundary_loop(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Return the oriented boundary cycle from consistently wound faces.

        The start vertex is geometry deterministic, while the direction follows
        the face winding.  Mapping that directed cycle counter-clockwise gives a
        positive Tutte embedding in the same orientation convention used by
        libigl.  Choosing the direction from vertex row order can globally
        reflect an otherwise valid initializer and make SLIM reject every step.
        """

        edge_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for tri in np.asarray(faces, dtype=np.int64):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = tuple(sorted((int(a), int(b))))
                edge_occurrences.setdefault(edge, []).append((int(a), int(b)))
        adjacency: dict[int, list[int]] = {}
        outgoing: dict[int, int] = {}
        incoming_count: dict[int, int] = {}
        for occurrences in edge_occurrences.values():
            if len(occurrences) != 1:
                continue
            a, b = occurrences[0]
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
            if a in outgoing:
                return np.empty(0, dtype=np.int32)
            outgoing[a] = b
            incoming_count[b] = incoming_count.get(b, 0) + 1
        if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return np.empty(0, dtype=np.int32)
        if set(outgoing) != set(adjacency) or any(incoming_count.get(vertex, 0) != 1 for vertex in adjacency):
            return np.empty(0, dtype=np.int32)

        points = np.asarray(vertices, dtype=np.float64)

        def key(vertex: int):
            neighbor_points = sorted(tuple(np.round(points[n], 12)) for n in adjacency[vertex])
            return (*tuple(np.round(points[vertex], 12)), *neighbor_points[0], *neighbor_points[1])

        start = min(adjacency, key=key)
        current = start
        loop: list[int] = []
        while True:
            loop.append(current)
            following = outgoing[current]
            if following == start:
                break
            if following in loop:
                return np.empty(0, dtype=np.int32)
            current = following
        if len(loop) != len(adjacency):
            return np.empty(0, dtype=np.int32)
        return np.asarray(loop, dtype=np.int32)
