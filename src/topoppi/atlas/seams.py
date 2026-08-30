"""Reconstruct mesh-edge seam topology from per-corner UV coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from topoppi.atlas.uv import as_corner_uv, canonical_geometry_corner_uv
from topoppi.mesh.provenance import OPTCUTS_GEOMETRY_VERTEX_IDS, SOURCE_VERTEX_IDS


@dataclass(frozen=True)
class UVSeamTopology:
    """Unique primal edges and their UV continuity state.

    ``incident_faces`` contains the two incident face indices for manifold
    internal edges.  Other rows are ``-1`` and are excluded from the face-dual
    graph.
    """

    edges: np.ndarray
    source_edges: np.ndarray
    occurrence_counts: np.ndarray
    incident_faces: np.ndarray
    edge_lengths_3d: np.ndarray
    edge_uv_length_sum: np.ndarray
    boundary_mask: np.ndarray
    internal_mask: np.ndarray
    seam_mask: np.ndarray


def uv_seam_topology(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    *,
    atol: float = 1e-9,
) -> UVSeamTopology:
    """Return exact edge incidences and UV discontinuities for a triangle mesh."""

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    corners = as_corner_uv(mesh, uv)
    predicate_corners, predicate_scale = canonical_geometry_corner_uv(mesh, corners)
    local_edges = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    edge_vertices = faces[:, local_edges]
    # Score continuity on the repaired topology. Diskification copies share a
    # geometry ID, while copies introduced to split a non-manifold vertex fan
    # remain distinct despite retaining the same root provenance ID.
    identity_key = OPTCUTS_GEOMETRY_VERTEX_IDS if OPTCUTS_GEOMETRY_VERTEX_IDS in mesh.metadata else SOURCE_VERTEX_IDS
    source_vertices = np.asarray(
        mesh.metadata.get(identity_key, np.arange(len(vertices))),
        dtype=np.int64,
    )
    if source_vertices.shape != (len(vertices),):
        raise ValueError(f"{identity_key} must contain one ID per mesh vertex.")
    edge_sources = source_vertices[edge_vertices]
    edge_uv = corners[:, local_edges]
    predicate_edge_uv = predicate_corners[:, local_edges]
    reverse = edge_sources[:, :, 0] > edge_sources[:, :, 1]
    ordered_source_edges = np.sort(edge_sources, axis=2).reshape(-1, 2)
    if np.any(ordered_source_edges[:, 0] == ordered_source_edges[:, 1]):
        raise ValueError("A mesh edge collapses to one source vertex ID.")
    ordered_vertex_edges = np.where(
        reverse[:, :, None],
        edge_vertices[:, :, ::-1],
        edge_vertices,
    ).reshape(-1, 2)
    ordered_uv = np.where(
        reverse[:, :, None, None],
        edge_uv[:, :, ::-1, :],
        edge_uv,
    ).reshape(-1, 2, 2)
    ordered_predicate_uv = np.where(
        reverse[:, :, None, None],
        predicate_edge_uv[:, :, ::-1, :],
        predicate_edge_uv,
    ).reshape(-1, 2, 2)

    unique_source_edges, inverse, counts = np.unique(
        ordered_source_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.any(counts > 2):
        raise ValueError("Source-edge provenance is non-manifold after mesh preparation.")
    occurrence_order = np.argsort(inverse, kind="stable")
    group_starts = np.cumsum(np.r_[0, counts[:-1]])
    first_occurrence = occurrence_order[group_starts]
    representative_edges = ordered_vertex_edges[first_occurrence]
    occurrence_edge_lengths = np.linalg.norm(
        vertices[ordered_vertex_edges[:, 1]] - vertices[ordered_vertex_edges[:, 0]],
        axis=1,
    )
    edge_lengths = (
        np.bincount(
            inverse,
            weights=occurrence_edge_lengths,
            minlength=len(unique_source_edges),
        )
        / counts
    )
    occurrence_uv_lengths = np.linalg.norm(ordered_uv[:, 1] - ordered_uv[:, 0], axis=1)
    edge_uv_length_sum = np.bincount(
        inverse,
        weights=occurrence_uv_lengths,
        minlength=len(unique_source_edges),
    ).astype(np.float64, copy=False)

    incident_faces = np.full((len(unique_source_edges), 2), -1, dtype=np.int64)
    seam_mask = np.zeros(len(unique_source_edges), dtype=bool)
    internal_mask = counts == 2
    internal_edges = np.flatnonzero(internal_mask)
    if len(internal_edges):
        first = occurrence_order[group_starts[internal_edges]]
        second = occurrence_order[group_starts[internal_edges] + 1]
        occurrence_faces = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
        incident_faces[internal_edges, 0] = occurrence_faces[first]
        incident_faces[internal_edges, 1] = occurrence_faces[second]
        continuous = (
            np.all(
                np.isclose(
                    ordered_predicate_uv[first],
                    ordered_predicate_uv[second],
                    atol=atol,
                    rtol=0.0,
                ),
                axis=(1, 2),
            )
            if np.isfinite(predicate_scale)
            else np.all(ordered_uv[first] == ordered_uv[second], axis=(1, 2))
        )
        seam_mask[internal_edges] = ~continuous

    return UVSeamTopology(
        edges=representative_edges,
        source_edges=unique_source_edges,
        occurrence_counts=counts,
        incident_faces=incident_faces,
        edge_lengths_3d=edge_lengths,
        edge_uv_length_sum=edge_uv_length_sum,
        boundary_mask=counts == 1,
        internal_mask=internal_mask,
        seam_mask=seam_mask,
    )
