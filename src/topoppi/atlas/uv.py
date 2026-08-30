"""Canonical UV helpers.

TopoPPI stores texture coordinates per face corner. This distinction is
essential for cut meshes: averaging several texture coordinates attached to one
3-D vertex would erase the seam that the parameterizer created.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import trimesh


def as_corner_uv(mesh: trimesh.Trimesh, uv: Optional[np.ndarray] = None, key: str = "uv") -> np.ndarray:
    """Return UV coordinates as ``(n_faces, 3, 2)`` without losing seams."""

    if uv is None:
        uv = mesh.metadata.get(key)
    if uv is None:
        raise ValueError(f"Mesh is missing UV data for key '{key}'.")

    arr = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if arr.shape == (len(faces), 3, 2):
        return np.ascontiguousarray(arr)
    if arr.shape == (len(mesh.vertices), 2):
        return np.ascontiguousarray(arr[faces])
    raise ValueError(
        f"UV data for '{key}' has shape {arr.shape}; expected ({len(mesh.vertices)}, 2) or ({len(faces)}, 3, 2)."
    )


def canonical_geometry_corner_uv(
    mesh: trimesh.Trimesh,
    uv: Optional[np.ndarray] = None,
    *,
    key: str = "uv",
) -> tuple[np.ndarray, float]:
    """Translate and uniformly scale UV for scale-free geometric predicates.

    The longest atlas extent becomes one.  This representation is only for
    orientation, continuity, and intersection predicates; reported UV lengths
    and metric Jacobians continue to use their explicitly documented scales.
    """

    corners = as_corner_uv(mesh, uv=uv, key=key).copy()
    points = corners.reshape(-1, 2)
    if not np.isfinite(points).all():
        return corners, float("nan")
    minimum = np.min(points, axis=0)
    scale = float(np.max(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale <= 0.0:
        return corners - minimum, float("nan")
    return (corners - minimum) / scale, scale


def corner_to_vertex_uv(
    mesh: trimesh.Trimesh,
    corner_uv: np.ndarray,
    *,
    atol: float = 1e-9,
) -> Optional[np.ndarray]:
    """Return vertex UV only when incident corners agree at atlas scale.

    ``atol`` is a fraction of the longest UV extent, matching the seam and
    injectivity predicates.  An absolute raw-coordinate tolerance would make
    the same atlas appear continuous or discontinuous after a uniform change
    of UV units.
    """

    corners = as_corner_uv(mesh, corner_uv)
    if not np.isfinite(corners).all():
        return None
    faces = np.asarray(mesh.faces, dtype=np.int64)
    flat_vertices = faces.reshape(-1)
    flat_corners = corners.reshape(-1, 2)
    first = np.full(len(mesh.vertices), len(flat_vertices), dtype=np.int64)
    np.minimum.at(first, flat_vertices, np.arange(len(flat_vertices), dtype=np.int64))
    if np.any(first == len(flat_vertices)):
        return None
    vertex_uv = flat_corners[first]
    extent = float(np.max(np.ptp(flat_corners, axis=0)))
    if extent == 0.0:
        return vertex_uv
    predicate_corners = (flat_corners - np.min(flat_corners, axis=0)) / extent
    predicate_vertex_uv = predicate_corners[first]
    if not np.all(
        np.isclose(
            predicate_corners,
            predicate_vertex_uv[flat_vertices],
            atol=atol,
            rtol=0.0,
        )
    ):
        return None
    return vertex_uv


def set_uv_layout(mesh: trimesh.Trimesh, uv: np.ndarray, key: str = "uv") -> np.ndarray:
    """Store canonical per-corner UV coordinates."""

    corners = as_corner_uv(mesh, uv)
    if not np.isfinite(corners).all():
        raise ValueError(f"UV data for '{key}' contains non-finite values.")
    mesh.metadata[key] = corners.copy()
    return corners


def unwrap_periodic_corner_uv(
    mesh: trimesh.Trimesh,
    vertex_uv: np.ndarray,
    *,
    axis: int = 0,
    period: float = 1.0,
) -> np.ndarray:
    """Unwrap a periodic coordinate independently on each face."""

    corners = as_corner_uv(mesh, vertex_uv).copy()
    half_period = period * 0.5
    values = corners[:, :, axis]
    wrap = (np.ptp(values, axis=1) > half_period)[:, None] & (values < half_period)
    corners[:, :, axis] += period * wrap
    return corners


def face_domain_hash(mesh: trimesh.Trimesh) -> str:
    """Stable identity for the exact scored 3-D face domain."""

    source_faces = mesh.metadata.get("source_face_ids")
    digest = hashlib.sha256()
    if source_faces is not None and len(source_faces) == len(mesh.faces):
        source_faces = np.asarray(source_faces, dtype=np.int64)
        order = np.argsort(source_faces, kind="stable")
        triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
        digest.update(b"source-face-domain-v2\0")
        digest.update(np.ascontiguousarray(source_faces[order], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(triangles[order], dtype=np.float64).tobytes())
    else:
        digest.update(b"geometry-v1\0")
        digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def uv_checksum(mesh: trimesh.Trimesh, uv: Optional[np.ndarray] = None, key: str = "uv") -> str:
    corners = as_corner_uv(mesh, uv=uv, key=key)
    return hashlib.sha256(np.ascontiguousarray(corners, dtype=np.float64).tobytes()).hexdigest()
