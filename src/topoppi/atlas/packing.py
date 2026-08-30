"""Deterministic chart packing with explicit transforms and padding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt
from typing import Iterable, List, Sequence, Tuple, cast

import numpy as np
import trimesh

from topoppi.atlas.uv import as_corner_uv, canonical_geometry_corner_uv, set_uv_layout


@dataclass(frozen=True)
class ChartTransform:
    chart_index: int
    rotated_90: bool
    local_scale: float
    affine_matrix: tuple[tuple[float, float], tuple[float, float]]
    translation_u: float
    translation_v: float
    packing_origin_u: float
    packing_origin_v: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _area_match_scale(mesh: trimesh.Trimesh, corner_uv: np.ndarray) -> float:
    tri, coordinate_scale = canonical_geometry_corner_uv(mesh, corner_uv)
    signed_twice = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (
        tri[:, 2, 0] - tri[:, 0, 0]
    )
    uv_area = float(np.sum(0.5 * np.abs(signed_twice)))
    mesh_area = float(mesh.area)
    if uv_area <= 1e-15 or mesh_area <= 1e-15 or not np.isfinite(coordinate_scale):
        raise ValueError("Cannot pack a chart with degenerate 2-D or 3-D area.")
    return sqrt(mesh_area / uv_area) / coordinate_scale


def resolved_chart_gap(meshes: Sequence[trimesh.Trimesh], gap_fraction: float) -> float:
    """Convert a dimensionless gap fraction to area-matched UV length."""

    total_area = float(sum(float(mesh.area) for mesh in meshes))
    return float(gap_fraction * sqrt(max(total_area, 0.0)))


def pack_mesh_charts(
    meshes: Sequence[trimesh.Trimesh],
    *,
    key: str = "uv",
    gap: float = 0.08,
    allow_rotate: bool = True,
) -> Tuple[List[np.ndarray], List[ChartTransform], dict[str, object]]:
    """Pack charts on deterministic shelves while preserving common texel density.

    Each chart is first scaled so its 2-D area matches its 3-D area.  That gives
    all charts one common physical scale.  Packing then applies only a 90-degree
    rotation and translation, so local distortion is unchanged.
    """

    if gap < 0.0:
        raise ValueError("Chart gap must be non-negative.")
    if not meshes:
        return [], [], {"status": "empty", "chart_count": 0}

    applied_gap = resolved_chart_gap(meshes, gap)
    charts = []
    for chart_index, mesh in enumerate(meshes):
        corners = as_corner_uv(mesh, key=key).copy()
        scale = _area_match_scale(mesh, corners)
        extent = np.ptp((corners * scale).reshape(-1, 2), axis=0)
        charts.append(
            {
                "index": chart_index,
                "corners": corners,
                "scale": scale,
                "width": float(extent[0]),
                "height": float(extent[1]),
            }
        )

    total_box_area = sum((item["width"] + applied_gap) * (item["height"] + applied_gap) for item in charts)
    max_width = max(item["width"] for item in charts)
    target_width = max(max_width, sqrt(total_box_area))
    # Avoid pathological one-column layouts for several similarly sized charts.
    if len(charts) > 1:
        target_width = max(target_width, max_width * min(len(charts), ceil(sqrt(len(charts)))))

    ordered = sorted(charts, key=lambda item: (-max(item["width"], item["height"]), -item["height"], item["index"]))
    packed: List[np.ndarray | None] = [None] * len(charts)
    transforms: List[ChartTransform | None] = [None] * len(charts)
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for item in ordered:
        width = float(item["width"])
        height = float(item["height"])
        rotate = False
        if allow_rotate and height > width and cursor_x + height <= target_width:
            width, height = height, width
            rotate = True
        if cursor_x > 0.0 and cursor_x + width > target_width:
            cursor_x = 0.0
            cursor_y += row_height + applied_gap
            row_height = 0.0

        source_corners = np.asarray(item["corners"], dtype=np.float64)
        if rotate:
            matrix = np.array([[0.0, float(item["scale"])], [-float(item["scale"]), 0.0]])
        else:
            matrix = np.eye(2, dtype=np.float64) * float(item["scale"])
        transformed = source_corners @ matrix.T
        transformed_min = np.min(transformed.reshape(-1, 2), axis=0)
        translation = np.array([cursor_x, cursor_y], dtype=np.float64) - transformed_min
        corners = transformed + translation
        index = int(item["index"])
        packed[index] = corners
        transforms[index] = ChartTransform(
            chart_index=index,
            rotated_90=rotate,
            local_scale=float(item["scale"]),
            affine_matrix=(
                (float(matrix[0, 0]), float(matrix[0, 1])),
                (float(matrix[1, 0]), float(matrix[1, 1])),
            ),
            translation_u=float(translation[0]),
            translation_v=float(translation[1]),
            packing_origin_u=float(cursor_x),
            packing_origin_v=float(cursor_y),
        )
        cursor_x += width + applied_gap
        row_height = max(row_height, height)

    packed_arrays = cast(List[np.ndarray], packed)
    final_transforms = cast(List[ChartTransform], transforms)
    all_points = np.concatenate([chart.reshape(-1, 2) for chart in packed_arrays], axis=0)
    atlas_min = np.min(all_points, axis=0)
    atlas_max = np.max(all_points, axis=0)
    report = {
        "status": "packed",
        "algorithm": "deterministic_shelf_area_matched_v1",
        "chart_count": int(len(packed_arrays)),
        "requested_gap_fraction": float(gap),
        "gap_reference_length": "sqrt(total_3d_chart_area)",
        "applied_gap_uv": float(applied_gap),
        "bounds_min": atlas_min.tolist(),
        "bounds_max": atlas_max.tolist(),
        "width": float(atlas_max[0] - atlas_min[0]),
        "height": float(atlas_max[1] - atlas_min[1]),
        "transforms": [transform.to_dict() for transform in final_transforms],
    }
    return packed_arrays, final_transforms, report


def apply_packed_uv(
    meshes: Sequence[trimesh.Trimesh],
    packed_uv: Iterable[np.ndarray],
    transforms: Sequence[ChartTransform],
    *,
    key: str = "uv_global",
) -> None:
    for mesh, corners, transform in zip(meshes, packed_uv, transforms, strict=True):
        set_uv_layout(mesh, corners, key=key)
        mesh.metadata["atlas_transform"] = transform.to_dict()


__all__ = ["ChartTransform", "apply_packed_uv", "pack_mesh_charts", "resolved_chart_gap"]
