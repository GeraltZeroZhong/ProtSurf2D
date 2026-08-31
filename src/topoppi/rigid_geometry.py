"""Geometry checks for rigid fits of matched protein coordinates."""

from __future__ import annotations

import numpy as np

MINIMUM_SECOND_TO_FIRST_SINGULAR_VALUE_RATIO = 1e-3


def point_cloud_spectrum(points: np.ndarray) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 3:
        raise ValueError("Rigid-fit geometry requires at least three 3D points.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Rigid-fit geometry contains non-finite coordinates.")
    singular_values = np.linalg.svd(coordinates - np.mean(coordinates, axis=0), compute_uv=False)
    ratio = (
        float(singular_values[1] / singular_values[0]) if len(singular_values) > 1 and singular_values[0] > 0.0 else 0.0
    )
    return singular_values, ratio


def require_stable_rigid_fit_geometry(
    reference: np.ndarray,
    mobile: np.ndarray,
    *,
    minimum_ratio: float = MINIMUM_SECOND_TO_FIRST_SINGULAR_VALUE_RATIO,
) -> dict[str, object]:
    reference_values, reference_ratio = point_cloud_spectrum(reference)
    mobile_values, mobile_ratio = point_cloud_spectrum(mobile)
    if min(reference_ratio, mobile_ratio) < minimum_ratio:
        raise ValueError("Matched point geometry is too close to collinear for a unique rigid placement.")
    return {
        "reference_geometry_singular_values": reference_values.tolist(),
        "mobile_geometry_singular_values": mobile_values.tolist(),
        "reference_geometry_second_to_first_ratio": reference_ratio,
        "mobile_geometry_second_to_first_ratio": mobile_ratio,
        "minimum_geometry_second_to_first_ratio": float(minimum_ratio),
    }


__all__ = [
    "MINIMUM_SECOND_TO_FIRST_SINGULAR_VALUE_RATIO",
    "point_cloud_spectrum",
    "require_stable_rigid_fit_geometry",
]
