import logging
import math

import numpy as np
import trimesh
import trimesh.smoothing
from scipy.spatial import cKDTree
from skimage import measure

from topoppi.config import SurfaceConfig
from topoppi.mesh.provenance import initialize_provenance

logger = logging.getLogger("Surface")


class SurfaceGenerator:
    """
    Generates a smooth Gaussian-density implicit molecular surface.

    This is not an exact solvent-excluded-surface construction.  The requested
    Gaussian sigma is expressed in Angstroms. Unit-peak atom Gaussians are
    evaluated directly on the sampling lattice, so sigma and the isovalue keep
    the same physical meaning when grid spacing changes.
    """

    def __init__(self, coords: np.ndarray, config: SurfaceConfig | None = None):
        """
        Args:
            coords: Atom coordinates (N, 3) numpy array.
        """
        self.coords = coords
        self.config = config or SurfaceConfig()
        self.last_report = {}

    @staticmethod
    def estimate_grid(coords: np.ndarray, config: SurfaceConfig | None = None) -> dict:
        """Estimate dense-grid geometry and memory without allocating the grid."""

        settings = config or SurfaceConfig()
        points = np.asarray(coords, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
            return {"status": "invalid_coordinates", "atom_count": int(len(points)) if points.ndim else 0}
        if not np.isfinite(points).all():
            return {"status": "nonfinite_coordinates", "atom_count": int(len(points))}

        min_bound = points.min(axis=0) - settings.padding
        max_bound = points.max(axis=0) + settings.padding
        extent = max_bound - min_bound
        target_resolution = float(settings.grid_resolution)
        requested_shape = np.ceil(extent / target_resolution).astype(np.int64)
        requested_voxels = math.prod(int(value) for value in requested_shape)
        shape = requested_shape.copy()
        voxel_count = requested_voxels
        adapted = False
        if voxel_count > settings.max_voxels and settings.adaptive_resolution:
            ratio = (float(voxel_count) / float(settings.max_voxels)) ** (1.0 / 3.0)
            target_resolution *= max(ratio * 1.01, 1.01)
            while True:
                shape = np.ceil(extent / target_resolution).astype(np.int64)
                voxel_count = math.prod(int(value) for value in shape)
                if voxel_count <= settings.max_voxels:
                    break
                target_resolution *= 1.01
            adapted = True
        spacing = extent / np.maximum(shape.astype(np.float64), 1.0)
        within_voxel_budget = bool(voxel_count <= settings.max_voxels)
        within_resolution_limit = bool(target_resolution <= settings.max_adaptive_resolution)
        # The deposited mass and filtered density share one float32 allocation.
        # scipy/skimage line buffers, marching-cubes output, and Python objects
        # are data-dependent and are not included in this lower bound.
        dense_field_lower_bound = int(voxel_count * np.dtype(np.float32).itemsize)
        return {
            "status": "ok" if within_voxel_budget and within_resolution_limit else "voxel_budget_exceeded",
            "atom_count": int(len(points)),
            "bounds_min_angstrom": min_bound.tolist(),
            "bounds_max_angstrom": max_bound.tolist(),
            "bounds_extent_angstrom": extent.tolist(),
            "requested_resolution_angstrom": float(settings.grid_resolution),
            "requested_grid_shape": [int(value) for value in requested_shape],
            "requested_voxel_count": requested_voxels,
            "effective_target_resolution_angstrom": float(target_resolution),
            "effective_grid_shape": [int(value) for value in shape],
            "effective_spacing_angstrom_xyz": [float(value) for value in spacing],
            "effective_voxel_count": int(voxel_count),
            "adaptive_resolution_used": adapted,
            "max_voxels": int(settings.max_voxels),
            "max_adaptive_resolution_angstrom": float(settings.max_adaptive_resolution),
            "within_voxel_budget": within_voxel_budget,
            "within_resolution_limit": within_resolution_limit,
            "estimated_dense_field_bytes_lower_bound": dense_field_lower_bound,
            "memory_estimate_scope": (
                "one in-place float32 density field; excludes scipy/skimage temporaries and marching-cubes output"
            ),
        }

    @staticmethod
    def _accumulate_unit_peak_gaussians(
        coords: np.ndarray,
        *,
        origin: np.ndarray,
        spacing: np.ndarray,
        shape: np.ndarray,
        sigma_angstrom: float,
        truncate_sigma: float = 4.0,
    ) -> np.ndarray:
        """Evaluate a locally truncated unit-peak Gaussian sum on a lattice."""

        grid = np.zeros(tuple(int(value) for value in shape), dtype=np.float32)
        cutoff = float(truncate_sigma * sigma_angstrom)
        inverse_two_sigma_sq = 0.5 / float(sigma_angstrom**2)
        for atom in np.asarray(coords, dtype=np.float64):
            lower = np.maximum(
                np.ceil((atom - cutoff - origin) / spacing).astype(np.int64),
                0,
            )
            upper = np.minimum(
                np.floor((atom + cutoff - origin) / spacing).astype(np.int64),
                shape - 1,
            )
            if np.any(lower > upper):
                continue
            axes = [origin[axis] + np.arange(lower[axis], upper[axis] + 1) * spacing[axis] for axis in range(3)]
            kernels = [
                np.exp(-((axis_points - atom[axis]) ** 2) * inverse_two_sigma_sq)
                for axis, axis_points in enumerate(axes)
            ]
            block = kernels[0][:, None, None] * kernels[1][None, :, None] * kernels[2][None, None, :]
            slices = tuple(slice(int(lower[axis]), int(upper[axis]) + 1) for axis in range(3))
            grid[slices] += np.asarray(block, dtype=np.float32)
        return grid

    def generate_mesh(self) -> trimesh.Trimesh | None:
        """
        Run the pipeline: Voxelization -> Density -> Isosurface -> Mesh.

        The configured isovalue stays fixed across structures. If the density
        field does not cross it, the method records the condition in
        ``last_report`` and returns ``None``.
        """
        config = self.config
        num_atoms = len(self.coords)
        logger.debug("Generating surface from %d atoms.", num_atoms)

        # 1. Define Grid Bounds with Padding and apply the exact same
        # allocation-free budget calculation exposed to benchmark preflight.
        grid_estimate = self.estimate_grid(self.coords, config)
        if grid_estimate["status"] in {"invalid_coordinates", "nonfinite_coordinates"}:
            self.last_report = grid_estimate
            logger.error("Cannot generate a surface from %s.", grid_estimate["status"])
            return None

        requested_resolution = float(config.grid_resolution)
        sigma_angstrom = float(config.sigma)
        level = float(config.level)
        min_bound = np.asarray(grid_estimate["bounds_min_angstrom"], dtype=np.float64)
        target_resolution = float(grid_estimate["effective_target_resolution_angstrom"])
        shape = np.asarray(grid_estimate["effective_grid_shape"], dtype=np.int64)
        requested_voxel_count = int(grid_estimate["requested_voxel_count"])
        voxel_count = int(grid_estimate["effective_voxel_count"])
        adapted = bool(grid_estimate["adaptive_resolution_used"])
        if grid_estimate["status"] != "ok":
            required = float(target_resolution)
            logger.error(
                "Grid requires %d voxels at %.4f A (budget=%d, max adaptive resolution=%.4f A).",
                voxel_count,
                required,
                config.max_voxels,
                config.max_adaptive_resolution,
            )
            self.last_report = {**grid_estimate, "required_resolution_angstrom": required}
            return None
        # The lattice divides each axis into ``shape`` cells.  Use its exact
        # per-axis spacing for atom deposition, Gaussian smoothing, and
        # marching-cubes geometry.
        spacing = np.asarray(grid_estimate["effective_spacing_angstrom_xyz"], dtype=np.float64)
        sigma_voxels = sigma_angstrom / spacing
        logger.debug(
            "Grid shape=%s, requested resolution=%.4f A, effective spacing=%s A, sigma=%.4f A (voxels=%s).",
            shape,
            requested_resolution,
            np.array2string(spacing, precision=4),
            sigma_angstrom,
            np.array2string(sigma_voxels, precision=4),
        )

        # The first density sample is at the centre of the first grid cell,
        # matching marching-cubes coordinates.
        density_sample_origin = min_bound + 0.5 * spacing
        density_field = self._accumulate_unit_peak_gaussians(
            self.coords,
            origin=density_sample_origin,
            spacing=spacing,
            shape=shape,
            sigma_angstrom=sigma_angstrom,
            truncate_sigma=4.0,
        )

        max_density = density_field.max()
        if max_density == 0:
            logger.error("Density field is empty (all zeros). Check coordinates.")
            self.last_report = {**grid_estimate, "status": "empty_density_field"}
            return None

        if level >= max_density:
            self.last_report = {
                **grid_estimate,
                "status": "isovalue_outside_density_range",
                "configured_isovalue": float(level),
                "maximum_density": float(max_density),
            }
            return None
        boundary_max_density = float(
            max(
                np.max(density_field[0, :, :]),
                np.max(density_field[-1, :, :]),
                np.max(density_field[:, 0, :]),
                np.max(density_field[:, -1, :]),
                np.max(density_field[:, :, 0]),
                np.max(density_field[:, :, -1]),
            )
        )
        if boundary_max_density >= level:
            self.last_report = {
                **grid_estimate,
                "status": "isovalue_intersects_grid_boundary",
                "configured_isovalue": float(level),
                "maximum_density": float(max_density),
                "boundary_maximum_density": boundary_max_density,
            }
            return None
        try:
            verts, faces, _normals, _values = measure.marching_cubes(
                density_field,
                level=level,
                spacing=tuple(float(x) for x in spacing),
                step_size=1,
            )
        except ValueError as exc:
            logger.warning("Marching cubes failed at configured level %.4f: %s", level, exc)
            self.last_report = {
                **grid_estimate,
                "status": "marching_cubes_failed",
                "configured_isovalue": float(level),
                "error": str(exc),
            }
            return None
        final_mesh = trimesh.Trimesh(
            vertices=density_sample_origin + verts,
            faces=faces,
            process=False,
        )

        if config.smoothing_iterations > 0:
            trimesh.smoothing.filter_laplacian(final_mesh, iterations=config.smoothing_iterations)

        nearest_atom = cKDTree(np.asarray(self.coords, dtype=np.float64)).query(
            np.asarray(final_mesh.vertices, dtype=np.float64),
            k=1,
        )[1]
        final_mesh.metadata["source_atom_indices"] = np.asarray(nearest_atom, dtype=np.int64)
        final_mesh.metadata["source_vertex_ids"] = np.arange(len(final_mesh.vertices), dtype=np.int64)
        final_mesh.metadata["source_face_ids"] = np.arange(len(final_mesh.faces), dtype=np.int64)
        initialize_provenance(final_mesh, stage="surface_generation")
        self.last_report = {
            "status": "ok",
            "surface_definition": "gaussian_density_isosurface",
            "density_convention": "direct_truncated_unit_peak_atom_gaussians_v2",
            "density_formula": "sum_i exp(-||x-r_i||^2/(2*sigma^2))",
            "atom_deposition": "direct_lattice_evaluation",
            "gaussian_truncate_sigma": 4.0,
            "requested_resolution_angstrom": float(requested_resolution),
            "effective_resolution_angstrom": float(np.max(spacing)),
            "effective_spacing_angstrom_xyz": [float(x) for x in spacing],
            "density_sample_origin_angstrom": [float(x) for x in density_sample_origin],
            "adaptive_resolution_used": bool(adapted),
            "sigma_angstrom": float(sigma_angstrom),
            "sigma_voxels_xyz": [float(x) for x in sigma_voxels],
            "isovalue": float(level),
            "boundary_maximum_density": boundary_max_density,
            "grid_shape": [int(x) for x in shape],
            "requested_voxel_count": int(requested_voxel_count),
            "effective_voxel_count": int(voxel_count),
            "estimated_grid_bytes_float32": int(voxel_count * np.dtype(np.float32).itemsize),
            "estimated_dense_field_bytes_lower_bound": grid_estimate["estimated_dense_field_bytes_lower_bound"],
            "memory_estimate_scope": grid_estimate["memory_estimate_scope"],
            "max_voxels": int(config.max_voxels),
            "atom_count": int(num_atoms),
            "vertex_count": int(len(final_mesh.vertices)),
            "face_count": int(len(final_mesh.faces)),
        }
        final_mesh.metadata["surface_generation"] = dict(self.last_report)
        logger.info(
            "Surface generated: %d verts, %d faces.",
            len(final_mesh.vertices),
            len(final_mesh.faces),
        )
        return final_mesh
