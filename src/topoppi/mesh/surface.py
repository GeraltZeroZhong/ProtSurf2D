import numpy as np
import trimesh
import trimesh.smoothing
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_closing
import logging

from topoppi.config import SurfaceConfig

logger = logging.getLogger("Surface")

class SurfaceGenerator:
    """
    Generates a smooth Solvent-Excluded Surface (SES) approximation 
    from atomic coordinates using Gaussian Density Fields.
    """
    
    def __init__(self, coords: np.ndarray, config: SurfaceConfig | None = None):
        """
        Args:
            coords: Atom coordinates (N, 3) numpy array.
        """
        self.coords = coords
        self.config = config or SurfaceConfig()

    def generate_mesh(self) -> trimesh.Trimesh:
        """
        Run the pipeline: Voxelization -> Density -> Isosurface -> Mesh.
        Includes robust retry logic for sparse surfaces.
        """
        config = self.config
        config.validate()
        grid_resolution = config.grid_resolution
        sigma = config.sigma
        level = config.level
        num_atoms = len(self.coords)
        logger.info(f"Generating surface from {num_atoms} atoms...")

        if num_atoms == 0:
            logger.error("No coordinates provided.")
            return None
        if grid_resolution <= 0:
            logger.error("grid_resolution must be > 0.")
            return None
        if sigma <= 0:
            logger.error("sigma must be > 0.")
            return None
        if not np.isfinite(self.coords).all():
            logger.error("Coordinates contain non-finite values.")
            return None

        # 1. Define Grid Bounds with Padding
        padding = config.padding
        min_bound = self.coords.min(axis=0) - padding
        max_bound = self.coords.max(axis=0) + padding
        
        # Calculate grid shape
        shape = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
        if np.any(shape <= 0):
            logger.error(f"Invalid grid shape computed: {shape}")
            return None
        voxel_count = int(np.prod(shape, dtype=np.int64))
        if voxel_count > config.max_voxels:
            logger.error(
                "Grid is too large (%d voxels). Increase --res to avoid excessive memory use.",
                voxel_count,
            )
            return None
        logger.info(f"Grid shape: {shape}, Resolution: {grid_resolution}A")

        # 2. Fast Voxelization
        grid, edges = np.histogramdd(
            self.coords, 
            bins=shape, 
            range=[(min_bound[i], max_bound[i]) for i in range(3)]
        )
        
        # 3. Compute Density Field
        density_field = gaussian_filter(grid.astype(float), sigma=sigma)
        
        max_density = density_field.max()
        if max_density == 0:
            logger.error("Density field is empty (all zeros). Check coordinates.")
            return None

        # 4. Voxel-space morphological closing before Marching Cubes.
        # This fills tiny interior gaps/tunnels and reduces downstream topology noise.
        closing_level = min(level, max_density * config.closing_fraction)
        occupancy = density_field >= closing_level
        closed_occupancy = binary_closing(
            occupancy,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=1
        )
        occupied_before = int(np.count_nonzero(occupancy))
        occupied_after = int(np.count_nonzero(closed_occupancy))
        if np.any(closed_occupancy):
            density_field = np.where(
                closed_occupancy,
                np.maximum(density_field, closing_level),
                0.0
            )
        logger.info(
            "Applied voxel-space binary closing "
            f"(level={closing_level:.6f}, occupied_voxels: {occupied_before} -> {occupied_after}, "
            f"delta={occupied_after - occupied_before})."
        )

        # --- Iterative Level Adjustment (Smart Retry) ---
        # Fix for 1AHW issue: If max_density is driven by outliers (e.g. clashing atoms),
        # the default level (0.1) might be too high for the rest of the surface.
        
        current_level = level
        # Safety: Ensure we start below the max
        if current_level >= max_density:
            current_level = max_density * config.retry_level_factor

        final_mesh = None
        
        # Heuristic: Expect at least 0.5 vertices per atom for a decent coarse surface
        min_expected_verts = min(config.min_expected_vertices_cap, num_atoms * config.min_vertices_per_atom) 

        for attempt in range(config.retry_attempts):
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    density_field, 
                    level=current_level,
                    step_size=1
                )
                
                n_verts = len(verts)
                # Check if surface is suspiciously small
                if n_verts < min_expected_verts and attempt < config.retry_attempts - 1:
                    logger.warning(f"Surface too small ({n_verts} verts) at level {current_level:.4f}. Reducing threshold...")
                    current_level *= config.retry_level_factor
                    continue 
                
                # If acceptable size or last attempt
                real_verts = min_bound + verts * grid_resolution
                final_mesh = trimesh.Trimesh(vertices=real_verts, faces=faces, vertex_normals=normals)
                break
                
            except ValueError as e:
                logger.warning(f"Marching Cubes failed at level {current_level}: {e}")
                current_level *= config.retry_level_factor
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        if final_mesh is None or len(final_mesh.vertices) == 0:
            logger.error("Failed to generate a valid mesh after retries.")
            return None

        # Optional: Basic smoothing
        try:
            if config.smoothing_iterations > 0:
                trimesh.smoothing.filter_laplacian(final_mesh, iterations=config.smoothing_iterations)
        except Exception as e:
            logger.warning(f"Mesh smoothing skipped: {e}")
        
        logger.info(f"Surface generated: {len(final_mesh.vertices)} verts, {len(final_mesh.faces)} faces.")
        return final_mesh

