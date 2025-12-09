import numpy as np
import trimesh
import trimesh.smoothing
from skimage import measure
from scipy.ndimage import gaussian_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Surface")

class SurfaceGenerator:
    """
    Generates a smooth Solvent-Excluded Surface (SES) approximation 
    from atomic coordinates using Gaussian Density Fields.
    """
    
    def __init__(self, coords: np.ndarray):
        """
        Args:
            coords: Atom coordinates (N, 3) numpy array.
        """
        self.coords = coords

    def generate_mesh(self, grid_resolution: float = 1.0, sigma: float = 1.5, level: float = 0.1) -> trimesh.Trimesh:
        """
        Run the pipeline: Voxelization -> Density -> Isosurface -> Mesh.
        
        Args:
            grid_resolution: Voxel size in Angstroms (smaller = detailed but slower).
            sigma: Gaussian smoothing radius (controls surface 'blobbiness').
            level: Isosurface threshold (usually 0.1 ~ 0.5).
            
        Returns:
            trimesh.Trimesh: The generated watertight mesh.
        """
        logger.info(f"Generating surface from {len(self.coords)} atoms...")
        
        if len(self.coords) == 0:
            logger.error("No coordinates provided.")
            return None

        # 1. Define Grid Bounds with Padding
        padding = 10.0
        min_bound = self.coords.min(axis=0) - padding
        max_bound = self.coords.max(axis=0) + padding
        
        # Calculate grid shape
        # shape = (max - min) / res
        shape = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
        
        logger.info(f"Grid shape: {shape}, Resolution: {grid_resolution}A")

        # 2. Fast Voxelization (The secret sauce for speed)
        # Instead of looping atoms, we use a 3D histogram to bin them instantly.
        grid, edges = np.histogramdd(
            self.coords, 
            bins=shape, 
            range=[(min_bound[i], max_bound[i]) for i in range(3)]
        )
        
        # 3. Compute Density Field (Gaussian Blur)
        # This simulates the electron density or probe rolling
        density_field = gaussian_filter(grid.astype(float), sigma=sigma)
        
        # --- CRITICAL FIX: Robust Level Selection ---
        # Calculate density range to prevent marching_cubes failure
        max_density = density_field.max()
        min_density = density_field.min()
        
        logger.debug(f"Density field range: [{min_density:.4f}, {max_density:.4f}]")
        
        if max_density == 0:
            logger.error("Density field is empty (all zeros). Check coordinates.")
            return None

        use_level = level
        # If the requested level is out of range (e.g. density is too diluted), adjust it.
        # This often happens with small proteins or high resolution grids.
        if use_level >= max_density:
            new_level = max_density * 0.5
            logger.warning(f"Requested level {level} > max density {max_density:.4f}. Auto-adjusting to {new_level:.4f}")
            use_level = new_level
        
        # 4. Marching Cubes (Extract Isosurface)
        try:
            verts, faces, normals, values = measure.marching_cubes(
                density_field, 
                level=use_level,
                step_size=1
            )
        except ValueError as e:
            logger.error(f"Marching Cubes failed: {e}")
            return None

        # 5. Transform vertices back to Real World Coordinates
        # verts currently are in grid index space (0..shape)
        # We need to map them back to Angstroms using min_bound and resolution
        
        # real_x = min_x + index_x * resolution
        real_verts = min_bound + verts * grid_resolution
        
        # 6. Create Mesh
        mesh = trimesh.Trimesh(vertices=real_verts, faces=faces, vertex_normals=normals)
        
        # Optional: Basic smoothing to remove grid artifacts
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        except Exception as e:
            logger.warning(f"Mesh smoothing skipped: {e}")
        
        logger.info(f"Surface generated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces.")
        return mesh

# --- Self-Contained Unit Test ---
if __name__ == "__main__":
    print("Running SurfaceGenerator Test...")
    
    # 1. Generate dummy atom coordinates (Two spheres of points)
    # Sphere 1
    theta = np.random.uniform(0, 2*np.pi, 100)
    phi = np.random.uniform(0, np.pi, 100)
    r = 5.0
    x1 = r * np.sin(phi) * np.cos(theta)
    y1 = r * np.sin(phi) * np.sin(theta)
    z1 = r * np.cos(phi)
    
    # Sphere 2 (shifted)
    x2 = x1 + 8.0
    
    coords = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y1, z1])
    ])
    
    # 2. Run Generator
    gen = SurfaceGenerator(coords)
    # Test with a very high level (0.5) to trigger the auto-adjust logic
    # Default density max is usually low (~0.1-0.2) depending on sigma
    mesh = gen.generate_mesh(grid_resolution=1.0, sigma=1.0, level=0.5)
    
    # 3. Verify
    if mesh is not None and len(mesh.vertices) > 0:
        print(f"[Test Result] Mesh Bounds: {mesh.bounds}")
        print(f"Is mesh watertight? {mesh.is_watertight}")
        # mesh.show() # Uncomment to view
        print("SUCCESS: Surface generated.")
    else:
        print("FAILURE: Mesh generation failed.")
