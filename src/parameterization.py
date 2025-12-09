import numpy as np
import igl
import trimesh
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSCM")

class Parameterizer:
    """
    Handles the flattening of 3D meshes into 2D UV coordinates using LSCM
    (Least Squares Conformal Maps), with a Harmonic fallback.
    """
    
    @staticmethod
    def flatten_patch(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Flatten a 3D mesh patch to 2D.
        
        Args:
            mesh: A single connected component (trimesh object).
            
        Returns:
            np.ndarray: UV coordinates of shape (N, 2), or None if failed.
        """
        # IGL is strict about types. Ensure correct C++ compatible types.
        # CRITICAL FIX: Use np.ascontiguousarray to strip Trimesh wrappers and ensure C-order.
        # CRITICAL FIX: Use np.int64 for indices (required by modern libigl bindings).
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        f = np.ascontiguousarray(mesh.faces, dtype=np.int64)

        # 1. Find Boundary Loop (LSCM needs a boundary)
        # igl.boundary_loop returns the ordered vertex indices of the boundary
        try:
            bnd = igl.boundary_loop(f)
        except Exception as e:
            logger.error(f"Failed to detect boundary: {e}")
            return None
        
        # Handle case where multiple boundaries might be returned (list of lists)
        if len(bnd) > 0 and isinstance(bnd[0], (list, np.ndarray)):
            bnd = sorted(bnd, key=lambda x: len(x), reverse=True)[0]
        
        bnd = np.array(bnd, dtype=np.int64)

        if len(bnd) < 3:
            logger.error("Mesh has no valid boundary (closed or degenerate).")
            return None

        # 2. Fix Boundary Conditions for LSCM
        # Strategy: Pin the two most distant points on the boundary.
        b1_idx = bnd[0]
        
        # Find point on boundary furthest from A (Euclidean distance)
        boundary_coords = v[bnd]
        dists = np.linalg.norm(boundary_coords - v[b1_idx], axis=1)
        b2_idx = bnd[np.argmax(dists)]
        
        # Constraints inputs: b (indices), bc (target coords)
        # FIX: Ensure 'b' is also int64 to match 'f'
        b = np.array([b1_idx, b2_idx], dtype=np.int64)
        bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        # 3. Run LSCM
        uv_normalized = None
        try:
            # IGL LSCM signature: lscm(V, F, b, bc) -> (success, UV)
            ret = igl.lscm(v, f, b, bc)
            
            # Robust unpacking
            if isinstance(ret, tuple) and len(ret) == 2:
                success, uv = ret
            else:
                success = True
                uv = ret

            # Handle numpy/bool ambiguity
            if isinstance(success, np.ndarray):
                is_success = success.all()
            else:
                is_success = bool(success)
            
            if is_success:
                uv_normalized = Parameterizer._normalize_uv(uv)
            else:
                logger.warning("IGL LSCM solver returned failure status.")
                
        except Exception as e:
            logger.warning(f"LSCM Exception: {e}")

        # 4. Fallback: Harmonic Parameterization
        if uv_normalized is None:
            logger.info("Attempting Harmonic Parameterization fallback...")
            uv_normalized = Parameterizer._flatten_harmonic(v, f, bnd)

        return uv_normalized

    @staticmethod
    def _flatten_harmonic(v, f, bnd):
        """
        Fallback method: Map boundary to circle and minimize Dirichlet energy.
        """
        try:
            # 1. Map boundary vertices to a circle
            # Ensure bnd is int64
            bnd = bnd.astype(np.int64)
            bnd_uv = igl.map_vertices_to_circle(v, bnd)
            
            # 2. Harmonic parameterization (power=1)
            # harmonic(V, F, b, bc, k)
            uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
            
            return Parameterizer._normalize_uv(uv)
        except Exception as e:
            logger.error(f"Harmonic Parameterization failed: {e}")
            return None

    @staticmethod
    def _normalize_uv(uv):
        """Helper to normalize UV to [0,1] range."""
        if uv is None or len(uv) == 0:
            return None
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        scale = uv_max - uv_min
        scale[scale < 1e-6] = 1.0 
        return (uv - uv_min) / scale

# --- Self-Contained Unit Test ---
if __name__ == "__main__":
    print("Running Parameterizer Test...")
    
    # 1. Create a dummy open surface (half-sphere cap)
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    
    # Cut it in half
    vertex_mask = mesh.vertices[:, 2] > 0
    face_mask = vertex_mask[mesh.faces].all(axis=1)
    
    sub = mesh.submesh([face_mask], append=True)
    if isinstance(sub, list): sub = trimesh.util.concatenate(sub)
    
    sub.process()
    logger.info(f"Created test mesh: {len(sub.vertices)} verts, {len(sub.faces)} faces")

    # 2. Run Flattening
    param = Parameterizer()
    uv = param.flatten_patch(sub)
    
    # 3. Verify
    if uv is not None:
        print(f"[Test Result] UV Shape: {uv.shape}")
        
        # Check for NaN
        if np.isnan(uv).any():
             print("FAILURE: UV contains NaNs.")
        else:
            print(f"  Range U: [{uv[:,0].min():.2f}, {uv[:,0].max():.2f}]")
            print(f"  Range V: [{uv[:,1].min():.2f}, {uv[:,1].max():.2f}]")
            
            if uv.shape[0] == len(sub.vertices) and uv.shape[1] == 2:
                print("SUCCESS: Parameterization worked (LSCM or Harmonic).")
            else:
                print("FAILURE: UV shape mismatch.")
    else:
        print("FAILURE: Parameterization returned None.")
