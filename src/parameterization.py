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
        # --- Step 0: Robust Mesh Sanitation ---
        # Critical Fix: Smoothing in TopologyManager can introduce degenerate (zero-area) faces.
        # These cause the LSCM linear solver to fail (singular matrix). 
        # We must clean the mesh immediately before parameterization to ensure stability.
        try:
            # 1. Merge Vertices (Topological Stitching)
            # Ensures that geometrically identical vertices are topologically merged.
            # This is crucial for LSCM which requires a single connected component.
            mesh.merge_vertices()

            # 2. Fill tiny holes so the selected boundary is truly an outer disk boundary.
            # Without this, keeping only the longest boundary loop still leaves inner holes,
            # and LSCM can fail with singular systems.
            trimesh.repair.fill_holes(mesh)

            # 3. Remove degenerate/sliver faces.
            # Marching Cubes can produce faces with non-zero area but pathological shape.
            # These make cotangent Laplacians ill-conditioned for LSCM.
            valid_faces = Parameterizer._face_quality_mask(
                np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                np.ascontiguousarray(mesh.faces, dtype=np.int64),
                mesh.area_faces,
                min_area=1e-5,
                min_angle_deg=2.0,
                max_aspect_ratio=50.0,
            )
            if not np.all(valid_faces):
                mesh.update_faces(valid_faces)
            
            # 4. Safe remove duplicate faces (Fix for AttributeError on older trimesh versions)
            if hasattr(mesh, 'remove_duplicate_faces'):
                mesh.remove_duplicate_faces()
            else:
                # Fallback: Process usually handles duplicates if specific method is missing,
                # but we avoid full process() to keep vertex order if possible.
                # If method is missing, we trust merge_vertices did enough.
                pass

            # 5. Remove unused vertices (Critical so indices match 0..N-1 for IGL)
            mesh.remove_unreferenced_vertices()
            
            # 6. Component Check: Ensure we really have one connected component
            # Sometimes 'remove_degenerate_faces' can disconnect the mesh.
            # LSCM cannot handle multiple disconnected components in one call.
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                # Keep only the largest component by vertex count.
                # IMPORTANT: mutate the original object in place so callers that
                # keep a reference to `mesh` (e.g., for plotting with patch.faces)
                # remain consistent with returned UV indexing.
                largest = max(components, key=lambda m: len(m.vertices))
                mesh.vertices = largest.vertices.copy()
                mesh.faces = largest.faces.copy()
                mesh.remove_unreferenced_vertices()
            
            # Check if mesh is still valid
            if len(mesh.vertices) < 3 or len(mesh.faces) == 0:
                logger.warning("Mesh became empty or degenerate after cleanup.")
                return None
                
        except Exception as e:
            logger.warning(f"Mesh sanitation in Parameterizer failed: {e}")
            return None

        # --- Step 1: Prepare IGL Data ---
        # IGL is strict about types. Ensure correct C++ compatible types.
        # Use np.ascontiguousarray to strip Trimesh wrappers and ensure C-order.
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        # libigl python bindings typically expect MatrixXi-compatible int32.
        # int64 indices can cause binding-level dtype mismatch and unstable behavior.
        f = np.ascontiguousarray(mesh.faces, dtype=np.int32)

        # 2. Find Boundary Loop (LSCM needs a boundary)
        # igl.boundary_loop returns the ordered vertex indices of the boundary
        try:
            bnd = igl.boundary_loop(f)
        except Exception as e:
            logger.error(f"Failed to detect boundary: {e}")
            return None
        
        # Handle libigl return variants robustly:
        # - flat loop: [i0, i1, ...]
        # - nested loops for multiple holes: [[...], (...), np.ndarray(...), ...]
        # We keep only the longest loop after mesh hole filling as a final safeguard.
        if len(bnd) > 0 and not np.isscalar(bnd[0]) and hasattr(bnd[0], "__iter__"):
            loops = []
            for loop in bnd:
                if np.isscalar(loop) or not hasattr(loop, "__iter__"):
                    continue
                loop_arr = np.asarray(loop).reshape(-1)
                if loop_arr.size > 0:
                    loops.append(loop_arr)
            if loops:
                bnd = max(loops, key=lambda x: x.size)

        # Delay strict dtype conversion until we are sure `bnd` is a single flat loop.
        bnd = np.asarray(bnd).reshape(-1)
        bnd = np.array(bnd, dtype=np.int32)

        if len(bnd) < 3:
            logger.error("Mesh has no valid boundary (closed or degenerate).")
            return None

        # 3. Fix Boundary Conditions for LSCM
        # Strategy: Pin two topologically opposite points on the boundary ring.
        # This avoids weak constraints on curled "C"-like boundaries.
        b1_idx = bnd[0]
        b2_idx = bnd[len(bnd) // 2]
        
        # Constraints inputs: b (indices), bc (target coords)
        # Keep indices int32 for libigl MatrixXi compatibility.
        b = np.array([b1_idx, b2_idx], dtype=np.int32)
        bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        # 4. Run LSCM
        uv_normalized = None
        try:
            # Python libigl bindings are inconsistent across versions:
            # - some return (success, uv)
            # - some return (uv, success)
            # - some return uv only
            ret = igl.lscm(v, f, b, bc)

            success = True
            uv = None

            if isinstance(ret, tuple) and len(ret) == 2:
                a, b_ret = ret

                # Identify UV output by array shape (N,2) where N = #V.
                if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2 and a.shape[0] == v.shape[0]:
                    uv = a
                    success = b_ret
                elif isinstance(b_ret, np.ndarray) and b_ret.ndim == 2 and b_ret.shape[1] == 2 and b_ret.shape[0] == v.shape[0]:
                    uv = b_ret
                    success = a
                else:
                    # Last resort: keep prior behavior but guard against bad types.
                    success = a
                    uv = b_ret
            else:
                uv = ret

            # Handle numpy/bool ambiguity for success flags.
            # Convert scalars, lists, and array-like wrappers (e.g., Eigen proxies)
            # to a NumPy array and reduce to a single truth value safely.
            is_success = bool(np.all(np.asarray(success)))

            # Some bindings return only UV (with no success flag), so accept
            # a valid UV array even if success parsing is uncertain.
            has_valid_uv = isinstance(uv, np.ndarray) and uv.ndim == 2 and uv.shape[0] == v.shape[0] and uv.shape[1] == 2

            if is_success or has_valid_uv:
                uv_normalized = Parameterizer._normalize_uv(uv)
            else:
                logger.warning("IGL LSCM solver returned failure status (Matrix likely singular).")

        except (ValueError, TypeError) as e:
            # Some binding-level failures throw ragged-array conversion errors
            # before robust tuple/array inspection can happen.
            logger.warning(f"LSCM returned non-rectangular/invalid output: {e}")
            uv_normalized = None
        except Exception as e:
            logger.warning(f"LSCM Exception: {e}")

        # 5. Fallback: Harmonic Parameterization
        # Only triggered if LSCM absolutely fails
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
            # Ensure bnd is int32
            bnd = bnd.astype(np.int32)
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

    @staticmethod
    def _face_quality_mask(vertices, faces, area_faces, min_area=1e-5, min_angle_deg=2.0, max_aspect_ratio=50.0):
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

# --- Self-Contained Unit Test ---
if __name__ == "__main__":
    pass
