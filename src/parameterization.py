import numpy as np
import igl
import trimesh
import logging
import importlib.metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LSCM")

class Parameterizer:
    """
    Handles the flattening of 3D meshes into 2D UV coordinates using LSCM
    (Least Squares Conformal Maps), with a Harmonic fallback.
    """
    _lscm_runtime_logged = False
    
    @staticmethod
    def flatten_patch(mesh: trimesh.Trimesh, method: str = "auto", return_info: bool = False):
        """
        Flatten a 3D mesh patch to 2D.
        
        Args:
            mesh: A single connected component (trimesh object).
            
        Returns:
            np.ndarray: UV coordinates of shape (N, 2), or None if failed.
        """
        diag = {
            "method": method,
            "diskification_triggered": False,
            "diskification_success": False,
            "topology_before": {},
            "topology_after": {},
            "face_count_before_topology_gate": 0,
            "face_count_after_topology_gate": 0,
            "vertex_count_before_topology_gate": 0,
            "vertex_count_after_topology_gate": 0,
            "area_before_topology_gate": 0.0,
            "area_after_topology_gate": 0.0,
            "failure_reason": None,
        }
        mode = (method or "auto").strip().lower()
        if mode not in {"auto", "lscm", "harmonic", "spherical", "cylindrical"}:
            logger.warning(f"Unknown parameterization method '{method}', fallback to auto.")
            mode = "auto"

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
                diag["failure_reason"] = "mesh_degenerate_after_cleanup"
                return (None, diag) if return_info else None
                
        except Exception as e:
            logger.warning(f"Mesh sanitation in Parameterizer failed: {e}")
            diag["failure_reason"] = "mesh_sanitation_failed"
            return (None, diag) if return_info else None

        # --- Step 1: Prepare IGL Data ---
        # IGL is strict about types. Ensure correct C++ compatible types.
        # Use np.ascontiguousarray to strip Trimesh wrappers and ensure C-order.
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        # libigl python bindings typically expect MatrixXi-compatible int32.
        # int64 indices can cause binding-level dtype mismatch and unstable behavior.
        f = np.ascontiguousarray(mesh.faces, dtype=np.int32)

        # Projection-only methods (no boundary/topology constraint required).
        if mode == "spherical":
            uv_sphere = Parameterizer._flatten_spherical(v)
            if uv_sphere is None:
                diag["failure_reason"] = "spherical_projection_failed"
            return (uv_sphere, diag) if return_info else uv_sphere
        if mode == "cylindrical":
            uv_cyl = Parameterizer._flatten_cylindrical(v)
            if uv_cyl is None:
                diag["failure_reason"] = "cylindrical_projection_failed"
            return (uv_cyl, diag) if return_info else uv_cyl

        # Topology gate before LSCM: a valid open disk patch should satisfy
        # Euler characteristic chi = V - E + F = 1, and have exactly one boundary loop.
        chi, n_edges, n_boundary_loops = Parameterizer._topology_stats(f, len(v))
        diag["face_count_before_topology_gate"] = int(len(f))
        diag["vertex_count_before_topology_gate"] = int(len(v))
        diag["area_before_topology_gate"] = float(mesh.area) if hasattr(mesh, "area") else 0.0
        diag["topology_before"] = {"chi": int(chi), "edges": int(n_edges), "boundary_loops": int(n_boundary_loops)}
        logger.info(
            "Euler topology stats before LSCM: "
            f"V={len(v)}, E={n_edges}, F={len(f)}, chi={chi}, boundary_loops={n_boundary_loops} "
            "(expected chi=1 and boundary_loops=1 for disk-like patch)."
        )
        if chi != 1 or n_boundary_loops != 1:
            diag["diskification_triggered"] = True
            logger.warning(
                "Patch not disk-like; attempting diskification before LSCM."
            )
            disk_mesh = Parameterizer._extract_largest_disk_region(mesh)
            if disk_mesh is not None and len(disk_mesh.faces) > 0:
                mesh.vertices = disk_mesh.vertices.copy()
                mesh.faces = disk_mesh.faces.copy()
                mesh.remove_unreferenced_vertices()
                v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                chi, n_edges, n_boundary_loops = Parameterizer._topology_stats(f, len(v))
                diag["diskification_success"] = bool(chi == 1 and n_boundary_loops == 1)
                logger.info(
                    "Euler topology stats after diskification: "
                    f"V={len(v)}, E={n_edges}, F={len(f)}, chi={chi}, boundary_loops={n_boundary_loops}."
                )

        diag["topology_after"] = {"chi": int(chi), "edges": int(n_edges), "boundary_loops": int(n_boundary_loops)}
        diag["face_count_after_topology_gate"] = int(len(f))
        diag["vertex_count_after_topology_gate"] = int(len(v))
        diag["area_after_topology_gate"] = float(mesh.area) if hasattr(mesh, "area") else 0.0
        if chi != 1 or n_boundary_loops != 1:
            logger.warning(
                "Euler characteristic gate failed; "
                f"V={len(v)}, E={n_edges}, F={len(f)}, chi={chi}, boundary_loops={n_boundary_loops}. "
                "Skipping LSCM."
            )
            diag["failure_reason"] = "topology_gate_failed"
            return (None, diag) if return_info else None

        # 2. Find Boundary Loop (LSCM needs a boundary)
        # igl.boundary_loop returns the ordered vertex indices of the boundary
        try:
            bnd = igl.boundary_loop(f)
        except Exception as e:
            logger.error(f"Failed to detect boundary: {e}")
            diag["failure_reason"] = "boundary_detection_failed"
            return (None, diag) if return_info else None
        
        # libigl 2.6.x returns one longest ordered boundary loop.
        bnd = np.asarray(bnd).reshape(-1)
        bnd = np.array(bnd, dtype=np.int32)

        if len(bnd) < 3:
            logger.error("Mesh has no valid boundary (closed or degenerate).")
            diag["failure_reason"] = "invalid_boundary"
            return (None, diag) if return_info else None

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
        if mode in {"auto", "lscm"}:
            try:
                if not Parameterizer._lscm_runtime_logged:
                    try:
                        igl_version = importlib.metadata.version("libigl")
                    except Exception:
                        igl_version = "unknown"
                    logger.info(f"LSCM runtime: libigl={igl_version}")
                    Parameterizer._lscm_runtime_logged = True

                # libigl 2.6.x python bindings return (V_uv, Q) and raise RuntimeError on failure.
                uv, _q = igl.lscm(v, f, b, bc)
                has_valid_uv = isinstance(uv, np.ndarray) and uv.ndim == 2 and uv.shape == (v.shape[0], 2)
                if has_valid_uv:
                    uv_normalized = Parameterizer._normalize_uv(uv)
                else:
                    logger.warning(
                        "IGL LSCM returned unexpected UV shape: "
                        f"{None if uv is None else getattr(uv, 'shape', type(uv))}"
                    )

            except RuntimeError as e:
                logger.warning(f"LSCM solver failed: {e}")
                uv_normalized = None
            except Exception as e:
                logger.warning(f"LSCM Exception: {e}")

        # 5. Fallback: Harmonic Parameterization
        # Only triggered if LSCM absolutely fails
        if uv_normalized is None and mode in {"auto", "harmonic"}:
            logger.info("Attempting Harmonic Parameterization fallback...")
            uv_normalized = Parameterizer._flatten_harmonic(v, f, bnd)

        if uv_normalized is None and diag["failure_reason"] is None:
            diag["failure_reason"] = "parameterization_failed"
        return (uv_normalized, diag) if return_info else uv_normalized

    @staticmethod
    def refine_patch_uv(mesh: trimesh.Trimesh, uv_init: np.ndarray = None, blend_strength: float = 0.8):
        """
        Joint-optimization friendly UV refinement entry.

        Strategy:
          1) Try a fresh robust flattening (LSCM/harmonic fallback).
          2) If uv_init exists, blend the new solution with uv_init to reduce jitter.
          3) Normalize to [0, 1] range.
        """
        uv_new = Parameterizer.flatten_patch(mesh)
        if uv_new is None:
            return None
        if uv_init is None or len(uv_init) != len(uv_new):
            return uv_new
        alpha = float(np.clip(blend_strength, 0.0, 1.0))
        blended = alpha * np.asarray(uv_new, dtype=np.float64) + (1.0 - alpha) * np.asarray(uv_init, dtype=np.float64)
        return Parameterizer._normalize_uv(blended)

    @staticmethod
    def uv_distortion_stats(mesh: trimesh.Trimesh, uv: np.ndarray):
        """
        Lightweight distortion proxy used by outer alternating optimizer.
        """
        if uv is None or len(uv) == 0 or len(mesh.faces) == 0:
            return {"mean": float("inf"), "max": float("inf")}
        f = np.asarray(mesh.faces, dtype=np.int64)
        v3 = np.asarray(mesh.vertices, dtype=np.float64)
        v2 = np.asarray(uv, dtype=np.float64)
        e3 = np.stack([
            np.linalg.norm(v3[f[:, 1]] - v3[f[:, 0]], axis=1),
            np.linalg.norm(v3[f[:, 2]] - v3[f[:, 1]], axis=1),
            np.linalg.norm(v3[f[:, 0]] - v3[f[:, 2]], axis=1),
        ], axis=1)
        e2 = np.stack([
            np.linalg.norm(v2[f[:, 1]] - v2[f[:, 0]], axis=1),
            np.linalg.norm(v2[f[:, 2]] - v2[f[:, 1]], axis=1),
            np.linalg.norm(v2[f[:, 0]] - v2[f[:, 2]], axis=1),
        ], axis=1)
        ratio = e2 / np.maximum(e3, 1e-8)
        val = np.abs(np.log(np.maximum(ratio, 1e-8))).mean(axis=1)
        return {"mean": float(np.mean(val)), "max": float(np.max(val))}

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
    def _flatten_spherical(v):
        """
        Spherical parameterization via direct angular projection.
        """
        try:
            pts = np.asarray(v, dtype=np.float64)
            if len(pts) < 3:
                return None
            c = pts.mean(axis=0, keepdims=True)
            d = pts - c
            r = np.linalg.norm(d, axis=1)
            if np.all(r < 1e-12):
                return None
            x, y, z = d[:, 0], d[:, 1], d[:, 2]
            theta = np.arctan2(y, x)  # [-pi, pi]
            rr = np.maximum(r, 1e-12)
            phi = np.arccos(np.clip(z / rr, -1.0, 1.0))  # [0, pi]
            uv = np.column_stack([(theta + np.pi) / (2.0 * np.pi), phi / np.pi])
            return Parameterizer._normalize_uv(uv)
        except Exception as e:
            logger.error(f"Spherical Parameterization failed: {e}")
            return None

    @staticmethod
    def _flatten_cylindrical(v):
        """
        Cylindrical parameterization using PCA major axis as cylinder axis.
        """
        try:
            pts = np.asarray(v, dtype=np.float64)
            if len(pts) < 3:
                return None
            c = pts.mean(axis=0, keepdims=True)
            d = pts - c
            cov = np.cov(d.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, int(np.argmax(eigvals))]
            axis = axis / max(np.linalg.norm(axis), 1e-12)

            # Build orthonormal basis (u, w) on plane orthogonal to axis.
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(seed, axis)) > 0.9:
                seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u = np.cross(axis, seed)
            u = u / max(np.linalg.norm(u), 1e-12)
            w = np.cross(axis, u)
            w = w / max(np.linalg.norm(w), 1e-12)

            radial_u = d @ u
            radial_w = d @ w
            angle = np.arctan2(radial_w, radial_u)  # [-pi, pi]
            h = d @ axis
            uv = np.column_stack([(angle + np.pi) / (2.0 * np.pi), h])
            return Parameterizer._normalize_uv(uv)
        except Exception as e:
            logger.error(f"Cylindrical Parameterization failed: {e}")
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

    @staticmethod
    def _euler_characteristic(faces, n_vertices):
        """
        Compute Euler characteristic chi = V - E + F for a triangular mesh patch.
        """
        if len(faces) == 0:
            return 0, 0
        edges = np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        n_edges = int(len(np.unique(edges, axis=0)))
        chi = int(n_vertices - n_edges + len(faces))
        return chi, n_edges

    @staticmethod
    def _topology_stats(faces, n_vertices):
        """
        Compute basic topology stats used for LSCM preflight.
        Returns: (chi, n_edges, n_boundary_loops)
        """
        if len(faces) == 0:
            return 0, 0, 0

        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        n_edges = int(len(unique_edges))
        chi = int(n_vertices - n_edges + len(faces))

        boundary_edges = unique_edges[counts == 1]
        if len(boundary_edges) == 0:
            return chi, n_edges, 0
        n_boundary_loops = Parameterizer._count_boundary_loops(boundary_edges)
        return chi, n_edges, n_boundary_loops

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
        for v in adj.keys():
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
    def _extract_largest_disk_region(mesh: trimesh.Trimesh):
        """
        Heuristic diskification:
        Grow connected face regions while preserving chi=1, then keep the largest such region.
        This converts annulus/handle-like patches into a largest simply-connected disk subset.
        """
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        n_faces = len(faces)
        if n_faces == 0:
            return None

        # Build edge -> faces map and face adjacency via shared edges.
        edge_to_faces = {}
        for fi, tri in enumerate(faces):
            for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                e = tuple(sorted((int(e[0]), int(e[1]))))
                edge_to_faces.setdefault(e, []).append(fi)

        neighbors = [set() for _ in range(n_faces)]
        for f_ids in edge_to_faces.values():
            if len(f_ids) == 2:
                a, b = f_ids
                neighbors[a].add(b)
                neighbors[b].add(a)

        best_set = set()
        visited_seed = set()
        for seed in range(n_faces):
            if seed in visited_seed:
                continue
            # Mark this connected dual component as visited for seed iteration.
            comp = set()
            stack = [seed]
            comp.add(seed)
            while stack:
                cur = stack.pop()
                for nei in neighbors[cur]:
                    if nei not in comp:
                        comp.add(nei)
                        stack.append(nei)
            visited_seed.update(comp)

            # Greedy region growth under chi=1 constraint.
            region = {seed}
            frontier = set(neighbors[seed])
            while frontier:
                cand = frontier.pop()
                if cand in region:
                    continue
                trial_faces = np.array(sorted(region | {cand}), dtype=np.int32)
                sub_f = faces[trial_faces]
                unique_v = np.unique(sub_f.reshape(-1))
                remap = -np.ones(int(np.max(unique_v)) + 1, dtype=np.int32)
                remap[unique_v] = np.arange(len(unique_v), dtype=np.int32)
                sub_f_local = remap[sub_f]
                chi, _, loops = Parameterizer._topology_stats(sub_f_local, len(unique_v))
                if chi == 1 and loops == 1:
                    region.add(cand)
                    frontier.update(neighbors[cand] - region)

            if len(region) > len(best_set):
                best_set = region

        if len(best_set) < 3:
            return None

        kept_faces = faces[np.array(sorted(best_set), dtype=np.int32)]
        unique_v = np.unique(kept_faces.reshape(-1))
        new_vertices = np.asarray(mesh.vertices, dtype=np.float64)[unique_v]
        remap = -np.ones(int(np.max(unique_v)) + 1, dtype=np.int32)
        remap[unique_v] = np.arange(len(unique_v), dtype=np.int32)
        new_faces = remap[kept_faces]
        disk_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        disk_mesh.remove_unreferenced_vertices()
        return disk_mesh

# --- Self-Contained Unit Test ---
if __name__ == "__main__":
    pass
