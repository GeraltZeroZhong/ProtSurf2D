import numpy as np
import trimesh
from scipy.spatial import KDTree
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Topology")

class TopologyManager:
    """
    Manages the extraction and topological processing of protein interface patches.
    """
    def __init__(self, mesh_A: trimesh.Trimesh, coords_B: np.ndarray):
        """
        Initialize the Topology Manager.
        
        Args:
            mesh_A: Full surface mesh of Chain A (trimesh.Trimesh).
            coords_B: Array of atom coordinates for Chain B (N, 3).
        """
        # Deep copy to avoid modifying the original mesh externally
        self.mesh_A = mesh_A.copy()
        self.coords_B = coords_B
        
        # Pre-process Mesh A: Ensure it's valid before we start slicing it.
        # This removes duplicate vertices which can break LSCM later.
        self.mesh_A.process(validate=True)
        
        # Build KDTree for Chain B once for fast queries
        self.tree_B = KDTree(self.coords_B)
        
        logger.info(f"Initialized TopologyManager with Mesh A ({len(self.mesh_A.vertices)} verts) and {len(coords_B)} atoms of B.")

    def get_interface_patches(self, distance_cutoff=5.0, min_patch_vertices=50) -> list:
        """
        Extract interface patches and split them into separate connected components.
        
        Args:
            distance_cutoff: Distance threshold in Angstroms (default: 5.0).
            min_patch_vertices: Minimum vertices to consider a patch valid (noise filter).
            
        Returns:
            List[trimesh.Trimesh]: A list of clean, topologically simple mesh patches.
        """
        logger.info(f"Extracting patches with cutoff={distance_cutoff}A...")
        
        # 1. Vertex Selection (KDTree Query)
        dists, _ = self.tree_B.query(self.mesh_A.vertices)
        vertex_mask = dists < distance_cutoff
        
        if not np.any(vertex_mask):
            logger.warning("No interface found! Check your cutoff or coordinates.")
            return []

        # 2. Face Selection (Expansion)
        face_mask = vertex_mask[self.mesh_A.faces].any(axis=1)
        
        # 3. Submesh Extraction
        raw_submesh = self.mesh_A.submesh([face_mask], append=True)
        
        # Robustness Fix: Handle list return type
        if isinstance(raw_submesh, list):
            if len(raw_submesh) == 0:
                return []
            raw_submesh = trimesh.util.concatenate(raw_submesh)

        if raw_submesh is None or len(raw_submesh.vertices) == 0:
            logger.warning("Submesh extraction resulted in empty mesh.")
            return []

        # 4. Connectivity Splitting
        components = raw_submesh.split(only_watertight=False)
        logger.info(f"Raw interface split into {len(components)} components.")
        
        valid_patches = []
        for i, comp in enumerate(components):
            # 5. Noise Filtering
            if len(comp.vertices) < min_patch_vertices:
                logger.info(f"  Dropped component {i} (Vertices: {len(comp.vertices)} < {min_patch_vertices})")
                continue
                
            # 6. Patch Sanitization
            cleaned_patch = self._sanitize_patch(comp)
            
            if cleaned_patch is not None:
                cleaned_patch.metadata['original_index'] = i 
                valid_patches.append(cleaned_patch)
        
        logger.info(f"Retained {len(valid_patches)} valid patches after filtering.")
        return valid_patches

    def _sanitize_patch(self, mesh: trimesh.Trimesh):
        """
        Clean a single patch to prepare it for parameterization.
        """
        try:
            # Remove vertices not used in any face
            mesh.remove_unreferenced_vertices()
            
            # Remove faces with zero area
            try:
                valid_faces = mesh.area_faces > 1e-9
                if not np.all(valid_faces):
                    mesh.update_faces(valid_faces)
            except Exception as e:
                logger.warning(f"Skipping degenerate face removal due to: {e}")
            
            mesh.merge_vertices()

            # Remove triangles adjacent to non-manifold edges (edge incidence > 2).
            # LSCM expects a 2-manifold patch; these faces make the linear system ill-posed.
            self._remove_nonmanifold_faces(mesh)

            # Keep the largest connected component after sanitation.
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                largest = max(components, key=lambda m: len(m.vertices))
                mesh.vertices = largest.vertices.copy()
                mesh.faces = largest.faces.copy()
                mesh.remove_unreferenced_vertices()

            # IMPORTANT: do not forcibly smooth open boundary patches.
            # Laplacian smoothing can shrink boundaries and create self-intersections/
            # near-zero-area triangles, which destabilize LSCM.
            return mesh
            
        except Exception as e:
            logger.error(f"Error sanitizing patch: {e}")
            logger.warning("Returning partially sanitized patch due to error.")
            return mesh

    @staticmethod
    def _remove_nonmanifold_faces(mesh: trimesh.Trimesh):
        """
        Remove faces that touch non-manifold edges (shared by >2 faces).
        """
        if len(mesh.faces) == 0:
            return

        edges = mesh.edges_sorted
        unique_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
        edge_counts = np.bincount(inverse, minlength=len(unique_edges))

        # Each face contributes 3 consecutive edges in mesh.edges_sorted.
        face_edge_counts = edge_counts[inverse].reshape(-1, 3)
        manifold_face_mask = np.all(face_edge_counts <= 2, axis=1)

        if not np.all(manifold_face_mask):
            mesh.update_faces(manifold_face_mask)
            mesh.remove_unreferenced_vertices()

# --- Self-Contained Unit Test ---
if __name__ == "__main__":
    print("Running TopologyManager Test...")
    mesh_a = trimesh.creation.icosphere(radius=10, subdivisions=4)
    b_pts1 = np.array([[11, 0, 0], [11, 1, 0], [11, 0, 1]]) 
    b_pts2 = np.array([[-11, 0, 0], [-11, 1, 0]])
    coords_b = np.vstack([b_pts1, b_pts2])
    
    topo = TopologyManager(mesh_a, coords_b)
    patches = topo.get_interface_patches(distance_cutoff=2.5, min_patch_vertices=5)
    
    print(f"\n[Test Result] Generated {len(patches)} patches.")
    for idx, p in enumerate(patches):
        print(f"  > Patch {idx}: {len(p.vertices)} vertices. Has NaNs? {np.isnan(p.vertices).any()}")
