import logging

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from topoppi.config import TopologyConfig
from topoppi.mesh.provenance import (
    connected_face_components,
    filter_faces,
    initialize_provenance,
    merge_duplicate_vertices,
    provenance_summary,
    remove_duplicate_faces,
    tracked_submesh,
    tracked_vertex_duplication,
)

logger = logging.getLogger("Topology")


class TopologyManager:
    """
    Manages the extraction and topological processing of protein interface patches.
    """

    def __init__(self, mesh_A: trimesh.Trimesh, coords_B: np.ndarray, config: TopologyConfig | None = None):
        """
        Initialize the Topology Manager.

        Args:
            mesh_A: Full surface mesh of Chain A (trimesh.Trimesh).
            coords_B: Array of atom coordinates for Chain B (N, 3).
        """
        self.config = config or TopologyConfig()
        self.mesh_A = mesh_A
        initialize_provenance(self.mesh_A, stage="surface_input")
        self.coords_B = coords_B
        self.last_report = {}
        self.component_provenance = []

        # Build KDTree for Chain B once for fast queries
        self.tree_B = cKDTree(self.coords_B)

        logger.debug(
            "Initialized TopologyManager with Mesh A (%d verts) and %d atoms of B.",
            len(self.mesh_A.vertices),
            len(coords_B),
        )

    def get_interface_patches(self) -> list:
        """
        Extract interface patches and split them into separate connected components.

        Args:
            Distance and filtering parameters are provided by ``TopologyConfig``.

        Returns:
            List[trimesh.Trimesh]: A list of clean, topologically simple mesh patches.
        """
        distance_cutoff = self.config.distance_cutoff
        min_patch_area = self.config.min_patch_area_angstrom2
        min_patch_vertices = self.config.min_patch_vertices
        self.component_provenance = []
        logger.debug("Extracting patches with cutoff=%g A.", distance_cutoff)

        # Select whole faces by the distance from their centroid.  This avoids
        # the one-triangle dilation introduced by an any-vertex rule while
        # keeping exact source-face provenance downstream.
        face_centroids = np.asarray(self.mesh_A.triangles_center, dtype=np.float64)
        face_distances, _ = self.tree_B.query(face_centroids)
        face_mask = face_distances <= distance_cutoff

        if not np.any(face_mask):
            logger.warning("No interface found! Check your cutoff or coordinates.")
            self.last_report = {
                "status": "no_interface_faces",
                "distance_cutoff_angstrom": float(distance_cutoff),
                "nearest_partner_distance_angstrom": float(np.min(face_distances)),
                "distance_definition": "triangle_centroid_to_nearest_partner_heavy_atom",
                "surface": self._compact_provenance(provenance_summary(self.mesh_A)),
                "selected_vertex_count": 0,
                "selected_face_count": 0,
                "components": [],
            }
            return []

        # Preserve complete source faces and their provenance.
        raw_submesh = tracked_submesh(
            self.mesh_A,
            np.flatnonzero(face_mask),
            stage="interface_face_selection",
        )

        if len(raw_submesh.vertices) == 0:
            logger.warning("Submesh extraction resulted in empty mesh.")
            self.last_report = {
                "status": "empty_interface_submesh",
                "distance_cutoff_angstrom": float(distance_cutoff),
                "distance_definition": "triangle_centroid_to_nearest_partner_heavy_atom",
                "surface": self._compact_provenance(provenance_summary(self.mesh_A)),
                "components": [],
            }
            return []

        # 4. Connectivity Splitting
        components = connected_face_components(raw_submesh)
        logger.debug("Raw interface split into %d components.", len(components))

        valid_patches = []
        component_records = []
        sanitized_component_count = 0
        split_source_component_count = 0
        for i, face_indices in enumerate(components):
            comp = tracked_submesh(raw_submesh, face_indices, stage="interface_component")
            before = provenance_summary(comp)
            rejection = self._patch_threshold_rejection(comp)
            if rejection is not None:
                logger.info(
                    "  Dropped component %d (area=%.6g A^2, vertices=%d).",
                    i,
                    float(comp.area),
                    len(comp.vertices),
                )
                self._record_dropped_component(component_records, i, before, rejection)
                continue

            # 6. Patch Sanitization
            cleaned_patches, sanitation = self._sanitize_patch(comp)
            if not cleaned_patches:
                self._record_dropped_component(
                    component_records,
                    i,
                    before,
                    str(sanitation.get("failure_reason", "sanitation_failed_or_empty")),
                    sanitation=sanitation,
                )
                continue
            child_denominators = self._partition_component_before_sanitation(comp, cleaned_patches)
            child_count = len(cleaned_patches)
            sanitized_component_count += child_count
            split_source_component_count += int(child_count > 1)
            for child_index, (cleaned_patch, child_before_mesh) in enumerate(
                zip(cleaned_patches, child_denominators, strict=True)
            ):
                patch_id = f"patch_{i:04d}" if child_count == 1 else f"patch_{i:04d}_part_{child_index:02d}"
                child_before = provenance_summary(child_before_mesh)
                child_sanitation = {
                    **sanitation,
                    "source_component_index": int(i),
                    "sanitized_subcomponent_index": int(child_index),
                    "sanitized_subcomponent_count": int(child_count),
                }
                cleaned_patch.metadata["topology_sanitation"] = child_sanitation
                after = provenance_summary(cleaned_patch)
                rejection = self._patch_threshold_rejection(
                    cleaned_patch,
                    suffix="_after_sanitation",
                )
                if rejection is not None:
                    self._record_dropped_component(
                        component_records,
                        i,
                        child_before,
                        rejection,
                        after,
                        sanitation=child_sanitation,
                        patch_id=patch_id,
                        parent_before=before,
                    )
                    continue
                self._record_accepted_component(
                    component_records,
                    i,
                    cleaned_patch,
                    child_before,
                    after,
                    patch_id=patch_id,
                    parent_before=before,
                )
                valid_patches.append(cleaned_patch)

        logger.debug("Retained %d valid patches after filtering.", len(valid_patches))
        surface_summary = provenance_summary(self.mesh_A)
        selected_summary = provenance_summary(raw_submesh)
        self.last_report = {
            "status": "ok" if valid_patches else "no_patch_after_filtering",
            "distance_cutoff_angstrom": float(distance_cutoff),
            "nearest_partner_distance_angstrom": float(np.min(face_distances)),
            "distance_definition": "triangle_centroid_to_nearest_partner_heavy_atom",
            "min_patch_area_angstrom2": float(min_patch_area),
            "min_patch_vertices": int(min_patch_vertices),
            "selected_vertex_count": int(len(raw_submesh.vertices)),
            "selected_face_count": int(np.count_nonzero(face_mask)),
            "surface": self._compact_provenance(surface_summary),
            "interface_face_selection": self._compact_provenance(selected_summary),
            "interface_face_area_retention_ratio": float(selected_summary["area"] / surface_summary["area"])
            if surface_summary["area"] > 0.0
            else float("nan"),
            "component_count": int(len(components)),
            "sanitized_component_count": int(sanitized_component_count),
            "post_filter_component_record_count": int(len(component_records)),
            "split_source_component_count": int(split_source_component_count),
            "accepted_patch_count": int(len(valid_patches)),
            "dropped_component_count": int(sum(record.get("status") != "accepted" for record in component_records)),
            "components": component_records,
        }
        return valid_patches

    def _patch_threshold_rejection(self, mesh, *, suffix: str = "") -> str | None:
        if float(mesh.area) < float(self.config.min_patch_area_angstrom2):
            return f"below_min_patch_area{suffix}"
        if len(mesh.vertices) < int(self.config.min_patch_vertices):
            return f"below_min_patch_vertices{suffix}"
        return None

    @staticmethod
    def _compact_provenance(summary: dict) -> dict:
        """Keep report-size stage evidence; full mappings stay on the mesh."""

        compact = dict(summary)
        for key in ("source_vertex_ids", "source_face_ids", "source_atom_indices"):
            values = compact.pop(key, [])
            compact[f"{key.removesuffix('_ids').removesuffix('_indices')}_count"] = int(len(values))
        return compact

    def _record_dropped_component(
        self,
        records,
        index,
        before,
        reason,
        after=None,
        sanitation=None,
        *,
        patch_id=None,
        parent_before=None,
    ):
        resolved_patch_id = str(patch_id or f"component_{index:04d}")
        report = {
            "component_index": int(index),
            "patch_id": resolved_patch_id,
            "status": "dropped",
            "reason": reason,
            "before_sanitation": self._compact_provenance(before),
        }
        if parent_before is not None:
            report["parent_component_before_sanitation"] = self._compact_provenance(parent_before)
        if after is not None:
            report["after_sanitation"] = self._compact_provenance(after)
        if sanitation is not None:
            report["sanitation"] = dict(sanitation)
            report["sanitized_subcomponent_index"] = int(sanitation.get("sanitized_subcomponent_index", 0))
            report["sanitized_subcomponent_count"] = int(sanitation.get("sanitized_subcomponent_count", 1))
        records.append(report)
        self.component_provenance.append(
            {
                "component_index": int(index),
                "patch_id": resolved_patch_id,
                "status": "dropped",
                "reason": reason,
                "before_sanitation": before,
                "after_sanitation": after,
                "parent_component_before_sanitation": parent_before,
                "sanitation": dict(sanitation) if sanitation is not None else None,
            }
        )

    def _record_accepted_component(
        self,
        records,
        index,
        patch,
        before,
        after,
        *,
        patch_id=None,
        parent_before=None,
    ):
        patch_id = str(patch_id or f"patch_{index:04d}")
        sanitation = dict(patch.metadata.get("topology_sanitation", {}))
        patch.metadata.update(
            {
                "original_index": index,
                "patch_id": patch_id,
                "topology_component_before": before,
                "topology_parent_component_before": parent_before or before,
                "topology_provenance": after,
            }
        )
        records.append(
            {
                "component_index": int(index),
                "patch_id": patch_id,
                "status": "accepted",
                "sanitized_subcomponent_index": int(sanitation.get("sanitized_subcomponent_index", 0)),
                "sanitized_subcomponent_count": int(sanitation.get("sanitized_subcomponent_count", 1)),
                "before_sanitation": self._compact_provenance(before),
                "parent_component_before_sanitation": self._compact_provenance(parent_before or before),
                "after_sanitation": self._compact_provenance(after),
                "face_retention_ratio": float(after["face_count"] / before["face_count"]),
                "materialized_vertex_count_ratio": float(after["vertex_count"] / before["vertex_count"]),
                "source_vertex_retention_ratio": float(
                    len(after["source_vertex_ids"]) / len(before["source_vertex_ids"])
                )
                if before["source_vertex_ids"]
                else float("nan"),
                "area_retention_ratio": (
                    float(after["area"] / before["area"]) if before["area"] > 0.0 else float("nan")
                ),
                "sanitation": sanitation,
            }
        )
        self.component_provenance.append(
            {
                "component_index": int(index),
                "patch_id": patch_id,
                "status": "accepted",
                "reason": None,
                "before_sanitation": before,
                "after_sanitation": after,
                "parent_component_before_sanitation": parent_before or before,
                "sanitation": sanitation,
            }
        )

    @staticmethod
    def _partition_component_before_sanitation(
        parent: trimesh.Trimesh,
        children: list[trimesh.Trimesh],
    ) -> list[trimesh.Trimesh]:
        """Partition the raw parent faces across sanitized child patches.

        Sanitization can remove a degenerate bridge or materialize distinct
        copies of a non-manifold vertex, turning one edge-connected input
        component into several valid components.  Each raw face is assigned to
        exactly one child so per-patch retention denominators still partition
        the original interface component.  Removed faces go to the child with
        the greatest source-vertex overlap, with a deterministic index tie-break.
        """

        if not children:
            return []
        initialize_provenance(parent)
        parent_source_faces = np.asarray(parent.metadata["source_face_ids"], dtype=np.int64)
        parent_source_vertices = np.asarray(parent.metadata["source_vertex_ids"], dtype=np.int64)
        child_face_sets = [
            set(int(value) for value in np.asarray(child.metadata["source_face_ids"], dtype=np.int64))
            for child in children
        ]
        child_vertex_sets = [
            set(int(value) for value in np.asarray(child.metadata["source_vertex_ids"], dtype=np.int64))
            for child in children
        ]
        assignments: list[list[int]] = [[] for _child in children]
        for face_index, source_face in enumerate(parent_source_faces):
            owners = [
                child_index
                for child_index, source_faces in enumerate(child_face_sets)
                if int(source_face) in source_faces
            ]
            if len(owners) == 1:
                assignments[owners[0]].append(int(face_index))
                continue
            if len(owners) > 1:
                raise RuntimeError("A source face was duplicated across sanitized components.")
            face_source_vertices = set(
                int(parent_source_vertices[vertex]) for vertex in np.asarray(parent.faces[face_index], dtype=np.int64)
            )
            owner = max(
                range(len(children)),
                key=lambda child_index: (
                    len(face_source_vertices & child_vertex_sets[child_index]),
                    -child_index,
                ),
            )
            assignments[owner].append(int(face_index))

        if sorted(index for group in assignments for index in group) != list(range(len(parent.faces))):
            raise RuntimeError("Raw interface faces were not partitioned exactly once.")
        return [
            tracked_submesh(
                parent,
                np.asarray(face_indices, dtype=np.int64),
                stage="interface_component_retention_partition",
            )
            for face_indices in assignments
        ]

    def _sanitize_patch(self, mesh: trimesh.Trimesh):
        """Clean a patch and return every valid edge-connected component."""
        working = merge_duplicate_vertices(mesh, stage="topology_merge_duplicate_vertices")
        valid_faces = working.area_faces > self.config.degenerate_face_area
        degenerate_face_count = int(np.count_nonzero(~valid_faces))
        if not np.all(valid_faces):
            working = filter_faces(working, valid_faces, stage="topology_remove_degenerate_faces")

        face_count_before_deduplication = int(len(working.faces))
        working = remove_duplicate_faces(working, stage="topology_remove_duplicate_faces")
        duplicate_face_count = int(face_count_before_deduplication - len(working.faces))
        nonmanifold = self._nonmanifold_edge_report(working)
        sanitation = {
            "degenerate_face_count_removed": degenerate_face_count,
            "duplicate_face_count_removed": duplicate_face_count,
            **nonmanifold,
            "failure_reason": None,
        }
        if int(nonmanifold["edge_count_above_allowed_incidence"]) > 0:
            sanitation["failure_reason"] = "nonmanifold_edge_incidence"
            return [], sanitation

        working = self._split_nonmanifold_vertices(working)

        components = connected_face_components(working)
        sanitation["connected_component_count_after_vertex_split"] = int(len(components))
        if not components:
            sanitation["failure_reason"] = "empty_after_sanitation"
            return [], sanitation
        sanitized = [
            tracked_submesh(
                working,
                face_indices,
                stage="topology_sanitized_component",
            )
            for face_indices in components
        ]
        for component in sanitized:
            component.metadata["topoppi_topology_sanitized"] = True
        return sanitized, sanitation

    def _nonmanifold_edge_report(self, mesh: trimesh.Trimesh) -> dict[str, int]:
        """Count invalid edge incidences; ambiguous sheets are never trimmed."""

        if len(mesh.faces) == 0:
            return {
                "maximum_edge_face_incidence": 0,
                "edge_count_above_allowed_incidence": 0,
            }

        edges = mesh.edges_sorted
        _unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        return {
            "maximum_edge_face_incidence": int(np.max(edge_counts)) if len(edge_counts) else 0,
            "edge_count_above_allowed_incidence": int(
                np.count_nonzero(edge_counts > int(self.config.max_edge_face_incidence))
            ),
        }

    @staticmethod
    def _split_nonmanifold_vertices(mesh: trimesh.Trimesh):
        """Separate disconnected incident-face fans without deleting surface faces.

        A manifold vertex has one connected fan ("umbrella") of incident faces.
        When several fans meet only at the same vertex, they represent distinct
        topological copies of that 3-D point.  Materializing those copies repairs
        the combinatorics while retaining every triangle, its area, and its root
        vertex/face/atom provenance.
        """
        if len(mesh.faces) == 0:
            return mesh

        faces = np.asarray(mesh.faces, dtype=np.int64)
        faces_per_vertex = [[] for _ in range(len(mesh.vertices))]
        for fi, tri in enumerate(faces):
            faces_per_vertex[tri[0]].append(fi)
            faces_per_vertex[tri[1]].append(fi)
            faces_per_vertex[tri[2]].append(fi)

        fans_to_split: list[tuple[int, list[list[int]]]] = []
        for v_idx, incident in enumerate(faces_per_vertex):
            if len(incident) <= 1:
                continue

            local_adj = {fi: set() for fi in incident}
            neighbor_faces: dict[int, list[int]] = {}
            for fi in incident:
                for neighbor in faces[fi]:
                    if neighbor != v_idx:
                        neighbor_faces.setdefault(int(neighbor), []).append(fi)
            for shared_faces in neighbor_faces.values():
                for fi in shared_faces:
                    local_adj[fi].update(fj for fj in shared_faces if fj != fi)

            # Find connected components of the incident-face fan graph.
            seen = set()
            components: list[list[int]] = []
            for fi in incident:
                if fi in seen:
                    continue
                component = []
                stack = [fi]
                seen.add(fi)
                while stack:
                    cur = stack.pop()
                    component.append(cur)
                    for nei in local_adj[cur]:
                        if nei not in seen:
                            seen.add(nei)
                            stack.append(nei)
                components.append(sorted(component))

            if len(components) > 1:
                components.sort(key=lambda component: (component[0], len(component)))
                fans_to_split.append((v_idx, components))

        if fans_to_split:
            vertices = np.asarray(mesh.vertices, dtype=np.float64).tolist()
            repaired_faces = faces.copy()
            vertex_to_input = list(range(len(mesh.vertices)))
            added_vertices = 0
            for vertex, components in fans_to_split:
                for component in components[1:]:
                    duplicate = len(vertices)
                    vertices.append(np.asarray(mesh.vertices[vertex], dtype=np.float64).tolist())
                    vertex_to_input.append(int(vertex))
                    for face_index in component:
                        locations = np.flatnonzero(repaired_faces[face_index] == vertex)
                        if len(locations) != 1:
                            raise RuntimeError("Invalid triangle incidence while splitting a vertex fan.")
                        repaired_faces[face_index, int(locations[0])] = duplicate
                    added_vertices += 1
            mesh = tracked_vertex_duplication(
                mesh,
                np.asarray(vertices, dtype=np.float64),
                repaired_faces,
                np.asarray(vertex_to_input, dtype=np.int64),
                stage="topology_split_nonmanifold_vertices",
            )
            mesh.metadata["topology_nonmanifold_vertex_split"] = {
                "source_vertex_count": int(len(fans_to_split)),
                "added_vertex_copy_count": int(added_vertices),
                "faces_removed": 0,
            }
            preview = [int(vertex) for vertex, _components in fans_to_split[:8]]
            logger.warning(
                "Split non-manifold vertex fans: source_vertices=%d, added_copies=%d, "
                "faces_removed=0, sample_vertex_ids=%s.",
                len(fans_to_split),
                added_vertices,
                preview,
            )
        return mesh
