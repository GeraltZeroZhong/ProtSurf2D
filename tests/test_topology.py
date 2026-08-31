import unittest

import numpy as np
import trimesh

from topoppi.config import TopologyConfig
from topoppi.mesh.provenance import initialize_provenance
from topoppi.mesh.topology import TopologyManager


class TopologyTests(unittest.TestCase):
    def test_extracts_synthetic_interface_patch(self):
        mesh_a = trimesh.creation.icosphere(radius=10.0, subdivisions=2)
        coords_b = np.array([[11.0, 0.0, 0.0], [11.0, 1.0, 0.0], [11.0, 0.0, 1.0]])

        topology_config = TopologyConfig(distance_cutoff=2.5, min_patch_vertices=5)
        manager = TopologyManager(mesh_a, coords_b, config=topology_config)
        patches = manager.get_interface_patches()

        self.assertGreaterEqual(len(patches), 1)
        self.assertTrue(all(len(p.vertices) >= 5 for p in patches))
        self.assertEqual(manager.last_report["status"], "ok")
        self.assertEqual(manager.last_report["accepted_patch_count"], len(patches))
        self.assertIn("interface_face_selection", manager.last_report)
        self.assertIn("source_face_ids", patches[0].metadata)
        self.assertEqual(patches[0].metadata["topology_component_before"]["face_count"] > 0, True)
        self.assertEqual(
            manager.last_report["distance_definition"],
            "triangle_centroid_to_nearest_partner_heavy_atom",
        )

    def test_no_interface_report_includes_the_nearest_surface_distance(self):
        mesh_a = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        manager = TopologyManager(
            mesh_a,
            np.asarray([[20.0, 0.0, 0.0]]),
            config=TopologyConfig(distance_cutoff=1.0),
        )

        self.assertEqual(manager.get_interface_patches(), [])
        self.assertEqual(manager.last_report["status"], "no_interface_faces")
        self.assertGreater(manager.last_report["nearest_partner_distance_angstrom"], 18.0)

    def test_patch_filter_uses_physical_area_before_tessellation_count(self):
        coarse = trimesh.creation.icosphere(radius=1.0, subdivisions=1)
        fine = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
        config = TopologyConfig(
            distance_cutoff=5.0,
            min_patch_area_angstrom2=10.0,
            min_patch_vertices=3,
        )
        partner = np.array([[0.0, 0.0, 0.0]])

        coarse_patches = TopologyManager(coarse, partner, config=config).get_interface_patches()
        fine_patches = TopologyManager(fine, partner, config=config).get_interface_patches()

        self.assertEqual(len(coarse_patches), 1)
        self.assertEqual(len(fine_patches), 1)
        self.assertNotEqual(len(coarse_patches[0].vertices), len(fine_patches[0].vertices))

    def test_nonmanifold_vertex_fans_are_split_without_face_or_area_loss(self):
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 3, 4]]),
            process=False,
        )
        mesh.metadata["source_atom_indices"] = np.arange(5, dtype=np.int64)
        initialize_provenance(mesh, stage="fixture")
        area_before = float(mesh.area)

        repaired = TopologyManager._split_nonmanifold_vertices(mesh)

        self.assertEqual(len(repaired.faces), 2)
        self.assertEqual(len(repaired.vertices), 6)
        self.assertAlmostEqual(float(repaired.area), area_before)
        self.assertEqual(len(np.unique(repaired.metadata["source_face_ids"])), 2)
        self.assertEqual(np.count_nonzero(repaired.metadata["source_vertex_ids"] == 0), 2)
        self.assertEqual(repaired.metadata["topology_nonmanifold_vertex_split"]["faces_removed"], 0)

    def test_nonmanifold_edge_component_is_rejected_without_trimming_faces(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.5, -1.0, 0.0],
                    [0.5, 0.0, 1.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
            process=False,
        )
        manager = TopologyManager(
            mesh,
            np.asarray([[0.5, 0.0, 0.0]]),
            config=TopologyConfig(
                distance_cutoff=5.0,
                min_patch_area_angstrom2=0.0,
                min_patch_vertices=3,
            ),
        )

        patches = manager.get_interface_patches()

        self.assertEqual(patches, [])
        self.assertEqual(manager.last_report["dropped_component_count"], 1)
        component = manager.last_report["components"][0]
        self.assertEqual(component["reason"], "nonmanifold_edge_incidence")
        self.assertEqual(component["before_sanitation"]["face_count"], 3)
        self.assertEqual(component["sanitation"]["edge_count_above_allowed_incidence"], 1)

    def test_degenerate_bridge_is_removed_and_valid_children_are_retained(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray(
                [
                    [0, 1, 3],
                    [0, 1, 2],
                    [1, 2, 4],
                ],
                dtype=np.int64,
            ),
            process=False,
        )
        mesh.metadata["source_atom_indices"] = np.arange(5, dtype=np.int64)
        manager = TopologyManager(
            mesh,
            np.asarray([[1.0, 0.25, 0.0]], dtype=np.float64),
            config=TopologyConfig(
                distance_cutoff=5.0,
                min_patch_area_angstrom2=0.0,
                min_patch_vertices=3,
            ),
        )

        patches = manager.get_interface_patches()

        self.assertEqual(len(patches), 2)
        self.assertEqual(sum(len(patch.faces) for patch in patches), 2)
        self.assertAlmostEqual(sum(float(patch.area) for patch in patches), 1.0)
        self.assertEqual(
            [patch.metadata["patch_id"] for patch in patches],
            ["patch_0000_part_00", "patch_0000_part_01"],
        )
        raw_face_sets = [set(patch.metadata["topology_component_before"]["source_face_ids"]) for patch in patches]
        self.assertFalse(raw_face_sets[0] & raw_face_sets[1])
        self.assertEqual(raw_face_sets[0] | raw_face_sets[1], {0, 1, 2})
        self.assertTrue(all(patch.metadata["topology_parent_component_before"]["face_count"] == 3 for patch in patches))
        self.assertEqual(manager.last_report["component_count"], 1)
        self.assertEqual(manager.last_report["sanitized_component_count"], 2)
        self.assertEqual(manager.last_report["post_filter_component_record_count"], 2)
        self.assertEqual(manager.last_report["split_source_component_count"], 1)
        self.assertEqual(manager.last_report["accepted_patch_count"], 2)
        self.assertEqual(manager.last_report["dropped_component_count"], 0)
        self.assertEqual(
            sum(record["before_sanitation"]["face_count"] for record in manager.last_report["components"]),
            3,
        )
        self.assertEqual(
            sum(record["after_sanitation"]["face_count"] for record in manager.last_report["components"]),
            2,
        )
        self.assertTrue(all(record["status"] == "accepted" for record in manager.last_report["components"]))
