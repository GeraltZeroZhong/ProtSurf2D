import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from topoppi.atlas.footprints import (
    analyze_residue_footprints,
    contact_partner_degrees,
    geometric_contact_partner_map,
    residue_aware_residue_weights,
    residue_footprint_pieces,
    residue_fragmentation_report,
    write_residue_footprint_sidecar,
)
from topoppi.atlas.metrics import UVAtlasMetrics
from topoppi.atlas.uv import set_uv_layout
from topoppi.io.io_loader import PDBLoader
from topoppi.mesh.provenance import OPTCUTS_GEOMETRY_VERTEX_IDS

FIXTURES = Path(__file__).parent / "fixtures"


class ResidueFootprintTests(unittest.TestCase):
    @staticmethod
    def _square_mesh() -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )

    def test_equal_split_has_half_fragmentation(self):
        mesh = self._square_mesh()
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([3.0, 0.0])

        records, seams = analyze_residue_footprints(mesh, corners, ["A:GLY:1"] * 4)

        self.assertEqual(np.count_nonzero(seams.seam_mask), 1)
        self.assertEqual(records[0]["baseline_component_count"], 1)
        self.assertEqual(records[0]["component_count_after_seams"], 2)
        self.assertEqual(records[0]["extra_component_count"], 1)
        self.assertAlmostEqual(records[0]["fragmentation_mass"] / records[0]["footprint_area"], 0.5)

        pieces = residue_footprint_pieces(
            mesh,
            corners,
            ["A:GLY:1"] * 4,
        )["A:GLY:1"]
        self.assertEqual(len(pieces), 2)
        self.assertAlmostEqual(sum(piece["footprint_mass_fraction"] for piece in pieces), 1.0)
        np.testing.assert_allclose(
            sorted(piece["uv_centroid"] for piece in pieces),
            [[2.0 / 3.0, 1.0 / 3.0], [10.0 / 3.0, 2.0 / 3.0]],
        )

    def test_geometric_contact_map_is_the_shared_residue_contact_definition(self):
        loader = PDBLoader(FIXTURES / "tiny_complex.pdb")
        coords_a, atoms_a = loader.get_chain_data("A")
        coords_b, atoms_b = loader.get_chain_data("B")

        partners = geometric_contact_partner_map(
            coords_a,
            atoms_a,
            coords_b,
            atoms_b,
            distance_cutoff=3.0,
        )

        self.assertEqual(
            partners,
            {
                "A:ALA:2": {"B:ALA:2": 3, "B:GLY:1": 8},
                "A:GLY:1": {"B:GLY:1": 7},
            },
        )
        self.assertEqual(
            contact_partner_degrees(partners),
            {"A:ALA:2": 2.0, "A:GLY:1": 1.0},
        )

    def test_piece_centroid_integrates_a_mixed_residue_corner_indicator(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            faces=np.asarray([[0, 1, 2]]),
            process=False,
        )

        pieces = residue_footprint_pieces(
            mesh,
            np.asarray(mesh.vertices[:, :2]),
            ["R", "X", "X"],
        )["R"]

        self.assertEqual(len(pieces), 1)
        self.assertAlmostEqual(pieces[0]["footprint_mass"], mesh.area / 3.0)
        np.testing.assert_allclose(pieces[0]["uv_centroid"], [1.0, 0.5])

    def test_tiny_piece_costs_less_than_balanced_split(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.2, 0.0, 0.0]]),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([4.0, 0.0])

        records, _seams = analyze_residue_footprints(mesh, corners, ["A:GLY:1"] * 4)
        score = records[0]["fragmentation_mass"] / records[0]["footprint_area"]

        self.assertAlmostEqual(score, 20.0 / 121.0)
        self.assertLess(score, 0.5)

    def test_natural_disconnections_are_the_zero_cost_baseline(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )

        records, _seams = analyze_residue_footprints(
            mesh,
            np.asarray(mesh.vertices[:, :2])[mesh.faces],
            ["A:GLY:1"] * 6,
        )

        self.assertEqual(records[0]["baseline_component_count"], 2)
        self.assertEqual(records[0]["component_count_after_seams"], 2)
        self.assertEqual(records[0]["extra_component_count"], 0)
        self.assertEqual(records[0]["fragmentation_mass"], 0.0)

    def test_face_support_does_not_connect_across_an_unlabelled_edge(self):
        mesh = self._square_mesh()
        labels = ["X", "R", "X", "R"]

        records, _seams = analyze_residue_footprints(
            mesh,
            np.asarray(mesh.vertices[:, :2])[mesh.faces],
            labels,
        )
        residue = next(record for record in records if record["residue"] == "R")

        self.assertEqual(residue["face_count"], 2)
        self.assertEqual(residue["dual_edge_count"], 0)
        self.assertEqual(residue["baseline_component_count"], 2)

    def test_cycles_expose_nonseparating_seam_crossings(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]),
            faces=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
            process=False,
        )
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[0] += np.asarray([10.0, 0.0])

        records, seams = analyze_residue_footprints(mesh, corners, ["A:GLY:1"] * 4)
        residue = records[0]

        self.assertEqual(np.count_nonzero(seams.seam_mask), 3)
        self.assertEqual(residue["cycle_rank"], 3)
        self.assertEqual(residue["extra_component_count"], 1)
        self.assertEqual(residue["nonseparating_seam_crossing_edge_count"], 2)
        self.assertAlmostEqual(residue["fragmentation_mass"] / residue["footprint_area"], 0.375)
        self.assertEqual(UVAtlasMetrics.seam_stats(mesh, corners)["seam_edge_count"], 3)

    def test_atlas_report_uses_source_atoms_and_interaction_weights(self):
        mesh = self._square_mesh()
        mesh.metadata["source_atom_indices"] = np.arange(4, dtype=np.int64)
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([3.0, 0.0])
        set_uv_layout(mesh, corners)

        report = residue_fragmentation_report(
            [mesh],
            ["A:GLY:1"] * 4,
            interaction_weights={"A:GLY:1": 3.0},
        )

        self.assertAlmostEqual(report["mean_fragmentation"], 0.5)
        self.assertAlmostEqual(report["area_weighted_fragmentation"], 0.5)
        self.assertAlmostEqual(report["interaction_weighted_fragmentation"], 0.5)
        self.assertEqual(report["interaction_weight_sum"], 3.0)
        self.assertAlmostEqual(report["objective_weighted_fragmentation"], 0.5)
        self.assertEqual(report["objective_weight_sum"], 4.0)
        self.assertEqual(report["nonlocality_audit"]["status"], "tree_like_footprints_only")

    def test_objective_weighted_fragmentation_includes_noncontact_residues(self):
        fragmented = self._square_mesh()
        fragmented.metadata["source_atom_indices"] = np.arange(4, dtype=np.int64)
        fragmented_corners = np.asarray(fragmented.vertices[:, :2])[fragmented.faces]
        fragmented_corners[1] += np.asarray([3.0, 0.0])
        set_uv_layout(fragmented, fragmented_corners)

        intact = self._square_mesh()
        intact.metadata["source_atom_indices"] = np.arange(4, 8, dtype=np.int64)
        set_uv_layout(intact, np.asarray(intact.vertices[:, :2])[intact.faces])

        report = residue_fragmentation_report(
            [fragmented, intact],
            ["A:GLY:1"] * 4 + ["A:ALA:2"] * 4,
            interaction_weights={"A:GLY:1": 3.0},
        )

        self.assertAlmostEqual(report["interaction_weighted_fragmentation"], 0.5)
        self.assertAlmostEqual(report["objective_weighted_fragmentation"], 0.4)
        self.assertEqual(report["interaction_weight_sum"], 3.0)
        self.assertEqual(report["objective_weight_sum"], 5.0)

    def test_cpp_sidecar_records_only_real_internal_seams(self):
        mesh = self._square_mesh()
        mesh.metadata["source_atom_indices"] = np.arange(4, dtype=np.int64)
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([3.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "footprints.txt"
            metadata = write_residue_footprint_sidecar(
                mesh,
                corners,
                ["A:GLY:1"] * 4,
                str(path),
                residue_weights={"A:GLY:1": 2.0},
            )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(metadata["internal_edge_count"], 1)
        self.assertEqual(metadata["initial_seam_edge_count"], 1)
        self.assertEqual(lines[0], "TOPOPPI_FOOTPRINT_V2")
        self.assertEqual(lines[1], "COUNTS 2 1 1 4")
        self.assertEqual(lines[2], "SOURCES 0 1 2 3")
        self.assertEqual(lines[3], "WEIGHTS 2")
        self.assertEqual(lines[-1], "EDGE 0 0 2 0 1 1 1 0")

    def test_source_edge_identity_recovers_a_preexisting_topology_cut(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [4, 5, 3]]),
            process=False,
        )
        mesh.metadata["source_vertex_ids"] = np.asarray([0, 1, 2, 3, 0, 2])
        mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.asarray([0, 1, 2, 3, 0, 2])
        mesh.metadata["source_atom_indices"] = np.asarray([0, 1, 2, 3, 0, 2])
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([3.0, 0.0])

        records, seams = analyze_residue_footprints(mesh, corners, ["A:GLY:1"] * 6)

        self.assertEqual(np.count_nonzero(seams.seam_mask), 1)
        np.testing.assert_array_equal(seams.source_edges[seams.seam_mask], [[0, 2]])
        self.assertEqual(records[0]["baseline_component_count"], 1)
        self.assertEqual(records[0]["component_count_after_seams"], 2)
        self.assertAlmostEqual(
            records[0]["fragmentation_mass"] / records[0]["footprint_area"],
            0.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "footprints.txt"
            write_residue_footprint_sidecar(
                mesh,
                corners,
                ["A:GLY:1"] * 4,
                str(path),
            )
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[2], "SOURCES 0 1 2 3 0 2")
        self.assertEqual(lines[-1], "EDGE 0 0 2 0 1 1 1 0")

    def test_repaired_fan_copies_do_not_create_a_false_internal_edge(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )
        mesh.metadata["source_vertex_ids"] = np.asarray([0, 1, 2, 0, 2, 3])
        mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.arange(6, dtype=np.int64)
        mesh.metadata["source_atom_indices"] = np.asarray([0, 1, 2, 0, 2, 3])
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]

        records, seams = analyze_residue_footprints(
            mesh,
            corners,
            ["A:GLY:1"] * 6,
        )

        self.assertEqual(np.count_nonzero(seams.internal_mask), 0)
        self.assertEqual(records[0]["baseline_component_count"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "footprints.txt"
            metadata = write_residue_footprint_sidecar(
                mesh,
                corners,
                ["A:GLY:1"] * 4,
                str(path),
            )
            text = path.read_text(encoding="utf-8")
        self.assertEqual(metadata["internal_edge_count"], 0)
        self.assertIn("SOURCES 0 1 2 3 4 5", text)

    def test_sidecar_can_map_a_source_collapsed_optcuts_input(self):
        mesh = self._square_mesh()
        mesh.metadata["source_vertex_ids"] = np.asarray([10, 11, 12, 13])
        mesh.metadata["source_atom_indices"] = np.arange(4, dtype=np.int64)
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "footprints.txt"
            metadata = write_residue_footprint_sidecar(
                mesh,
                corners,
                ["A:GLY:1"] * 4,
                str(path),
                input_source_vertices=[10, 11, 12, 13],
            )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[1], "COUNTS 2 1 1 4")
        self.assertEqual(lines[2], "SOURCES 10 11 12 13")
        self.assertEqual(metadata["mesh_vertex_count"], 4)
        self.assertEqual(metadata["input_vertex_count"], 4)

    def test_residue_aware_weights_keep_a_unit_residue_baseline(self):
        weights = residue_aware_residue_weights(
            ["A:GLY:1", "A:GLY:1", "A:LYS:2"],
            {"A:GLY:1": 3.0},
        )

        self.assertEqual(weights, {"A:GLY:1": 4.0, "A:LYS:2": 1.0})


if __name__ == "__main__":
    unittest.main()
