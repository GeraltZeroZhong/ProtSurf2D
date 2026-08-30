import unittest

import numpy as np
import trimesh

from topoppi.atlas.metrics import UVAtlasMetrics
from topoppi.atlas.uv import face_domain_hash
from topoppi.config import ParameterizationConfig
from topoppi.mesh.parameterization import Parameterizer
from topoppi.mesh.provenance import (
    OPTCUTS_GEOMETRY_VERTEX_IDS,
    initialize_provenance,
    tracked_submesh,
    tracked_vertex_duplication,
)


def square_mesh():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.1, 1.0, 0.03], [0.0, 1.0, -0.1]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    mesh.metadata["source_atom_indices"] = np.array([10, 11, 12, 13])
    initialize_provenance(mesh, stage="fixture")
    return mesh


class ParameterizationDomainTests(unittest.TestCase):
    def test_tracked_vertex_duplication_rejects_reordered_face_correspondence(self):
        mesh = square_mesh()
        vertices = np.vstack([mesh.vertices, mesh.vertices[0]])
        vertex_to_input = np.asarray([0, 1, 2, 3, 0], dtype=np.int64)
        reordered_faces = np.asarray([[4, 2, 3], [0, 1, 2]], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "face order and corner correspondence"):
            tracked_vertex_duplication(
                mesh,
                vertices,
                reordered_faces,
                vertex_to_input,
                stage="invalid_cut_fixture",
            )

    def test_tracked_vertex_duplication_rejects_changed_geometry(self):
        mesh = square_mesh()
        vertices = np.asarray(mesh.vertices).copy()
        vertices[0, 2] += 0.1

        with self.assertRaisesRegex(ValueError, "must not change"):
            tracked_vertex_duplication(
                mesh,
                vertices,
                np.asarray(mesh.faces),
                np.arange(len(vertices)),
                stage="invalid_geometry_fixture",
            )

    def test_angular_comparator_frame_is_rigid_motion_and_permutation_equivariant(self):
        rng = np.random.default_rng(20260817)
        points = rng.normal(size=(40, 3)) * np.array([3.0, 2.0, 1.0]) + np.array([5.0, -2.0, 8.0])
        rotation, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 0] *= -1.0
        transformed = points @ rotation.T + np.array([10.0, 4.0, -7.0])
        permutation = rng.permutation(len(points))
        inverse = np.argsort(permutation)

        for flatten in (Parameterizer._flatten_spherical, Parameterizer._flatten_cylindrical):
            reference = flatten(points)
            moved = flatten(transformed)
            permuted = flatten(points[permutation])[inverse]
            self.assertIsNotNone(reference)
            self.assertIsNotNone(moved)
            self.assertIsNotNone(permuted)
            for candidate in (moved, permuted):
                periodic_delta = (candidate[:, 0] - reference[:, 0] + 0.5) % 1.0 - 0.5
                periodic_delta -= np.median(periodic_delta)
                np.testing.assert_allclose(periodic_delta, 0.0, atol=1e-12)
                np.testing.assert_allclose(candidate[:, 1], reference[:, 1], atol=1e-12)

    def test_angular_comparators_reject_intrinsically_ambiguous_frames(self):
        regular_tetrahedron = np.asarray([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        circular_prism = np.vstack(
            [np.column_stack([np.cos(angles), np.sin(angles), np.full(len(angles), height)]) for height in (-4.0, 4.0)]
        )

        rectangular_box = np.asarray([[x, y, z] for x in (-3.0, 3.0) for y in (-2.0, 2.0) for z in (-1.0, 1.0)])

        for points in (regular_tetrahedron, circular_prism, rectangular_box):
            self.assertIsNone(Parameterizer._flatten_spherical(points))
            self.assertIsNone(Parameterizer._flatten_cylindrical(points))

        permutation = np.random.default_rng(20260819).permutation(len(rectangular_box))
        self.assertIsNone(Parameterizer._flatten_cylindrical(rectangular_box[permutation]))

    def test_provenance_length_mismatch_is_not_silently_reset(self):
        mesh = square_mesh()
        mesh.metadata["source_vertex_ids"] = np.array([0], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "source_vertex_ids"):
            initialize_provenance(mesh)

    def test_every_method_uses_the_identical_prepared_face_domain(self):
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10, slim_iterations=2))
        prepared = square_mesh()
        output, info = parameterizer.prepare_patch(prepared, return_info=True)
        self.assertIs(output, prepared)
        expected_hash = face_domain_hash(prepared)

        for method in ("lscm", "harmonic", "slim", "spherical", "cylindrical"):
            candidate = prepared.copy()
            uv, diagnostic = parameterizer.flatten_patch(candidate, method=method, return_info=True)
            self.assertIsNotNone(uv, msg=f"{method}: {diagnostic}")
            self.assertEqual(face_domain_hash(candidate), expected_hash)
            self.assertEqual(diagnostic["source_face_hash"], expected_hash)

        self.assertEqual(info["face_retention_ratio"], 1.0)
        self.assertEqual(info["source_atom_retention_ratio"], 1.0)

    def test_uniform_harmonic_is_injective_and_initializes_slim(self):
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.1],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, -0.1],
                    [0.0, 0.0, 0.4],
                ]
            ),
            faces=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
            process=False,
        )
        initialize_provenance(mesh, stage="fixture")
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10, slim_iterations=2))

        harmonic_mesh = mesh.copy()
        harmonic_uv, harmonic_info = parameterizer.flatten_patch(
            harmonic_mesh,
            method="harmonic",
            return_info=True,
        )
        self.assertIsNotNone(harmonic_uv, msg=harmonic_info)
        self.assertTrue(
            UVAtlasMetrics.parameterization_injectivity_stats(harmonic_mesh, harmonic_uv)["globally_injective"]
        )

        slim_mesh = mesh.copy()
        slim_uv, slim_info = parameterizer.flatten_patch(slim_mesh, method="slim", return_info=True)
        self.assertIsNotNone(slim_uv, msg=slim_info)
        self.assertEqual(slim_info["initializer"], "uniform_weight_tutte_harmonic")

    def test_slim_receives_positive_tutte_orientation_and_improves_it(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.1],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, -0.1],
                    [0.7, 0.2, 0.3],
                ]
            ),
            faces=np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
            process=False,
        )
        initialize_provenance(mesh, stage="asymmetric_disk")
        parameterizer = Parameterizer(ParameterizationConfig(slim_iterations=20))

        harmonic_mesh = mesh.copy()
        harmonic_uv = parameterizer.flatten_patch(harmonic_mesh, method="harmonic")
        slim_mesh = mesh.copy()
        slim_uv = parameterizer.flatten_patch(slim_mesh, method="slim")

        harmonic_corners = np.asarray(harmonic_uv)[harmonic_mesh.faces]
        first_edge = harmonic_corners[:, 1] - harmonic_corners[:, 0]
        second_edge = harmonic_corners[:, 2] - harmonic_corners[:, 0]
        signed_area = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
        self.assertTrue(np.all(signed_area > 0.0))
        harmonic_stats = UVAtlasMetrics.symmetric_dirichlet_stats(
            harmonic_mesh,
            harmonic_uv,
        )
        slim_stats = UVAtlasMetrics.symmetric_dirichlet_stats(slim_mesh, slim_uv)
        self.assertLess(slim_stats["mean"], harmonic_stats["mean"] - 0.1)

    def test_boundary_orientation_rejects_inconsistent_face_winding(self):
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        inconsistent_faces = np.asarray([[0, 1, 2], [0, 3, 2]])

        boundary = Parameterizer._ordered_boundary_loop(vertices, inconsistent_faces)

        self.assertEqual(len(boundary), 0)

    def test_harmonic_boundary_uses_three_dimensional_arc_length(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.5, 0.2],
            ]
        )
        boundary = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
        faces = np.asarray(
            [[0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 0, 5]],
            dtype=np.int32,
        )

        uv = Parameterizer._flatten_harmonic(vertices, faces, boundary)
        pins = Parameterizer._boundary_antipodal_pins(vertices, boundary)

        self.assertIsNotNone(uv)
        boundary_angles = np.unwrap(np.arctan2(uv[boundary, 1], uv[boundary, 0]))
        angular_steps = np.diff(boundary_angles)
        edge_lengths = np.linalg.norm(np.diff(vertices[boundary], axis=0), axis=1)
        np.testing.assert_allclose(
            angular_steps / angular_steps.sum(),
            edge_lengths / edge_lengths.sum(),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(int(pins[0]), 0)
        self.assertEqual(int(pins[1]), 3)

    def test_tracked_submesh_retains_root_face_vertex_and_atom_ids(self):
        mesh = square_mesh()
        subset = tracked_submesh(mesh, [1], stage="fixture_subset")

        np.testing.assert_array_equal(subset.metadata["source_face_ids"], [1])
        np.testing.assert_array_equal(subset.metadata["source_vertex_ids"], [0, 2, 3])
        np.testing.assert_array_equal(subset.metadata["source_atom_indices"], [10, 12, 13])
        self.assertEqual(subset.metadata["provenance_history"][-1]["stage"], "fixture_subset")

    def test_diskification_is_deterministic_and_retains_every_face(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ]
        )
        source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        initialize_provenance(source, stage="annulus")
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10))

        outputs = []
        prepared_meshes = []
        diagnostics = []
        for _ in range(2):
            candidate = source.copy()
            output, info = parameterizer.prepare_patch(candidate, return_info=True)
            self.assertIsNotNone(output, msg=info)
            outputs.append(np.asarray(output.metadata["source_face_ids"]))
            prepared_meshes.append(output)
            diagnostics.append(info)

        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertTrue(diagnostics[0]["diskification_triggered"])
        self.assertTrue(diagnostics[0]["diskification_success"])
        self.assertEqual(diagnostics[0]["diskification_faces_removed"], 0)
        self.assertEqual(diagnostics[0]["face_retention_ratio"], 1.0)
        self.assertEqual(diagnostics[0]["area_retention_ratio"], 1.0)
        self.assertEqual(diagnostics[0]["source_vertex_retention_ratio"], 1.0)
        self.assertGreater(diagnostics[0]["materialized_vertex_count_ratio"], 1.0)
        self.assertEqual(len(outputs[0]), len(faces))
        self.assertGreater(diagnostics[0]["diskification_added_vertex_count"], 0)
        self.assertEqual(
            len(np.unique(prepared_meshes[0].metadata[OPTCUTS_GEOMETRY_VERTEX_IDS])),
            len(source.vertices),
        )
        self.assertEqual(
            len(prepared_meshes[0].metadata[OPTCUTS_GEOMETRY_VERTEX_IDS]),
            len(prepared_meshes[0].vertices),
        )
        source_ids = np.asarray(source.metadata["source_vertex_ids"], dtype=np.int64)
        prepared = source.copy()
        prepared, report = parameterizer.prepare_patch(prepared, return_info=True)
        current_sources = np.asarray(prepared.metadata["source_vertex_ids"], dtype=np.int64)
        boundary_source_edges = []
        edge_counts = {}
        for tri in np.asarray(prepared.faces, dtype=np.int64):
            for left, right in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                current_edge = tuple(sorted((int(left), int(right))))
                edge_counts[current_edge] = edge_counts.get(current_edge, 0) + 1
        for current_edge, count in edge_counts.items():
            if count == 1:
                boundary_source_edges.append(tuple(sorted(int(current_sources[index]) for index in current_edge)))
        duplicated_boundary_sources = {edge for edge in boundary_source_edges if boundary_source_edges.count(edge) == 2}
        expected_cut_sources = {tuple(edge) for edge in report["diskification_cut_edges_source_vertex_ids"]}
        self.assertEqual(duplicated_boundary_sources, expected_cut_sources)
        self.assertTrue(all(edge[0] in source_ids and edge[1] in source_ids for edge in expected_cut_sources))

    def test_diskification_cut_is_invariant_to_vertex_and_face_permutation(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        base = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        initialize_provenance(base, stage="annulus")

        vertex_order = np.array([5, 2, 7, 0, 4, 1, 6, 3])
        old_to_new = np.empty(len(vertex_order), dtype=np.int64)
        old_to_new[vertex_order] = np.arange(len(vertex_order))
        face_order = np.array([4, 0, 6, 2, 7, 3, 1, 5])
        permuted = trimesh.Trimesh(
            vertices=vertices[vertex_order],
            faces=old_to_new[faces[face_order]],
            process=False,
        )
        permuted.metadata["source_vertex_ids"] = vertex_order.copy()
        permuted.metadata["source_face_ids"] = face_order.copy()
        initialize_provenance(permuted, stage="annulus_permuted")

        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10))
        first, first_info = parameterizer.prepare_patch(base, return_info=True)
        second, second_info = parameterizer.prepare_patch(permuted, return_info=True)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(
            first_info["diskification_cut_edges_source_vertex_ids"],
            second_info["diskification_cut_edges_source_vertex_ids"],
        )
        self.assertEqual(face_domain_hash(first), face_domain_hash(second))
        self.assertEqual(first_info["face_retention_ratio"], 1.0)
        self.assertEqual(second_info["face_retention_ratio"], 1.0)

    def test_closed_genus_zero_surface_is_opened_without_deleting_a_face(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]),
            faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
            process=False,
        )
        initialize_provenance(mesh, stage="closed_tetrahedron")
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10))

        prepared, report = parameterizer.prepare_patch(mesh, return_info=True)

        self.assertIsNotNone(prepared, msg=report)
        self.assertTrue(report["diskification_closed_surface_opening"])
        self.assertTrue(report["diskification_success"])
        self.assertEqual(report["diskification_faces_removed"], 0)
        self.assertEqual(report["topology_after"]["chi"], 1)
        self.assertEqual(report["topology_after"]["boundary_loops"], 1)

    def test_preparation_does_not_remerge_intentional_topology_copies(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]),
            faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
            process=False,
        )
        initialize_provenance(mesh, stage="closed_tetrahedron")
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10))
        prepared, first_report = parameterizer.prepare_patch(mesh, return_info=True)
        self.assertIsNotNone(prepared, msg=first_report)
        self.assertGreater(
            len(prepared.metadata["source_vertex_ids"]) - len(np.unique(prepared.metadata["source_vertex_ids"])),
            0,
        )

        replay = prepared.copy()
        replay.metadata.pop("topoppi_parameterization_prepared", None)
        replay.metadata.pop("parameterization_preparation", None)
        replay, second_report = parameterizer.prepare_patch(replay, return_info=True)

        self.assertIsNotNone(replay, msg=second_report)
        self.assertTrue(second_report["topological_vertex_copies_preserved"])
        self.assertEqual(second_report["face_retention_ratio"], 1.0)
        self.assertEqual(second_report["topology_after"], first_report["topology_after"])
        self.assertIsNotNone(replay.copy())

    def test_topology_gate_rejects_nonmanifold_complex_with_disk_like_scalars(self):
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
        initialize_provenance(mesh, stage="three_faces_one_edge")
        parameterizer = Parameterizer(ParameterizationConfig(min_face_area=1e-10))

        topology = parameterizer._topology_report(mesh.faces, len(mesh.vertices))
        prepared, report = parameterizer.prepare_patch(mesh, return_info=True)

        self.assertEqual(
            (topology["chi"], topology["edges"], topology["boundary_loops"]),
            (1, 7, 1),
        )
        self.assertEqual(topology["nonmanifold_edge_count"], 1)
        self.assertFalse(topology["is_connected_two_manifold"])
        self.assertIsNone(prepared)
        self.assertEqual(
            report["diskification_failure"],
            "input_is_not_a_connected_two_manifold",
        )
        self.assertEqual(report["failure_reason"], "topology_gate_failed")


if __name__ == "__main__":
    unittest.main()
