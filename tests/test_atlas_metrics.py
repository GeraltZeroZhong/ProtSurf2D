import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from shapely.errors import ShapelyError

from topoppi.atlas.metrics import UVAtlasMetrics
from topoppi.atlas.packing import apply_packed_uv, pack_mesh_charts, resolved_chart_gap
from topoppi.atlas.uv import set_uv_layout
from topoppi.mesh.provenance import OPTCUTS_GEOMETRY_VERTEX_IDS


class AtlasMetricTests(unittest.TestCase):
    @staticmethod
    def _square_mesh():
        return trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )

    def test_global_reflection_is_legal_but_local_flip_is_detected(self):
        mesh = self._square_mesh()
        uv_ok = np.asarray(mesh.vertices[:, :2])
        uv_reflected = uv_ok * np.array([-1.0, 1.0])
        uv_local_flip = uv_ok[mesh.faces].copy()
        uv_local_flip[1, [1, 2]] = uv_local_flip[1, [2, 1]]

        self.assertEqual(UVAtlasMetrics.flip_rate(mesh, uv_ok), 0.0)
        self.assertEqual(UVAtlasMetrics.flip_rate(mesh, uv_reflected), 0.0)
        self.assertAlmostEqual(UVAtlasMetrics.flip_rate(mesh, uv_local_flip), 0.5)

    def test_distortion_is_invariant_to_global_similarity(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )

        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        transformed = 7.5 * (uv @ rotation.T) + np.array([12.0, -4.0])
        original = UVAtlasMetrics.distortion_stats(mesh, uv)
        changed = UVAtlasMetrics.distortion_stats(mesh, transformed)

        self.assertAlmostEqual(original["mean"], changed["mean"], places=12)
        self.assertAlmostEqual(
            UVAtlasMetrics.area_distortion_stats(mesh, uv)["mean"],
            UVAtlasMetrics.area_distortion_stats(mesh, transformed)["mean"],
            places=12,
        )
        self.assertAlmostEqual(
            UVAtlasMetrics.symmetric_dirichlet_stats(mesh, uv)["mean"],
            UVAtlasMetrics.symmetric_dirichlet_stats(mesh, transformed)["mean"],
            places=12,
        )

    def test_scale_fair_metrics_handle_extreme_uv_units(self):
        mesh = self._square_mesh()
        uv = np.asarray(mesh.vertices[:, :2])
        expected = {
            "distortion": UVAtlasMetrics.distortion_stats(mesh, uv)["mean"],
            "symmetric_dirichlet": UVAtlasMetrics.symmetric_dirichlet_stats(mesh, uv)["mean"],
            "angle": UVAtlasMetrics.angle_distortion_stats(mesh, uv)["mean"],
            "area": UVAtlasMetrics.area_distortion_stats(mesh, uv)["mean"],
        }

        for scale in (1e-12, 1e12):
            scaled = uv * scale
            observed = {
                "distortion": UVAtlasMetrics.distortion_stats(mesh, scaled)["mean"],
                "symmetric_dirichlet": UVAtlasMetrics.symmetric_dirichlet_stats(mesh, scaled)["mean"],
                "angle": UVAtlasMetrics.angle_distortion_stats(mesh, scaled)["mean"],
                "area": UVAtlasMetrics.area_distortion_stats(mesh, scaled)["mean"],
            }
            for metric, value in observed.items():
                self.assertAlmostEqual(value, expected[metric], places=12, msg=f"{metric=}, {scale=}")

    def test_packing_is_invariant_to_input_uv_units(self):
        mesh = self._square_mesh()
        uv = np.asarray(mesh.vertices[:, :2])
        packed_by_scale = []
        for scale in (1e-12, 1.0, 1e12):
            candidate = mesh.copy()
            set_uv_layout(candidate, uv * scale)
            packed, _transforms, _report = pack_mesh_charts([candidate], gap=0.0)
            packed_by_scale.append(packed[0])

        np.testing.assert_allclose(packed_by_scale[0], packed_by_scale[1], atol=1e-12)
        np.testing.assert_allclose(packed_by_scale[2], packed_by_scale[1], atol=1e-12)

    def test_symmetric_dirichlet_uses_the_analytic_best_global_scale(self):
        mesh = self._square_mesh()
        anisotropic = np.asarray(mesh.vertices[:, :2]) * np.array([4.0, 0.5])
        stats = UVAtlasMetrics.symmetric_dirichlet_stats(mesh, anisotropic)

        # One global rescaling can remove the product of the two stretches but
        # cannot remove their anisotropy.  The optimum has stretches sqrt(8)
        # and 1/sqrt(8), hence E_SD = 8 + 1/8.
        self.assertAlmostEqual(stats["mean"], 8.0 + 1.0 / 8.0, places=12)
        self.assertEqual(
            stats["scale_alignment"],
            "analytic_global_symmetric_dirichlet_minimum",
        )

    def test_internal_seam_is_separate_from_outer_boundary(self):
        mesh = self._square_mesh()
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        continuous = UVAtlasMetrics.seam_stats(mesh, corners)
        corners[1, 0] += np.array([3.0, 0.0])
        discontinuous = UVAtlasMetrics.seam_stats(mesh, corners)

        self.assertEqual(continuous["seam_edge_count"], 0)
        self.assertEqual(continuous["boundary_edge_count"], 4)
        self.assertEqual(discontinuous["seam_edge_count"], 1)

    def test_packing_has_exact_zero_overlap_and_reproducible_affines(self):
        meshes = []
        original = []
        for offset in (0.0, 3.0):
            mesh = trimesh.Trimesh(
                vertices=np.array([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]]),
                faces=np.array([[0, 1, 2]]),
                process=False,
            )
            uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            set_uv_layout(mesh, uv)
            meshes.append(mesh)
            original.append(uv[mesh.faces])

        packed, transforms, _report = pack_mesh_charts(meshes, gap=0.2)
        apply_packed_uv(meshes, packed, transforms)
        applied_gap = resolved_chart_gap(meshes, 0.2)
        stats = UVAtlasMetrics.atlas_geometry_stats(meshes, padding=applied_gap)

        self.assertEqual(stats["overlap_area"], 0.0)
        self.assertEqual(stats["padding_violations"], 0)
        self.assertGreaterEqual(stats["min_chart_gap"], applied_gap - 1e-10)
        for source, target, transform in zip(original, packed, transforms, strict=True):
            matrix = np.asarray(transform.affine_matrix)
            translation = np.array([transform.translation_u, transform.translation_v])
            np.testing.assert_allclose(source @ matrix.T + translation, target)

    def test_chart_gap_scales_with_surface_area(self):
        mesh = self._square_mesh()
        scaled = mesh.copy()
        scaled.apply_scale(3.0)

        self.assertAlmostEqual(resolved_chart_gap([scaled], 0.08), 3.0 * resolved_chart_gap([mesh], 0.08))

    def test_packing_rejects_a_degenerate_chart(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        set_uv_layout(mesh, np.zeros((3, 2)))

        with self.assertRaisesRegex(ValueError, "degenerate"):
            pack_mesh_charts([mesh])

    def test_unique_overlap_and_multiplicity_overdraw_are_distinct(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            process=False,
        )
        repeated_triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        set_uv_layout(mesh, np.repeat(repeated_triangle[None, :, :], 3, axis=0))

        stats = UVAtlasMetrics.atlas_geometry_stats([mesh], key="uv")

        self.assertAlmostEqual(stats["overlap_area"], 0.5)
        self.assertAlmostEqual(stats["within_chart_overlap_area"], 0.5)
        self.assertAlmostEqual(stats["overdraw_area"], 1.0)
        self.assertAlmostEqual(stats["within_chart_overdraw_area"], 1.0)

    def test_global_injectivity_rejects_overlap_without_local_flips(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.array([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )
        repeated = np.repeat(
            np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
            2,
            axis=0,
        )

        stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, repeated)

        self.assertEqual(stats["flip_face_count"], 0)
        self.assertGreater(stats["overdraw_ratio"], 0.0)
        self.assertFalse(stats["globally_injective"])

    def test_global_injectivity_accepts_a_simple_square(self):
        mesh = self._square_mesh()

        stats = UVAtlasMetrics.parameterization_injectivity_stats(
            mesh,
            np.asarray(mesh.vertices[:, :2]),
        )

        self.assertTrue(stats["globally_injective"])
        self.assertEqual(stats["flip_face_count"], 0)
        self.assertEqual(stats["overdraw_ratio"], 0.0)
        self.assertEqual(stats["source_distinct_zero_measure_contact_pair_count"], 0)

    def test_optcuts_constraint_energy_retains_raw_scale_and_identity_four(self):
        mesh = self._square_mesh()
        identity = np.asarray(mesh.vertices[:, :2])
        stretched = identity * np.asarray([3.0, 1.0])

        self.assertAlmostEqual(
            UVAtlasMetrics.optcuts_constraint_energy(mesh, identity),
            4.0,
        )
        self.assertGreater(
            UVAtlasMetrics.optcuts_constraint_energy(mesh, stretched),
            4.1,
        )

    def test_global_injectivity_rejects_source_distinct_point_contact(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )
        corners = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],
            ]
        )

        stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, corners)

        self.assertEqual(stats["within_chart_overdraw_area"], 0.0)
        self.assertEqual(stats["source_distinct_zero_measure_contact_pair_count"], 1)
        self.assertFalse(stats["globally_injective"])

    def test_global_injectivity_does_not_collapse_repaired_vertex_fan_copies(self):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )
        mesh.metadata["source_vertex_ids"] = np.asarray([10, 11, 12, 10, 13, 14])
        mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.arange(6, dtype=np.int64)
        corners = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]],
            ]
        )

        stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, corners)

        self.assertEqual(stats["within_chart_overdraw_area"], 0.0)
        self.assertEqual(stats["source_distinct_zero_measure_contact_pair_count"], 1)
        self.assertFalse(stats["globally_injective"])

    def test_continuity_is_measured_on_the_pre_cut_geometry_domain(self):
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
        mesh.metadata[OPTCUTS_GEOMETRY_VERTEX_IDS] = np.asarray([0, 1, 2, 3, 0, 2])
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1] += np.asarray([3.0, 0.0])

        stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, corners)

        self.assertTrue(stats["globally_injective"])
        self.assertTrue(stats["continuous_on_materialized_cut_mesh"])
        self.assertFalse(stats["continuous_on_pre_cut_geometry_domain"])
        self.assertFalse(stats["continuous_on_input_mesh"])
        self.assertEqual(stats["continuity_vertex_identity"], "optcuts_geometry_vertex_ids")

    def test_injectivity_predicates_are_invariant_to_uniform_uv_scale(self):
        mesh = self._square_mesh()
        uv = np.asarray(mesh.vertices[:, :2])

        for scale in (1e-12, 1.0, 1e12):
            stats = UVAtlasMetrics.parameterization_injectivity_stats(
                mesh,
                uv * scale,
            )
            self.assertTrue(stats["globally_injective"], msg=f"scale={scale:g}")
            self.assertEqual(stats["flip_face_count"], 0)

    def test_overlap_rejection_is_invariant_to_uniform_uv_scale(self):
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]]),
            process=False,
        )
        repeated = np.repeat(
            np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
            2,
            axis=0,
        )

        for scale in (1e-12, 1.0, 1e12):
            stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, repeated * scale)
            self.assertFalse(stats["globally_injective"], msg=f"scale={scale:g}")
            self.assertGreater(stats["overdraw_ratio"], 0.0)

    def test_seam_detection_is_invariant_to_uniform_uv_scale(self):
        mesh = self._square_mesh()
        corners = np.asarray(mesh.vertices[:, :2])[mesh.faces]
        corners[1, 0] += np.array([0.25, 0.0])

        for scale in (1e-12, 1.0, 1e12):
            stats = UVAtlasMetrics.seam_stats(mesh, corners * scale)
            self.assertEqual(stats["seam_edge_count"], 1, msg=f"scale={scale:g}")

    def test_global_injectivity_reports_a_whole_map_reflection(self):
        mesh = self._square_mesh()
        reflected = np.asarray(mesh.vertices[:, :2]).copy()
        reflected[:, 0] *= -1.0

        stats = UVAtlasMetrics.parameterization_injectivity_stats(mesh, reflected)

        self.assertTrue(stats["globally_injective"])
        self.assertTrue(stats["global_reflection_required_for_positive_orientation"])

    def test_exact_geometry_failure_is_conservative_and_does_not_escape(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with patch("topoppi.atlas.metrics.unary_union", side_effect=ShapelyError("synthetic")):
            injectivity = UVAtlasMetrics.parameterization_injectivity_stats(mesh, uv)
            atlas = UVAtlasMetrics.atlas_geometry_stats([mesh], uv_arrays=[uv])

        self.assertFalse(injectivity["globally_injective"])
        self.assertEqual(injectivity["geometry_evaluation_status"], "failed")
        self.assertEqual(atlas["status"], "geometry_evaluation_failed")

    def test_disjointness_certificate_skips_pairwise_triangle_intersections(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with patch.object(
            UVAtlasMetrics,
            "_pairwise_overlap_geometry",
            side_effect=AssertionError("pairwise intersections were unnecessary"),
        ):
            stats = UVAtlasMetrics.atlas_geometry_stats([mesh], uv_arrays=[uv])

        self.assertEqual(stats["geometry_evaluation_status"], "ok")
        self.assertEqual(stats["overdraw_area"], 0.0)
