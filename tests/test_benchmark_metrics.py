import unittest

import numpy as np
import trimesh

from topoppi.atlas.metrics import UVAtlasMetrics
from topoppi.atlas.uv import set_uv_layout
from topoppi.benchmarking.metrics_utils import (
    agg_geo_stability,
    jacobian_stability_stats,
    quality_block,
    rasterize_feature_maps,
    symmetric_dirichlet_energy,
)


def triangle_mesh():
    return trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )


class BenchmarkMetricTests(unittest.TestCase):
    def test_jacobian_ratios_use_total_positive_face_area_at_both_levels(self):
        vertices = np.vstack(
            [
                np.asarray([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]])
                for offset in (0.0, 2.0, 4.0)
            ]
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            process=False,
        )
        corner_uv = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        set_uv_layout(mesh, corner_uv)

        patch_stats = jacobian_stability_stats(mesh, corner_uv)
        aggregate_stats = agg_geo_stability([mesh])

        for stats in (patch_stats, aggregate_stats):
            self.assertAlmostEqual(stats["negative_jacobian_ratio"], 1.0 / 3.0)
            self.assertAlmostEqual(stats["invalid_jacobian_area_ratio"], 1.0 / 3.0)

    def test_symmetric_dirichlet_is_global_scale_invariant(self):
        mesh = triangle_mesh()
        uv = np.asarray(mesh.vertices[:, :2])
        transformed = uv * 9.0 + np.array([30.0, -8.0])

        self.assertAlmostEqual(
            symmetric_dirichlet_energy(mesh, uv),
            symmetric_dirichlet_energy(mesh, transformed),
            places=12,
        )

    def test_invalid_jacobian_is_not_silently_dropped(self):
        mesh = triangle_mesh()
        set_uv_layout(mesh, np.zeros((3, 2)))
        quality = quality_block([mesh], patch_gap=0.1)

        self.assertTrue(np.isinf(quality["distortion"]["mean"]))
        self.assertTrue(np.isinf(quality["angle_distortion"]["mean"]))
        self.assertTrue(np.isinf(quality["area_distortion"]["mean"]))
        self.assertEqual(quality["distortion"]["invalid_area_ratio"], 1.0)

    def test_collinear_uv_triangle_is_invalid_for_angle_metric(self):
        mesh = triangle_mesh()
        collinear_uv = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])

        angle_values, _weights = UVAtlasMetrics.angle_distortion_samples(mesh, collinear_uv)

        self.assertTrue(np.isinf(angle_values[0]))

    def test_rasterization_fills_triangles_not_only_vertices(self):
        mesh = triangle_mesh()
        set_uv_layout(mesh, np.asarray(mesh.vertices[:, :2]))
        raster = rasterize_feature_maps([mesh], size=64)

        self.assertGreater(np.count_nonzero(raster), 100)
        self.assertGreater(float(np.max(raster)), 0.0)

    def test_multichart_seam_normalization_uses_total_area_scale(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        corner_uv = np.asarray(mesh.vertices[mesh.faces, :2])
        corner_uv[1] += np.array([3.0, 0.0])
        set_uv_layout(mesh, corner_uv)

        quality = quality_block([mesh.copy(), mesh.copy()], patch_gap=0.1)

        self.assertAlmostEqual(quality["seam"]["seam_length_3d"], 2.0 * np.sqrt(2.0))
        self.assertAlmostEqual(quality["seam"]["seam_length_3d_normalized"], 2.0)


if __name__ == "__main__":
    unittest.main()
