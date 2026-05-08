import unittest

import numpy as np
import trimesh

from topoppi.atlas.metrics import UVAtlasMetrics


class AtlasMetricTests(unittest.TestCase):
    def test_flip_rate_and_distortion_on_single_triangle(self):
        mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )

        uv_ok = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        uv_flipped = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

        self.assertEqual(UVAtlasMetrics.flip_rate(mesh, uv_ok), 0.0)
        self.assertEqual(UVAtlasMetrics.flip_rate(mesh, uv_flipped), 1.0)
        stats = UVAtlasMetrics.distortion_stats(mesh, uv_ok)
        self.assertTrue(np.isfinite(stats["mean"]))

    def test_atlas_bbox_metrics(self):
        uv_a = np.array([[0.0, 0.0], [1.0, 1.0]])
        uv_b = np.array([[0.5, 0.5], [1.5, 1.5]])

        self.assertGreater(UVAtlasMetrics.atlas_bbox_overlap_area([uv_a, uv_b]), 0.0)
        self.assertEqual(UVAtlasMetrics.padding_violations([uv_a, uv_b], padding=0.1), 1)
