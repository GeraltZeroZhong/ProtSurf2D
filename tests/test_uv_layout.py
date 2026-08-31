import unittest

import numpy as np
import trimesh

from topoppi.atlas.uv import as_corner_uv, corner_to_vertex_uv, set_uv_layout


class UVLayoutTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )

    def test_discontinuous_corner_uv_is_never_averaged(self):
        corners = np.asarray(self.mesh.vertices[:, :2])[self.mesh.faces]
        corners[1, 0] = np.array([4.0, 5.0])

        self.assertIsNone(corner_to_vertex_uv(self.mesh, corners))
        set_uv_layout(self.mesh, corners)

        np.testing.assert_array_equal(as_corner_uv(self.mesh), corners)
        np.testing.assert_array_equal(self.mesh.metadata["uv"], corners)
        self.assertNotIn("uv_corners", self.mesh.metadata)
        self.assertNotIn("uv_is_continuous", self.mesh.metadata)

    def test_continuous_vertex_uv_round_trips(self):
        vertex_uv = np.asarray(self.mesh.vertices[:, :2])
        set_uv_layout(self.mesh, vertex_uv)

        np.testing.assert_array_equal(self.mesh.metadata["uv"], vertex_uv[self.mesh.faces])
        np.testing.assert_array_equal(as_corner_uv(self.mesh), vertex_uv[self.mesh.faces])

    def test_corner_continuity_is_invariant_to_uniform_uv_scale(self):
        continuous = np.asarray(self.mesh.vertices[:, :2])[self.mesh.faces]
        discontinuous = continuous.copy()
        discontinuous[1, 0] += np.asarray([0.25, 0.0])

        for scale in (1e-12, 1.0, 1e12):
            self.assertIsNotNone(corner_to_vertex_uv(self.mesh, continuous * scale))
            self.assertIsNone(corner_to_vertex_uv(self.mesh, discontinuous * scale))


if __name__ == "__main__":
    unittest.main()
