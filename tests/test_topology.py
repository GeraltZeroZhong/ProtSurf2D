import unittest

import numpy as np
import trimesh

from topoppi.config import TopologyConfig
from topoppi.mesh.topology import TopologyManager


class TopologyTests(unittest.TestCase):
    def test_extracts_synthetic_interface_patch(self):
        mesh_a = trimesh.creation.icosphere(radius=10.0, subdivisions=2)
        coords_b = np.array([[11.0, 0.0, 0.0], [11.0, 1.0, 0.0], [11.0, 0.0, 1.0]])

        topology_config = TopologyConfig(distance_cutoff=2.5, min_patch_vertices=5)
        patches = TopologyManager(mesh_a, coords_b, config=topology_config).get_interface_patches()

        self.assertGreaterEqual(len(patches), 1)
        self.assertTrue(all(len(p.vertices) >= 5 for p in patches))
