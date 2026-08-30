import unittest

import numpy as np

from topoppi.config import SurfaceConfig
from topoppi.mesh.surface import SurfaceGenerator


class AdaptiveSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 1.5],
            ]
        )

    def test_voxel_budget_can_fail_with_an_actionable_report(self):
        generator = SurfaceGenerator(
            self.coords,
            SurfaceConfig(
                grid_resolution=0.2,
                padding=2.0,
                max_voxels=500,
                adaptive_resolution=False,
                max_adaptive_resolution=4.0,
            ),
        )
        mesh = generator.generate_mesh()

        self.assertIsNone(mesh)
        self.assertEqual(generator.last_report["status"], "voxel_budget_exceeded")
        self.assertGreater(generator.last_report["requested_voxel_count"], 500)

    def test_grid_estimate_matches_generation_budget_without_allocating(self):
        config = SurfaceConfig(
            grid_resolution=0.2,
            padding=2.0,
            max_voxels=5000,
            adaptive_resolution=True,
            max_adaptive_resolution=4.0,
        )
        estimate = SurfaceGenerator.estimate_grid(self.coords, config)

        self.assertEqual(estimate["status"], "ok")
        self.assertLessEqual(estimate["effective_voxel_count"], config.max_voxels)
        self.assertGreater(estimate["estimated_dense_field_bytes_lower_bound"], 0)
        self.assertIn("marching-cubes", estimate["memory_estimate_scope"])

    def test_adaptive_surface_uses_exact_physical_bin_spacing(self):
        generator = SurfaceGenerator(
            self.coords,
            SurfaceConfig(
                grid_resolution=0.2,
                sigma=0.6,
                level=0.01,
                padding=2.2,
                max_voxels=5000,
                adaptive_resolution=True,
                max_adaptive_resolution=4.0,
                smoothing_iterations=0,
            ),
        )
        mesh = generator.generate_mesh()

        self.assertIsNotNone(mesh, msg=generator.last_report)
        report = generator.last_report
        self.assertTrue(report["adaptive_resolution_used"])
        self.assertLessEqual(report["effective_voxel_count"], report["max_voxels"])
        self.assertEqual(len(report["effective_spacing_angstrom_xyz"]), 3)
        for spacing, sigma_voxels in zip(
            report["effective_spacing_angstrom_xyz"],
            report["sigma_voxels_xyz"],
            strict=True,
        ):
            self.assertAlmostEqual(spacing * sigma_voxels, report["sigma_angstrom"], places=10)
        estimate = SurfaceGenerator.estimate_grid(self.coords, generator.config)
        expected_origin = np.asarray(estimate["bounds_min_angstrom"]) + 0.5 * np.asarray(
            report["effective_spacing_angstrom_xyz"]
        )
        np.testing.assert_allclose(report["density_sample_origin_angstrom"], expected_origin)
        self.assertEqual(len(mesh.metadata["source_atom_indices"]), len(mesh.vertices))
        self.assertEqual(
            report["density_convention"],
            "direct_truncated_unit_peak_atom_gaussians_v2",
        )

    def test_density_isovalue_has_resolution_independent_physical_meaning(self):
        atom = np.array([[0.0, 0.0, 0.0]])
        radii = []
        for resolution in (0.4, 0.5, 0.625):
            generator = SurfaceGenerator(
                atom,
                SurfaceConfig(
                    grid_resolution=resolution,
                    sigma=1.0,
                    level=0.2,
                    padding=4.0,
                    max_voxels=100_000,
                    adaptive_resolution=False,
                    max_adaptive_resolution=resolution,
                    smoothing_iterations=0,
                ),
            )
            mesh = generator.generate_mesh()
            self.assertIsNotNone(mesh, msg=generator.last_report)
            radii.append(float(np.mean(np.linalg.norm(mesh.vertices, axis=1))))

        expected = np.sqrt(2.0 * np.log(1.0 / 0.2))
        self.assertLess(max(radii) - min(radii), 0.12)
        self.assertLess(abs(float(np.mean(radii)) - expected), 0.15)

    def test_direct_lattice_surface_is_translation_covariant(self):
        config = SurfaceConfig(
            grid_resolution=0.5,
            sigma=1.0,
            level=0.2,
            padding=4.0,
            max_voxels=100_000,
            adaptive_resolution=False,
            max_adaptive_resolution=0.5,
            smoothing_iterations=0,
        )
        radii = []
        for shift in (0.0, 0.17):
            point = np.array([[shift, shift, shift]])
            mesh = SurfaceGenerator(point, config).generate_mesh()
            self.assertIsNotNone(mesh)
            radii.append(float(np.mean(np.linalg.norm(mesh.vertices - point, axis=1))))
        self.assertLess(abs(radii[0] - radii[1]), 0.05)

    def test_isovalue_is_not_silently_lowered_per_structure(self):
        generator = SurfaceGenerator(
            np.array([[0.0, 0.0, 0.0]]),
            SurfaceConfig(
                grid_resolution=0.5,
                sigma=1.0,
                level=2.0,
                padding=4.0,
                max_voxels=100_000,
                adaptive_resolution=False,
                max_adaptive_resolution=0.5,
            ),
        )

        self.assertIsNone(generator.generate_mesh())
        self.assertEqual(generator.last_report["status"], "isovalue_outside_density_range")
        self.assertEqual(generator.last_report["configured_isovalue"], 2.0)

    def test_surface_touching_the_sampling_boundary_is_rejected(self):
        generator = SurfaceGenerator(
            np.array([[0.0, 0.0, 0.0]]),
            SurfaceConfig(
                grid_resolution=0.5,
                sigma=1.0,
                level=0.2,
                padding=1.0,
                max_voxels=100_000,
                adaptive_resolution=False,
                max_adaptive_resolution=0.5,
                smoothing_iterations=0,
            ),
        )

        self.assertIsNone(generator.generate_mesh())
        self.assertEqual(generator.last_report["status"], "isovalue_intersects_grid_boundary")
        self.assertGreaterEqual(
            generator.last_report["boundary_maximum_density"],
            generator.last_report["configured_isovalue"],
        )


if __name__ == "__main__":
    unittest.main()
