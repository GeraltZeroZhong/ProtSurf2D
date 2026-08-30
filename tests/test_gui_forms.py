import os
import tempfile
import unittest

from topoppi.errors import ConfigurationError
from topoppi.gui_app.forms import parse_benchmark_form, parse_single_run_form

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_PDB = os.path.join(FIXTURE_DIR, "tiny_complex.pdb")


class GUIFormParsingTests(unittest.TestCase):
    def test_single_run_form_uses_gui_patch_and_display_defaults(self):
        form = parse_single_run_form(
            {
                "path": TINY_PDB,
                "chain_a": "A",
                "chain_b": "B",
                "res": "2",
                "sigma": "1",
            }
        )

        config = form.to_config()
        self.assertEqual(form.cutoff, 4.0)
        self.assertEqual(form.min_points, 1)
        self.assertEqual(config.topology.distance_cutoff, 4.0)
        self.assertEqual(config.visualization.min_points, 1)
        self.assertEqual(config.optcuts.optcuts_mode, config.optcuts.optcuts_headless_mode)

    def test_single_run_form_parses_typed_values(self):
        form = parse_single_run_form(
            {
                "path": TINY_PDB,
                "chain_a": " A ",
                "chain_b": "B",
                "cutoff": "9.5",
                "contact_distance_angstrom": "5.5",
                "res": "2.0",
                "sigma": "1.0",
                "surface_level": "0.2",
                "max_voxels": "12345",
                "parameterization_method": "slim",
                "slim_boundary_constraint_weight": "2500",
                "min_points": "3",
                "optcuts_bin": "",
                "patch_gap": "0.15",
                "optcuts_lambda": "0.9",
                "optcuts_distortion_bound": "4.5",
                "optcuts_initial_cut_option": "1",
                "optcuts_use_bijectivity": False,
                "optcuts_initialization": "automatic",
                "save_optcuts_frames": True,
                "optcuts_frame_stride": "2",
                "optcuts_min_frame_long_edge": "0",
                "output_dir": FIXTURE_DIR,
            }
        )

        self.assertEqual(form.chain_a, "A")
        self.assertEqual(form.chain_b, "B")
        self.assertEqual(form.cutoff, 9.5)
        self.assertEqual(form.contact_distance_angstrom, 5.5)
        self.assertEqual(form.min_points, 3)
        self.assertEqual(form.surface_level, 0.2)
        self.assertEqual(form.max_voxels, 12345)
        self.assertEqual(form.parameterization_method, "slim")
        self.assertEqual(form.slim_boundary_constraint_weight, 2500.0)
        self.assertEqual(form.patch_gap, 0.15)
        self.assertFalse(form.optcuts_use_bijectivity)
        self.assertEqual(form.optcuts_initialization, "automatic")
        self.assertTrue(form.save_optcuts_frames)
        config = form.to_config()
        self.assertEqual(config.surface.grid_resolution, 2.0)
        self.assertEqual(config.parameterization.method, "slim")
        self.assertEqual(config.parameterization.slim_boundary_constraint_weight, 2500.0)
        self.assertEqual(config.optcuts.residue_fragmentation_weight, 20.0)
        self.assertEqual(config.optcuts.optcuts_mode, config.optcuts.optcuts_headless_mode)
        self.assertEqual(config.visualization.min_points, 3)
        self.assertFalse(config.optcuts.optcuts_use_bijectivity)

    def test_single_run_form_uses_complete_topoppi_defaults(self):
        form = parse_single_run_form(
            {
                "path": TINY_PDB,
                "chain_a": "A",
                "chain_b": "B",
                "cutoff": "9",
                "res": "2",
                "sigma": "1",
                "min_points": "3",
            }
        )

        self.assertEqual(form.residue_fragmentation_weight, 20.0)
        self.assertEqual(form.to_config().optcuts.residue_fragmentation_weight, 20.0)
        self.assertTrue(form.to_config().visualization.color_by_interaction_type)
        self.assertFalse(form.to_config().visualization.use_geometric_interaction_fallback)

    def test_single_run_form_applies_core_optcuts_ranges(self):
        with self.assertRaisesRegex(ConfigurationError, "must be in"):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "min_points": "3",
                    "optcuts_lambda": "1.0",
                }
            )

    def test_single_run_form_requires_boolean_widget_values(self):
        with self.assertRaisesRegex(ConfigurationError, "boolean"):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "min_points": "3",
                    "save_optcuts_frames": "false",
                }
            )

    def test_single_run_form_rejects_invalid_numbers_before_worker(self):
        with self.assertRaises(ConfigurationError):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "not-a-number",
                    "res": "2.0",
                    "sigma": "1.0",
                    "min_points": "3",
                }
            )

    def test_single_run_form_rejects_empty_chain(self):
        with self.assertRaises(ConfigurationError):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "min_points": "3",
                }
            )

    def test_single_run_form_rejects_same_chain(self):
        with self.assertRaises(ConfigurationError):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "A",
                    "chain_b": "A",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "min_points": "3",
                }
            )

    def test_single_run_form_rejects_non_finite_float(self):
        with self.assertRaises(ConfigurationError):
            parse_single_run_form(
                {
                    "path": TINY_PDB,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "nan",
                    "res": "2",
                    "sigma": "1",
                    "min_points": "3",
                }
            )

    def test_benchmark_form_allows_new_output_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "benchmark_results")
            form = parse_benchmark_form(
                {
                    "folder": tmpdir,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "output_root": output_root,
                    "run_mode": "new",
                    "max_workers": "2",
                }
            )
            config = form.to_config()

        self.assertEqual(form.output_root, output_root)
        self.assertEqual(form.run_mode, "new")
        self.assertEqual(form.max_workers, 2)
        self.assertEqual(
            form.optcuts_variants,
            ("optcuts_automatic", "optcuts_lscm_initialized", "residue_aware_optcuts"),
        )
        self.assertFalse(config.resume)
        self.assertEqual(config.max_workers, 2)
        self.assertEqual(config.resolved_optcuts_variants(), form.optcuts_variants)

    def test_quality_benchmark_accepts_parallel_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            form = parse_benchmark_form(
                {
                    "folder": tmpdir,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "benchmark_purpose": "quality",
                    "repetitions": "1",
                    "warmup_runs": "0",
                    "max_workers": "8",
                    "threads_per_worker": "1",
                    "worker_memory_limit_mb": "2048",
                }
            )
            config = form.to_config()

        self.assertEqual(config.benchmark_purpose, "quality")
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.worker_memory_limit_mb, 2048.0)

    def test_operational_profile_requires_one_automatic_arm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            form = parse_benchmark_form(
                {
                    "folder": tmpdir,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "benchmark_purpose": "performance",
                    "execution_profile": "operational_optcuts",
                    "optcuts_variants": ["residue_aware_optcuts"],
                    "include_topology_ablation": False,
                }
            )

            self.assertEqual(form.to_config().resolved_optcuts_variants(), ("residue_aware_optcuts",))
            with self.assertRaisesRegex(ConfigurationError, "exactly one automatic"):
                parse_benchmark_form(
                    {
                        "folder": tmpdir,
                        "chain_a": "A",
                        "chain_b": "B",
                        "cutoff": "9",
                        "res": "2",
                        "sigma": "1",
                        "benchmark_purpose": "performance",
                        "execution_profile": "operational_optcuts",
                        "optcuts_variants": ["optcuts_automatic", "residue_aware_optcuts"],
                        "include_topology_ablation": False,
                    }
                )

    def test_formal_benchmark_form_enforces_manifest_repeats_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.csv")
            coordinate_audit = os.path.join(tmpdir, "coordinate_audit.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write("pdb,chain_a,chain_b,input_sha256,cluster_id\n")
            with open(coordinate_audit, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            form = parse_benchmark_form(
                {
                    "folder": tmpdir,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "run_mode": "new",
                    "chain_selection_mode": "manifest",
                    "manifest_path": manifest,
                    "formal_mode": True,
                    "repetitions": "3",
                    "warmup_runs": "1",
                    "max_workers": "1",
                    "expected_optcuts_sha256": "a" * 64,
                    "coordinate_audit_path": coordinate_audit,
                    "expected_coordinate_audit_sha256": "b" * 64,
                }
            )

        self.assertTrue(form.formal_mode)
        self.assertEqual(form.chain_selection_mode, "manifest")
        self.assertEqual(form.expected_optcuts_sha256, "a" * 64)
        self.assertEqual(form.coordinate_audit_path, coordinate_audit)

    def test_formal_quality_form_uses_quality_repetition_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.csv")
            coordinate_audit = os.path.join(tmpdir, "coordinate_audit.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write("pdb,chain_a,chain_b,input_sha256,cluster_id\n")
            with open(coordinate_audit, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            form = parse_benchmark_form(
                {
                    "folder": tmpdir,
                    "chain_a": "A",
                    "chain_b": "B",
                    "cutoff": "9",
                    "res": "2",
                    "sigma": "1",
                    "run_mode": "new",
                    "chain_selection_mode": "manifest",
                    "manifest_path": manifest,
                    "formal_mode": True,
                    "benchmark_purpose": "quality",
                    "repetitions": "1",
                    "warmup_runs": "0",
                    "max_workers": "8",
                    "expected_optcuts_sha256": "a" * 64,
                    "coordinate_audit_path": coordinate_audit,
                    "expected_coordinate_audit_sha256": "b" * 64,
                }
            )

        self.assertEqual(form.benchmark_purpose, "quality")
        self.assertEqual(form.max_workers, 8)

    def test_formal_benchmark_form_rejects_overwrite_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = os.path.join(tmpdir, "manifest.csv")
            with open(manifest, "w", encoding="utf-8") as handle:
                handle.write("pdb,chain_a,chain_b,input_sha256,cluster_id\n")
            with self.assertRaisesRegex(ConfigurationError, "cannot use overwrite"):
                parse_benchmark_form(
                    {
                        "folder": tmpdir,
                        "chain_a": "A",
                        "chain_b": "B",
                        "cutoff": "9",
                        "res": "2",
                        "sigma": "1",
                        "run_mode": "overwrite",
                        "chain_selection_mode": "manifest",
                        "manifest_path": manifest,
                        "formal_mode": True,
                        "repetitions": "3",
                        "warmup_runs": "1",
                        "max_workers": "1",
                        "expected_optcuts_sha256": "a" * 64,
                    }
                )
