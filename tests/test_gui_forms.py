import os
import tempfile
import unittest

from topoppi.errors import ConfigurationError
from topoppi.gui_app.forms import parse_benchmark_form, parse_single_run_form

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_PDB = os.path.join(FIXTURE_DIR, "tiny_complex.pdb")


class GUIFormParsingTests(unittest.TestCase):
    def test_single_run_form_parses_typed_values(self):
        form = parse_single_run_form(
            {
                "path": TINY_PDB,
                "chain_a": " A ",
                "chain_b": "B",
                "cutoff": "9.5",
                "res": "2.0",
                "sigma": "1.0",
                "min_points": "3",
                "optcuts_bin": "",
                "save_optcuts_frames": True,
                "optcuts_frame_stride": "2",
                "optcuts_min_frame_long_edge": "0",
                "output_dir": FIXTURE_DIR,
            }
        )

        self.assertEqual(form.chain_a, "A")
        self.assertEqual(form.chain_b, "B")
        self.assertEqual(form.cutoff, 9.5)
        self.assertEqual(form.min_points, 3)
        self.assertTrue(form.save_optcuts_frames)

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

        self.assertEqual(form.output_root, output_root)
        self.assertEqual(form.run_mode, "new")
        self.assertEqual(form.max_workers, 2)
