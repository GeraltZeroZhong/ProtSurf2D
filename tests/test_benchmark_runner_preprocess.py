import json
import os
import shutil
import tempfile
import unittest

from topoppi.benchmarking import BenchmarkRunner
from topoppi.config import BenchmarkConfig

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_PDB = os.path.join(FIXTURE_DIR, "tiny_complex.pdb")


class BenchmarkPreprocessTests(unittest.TestCase):
    def test_no_valid_jobs_writes_preprocessing_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(TINY_PDB, os.path.join(tmpdir, "tiny_complex.pdb"))
            output_root = os.path.join(tmpdir, "out")
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    input_folder=tmpdir,
                    output_root=output_root,
                    chain_a="A",
                    chain_b="B",
                    min_chain_residues=11,
                )
            )

            with self.assertRaises(ValueError):
                runner.run()

            report_path = os.path.join(output_root, "benchmark_report.json")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(report["summary"]["valid_structure_count"], 0)
        self.assertEqual(report["preprocessing"]["accepted_files"], 0)
        self.assertEqual(report["runtime"]["worker_count"], 0)
        self.assertIn("config_fingerprint", report["runtime"])

    def test_resume_ignores_mismatched_config_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "out")
            os.makedirs(output_root, exist_ok=True)
            checkpoint_path = os.path.join(output_root, "benchmark_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config_fingerprint": "stale",
                        "files": [{"pdb": "tiny_complex.pdb", "patch_count": 1}],
                    },
                    handle,
                )

            runner = BenchmarkRunner(
                BenchmarkConfig(
                    input_folder=FIXTURE_DIR,
                    output_root=output_root,
                    chain_a="A",
                    chain_b="B",
                )
            )
            jobs = [{"pdb": "tiny_complex.pdb", "chain_a": "A", "chain_b": "B"}]
            completed, remaining = runner._load_resume_state(jobs)

        self.assertEqual(completed, [])
        self.assertEqual(remaining, jobs)
