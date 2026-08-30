import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from topoppi.benchmarking import BenchmarkRunner
from topoppi.config import BenchmarkConfig

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_PDB = os.path.join(FIXTURE_DIR, "tiny_complex.pdb")


class BenchmarkPreprocessTests(unittest.TestCase):
    def test_cpu_affinity_is_optional_on_macos(self):
        without_affinity = types.SimpleNamespace()
        psutil_without_affinity = types.SimpleNamespace(Process=lambda: types.SimpleNamespace())
        with (
            mock.patch("topoppi.benchmarking.runner.os", without_affinity),
            mock.patch("topoppi.benchmarking.runner.psutil", psutil_without_affinity),
        ):
            self.assertEqual(BenchmarkRunner._ordered_available_cpus(), [])

    def test_run_reuses_cached_preflight_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "out")
            runner = BenchmarkRunner(BenchmarkConfig(tmpdir, output_root))
            runner._preflight_jobs = [{"pdb": "cached.pdb"}]
            runner._preflight_preprocessing = {"accepted_files": 1}
            with (
                mock.patch.object(
                    runner,
                    "preflight",
                    return_value={"ready": True, "structure_file_count": 1, "blockers": []},
                ),
                mock.patch.object(runner, "_prepare_benchmark_jobs") as prepare_jobs,
                mock.patch.object(runner, "_load_resume_state", return_value=([], [])),
                mock.patch.object(runner, "_save_checkpoint", return_value=True),
                mock.patch.object(runner, "_resolve_worker_count", return_value=1),
                mock.patch.object(runner, "_safe_progress"),
                mock.patch.object(runner, "_build_output", return_value={"status": "ok"}),
                mock.patch.object(runner, "_write_outputs"),
            ):
                output = runner.run()

        prepare_jobs.assert_not_called()
        self.assertEqual(output, {"status": "ok"})

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
