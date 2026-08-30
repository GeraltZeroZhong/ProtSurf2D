import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from topoppi import __version__, benchmark_cli, cli, gui
from topoppi.benchmarking.evidence_bundle import BENCHMARK_ARTIFACT_FILENAMES


def _write_evidence_bundle(root: Path) -> Path:
    config = dict(BENCHMARK_ARTIFACT_FILENAMES)
    config["artifact_checksums_filename"] = "benchmark_artifact_checksums.json"
    report_path = root / config["report_filename"]
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "topoppi_version": __version__,
                "config": config,
                "runtime": {"config_fingerprint": "test-fingerprint"},
                "files": [],
                "summary": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for field, filename in BENCHMARK_ARTIFACT_FILENAMES.items():
        if field != "report_filename":
            (root / filename).write_text(f"{field}\n", encoding="utf-8")
    artifacts = []
    for filename in BENCHMARK_ARTIFACT_FILENAMES.values():
        artifact = root / filename
        artifacts.append(
            {
                "filename": filename,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
    (root / config["artifact_checksums_filename"]).write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "config_fingerprint": "test-fingerprint",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return report_path


class CliTests(unittest.TestCase):
    def test_help_locates_generated_prolif_beside_the_output_image(self):
        help_text = " ".join(cli.build_parser().format_help().split())

        self.assertIn("generate one beside the output image", help_text)

    def test_commands_report_the_installed_version(self):
        for command in (cli.main, gui.main, benchmark_cli.main):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaisesRegex(SystemExit, "0"):
                command(["--version"])
            self.assertIn(__version__, output.getvalue())

    def test_cli_uses_headless_optcuts_mode(self):
        with mock.patch("topoppi.cli.run_interface_mapping") as run_interface_mapping:
            exit_code = cli.main(["input.pdb", "-A", "A", "-B", "B", "--optcuts-bin", "OptCuts_bin"])

        self.assertEqual(exit_code, 0)
        config = run_interface_mapping.call_args.args[0]
        self.assertEqual(config.optcuts.optcuts_mode, config.optcuts.optcuts_headless_mode)
        self.assertEqual(config.optcuts.residue_fragmentation_weight, 20.0)
        self.assertEqual(config.visualization.residue_scope, "interaction")
        self.assertEqual(config.topology.distance_cutoff, 4.0)
        self.assertEqual(config.visualization.min_points, 1)
        self.assertTrue(config.visualization.color_by_interaction_type)
        self.assertFalse(config.visualization.use_geometric_interaction_fallback)

    def test_cli_propagates_surface_parameterization_and_optcuts_controls(self):
        with mock.patch("topoppi.cli.run_interface_mapping") as run_interface_mapping:
            exit_code = cli.main(
                [
                    "input.pdb",
                    "--surface-level",
                    "0.2",
                    "--max-voxels",
                    "1234",
                    "--no-adaptive-resolution",
                    "--parameterization",
                    "slim",
                    "--slim-iterations",
                    "7",
                    "--slim-boundary-constraint-weight",
                    "1000000",
                    "--optcuts-lambda",
                    "0.9",
                    "--optcuts-distortion-bound",
                    "4.2",
                    "--optcuts-initial-cut-option",
                    "1",
                    "--no-optcuts-bijectivity",
                    "--optcuts-initialization",
                    "automatic",
                    "--optcuts-timeout",
                    "42",
                    "--residue-fragmentation-weight",
                    "0.75",
                    "--geometric-fallback-distance",
                    "5.5",
                    "--residue-scope",
                    "patch",
                    "--min-points",
                    "3",
                    "--uniform-residue-color",
                    "--geometric-interaction-fallback",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = run_interface_mapping.call_args.args[0]
        self.assertEqual(config.surface.level, 0.2)
        self.assertEqual(config.surface.max_voxels, 1234)
        self.assertFalse(config.surface.adaptive_resolution)
        self.assertEqual(config.parameterization.method, "slim")
        self.assertEqual(config.parameterization.slim_iterations, 7)
        self.assertEqual(config.parameterization.slim_boundary_constraint_weight, 1_000_000.0)
        self.assertEqual(config.optcuts.optcuts_lambda_init, 0.9)
        self.assertEqual(config.optcuts.optcuts_distortion_bound, 4.2)
        self.assertEqual(config.optcuts.optcuts_initial_cut_option, 1)
        self.assertFalse(config.optcuts.optcuts_use_bijectivity)
        self.assertFalse(config.optcuts.use_input_uv)
        self.assertEqual(config.optcuts.timeout_sec, 42.0)
        self.assertEqual(config.optcuts.residue_fragmentation_weight, 0.75)
        self.assertEqual(config.contact_distance_angstrom, 5.5)
        self.assertEqual(config.visualization.residue_scope, "patch")
        self.assertEqual(config.visualization.min_points, 3)
        self.assertFalse(config.visualization.color_by_interaction_type)
        self.assertTrue(config.visualization.use_geometric_interaction_fallback)

    def test_benchmark_config_errors_are_single_line_and_name_the_field(self):
        cases = (
            ({"typo_workers": 2}, "Unknown benchmark config field: typo_workers."),
            ({"surface": "fine"}, "Benchmark config field 'surface' must be a JSON object."),
            (
                {"surface": {"grid_resoluton": 1.0}},
                "Unknown benchmark config field: surface.grid_resoluton.",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (payload, message) in enumerate(cases):
                with self.subTest(payload=payload):
                    config_path = Path(tmp) / f"invalid-{index}.json"
                    config_path.write_text(json.dumps(payload), encoding="utf-8")
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = benchmark_cli.main(["preflight", str(config_path)])

                    self.assertEqual(exit_code, 2)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertEqual(stderr.getvalue(), f"topoppi-benchmark: {message}\n")
                    self.assertNotIn("Traceback", stderr.getvalue())

    def test_benchmark_preflight_output_modes(self):
        payload = {
            "ready": True,
            "accepted_job_count": 4,
            "resumed_structure_count": 1,
            "remaining_job_count": 3,
            "planned_worker_process_count": 2,
            "blockers": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "preflight.json"
            for extra_args, expected_output in (
                (
                    [],
                    "Preflight: ready\nStructures: 4 accepted, 1 resumed, 3 remaining\nWorker processes: 2 planned\n",
                ),
                (["--json"], None),
                (["--output-json", str(output_path)], f"JSON report: {output_path.resolve()}\n"),
            ):
                with self.subTest(extra_args=extra_args):
                    stdout = io.StringIO()
                    runner = mock.Mock()
                    runner.preflight.return_value = payload
                    with (
                        mock.patch.object(benchmark_cli, "load_benchmark_config", return_value=object()),
                        mock.patch.object(benchmark_cli, "BenchmarkRunner", return_value=runner),
                        redirect_stdout(stdout),
                    ):
                        exit_code = benchmark_cli.main(["preflight", "benchmark.json", *extra_args])

                    self.assertEqual(exit_code, 0)
                    if extra_args == ["--json"]:
                        self.assertEqual(json.loads(stdout.getvalue()), payload)
                    else:
                        self.assertEqual(stdout.getvalue(), expected_output)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), payload)

    def test_benchmark_run_output_modes(self):
        config = SimpleNamespace(
            formal_mode=False,
            output_root="benchmark-output",
            report_filename="benchmark_report.json",
            execution_profile="comparative",
        )
        preflight = {"resumed_structure_count": 2}
        output = {
            "summary": {
                "attempted_structure_count": 5,
                "failed_structure_count": 1,
                "complete_comparison_structure_count": 4,
            }
        }
        for extra_args in ([], ["--json"]):
            with self.subTest(extra_args=extra_args):
                stdout = io.StringIO()
                runner = mock.Mock()
                runner.preflight.return_value = preflight
                runner.run.return_value = output
                with (
                    mock.patch.object(
                        benchmark_cli,
                        "load_benchmark_config",
                        return_value=config,
                    ),
                    mock.patch.object(
                        benchmark_cli,
                        "BenchmarkRunner",
                        return_value=runner,
                    ) as runner_type,
                    redirect_stdout(stdout),
                ):
                    exit_code = benchmark_cli.main(["run", "benchmark.json", *extra_args])

                self.assertEqual(exit_code, 0)
                runner_type.assert_called_once_with(config)
                if extra_args:
                    self.assertEqual(
                        json.loads(stdout.getvalue()),
                        {
                            "status": "complete",
                            "report": "benchmark-output/benchmark_report.json",
                            "resumed_structure_count": 2,
                            "summary": output["summary"],
                        },
                    )
                else:
                    self.assertEqual(
                        stdout.getvalue(),
                        "Benchmark: complete\n"
                        "Report: benchmark-output/benchmark_report.json\n"
                        "Structures: 5 attempted, 2 resumed, 1 failed\n"
                        "Complete comparisons: 4/5\n",
                    )

    def test_sensitivity_output_modes(self):
        preflight = {
            "ready": True,
            "scenario_count": 3,
            "planned_worker_process_count": 2,
            "scenarios": [],
        }
        result = {
            "scenario_count": 3,
            "scenarios": [
                {"scenario_id": "baseline", "status": "complete"},
                {"scenario_id": "surface", "status": "failed"},
                {"scenario_id": "topology", "status": "complete"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "sensitivity_plan.json"
            for command, payload, summary in (
                (
                    "preflight-sensitivity",
                    preflight,
                    "Sensitivity preflight: ready\nScenarios: 3\nWorker processes: 2 planned\n",
                ),
                (
                    "run-sensitivity",
                    result,
                    "Sensitivity study: complete\n"
                    f"Results: {plan_path.parent / 'sensitivity_results.json'}\n"
                    "Scenarios: 3 total, 1 failed\n",
                ),
            ):
                for extra_args in ([], ["--json"]):
                    with self.subTest(command=command, extra_args=extra_args):
                        stdout = io.StringIO()
                        runner = mock.Mock()
                        if command == "preflight-sensitivity":
                            runner.preflight.return_value = payload
                        else:
                            runner.run.return_value = payload
                        with (
                            mock.patch.object(
                                benchmark_cli,
                                "SensitivityBenchmarkRunner",
                                return_value=runner,
                            ) as runner_type,
                            redirect_stdout(stdout),
                        ):
                            exit_code = benchmark_cli.main([command, str(plan_path), *extra_args])

                        self.assertEqual(exit_code, 0)
                        runner_type.assert_called_once_with(str(plan_path))
                        if extra_args:
                            self.assertEqual(json.loads(stdout.getvalue()), payload)
                        else:
                            self.assertEqual(stdout.getvalue(), summary)

    def test_verify_evidence_bundle_output_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = _write_evidence_bundle(root)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = benchmark_cli.main(["verify", str(report_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("Evidence bundle: verified\n", stdout.getvalue())
            self.assertIn(f"Report: {report_path.resolve()}\n", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = benchmark_cli.main(["verify", str(report_path), "--json"])
            self.assertEqual(exit_code, 0)
            verified = json.loads(stdout.getvalue())
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(verified["report"], str(report_path.resolve()))
            self.assertEqual(verified["schema_version"], "2.0")

            summary_path = root / BENCHMARK_ARTIFACT_FILENAMES["summary_filename"]
            summary_path.write_text("tampered\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = benchmark_cli.main(["verify", str(report_path)])
            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("artifact checksum differs", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
