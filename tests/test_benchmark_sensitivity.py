import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from topoppi.benchmark_cli import load_benchmark_config, main
from topoppi.benchmarking.sensitivity import (
    SENSITIVITY_AXES,
    SensitivityBenchmarkRunner,
    SensitivityScenario,
    _paired_sensitivity_analysis,
    build_sensitivity_scenarios,
    load_sensitivity_plan,
    write_sensitivity_plan,
)
from topoppi.config import BenchmarkConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class BenchmarkSensitivityTests(unittest.TestCase):
    @staticmethod
    def _sensitivity_quality(value):
        return {
            "distortion": {"mean": value},
            "symmetric_dirichlet": {"mean": 2.0 + value},
            "angle_distortion": {"mean": value},
            "area_distortion": {"mean": value},
            "flip_rate": 0.0,
            "seam": {"seam_length_3d_normalized": value},
            "residue_footprint_fragmentation": {
                "objective_weighted_fragmentation": value,
            },
            "injectivity": {"all_patches_globally_injective": True},
        }

    def test_paired_sensitivity_tracks_residue_aware_optcuts_effect_stability(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(
                str(FIXTURE_DIR),
                str(Path(tmp) / "baseline"),
                repetitions=1,
                optcuts=replace(
                    BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "unused")).optcuts,
                    residue_fragmentation_weight=1.0,
                ),
                optcuts_variants=("optcuts_automatic", "residue_aware_optcuts"),
            )
            changed = replace(base, output_root=str(Path(tmp) / "changed"))
            scenarios = [
                SensitivityScenario("baseline", {}, base),
                SensitivityScenario("changed", {"surface.sigma": 1.2}, changed),
            ]

            def row(standard, topoppi):
                standard_quality = self._sensitivity_quality(standard)
                residue_aware_quality = self._sensitivity_quality(topoppi)
                return {
                    "manifest_record_id": "record-1",
                    "pdb": "record-1.pdb",
                    "input_sha256": "a" * 64,
                    "status": "ok",
                    "chain_selection": {"chain_a": "A", "chain_b": "B"},
                    "family_id": "family-1",
                    "sequence_cluster_a": "cluster-a",
                    "sequence_cluster_b": "cluster-b",
                    "inference_sequence_cluster_a": "pdep-a",
                    "inference_sequence_cluster_b": "pdep-b",
                    "inference_family_id": "pifam-1",
                    "analysis_split": "test",
                    "optcuts_automatic": self._sensitivity_quality(99.0),
                    "residue_aware_optcuts": self._sensitivity_quality(99.0),
                    "independent_optcuts_arm_quality": {
                        "optcuts_automatic": {
                            "domain_complete": True,
                            "metric_finite": True,
                            "globally_injective": True,
                            "usable": True,
                            "domain_signature": "domain-1",
                            "quality": standard_quality,
                        },
                        "residue_aware_optcuts": {
                            "domain_complete": True,
                            "metric_finite": True,
                            "globally_injective": True,
                            "usable": True,
                            "domain_signature": "domain-1",
                            "quality": residue_aware_quality,
                        },
                    },
                }

            analysis = _paired_sensitivity_analysis(
                scenarios,
                {
                    "baseline": {"files": [row(0.5, 0.4)]},
                    "changed": {"files": [row(0.6, 0.3)]},
                },
                bootstrap_iterations=100,
                seed=7,
            )

            noninjective = row(0.6, 0.3)
            arm = noninjective["independent_optcuts_arm_quality"]["optcuts_automatic"]
            arm["globally_injective"] = False
            arm["usable"] = False
            arm["quality"]["injectivity"]["all_patches_globally_injective"] = False
            noninjective_analysis = _paired_sensitivity_analysis(
                scenarios,
                {
                    "baseline": {"files": [row(0.5, 0.4)]},
                    "changed": {"files": [noninjective]},
                },
                bootstrap_iterations=100,
                seed=7,
            )

            mismatched_component = row(0.6, 0.3)
            mismatched_component["analysis_split_component_id"] = "changed-component"
            with self.assertRaisesRegex(ValueError, "identical attempted structure identities"):
                _paired_sensitivity_analysis(
                    scenarios,
                    {
                        "baseline": {"files": [row(0.5, 0.4)]},
                        "changed": {"files": [mismatched_component]},
                    },
                    bootstrap_iterations=100,
                    seed=7,
                )

        changed_analysis = analysis["scenario_vs_baseline"]["changed"]
        self.assertNotIn("baseline", analysis["scenario_vs_baseline"])
        self.assertIn("descriptive only", analysis["multiplicity_policy"])
        stability = changed_analysis["residue_aware_treatment_effect_stability"]
        endpoint = stability["pairwise_common_complete_structure_comparisons"]["objective_weighted_fragmentation"]
        self.assertAlmostEqual(endpoint["mean_paired_difference"], 0.2)
        self.assertIn("residue_aware_optcuts", changed_analysis["methods"])
        self.assertNotIn(
            "flip_rate",
            changed_analysis["methods"]["optcuts_automatic"]["pairwise_common_complete_structure_comparisons"],
        )
        self.assertEqual(stability["pairwise_identical_source_domain_structure_count"], 1)
        noninjective_method = noninjective_analysis["scenario_vs_baseline"]["changed"]["methods"]["optcuts_automatic"]
        self.assertEqual(noninjective_method["pairwise_common_complete_structure_count"], 1)
        self.assertEqual(noninjective_method["excluded_from_pairwise_efficacy_count"], 0)
        self.assertEqual(
            noninjective_method["all_attempted_unusable_output_comparison"]["mean_paired_difference"],
            -1.0,
        )

    def test_one_factor_plan_changes_only_one_supported_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(
                input_folder=str(FIXTURE_DIR),
                output_root=str(Path(tmp) / "results"),
                repetitions=1,
            )
            scenarios = build_sensitivity_scenarios(
                base,
                {
                    "cutoff": [base.topology.distance_cutoff, 8.0, 10.0],
                    "sigma": [base.surface.sigma, 1.5],
                },
                output_root=str(Path(tmp) / "study"),
            )

        self.assertEqual(scenarios[0].scenario_id, "baseline")
        self.assertEqual(len(scenarios), 4)
        self.assertTrue(all(len(scenario.changes) <= 1 for scenario in scenarios))
        self.assertEqual(base.topology.distance_cutoff, 4.0)
        self.assertEqual(base.surface.sigma, 1.0)
        for scenario in scenarios:
            self.assertTrue(scenario.config.resume)
            self.assertIn(scenario.scenario_id, scenario.config.output_root)

    def test_plan_requires_the_automatic_baseline_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(
                input_folder=str(FIXTURE_DIR),
                output_root=str(Path(tmp) / "results"),
                execution_profile="operational_optcuts",
                optcuts_variants=("residue_aware_optcuts",),
                include_topology_ablation=False,
                repetitions=1,
                optcuts=replace(
                    BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "unused")).optcuts,
                    residue_fragmentation_weight=1.0,
                ),
            )

            plan_root = Path(tmp) / "study"
            with self.assertRaisesRegex(ValueError, "require optcuts_automatic as the baseline arm"):
                write_sensitivity_plan(base, {"sigma": [1.0, 1.2]}, str(plan_root))
            self.assertFalse(plan_root.exists())

    def test_plan_loader_rejects_a_missing_automatic_baseline_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = BenchmarkConfig(str(FIXTURE_DIR), str(root / "results"), repetitions=1)
            plan_path = Path(write_sensitivity_plan(base, {"sigma": [1.0]}, str(root / "plan")))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            baseline = plan["baseline_config"]
            baseline["execution_profile"] = "operational_optcuts"
            baseline["optcuts_variants"] = ["residue_aware_optcuts"]
            baseline["include_topology_ablation"] = False
            baseline["optcuts"]["residue_fragmentation_weight"] = 1.0
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "require optcuts_automatic as the baseline arm"):
                load_sensitivity_plan(str(plan_path))

    def test_run_reuses_the_preflighted_runner_for_each_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = BenchmarkConfig(str(FIXTURE_DIR), str(root / "baseline"), repetitions=1)
            changed = replace(base, output_root=str(root / "changed"))
            scenarios = [
                SensitivityScenario("baseline", {}, base),
                SensitivityScenario("changed", {"surface.sigma": 1.2}, changed),
            ]
            runners = [Mock(), Mock()]
            for index, runner in enumerate(runners):
                runner.preflight.return_value = {
                    "ready": True,
                    "blockers": [],
                    "config_fingerprint": f"config-{index}",
                    "planned_worker_process_count": 1,
                }
                runner.run.return_value = {"files": []}

            with (
                patch(
                    "topoppi.benchmarking.sensitivity.load_sensitivity_plan",
                    return_value=({}, scenarios),
                ),
                patch(
                    "topoppi.benchmarking.sensitivity.BenchmarkRunner",
                    side_effect=runners,
                ) as runner_type,
                patch(
                    "topoppi.benchmarking.sensitivity._sensitivity_summary_row",
                    side_effect=lambda scenario, _report: {
                        "scenario_id": scenario.scenario_id,
                        "status": "ok",
                    },
                ),
                patch(
                    "topoppi.benchmarking.sensitivity._paired_sensitivity_analysis",
                    return_value={"status": "evaluated", "scenario_vs_baseline": {}},
                ),
                patch("topoppi.benchmarking.sensitivity.sha256_file", return_value="a" * 64),
                patch("topoppi.benchmarking.sensitivity.dump_json_atomic"),
                patch("topoppi.benchmarking.sensitivity._write_sensitivity_csv"),
            ):
                result = SensitivityBenchmarkRunner(str(root / "plan.json")).run()

        self.assertEqual(runner_type.call_count, len(scenarios))
        self.assertEqual(result["scenario_count"], len(scenarios))
        for runner in runners:
            runner.preflight.assert_called_once_with()
            runner.run.assert_called_once_with()

    def test_unconfirmed_formal_run_stops_before_preflight(self):
        base = BenchmarkConfig(str(FIXTURE_DIR), "unused", repetitions=1)
        scenario = SensitivityScenario(
            "formal",
            {},
            replace(base, formal_mode=True),
        )
        scenario_runner = Mock()
        runner = SensitivityBenchmarkRunner.__new__(SensitivityBenchmarkRunner)
        runner.scenarios = [scenario]
        runner._scenario_runners = {scenario.scenario_id: scenario_runner}

        with self.assertRaisesRegex(RuntimeError, "confirm_formal=True"):
            runner.run()

        scenario_runner.preflight.assert_not_called()

    def test_factorial_plan_uses_cartesian_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "results"), repetitions=1)
            scenarios = build_sensitivity_scenarios(
                base,
                {"sigma": [0.8, 1.2], "isovalue": [0.05, 0.1]},
                design="factorial",
                output_root=str(Path(tmp) / "study"),
            )

        self.assertEqual(len(scenarios), 5)
        self.assertEqual(scenarios[0].scenario_id, "baseline")
        self.assertTrue(all(set(scenario.changes).issubset(SENSITIVITY_AXES) for scenario in scenarios))

    def test_cli_writes_a_plan_but_does_not_create_result_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "benchmark.json"
            axes_path = root / "axes.json"
            plan_root = root / "plan"
            config_path.write_text(
                json.dumps(
                    {
                        "input_folder": str(FIXTURE_DIR),
                        "output_root": "unused-results",
                        "repetitions": 1,
                        "optcuts": {"optcuts_bin": str(Path.cwd() / "tools" / "OptCuts" / "OptCuts_bin")},
                    }
                ),
                encoding="utf-8",
            )
            axes_path.write_text(json.dumps({"sigma": [0.8, 1.0, 1.2]}), encoding="utf-8")

            exit_code = main(
                [
                    "plan-sensitivity",
                    str(config_path),
                    str(axes_path),
                    "--plan-root",
                    str(plan_root),
                ]
            )
            plan, scenarios = load_sensitivity_plan(str(plan_root / "sensitivity_plan.json"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(plan["scenario_count"], 3)
            self.assertEqual(len(scenarios), 3)
            self.assertFalse((plan_root / "results" / "baseline").exists())

    def test_config_paths_are_resolved_relative_to_the_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inputs").mkdir()
            config_path = root / "benchmark.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_folder": "inputs",
                        "output_root": "outputs",
                        "repetitions": 1,
                    }
                ),
                encoding="utf-8",
            )

            config = load_benchmark_config(str(config_path))

        self.assertEqual(config.input_folder, str(root / "inputs"))
        self.assertEqual(config.output_root, str(root / "outputs"))

    def test_plan_loader_rejects_a_tampered_scenario_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "results"), repetitions=1)
            plan_path = write_sensitivity_plan(base, {"sigma": [1.0, 1.2]}, str(Path(tmp) / "plan"))
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            config_path = Path(plan_path).parent / plan["scenarios"][0]["config_file"]
            config_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_sensitivity_plan(plan_path)

    def test_plan_loader_rejects_hidden_non_axis_config_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "results"), repetitions=1)
            plan_path = write_sensitivity_plan(base, {"sigma": [1.0]}, str(Path(tmp) / "plan"))
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            record = plan["scenarios"][0]
            config_path = Path(plan_path).parent / record["config_file"]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["random_seed"] += 1
            config_path.write_text(json.dumps(config), encoding="utf-8")
            record["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            Path(plan_path).write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside its declared axes"):
                load_sensitivity_plan(plan_path)

    def test_plan_loader_rejects_output_root_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "results"), repetitions=1)
            plan_path = write_sensitivity_plan(base, {"sigma": [1.0]}, str(Path(tmp) / "plan"))
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            record = plan["scenarios"][0]
            config_path = Path(plan_path).parent / record["config_file"]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["output_root"] = str(Path(tmp) / "shared-results")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            record["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            Path(plan_path).write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside its declared axes"):
                load_sensitivity_plan(plan_path)

    def test_plan_loader_rejects_a_non_integer_scenario_count_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = BenchmarkConfig(str(FIXTURE_DIR), str(Path(tmp) / "results"), repetitions=1)
            plan_path = write_sensitivity_plan(base, {"sigma": [1.0]}, str(Path(tmp) / "plan"))
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            plan["scenario_count"] = None
            Path(plan_path).write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenario_count must be an integer"):
                load_sensitivity_plan(plan_path)

    def test_formal_config_still_requires_explicit_runner_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text("pdb,chain_a,chain_b,input_sha256,cluster_id\n", encoding="utf-8")
            base = BenchmarkConfig(
                input_folder=tmp,
                output_root=str(Path(tmp) / "out"),
                chain_selection_mode="manifest",
                manifest_path=str(manifest),
                repetitions=3,
                warmup_runs=1,
                formal_mode=True,
                optcuts=replace(
                    BenchmarkConfig(tmp, str(Path(tmp) / "x")).optcuts,
                    expected_binary_sha256="a" * 64,
                ),
            )
            scenarios = build_sensitivity_scenarios(base, {"sigma": [1.0, 1.2]})

        self.assertTrue(all(scenario.config.formal_mode for scenario in scenarios))
        self.assertTrue(all(scenario.config.max_workers == 1 for scenario in scenarios))

    def test_cli_refuses_formal_run_without_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text("pdb,chain_a,chain_b\n", encoding="utf-8")
            config_path = root / "formal.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_folder": str(root),
                        "output_root": str(root / "out"),
                        "chain_selection_mode": "manifest",
                        "manifest_path": str(manifest),
                        "formal_mode": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised:
                main(["run", str(config_path)])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
