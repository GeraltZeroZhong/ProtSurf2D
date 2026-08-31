import csv
import tempfile
import unittest
from pathlib import Path

from topoppi.benchmarking.reporting import aggregate_results, write_csv


class BenchmarkReportingTests(unittest.TestCase):
    def test_flat_csv_separates_top_level_and_exact_residue_aware_pair_domains(self):
        top_level_standard = {
            "distortion": {"mean": 99.0},
            "symmetric_dirichlet": {"mean": 99.0},
        }
        exact_standard = {
            "distortion": {"mean": 1.0},
            "symmetric_dirichlet": {"mean": 2.1},
            "angle_distortion": {"mean": 0.1},
            "area_distortion": {"mean": 0.2},
            "flip_rate": 0.0,
            "seam": {"seam_length_3d_normalized": 0.3},
            "residue_footprint_fragmentation": {"objective_weighted_fragmentation": 0.4},
        }
        exact_topoppi = {
            **exact_standard,
            "distortion": {"mean": 0.8},
            "residue_footprint_fragmentation": {"objective_weighted_fragmentation": 0.2},
        }
        row = {
            "pdb": "pair.pdb",
            "experimental_methods_json": '["SOLUTION NMR"]',
            "experimental_method_group": "solution_nmr",
            "experimental_method_contains_nmr": True,
            "comparison_domain": {"signature": "standard-domain", "complete": True},
            "residue_aware_comparison_domain": {"signature": "topoppi-domain", "complete": True},
            "optcuts_automatic": top_level_standard,
            "residue_aware_optcuts": exact_topoppi,
            "residue_aware_pair_quality": {
                "status": "evaluated",
                "complete": True,
                "expected_patch_count": 1,
                "common_patch_count": 1,
                "domain_signature": "topoppi-domain",
                "arms": {
                    "optcuts_automatic": {
                        "domain_complete": True,
                        "metric_finite": True,
                        "globally_injective": True,
                        "usable": True,
                    },
                    "residue_aware_optcuts": {
                        "domain_complete": True,
                        "metric_finite": True,
                        "globally_injective": True,
                        "usable": True,
                    },
                },
                "methods": {
                    "optcuts_automatic": exact_standard,
                    "residue_aware_optcuts": exact_topoppi,
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            write_csv([row], tmp)
            with Path(tmp, "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
                flat = next(csv.DictReader(handle))

        self.assertEqual(float(flat["optcuts_automatic_distortion_mean"]), 99.0)
        self.assertEqual(float(flat["residue_aware_pair_optcuts_automatic_distortion_mean"]), 1.0)
        self.assertEqual(float(flat["residue_aware_pair_residue_aware_optcuts_distortion_mean"]), 0.8)
        self.assertEqual(flat["optcuts_automatic_metric_domain_signature"], "standard-domain")
        self.assertEqual(flat["residue_aware_pair_domain_signature"], "topoppi-domain")
        self.assertEqual(flat["experimental_method_group"], "solution_nmr")
        self.assertEqual(flat["experimental_method_contains_nmr"], "True")

    def test_empty_aggregate_is_stable(self):
        summary = aggregate_results([])
        self.assertEqual(summary["valid_structure_count"], 0)
        self.assertEqual(summary["attempted_structure_count"], 0)
        self.assertEqual(summary["residue_aware_optcuts_comparisons"]["status"], "not_evaluated")

    def test_prespecified_method_sensitivity_excludes_any_nmr_entry(self):
        rows = [
            {
                "pdb": "xray.pdb",
                "analysis_split": "test",
                "experimental_method_contains_nmr": False,
                "comparison_domain": {"common_patch_count": 1, "complete": True},
                "lscm": {"distortion": {"mean": 1.0}},
            },
            {
                "pdb": "nmr.pdb",
                "analysis_split": "test",
                "experimental_method_contains_nmr": "True",
                "comparison_domain": {"common_patch_count": 1, "complete": True},
                "lscm": {"distortion": {"mean": 2.0}},
            },
        ]

        summary = aggregate_results(rows, methods=("lscm",), bootstrap_iterations=10)
        sensitivity = summary["experimental_method_sensitivity"]

        self.assertEqual(sensitivity["status"], "evaluated")
        self.assertEqual(sensitivity["all_splits_excluded_nmr_structure_count"], 1)
        self.assertEqual(sensitivity["filtered_summary"]["attempted_structure_count"], 1)
        self.assertEqual(
            sensitivity["filtered_summary"]["experimental_method_sensitivity"]["status"],
            "not_repeated_inside_filtered_summary",
        )

    def test_error_rows_are_excluded_from_valid_count(self):
        summary = aggregate_results([{"pdb": "x.pdb", "patch_count": 0, "error": "failed"}])
        self.assertEqual(summary["valid_structure_count"], 0)
        self.assertEqual(summary["failed_structure_count"], 1)

    def test_operational_runtime_does_not_require_a_quality_comparison_domain(self):
        row = {
            "pdb": "timed.pdb",
            "status": "ok",
            "analysis_split": "test",
            "benchmark_purpose": "performance",
            "execution_profile": "operational_optcuts",
            "operational_method": "residue_aware_optcuts",
            "execution_certificate": {"scientifically_usable": True},
            "comparison_domain": {"common_patch_count": 0, "complete": False},
            "timing": {"isolated_repetitions": {"wall_sec_median": 4.0}},
            "memory": {"peak_rss_mb": 128.0},
        }

        summary = aggregate_results(
            [row],
            methods=("optcuts_automatic",),
            bootstrap_iterations=10,
        )

        self.assertEqual(summary["valid_structure_count"], 0)
        self.assertEqual(summary["performance_timing_structure_count"], 1)
        self.assertEqual(summary["isolated_end_to_end_wall_sec"]["median"], 4.0)
        self.assertEqual(summary["performance_execution_profiles"], ["operational_optcuts"])
        self.assertEqual(summary["performance_operational_methods"], ["residue_aware_optcuts"])
        self.assertEqual(summary["operational_scientifically_usable_structure_count"], 1)
        self.assertEqual(summary["excluded_without_common_domain_count"], 0)

    def test_performance_summary_retains_censored_runtime_observations(self):
        completed = {
            "pdb": "completed.pdb",
            "status": "ok",
            "analysis_split": "test",
            "benchmark_purpose": "performance",
            "execution_profile": "operational_optcuts",
            "timing": {"isolated_repetitions": {"wall_sec_median": 4.0}},
            "memory": {"peak_rss_mb": 128.0},
            "worker_measurements": [
                {
                    "warmup": False,
                    "wall_sec": 4.0,
                    "peak_rss_mb": 128.0,
                    "worker_completed": True,
                    "right_censored": False,
                    "termination_reason": None,
                }
            ],
        }
        censored = {
            "pdb": "timeout.pdb",
            "status": "failed",
            "error": "timeout",
            "analysis_split": "test",
            "benchmark_purpose": "performance",
            "execution_profile": "operational_optcuts",
            "worker_measurements": [
                {
                    "warmup": False,
                    "wall_sec": 10.2,
                    "runtime_observation_sec": 10.0,
                    "peak_rss_mb": 256.0,
                    "worker_completed": False,
                    "right_censored": True,
                    "termination_reason": "timeout",
                    "censoring_threshold_sec": 10.0,
                    "censoring_event_elapsed_sec": 10.0,
                }
            ],
        }
        quality_failure = {
            "pdb": "quality.pdb",
            "status": "failed",
            "error": "failure",
            "analysis_split": "test",
            "benchmark_purpose": "quality",
            "worker_measurements": [{"warmup": False, "wall_sec": 99.0, "peak_rss_mb": 512.0}],
        }

        summary = aggregate_results(
            [completed, censored, quality_failure],
            methods=("optcuts_automatic",),
            bootstrap_iterations=10,
        )

        self.assertEqual(summary["performance_attempted_structure_count"], 2)
        self.assertEqual(summary["performance_timing_structure_count"], 1)
        self.assertEqual(summary["performance_observed_run_count"], 2)
        self.assertEqual(summary["performance_right_censored_run_count"], 1)
        self.assertEqual(
            summary["performance_termination_reason_counts"],
            {"completed": 1, "timeout": 1},
        )
        self.assertEqual(
            summary["isolated_runtime_observation_sec_including_censored_lower_bounds"]["count"],
            2,
        )
        self.assertEqual(
            summary["isolated_runtime_observation_sec_including_censored_lower_bounds"]["median"],
            7.0,
        )
        self.assertEqual(summary["isolated_supervisor_wall_sec_all_observed_runs"]["median"], 7.1)

        with tempfile.TemporaryDirectory() as tmp:
            write_csv([censored], tmp)
            with Path(tmp, "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
                flat = next(csv.DictReader(handle))
        self.assertEqual(flat["worker_runtime_observation_count"], "1")
        self.assertEqual(flat["worker_right_censored"], "True")
        self.assertEqual(flat["worker_termination_reason"], "timeout")
        self.assertEqual(float(flat["worker_censoring_threshold_sec"]), 10.0)
        self.assertEqual(float(flat["worker_censoring_event_elapsed_sec"]), 10.0)
        self.assertEqual(float(flat["worker_runtime_observation_sec_last"]), 10.0)
        self.assertEqual(float(flat["worker_supervisor_wall_sec_last"]), 10.2)

    def test_strict_json_null_retention_values_do_not_break_aggregation(self):
        methods = (
            "lscm",
            "harmonic",
            "slim",
            "spherical",
            "cylindrical",
            "optcuts_automatic",
            "optcuts_lscm_initialized",
        )
        quality = {
            "distortion": {"mean": 1.0},
            "angle_distortion": {"mean": 0.1},
            "area_distortion": {"mean": 0.2},
            "flip_rate": 0.0,
        }
        row = {
            "comparison_domain": {"common_patch_count": 1, "complete": True},
            "patch_records": [{"face_retention_ratio": None, "residue_retention_ratio": 1.0}],
            "method_execution": {},
            **{method: quality for method in methods},
        }

        summary = aggregate_results([row], bootstrap_iterations=10)

        self.assertEqual(summary["topology_biological_retention"]["face_retention_ratio"]["count"], 0)
        self.assertEqual(summary["topology_biological_retention"]["residue_retention_ratio"]["mean"], 1.0)

    def test_pooled_retention_uses_all_component_denominators(self):
        rows = [
            {
                "pdb": "x.pdb",
                "error": "all components rejected",
                "patch_records": [
                    {
                        "face_count_before": 90,
                        "face_count_after_topology_sanitation": 80,
                        "face_count_after": 70,
                    },
                    {
                        "face_count_before": 10,
                        "face_count_after_topology_sanitation": 0,
                        "face_count_after": 0,
                    },
                    {
                        # A corrupt zero-denominator row must not inflate the
                        # pooled numerator or define a retention ratio.
                        "face_count_before": 0,
                        "face_count_after_topology_sanitation": 5,
                        "face_count_after": 5,
                    },
                ],
            }
        ]

        summary = aggregate_results(rows)
        pooled = summary["topology_biological_retention_pooled_component_incidence"]["face"]

        self.assertEqual(pooled["overall"]["denominator_total"], 100.0)
        self.assertEqual(pooled["overall"]["retained_total"], 70.0)
        self.assertAlmostEqual(pooled["overall"]["retention_ratio"], 0.7)
        self.assertAlmostEqual(pooled["topology_sanitation"]["retention_ratio"], 0.8)
        self.assertAlmostEqual(pooled["parameterization"]["retention_ratio"], 70.0 / 80.0)

    def test_atlas_summary_uses_declared_reference_method(self):
        row = {
            "analysis_split": "test",
            "comparison_domain": {"common_patch_count": 2, "complete": True},
            "patch_records": [],
            "method_execution": {},
            "atlas_trainability": {"reference_method": "lscm"},
            "lscm": {
                "distortion": {"mean": 1.0},
                "symmetric_dirichlet": {"mean": 2.0},
                "angle_distortion": {"mean": 0.0},
                "area_distortion": {"mean": 0.0},
                "flip_rate": 0.0,
                "atlas": {
                    "utilization": 0.75,
                    "overlap_ratio": 0.0,
                    "overdraw_ratio": 1.0,
                    "min_chart_gap": 0.1,
                    "padding_violations": 0,
                },
            },
        }

        summary = aggregate_results([row], methods=("lscm",), bootstrap_iterations=10)

        self.assertEqual(summary["multi_patch_atlas"]["reference_methods"], ["lscm"])
        self.assertEqual(summary["multi_patch_atlas"]["utilization"]["mean"], 0.75)

    def test_paired_geometry_strata_are_reported_without_filtering(self):
        rows = [
            {
                "pdb": "high.pdb",
                "analysis_split": "test",
                "paired_geometry_stratum": "high_fidelity",
                "comparison_domain": {"common_patch_count": 0, "complete": False},
            },
            {
                "pdb": "stress.pdb",
                "analysis_split": "test",
                "paired_geometry_stratum": "geometry_stress_test",
                "comparison_domain": {"common_patch_count": 0, "complete": False},
            },
        ]

        summary = aggregate_results(rows, methods=("lscm",), bootstrap_iterations=10)

        self.assertEqual(summary["paired_geometry_strata"]["high_fidelity"]["attempted_structure_count"], 1)
        self.assertEqual(
            summary["paired_geometry_strata"]["geometry_stress_test"]["attempted_structure_count"],
            1,
        )

    def test_topology_ablation_retains_all_configured_structures_in_completion_endpoint(self):
        quality_without = {
            "distortion": {"mean": 1.1},
            "symmetric_dirichlet": {"mean": 2.2},
            "seam": {"seam_length_3d_normalized": 0.4},
            "flip_rate": 0.0,
            "runtime": {"wall_sec": 2.0},
            "injectivity": {"all_patches_globally_injective": True},
        }
        quality_with = {
            "distortion": {"mean": 1.0},
            "symmetric_dirichlet": {"mean": 2.1},
            "seam": {"seam_length_3d_normalized": 0.3},
            "flip_rate": 0.0,
            "runtime": {"wall_sec": 2.5},
            "injectivity": {"all_patches_globally_injective": True},
        }
        exact_pair = {
            "analysis_split": "test",
            "topology_ablation_configured": True,
            "topology_preprocessing_pair_quality": {
                "status": "evaluated",
                "complete": True,
                "expected_patch_count": 1,
                "raw_success_patch_count": 1,
                "prepared_success_patch_count": 1,
                "unique_patch_ids": True,
                "methods": {
                    "optcuts_without_topology_preparation": quality_without,
                    "optcuts_with_topology_preparation": quality_with,
                },
            },
        }
        prepared_failure = {
            "analysis_split": "test",
            "topology_ablation_configured": True,
            "topology_preprocessing_pair_quality": {
                "status": "ineligible_or_incomplete",
                "complete": False,
                "expected_patch_count": 1,
                "raw_success_patch_count": 1,
                "prepared_success_patch_count": 0,
                "unique_patch_ids": True,
                "methods": {},
            },
        }
        upstream_failure = {
            "analysis_split": "test",
            "topology_ablation_configured": True,
            "status": "failed",
            "error": "surface failure",
        }

        summary = aggregate_results(
            [exact_pair, prepared_failure, upstream_failure],
            methods=("optcuts_automatic",),
            bootstrap_iterations=10,
        )
        topology = summary["topology_preprocessing_ablation"]

        self.assertEqual(topology["configured_test_structure_count"], 3)
        self.assertEqual(topology["ablation_reached_test_structure_count"], 2)
        self.assertEqual(topology["not_reached_test_structure_count"], 1)
        self.assertEqual(topology["complete_test_structure_count"], 1)
        self.assertEqual(
            topology["all_configured_structure_completion"],
            {
                "without_topology_preparation_count": 2,
                "with_topology_preparation_count": 1,
                "both_complete_count": 1,
                "gained_after_preparation_count": 0,
                "lost_after_preparation_count": 1,
                "neither_complete_count": 1,
                "rule": (
                    "an arm is complete only when it emits one uniquely identified output for every "
                    "extracted patch; failures before the ablation are retained as neither complete"
                ),
            },
        )
