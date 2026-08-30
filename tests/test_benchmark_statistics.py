import unittest

import numpy as np

from topoppi.benchmarking.reporting import aggregate_results
from topoppi.benchmarking.statistics import _dyadic_robust_mean_sensitivity, paired_method_comparison


def quality(value):
    return {
        "distortion": {"mean": value},
        "symmetric_dirichlet": {"mean": value},
        "angle_distortion": {"mean": value},
        "area_distortion": {"mean": value},
        "flip_rate": 0.0,
    }


def add_standard_pairs(row, baselines):
    treatment = "optcuts_automatic"
    row["standard_method_pair_quality"] = {
        f"{baseline}_vs_{treatment}": {
            "status": "evaluated",
            "complete": True,
            "arms": {
                baseline: {
                    "domain_complete": True,
                    "metric_finite": True,
                    "globally_injective": True,
                    "usable": True,
                },
                treatment: {
                    "domain_complete": True,
                    "metric_finite": True,
                    "globally_injective": True,
                    "usable": True,
                },
            },
            "methods": {
                baseline: row[baseline],
                treatment: row[treatment],
            },
        }
        for baseline in baselines
    }


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_one_inferential_unit_does_not_produce_a_zero_width_primary_interval(self):
        result = paired_method_comparison(
            [
                {
                    "family_id": "only-family",
                    "sequence_cluster_a": "only-a",
                    "sequence_cluster_b": "only-b",
                    "a": quality(2.0),
                    "b": quality(1.0),
                }
            ],
            baseline="a",
            treatment="b",
            bootstrap_iterations=20,
        )

        self.assertTrue(np.isnan(result["primary_mean_difference_ci95"]).all())
        self.assertTrue(np.isnan(result["wilcoxon_p_value"]))
        sensitivity = result["shared_protein_dependency_sensitivity"]["mean_difference"]
        self.assertEqual(sensitivity["status"], "unavailable")
        self.assertEqual(sensitivity["reason"], "fewer_than_two_heterotypic_families")

    def test_dyadic_robust_sandwich_matches_direct_shared_member_sum(self):
        values = np.asarray([1.0, 2.0, 4.0])
        partner_a = np.asarray(["a", "a", "d"], dtype=object)
        partner_b = np.asarray(["b", "c", "e"], dtype=object)

        observed = _dyadic_robust_mean_sensitivity(
            values,
            partner_a,
            partner_b,
        )

        residual = values - np.mean(values)
        expected_meat = float(np.dot(residual, residual) + 2.0 * residual[0] * residual[1])
        self.assertEqual(observed["status"], "evaluated")
        self.assertAlmostEqual(observed["raw_sandwich_meat"], expected_meat)
        self.assertAlmostEqual(observed["raw_variance"], expected_meat / 9.0)

    def test_dyadic_sensitivity_excludes_homotypic_families_explicitly(self):
        observed = _dyadic_robust_mean_sensitivity(
            np.asarray([100.0, 1.0, 2.0, 4.0]),
            np.asarray(["a", "a", "a", "d"], dtype=object),
            np.asarray(["a", "b", "c", "e"], dtype=object),
        )

        self.assertEqual(observed["homotypic_family_count_excluded"], 1)
        self.assertEqual(observed["heterotypic_family_count"], 3)
        self.assertAlmostEqual(observed["mean"], 7.0 / 3.0)

    def test_resolved_cluster_bootstrap_remains_primary_when_node_sensitivity_is_available(self):
        rows = [
            {
                "family_id": f"family-{index}",
                "sequence_cluster_a": f"a-{index}",
                "sequence_cluster_b": f"b-{index}",
                "a": quality(float(index + 2)),
                "b": quality(1.0),
            }
            for index in range(3)
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            bootstrap_iterations=100,
            seed=7,
        )

        sensitivity = result["shared_protein_dependency_sensitivity"]["mean_difference"]
        self.assertEqual(sensitivity["status"], "evaluated")
        self.assertEqual(sensitivity["protein_cluster_count"], 6)
        self.assertEqual(result["primary_interval_method"], "resolved_cluster_bootstrap")
        self.assertEqual(
            sensitivity["role"],
            "shared_protein_dependency_sensitivity_not_primary",
        )

    def test_inference_uses_cluster_means_not_pooled_structures(self):
        rows = [{"cluster_id": "large", "a": quality(10.0), "b": quality(0.0)} for _ in range(10)]
        rows.append({"cluster_id": "small", "a": quality(0.0), "b": quality(10.0)})
        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            bootstrap_iterations=100,
            seed=4,
        )

        self.assertGreater(result["mean_paired_difference"], 0.0)
        self.assertAlmostEqual(result["mean_cluster_difference"], 0.0)
        self.assertEqual(result["inferential_unit"], "cluster_id_mean")
        self.assertNotEqual(result["wilcoxon_p_value"], 0.0)

    def test_tiny_nonzero_cluster_effect_is_not_reclassified_as_exact_zero(self):
        rows = [
            {
                "family_id": f"family-{index}",
                "a": quality(1.0 + 1e-10),
                "b": quality(1.0),
            }
            for index in range(8)
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            bootstrap_iterations=20,
        )

        self.assertLess(result["wilcoxon_p_value"], 1.0)

    def test_mixed_family_metadata_is_not_labelled_family_level(self):
        rows = [
            {"family_id": "family-1", "cluster_id": "cluster-1", "a": quality(2.0), "b": quality(1.0)},
            {"family_id": "", "cluster_id": "cluster-2", "a": quality(2.0), "b": quality(1.0)},
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            bootstrap_iterations=20,
        )

        self.assertEqual(result["inferential_unit"], "resolved_cluster_mean")
        self.assertFalse(result["all_requested_cluster_keys_available"])
        self.assertEqual(result["cluster_source_counts"], {"cluster_id": 1, "family_id": 1})

    def test_union_component_primary_disables_single_node_bootstrap(self):
        rows = [
            {
                "analysis_split_component_id": f"component-{index}",
                "family_id": f"family-{index}",
                "sequence_cluster_a": f"experimental-a-{index}",
                "sequence_cluster_b": f"experimental-b-{index}",
                "inference_sequence_cluster_a": f"prediction-a-{index}",
                "inference_sequence_cluster_b": f"prediction-b-{index}",
                "a": quality(2.0 + index),
                "b": quality(1.0 + index),
            }
            for index in range(3)
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            cluster_key="analysis_split_component_id",
            bootstrap_iterations=100,
            seed=7,
        )

        self.assertEqual(result["inferential_unit"], "analysis_split_component_id_mean")
        sensitivity = result["shared_protein_dependency_sensitivity"]["mean_difference"]
        self.assertEqual(sensitivity["status"], "unavailable")
        self.assertEqual(
            sensitivity["reason"],
            "cluster_to_partner_mapping_missing_or_inconsistent",
        )
        self.assertEqual(result["partner_cluster_source_counts"], {"unavailable": 3})

    def test_prediction_dependencies_are_an_explicit_inference_unit(self):
        rows = [
            {
                "family_id": f"reference-family-{index}",
                "sequence_cluster_a": f"reference-a-{index}",
                "sequence_cluster_b": f"reference-b-{index}",
                "inference_family_id": "shared-prediction-family",
                "inference_sequence_cluster_a": "predicted-a",
                "inference_sequence_cluster_b": "predicted-b",
                "a": quality(float(index + 2)),
                "b": quality(1.0),
            }
            for index in range(2)
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            cluster_key="inference_family_id",
            bootstrap_iterations=20,
        )

        self.assertEqual(result["cluster_count"], 1)
        self.assertEqual(result["inferential_unit"], "prediction_dependency_family_mean")
        self.assertEqual(result["cluster_source_counts"], {"inference_family_id": 2})
        self.assertEqual(
            result["partner_cluster_source_counts"],
            {"inference_sequence_cluster": 2},
        )

        experimental_result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            cluster_key="family_id",
            bootstrap_iterations=20,
        )
        self.assertEqual(experimental_result["cluster_count"], 2)
        self.assertEqual(experimental_result["inferential_unit"], "interaction_family_mean")
        self.assertEqual(experimental_result["cluster_source_counts"], {"family_id": 2})
        self.assertEqual(
            experimental_result["partner_cluster_source_counts"],
            {"sequence_cluster": 2},
        )

    def test_predicted_report_uses_prediction_dependency_families(self):
        rows = []
        for index in range(2):
            row = {
                "structure_type": "afdb",
                "analysis_split": "test",
                "family_id": f"experimental-family-{index}",
                "sequence_cluster_a": f"experimental-a-{index}",
                "sequence_cluster_b": f"experimental-b-{index}",
                "inference_family_id": "shared-prediction-family",
                "inference_sequence_cluster_a": "predicted-a",
                "inference_sequence_cluster_b": "predicted-b",
                "comparison_domain": {"common_patch_count": 1, "complete": True},
                "patch_records": [],
                "method_execution": {},
                "lscm": quality(float(index + 2)),
                "optcuts_automatic": quality(1.0),
            }
            add_standard_pairs(row, ("lscm",))
            rows.append(row)

        summary = aggregate_results(
            rows,
            methods=("lscm", "optcuts_automatic"),
            bootstrap_iterations=20,
        )
        comparison = summary["paired_cluster_aware_comparisons"]["lscm_vs_optcuts_automatic"]

        self.assertEqual(summary["inferential_cluster_key"], "inference_family_id")
        self.assertEqual(comparison["cluster_key"], "inference_family_id")
        self.assertEqual(comparison["cluster_count"], 1)
        self.assertEqual(comparison["inferential_unit"], "prediction_dependency_family_mean")

    def test_symmetric_dirichlet_relative_effect_uses_identity_excess(self):
        rows = [
            {
                "family_id": "family-1",
                "a": quality(2.4),
                "b": quality(2.2),
            }
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            relative_reference=2.0,
            bootstrap_iterations=20,
        )

        self.assertAlmostEqual(result["relative_improvement_of_cluster_means"], 0.5)
        self.assertEqual(result["relative_improvement_reference"], 2.0)

    def test_binary_endpoint_reports_exact_discordant_pair_counts(self):
        rows = [
            {"family_id": "f1", "a": {"event": 1.0}, "b": {"event": 0.0}},
            {"family_id": "f2", "a": {"event": 1.0}, "b": {"event": 0.0}},
            {"family_id": "f3", "a": {"event": 0.0}, "b": {"event": 1.0}},
            {"family_id": "f4", "a": {"event": 0.0}, "b": {"event": 0.0}},
        ]

        result = paired_method_comparison(
            rows,
            baseline="a",
            treatment="b",
            metric_path=("event",),
            binary_endpoint=True,
            bootstrap_iterations=20,
        )

        self.assertEqual(result["baseline_event_treatment_nonevent_count"], 2)
        self.assertEqual(result["baseline_nonevent_treatment_event_count"], 1)
        self.assertEqual(result["discordant_pair_count"], 3)
        self.assertEqual(result["concordant_pair_count"], 1)

    def test_summary_adds_bh_adjusted_p_values(self):
        rows = []
        methods = (
            "lscm",
            "harmonic",
            "slim",
            "spherical",
            "cylindrical",
            "optcuts_automatic",
            "optcuts_lscm_initialized",
        )
        for index in range(8):
            row = {
                "cluster_id": f"cluster-{index // 2}",
                "comparison_domain": {"common_patch_count": 1, "complete": True},
                "patch_records": [],
                "method_execution": {},
            }
            for method_index, method in enumerate(methods):
                row[method] = quality(float(method_index + index + 1))
            add_standard_pairs(row, methods[:5])
            rows.append(row)

        summary = aggregate_results(rows, bootstrap_iterations=100, random_seed=7)
        comparisons = summary["paired_cluster_aware_comparisons"]
        self.assertEqual(len(comparisons), 5)
        for block in comparisons.values():
            self.assertIn("wilcoxon_q_value_bh", block)
            self.assertGreaterEqual(block["wilcoxon_q_value_bh"], block["wilcoxon_p_value"])

    def test_residue_aware_comparisons_form_a_separate_summary(self):
        def interaction_quality(value):
            return {
                **quality(value),
                "seam": {"seam_length_3d_normalized": value},
                "residue_footprint_fragmentation": {
                    "interaction_weighted_fragmentation": value,
                    "objective_weighted_fragmentation": value,
                },
            }

        row = {
            "cluster_id": "cluster-1",
            "comparison_domain": {"common_patch_count": 1, "complete": True},
            "residue_aware_comparison_domain": {"common_patch_count": 1, "complete": True},
            "analysis_split": "test",
            "patch_records": [],
            "method_execution": {},
            "optcuts_automatic": interaction_quality(1.0),
            "residue_aware_optcuts": interaction_quality(0.5),
            "residue_aware_pair_quality": {
                "status": "evaluated",
                "complete": True,
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
                    "optcuts_automatic": interaction_quality(1.0),
                    "residue_aware_optcuts": interaction_quality(0.5),
                },
            },
        }

        block = aggregate_results([row], bootstrap_iterations=10)["residue_aware_optcuts_comparisons"]

        self.assertEqual(block["status"], "evaluated")
        self.assertEqual(len(block["comparisons"]), 4)
        self.assertNotIn("automatic_flip_rate", block["comparisons"])
        self.assertIn(
            "automatic_objective_weighted_fragmentation",
            block["comparisons"],
        )
        self.assertTrue(all(comparison["paired_structure_count"] == 1 for comparison in block["comparisons"].values()))
        self.assertEqual(
            block["primary_comparison"],
            "automatic_objective_weighted_fragmentation",
        )
        self.assertNotIn(
            "wilcoxon_q_value_bh",
            block["comparisons"][block["primary_comparison"]],
        )
        self.assertEqual(
            block["paired_reliability_comparisons"]["automatic_unusable_output"]["paired_structure_count"],
            1,
        )

    def test_interaction_statistics_reject_legacy_top_level_fallback(self):
        def interaction_quality(value):
            return {
                **quality(value),
                "seam": {"seam_length_3d_normalized": value},
                "residue_footprint_fragmentation": {
                    "objective_weighted_fragmentation": value,
                },
            }

        row = {
            "analysis_split": "test",
            "comparison_domain": {"common_patch_count": 1, "complete": True},
            "residue_aware_comparison_domain": {"common_patch_count": 1, "complete": True},
            "patch_records": [],
            "method_execution": {},
            "optcuts_automatic": interaction_quality(1.0),
            "residue_aware_optcuts": interaction_quality(0.5),
        }

        block = aggregate_results(
            [row],
            methods=("optcuts_automatic", "residue_aware_optcuts"),
            bootstrap_iterations=10,
        )["residue_aware_optcuts_comparisons"]

        self.assertEqual(block["status"], "no_exact_pair_quality_rows")
        self.assertEqual(block["complete_domain_test_structure_count"], 1)
        self.assertEqual(block["complete_test_structure_count"], 0)
        self.assertEqual(block["excluded_without_exact_pair_quality_count"], 1)

    def test_interaction_statistics_use_exact_pair_domain_independently(self):
        def interaction_quality(value):
            return {
                **quality(value),
                "seam": {"seam_length_3d_normalized": value},
                "residue_footprint_fragmentation": {
                    "interaction_weighted_fragmentation": value,
                    "objective_weighted_fragmentation": value,
                },
            }

        pair_baseline = interaction_quality(1.0)
        pair_treatment = interaction_quality(0.5)
        row = {
            "cluster_id": "cluster-1",
            "family_id": "family-1",
            "sequence_cluster_a": "protein-a",
            "sequence_cluster_b": "protein-b",
            "comparison_domain": {"common_patch_count": 0, "complete": False},
            "residue_aware_comparison_domain": {"common_patch_count": 1, "complete": True},
            "analysis_split": "test",
            "patch_records": [],
            "method_execution": {},
            "optcuts_automatic": interaction_quality(100.0),
            "residue_aware_optcuts": pair_treatment,
            "residue_aware_pair_quality": {
                "status": "evaluated",
                "complete": True,
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
                    "optcuts_automatic": pair_baseline,
                    "residue_aware_optcuts": pair_treatment,
                },
            },
        }

        summary = aggregate_results(
            [row],
            methods=("optcuts_automatic", "residue_aware_optcuts"),
            bootstrap_iterations=10,
        )
        block = summary["residue_aware_optcuts_comparisons"]
        comparison = block["comparisons"]["automatic_objective_weighted_fragmentation"]

        self.assertEqual(summary["complete_test_structure_count"], 0)
        self.assertEqual(summary["complete_residue_aware_test_structure_count"], 1)
        self.assertEqual(block["exact_pair_quality_structure_count"], 1)
        self.assertEqual(comparison["mean_baseline"], 1.0)
        self.assertEqual(comparison["mean_treatment"], 0.5)

    def test_interaction_reliability_keeps_incomplete_residue_aware_pairs(self):
        row = {
            "cluster_id": "cluster-1",
            "family_id": "family-1",
            "analysis_split": "test",
            "comparison_domain": {"common_patch_count": 0, "complete": False},
            "residue_aware_comparison_domain": {"common_patch_count": 0, "complete": False},
            "patch_records": [],
            "method_execution": {},
            "residue_aware_pair_quality": {
                "status": "incomplete_comparison",
                "complete": False,
                "arms": {
                    "optcuts_automatic": {
                        "domain_complete": True,
                        "metric_finite": True,
                        "globally_injective": True,
                        "usable": True,
                    },
                    "residue_aware_optcuts": {
                        "domain_complete": False,
                        "metric_finite": False,
                        "globally_injective": False,
                        "usable": False,
                    },
                },
                "methods": {},
            },
        }

        block = aggregate_results(
            [row],
            methods=("optcuts_automatic", "residue_aware_optcuts"),
            bootstrap_iterations=10,
        )["residue_aware_optcuts_comparisons"]
        reliability = block["paired_reliability_comparisons"]["automatic_unusable_output"]

        self.assertEqual(reliability["attempted_pair_structure_count"], 1)
        self.assertEqual(reliability["paired_structure_count"], 1)
        self.assertEqual(
            reliability["baseline_nonevent_treatment_event_count"],
            1,
        )

    def test_interaction_reliability_keeps_whole_structure_worker_failures(self):
        row = {
            "cluster_id": "cluster-1",
            "family_id": "family-1",
            "sequence_cluster_a": "a",
            "sequence_cluster_b": "b",
            "analysis_split": "test",
            "patch_records": [],
            "method_execution": {},
            "error": "worker failed before methods",
        }

        block = aggregate_results(
            [row],
            methods=("optcuts_automatic", "residue_aware_optcuts"),
            bootstrap_iterations=10,
        )["residue_aware_optcuts_comparisons"]
        reliability = block["paired_reliability_comparisons"]["automatic_unusable_output"]

        self.assertEqual(reliability["attempted_pair_structure_count"], 1)
        self.assertEqual(reliability["paired_structure_count"], 1)
        self.assertEqual(reliability["concordant_pair_count"], 1)

    def test_execution_and_stratum_attempt_denominators_are_test_only(self):
        rows = []
        for split, attempted in (("development", 100), ("test", 2)):
            rows.append(
                {
                    "analysis_split": split,
                    "structure_type": "experimental",
                    "comparison_domain": {"common_patch_count": 0, "complete": False},
                    "patch_records": [],
                    "method_execution": {"optcuts_automatic": {"attempted": attempted, "success": attempted}},
                }
            )

        summary = aggregate_results(
            rows,
            methods=("optcuts_automatic",),
            bootstrap_iterations=10,
        )

        execution = summary["method_execution_all_attempted"]["optcuts_automatic"]
        self.assertEqual(execution["attempted_patch_count"], 2)
        self.assertEqual(execution["attempted_patch_count_all_splits"], 102)
        stratum = summary["structure_type_strata"]["experimental"]
        self.assertEqual(stratum["attempted_structure_count"], 1)
        self.assertEqual(stratum["attempted_all_splits_structure_count"], 2)

    def test_development_rows_are_excluded_from_confirmatory_statistics(self):
        methods = (
            "lscm",
            "harmonic",
            "slim",
            "spherical",
            "cylindrical",
            "optcuts_automatic",
            "optcuts_lscm_initialized",
        )
        rows = []
        for split, baseline_value in (("development", 100.0), ("test", 2.0)):
            row = {
                "cluster_id": split,
                "analysis_split": split,
                "comparison_domain": {"common_patch_count": 1, "complete": True},
                "patch_records": [],
                "method_execution": {},
            }
            row.update({method: quality(baseline_value) for method in methods})
            row["optcuts_automatic"] = quality(1.0)
            add_standard_pairs(row, methods[:5])
            rows.append(row)

        summary = aggregate_results(rows, bootstrap_iterations=10)

        self.assertEqual(summary["complete_comparison_structure_count"], 2)
        self.assertEqual(summary["complete_test_structure_count"], 1)
        self.assertEqual(summary["complete_development_structure_count"], 1)
        self.assertEqual(
            summary["method_distributions"]["lscm"]["distortion_mean"]["mean"],
            2.0,
        )
        self.assertTrue(
            all(block["paired_structure_count"] == 1 for block in summary["paired_cluster_aware_comparisons"].values())
        )

    def test_each_standard_pair_is_independent_of_unrelated_method_failure(self):
        row = {
            "analysis_split": "test",
            "comparison_domain": {"common_patch_count": 0, "complete": False},
            "patch_records": [],
            "method_execution": {},
            "harmonic": quality(2.0),
            "optcuts_automatic": quality(1.0),
            "standard_method_pair_quality": {
                "harmonic_vs_optcuts_automatic": {
                    "status": "evaluated",
                    "complete": True,
                    "arms": {
                        "harmonic": {
                            "domain_complete": True,
                            "metric_finite": True,
                            "globally_injective": True,
                            "usable": True,
                        },
                        "optcuts_automatic": {
                            "domain_complete": True,
                            "metric_finite": True,
                            "globally_injective": True,
                            "usable": True,
                        },
                    },
                    "methods": {
                        "harmonic": quality(2.0),
                        "optcuts_automatic": quality(1.0),
                    },
                }
            },
        }

        summary = aggregate_results(
            [row],
            methods=("harmonic", "optcuts_automatic"),
            bootstrap_iterations=10,
        )
        comparison = summary["paired_cluster_aware_comparisons"]["harmonic_vs_optcuts_automatic"]

        self.assertEqual(summary["complete_test_structure_count"], 0)
        self.assertEqual(comparison["paired_structure_count"], 1)
        self.assertEqual(comparison["mean_paired_difference"], 1.0)

    def test_reliability_endpoint_counts_incomplete_method_arm_as_invalid(self):
        row = {
            "analysis_split": "test",
            "comparison_domain": {"common_patch_count": 0, "complete": False},
            "patch_records": [],
            "method_execution": {},
            "standard_method_pair_quality": {
                "lscm_vs_optcuts_automatic": {
                    "status": "ineligible_or_incomplete",
                    "complete": False,
                    "arms": {
                        "lscm": {
                            "domain_complete": False,
                            "metric_finite": False,
                            "globally_injective": False,
                            "usable": False,
                        },
                        "optcuts_automatic": {
                            "domain_complete": True,
                            "metric_finite": True,
                            "globally_injective": True,
                            "usable": True,
                        },
                    },
                    "methods": {},
                }
            },
        }

        validity = aggregate_results(
            [row],
            methods=("lscm", "optcuts_automatic"),
            bootstrap_iterations=10,
        )["paired_unusable_output_comparisons"]["lscm_vs_optcuts_automatic"]

        self.assertEqual(validity["reliability_pair_structure_count"], 1)
        self.assertEqual(validity["mean_paired_difference"], 1.0)


if __name__ == "__main__":
    unittest.main()
