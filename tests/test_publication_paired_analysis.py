from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from topoppi import __version__


def _load_script():
    path = Path(__file__).parents[1] / "tools" / "publication" / "analyze_paired_benchmarks.py"
    spec = importlib.util.spec_from_file_location("test_analyze_paired_benchmarks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_script()


def _quality(value: float, *, injective: bool = True):
    return {
        "residue_footprint_fragmentation": {
            "objective_weighted_fragmentation": value,
        },
        "symmetric_dirichlet": {"mean": 2.0 + value},
        "distortion": {"mean": value},
        "angle_distortion": {"mean": value},
        "area_distortion": {"mean": value},
        "flip_rate": 0.0,
        "seam": {"seam_length_3d_normalized": value},
        "injectivity": {"all_patches_globally_injective": injective},
    }


def _experimental(record_id: str, standard: float = 0.5, topoppi: float = 0.3):
    standard_quality = _quality(standard)
    residue_aware_quality = _quality(topoppi)
    inference_a = f"pdep-a-{record_id}"
    inference_b = f"pdep-b-{record_id}"
    return {
        "manifest_record_id": record_id,
        "structure_type": "experimental",
        "status": "ok",
        "analysis_split": "test",
        "analysis_split_component_id": f"split-{record_id}",
        "analysis_split_basis": "experimental_homology_and_reused_afdb_accession_component",
        "cluster_id": f"cluster-{record_id}",
        "family_id": f"family-{record_id}",
        "sequence_cluster_a": f"a-{record_id}",
        "sequence_cluster_b": f"b-{record_id}",
        "inference_sequence_cluster_a": inference_a,
        "inference_sequence_cluster_b": inference_b,
        "inference_family_id": ANALYZE.inference_family_id(inference_a, inference_b),
        "inference_dependency_basis": ANALYZE.INFERENCE_DEPENDENCY_BASIS,
        "paired_record_id": f"pair-{record_id}",
        "experimental_methods_json": '["X-RAY DIFFRACTION"]',
        "experimental_method_group": "x_ray_diffraction",
        "experimental_method_contains_nmr": False,
        "optcuts_automatic": _quality(99.0),
        "residue_aware_optcuts": _quality(99.0),
        "independent_optcuts_arm_quality": {
            "optcuts_automatic": {
                "domain_complete": True,
                "metric_finite": True,
                "globally_injective": True,
                "usable": True,
                "quality": standard_quality,
            },
            "residue_aware_optcuts": {
                "domain_complete": True,
                "metric_finite": True,
                "globally_injective": True,
                "usable": True,
                "quality": residue_aware_quality,
            },
        },
        "residue_aware_pair_quality": {
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
                "optcuts_automatic": standard_quality,
                "residue_aware_optcuts": residue_aware_quality,
            },
        },
    }


def _predicted(
    record_id: str,
    reference_id: str,
    *,
    standard: float = 0.7,
    topoppi: float = 0.4,
):
    reference = _experimental(reference_id)
    standard_quality = _quality(standard)
    residue_aware_quality = _quality(topoppi)
    inference_a = reference["inference_sequence_cluster_a"]
    inference_b = reference["inference_sequence_cluster_b"]
    return {
        "manifest_record_id": record_id,
        "structure_type": "afdb_monomer_replacement",
        "paired_experimental_record_id": reference_id,
        "paired_record_id": reference["paired_record_id"],
        "experimental_methods_json": reference["experimental_methods_json"],
        "experimental_method_group": reference["experimental_method_group"],
        "experimental_method_contains_nmr": reference["experimental_method_contains_nmr"],
        "status": "ok",
        "analysis_split": "test",
        "analysis_split_component_id": reference["analysis_split_component_id"],
        "analysis_split_basis": reference["analysis_split_basis"],
        "cluster_id": reference["cluster_id"],
        "family_id": reference["family_id"],
        "sequence_cluster_a": reference["sequence_cluster_a"],
        "sequence_cluster_b": reference["sequence_cluster_b"],
        "inference_sequence_cluster_a": inference_a,
        "inference_sequence_cluster_b": inference_b,
        "inference_family_id": ANALYZE.inference_family_id(inference_a, inference_b),
        "inference_dependency_basis": ANALYZE.INFERENCE_DEPENDENCY_BASIS,
        "candidate_chain_pair_count": 1,
        "selected_residue_contact_fraction": 0.9,
        "chain_b_residue_count": 20,
        "paired_geometry_stratum": "high_fidelity",
        "paired_contact_cutoff_angstrom": 6.0,
        "paired_predicted_contact_count_total": 12,
        "paired_alignment_a_optimal_correspondence_count": 1,
        "paired_alignment_b_optimal_correspondence_count": 1,
        "paired_alignment_a_selected_pair_consensus_fraction": 1.0,
        "paired_alignment_b_selected_pair_consensus_fraction": 1.0,
        "confidence_preflight": {"summary_unit": "residue", "mean": 85.0},
        "optcuts_automatic": _quality(99.0),
        "residue_aware_optcuts": _quality(99.0),
        "independent_optcuts_arm_quality": {
            "optcuts_automatic": {
                "domain_complete": True,
                "metric_finite": True,
                "globally_injective": True,
                "usable": True,
                "quality": standard_quality,
            },
            "residue_aware_optcuts": {
                "domain_complete": True,
                "metric_finite": True,
                "globally_injective": True,
                "usable": True,
                "quality": residue_aware_quality,
            },
        },
        "residue_aware_pair_quality": {
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
                "optcuts_automatic": standard_quality,
                "residue_aware_optcuts": residue_aware_quality,
            },
        },
    }


def test_exact_pairing_and_signed_transport_effects():
    experimental = {"files": [_experimental("e1"), _experimental("unused")]}
    predicted = {"files": [_predicted("p1", "e1")]}

    pairs = ANALYZE.pair_rows(experimental, predicted)
    result = ANALYZE.analyze_pairs(pairs, bootstrap_iterations=100, seed=7)

    standard = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    endpoint = standard["continuous_endpoints"]["objective_weighted_fragmentation"]
    assert endpoint["mean_paired_difference"] == pytest.approx(0.2)
    assert "flip_rate" not in standard["continuous_endpoints"]
    effect = result["main_all_frozen_test_pairs"]["residue_aware_effect_transport"]
    effect_endpoint = effect["continuous_endpoints"]["objective_weighted_fragmentation"]
    assert effect_endpoint["mean_paired_difference"] == pytest.approx(0.1)
    efficacy = result["main_all_frozen_test_pairs"]["residue_aware_efficacy_by_structure_source"]
    assert efficacy["experimental"]["continuous_endpoints"]["objective_weighted_fragmentation"][
        "mean_paired_difference"
    ] == pytest.approx(0.2)
    assert efficacy["predicted"]["continuous_endpoints"]["objective_weighted_fragmentation"][
        "mean_paired_difference"
    ] == pytest.approx(0.3)
    assert result["sensitivity_cohorts"]["strict_binary_interface"]["attrition"]["retained_pair_count"] == 1
    assert result["sensitivity_cohorts"]["unique_sequence_correspondence"]["attrition"]["retained_pair_count"] == 1
    semantics = result["predicted_cohort_semantics"]
    assert semantics["analysis_role"] == "controlled fixed-pose conformational sensitivity analysis"
    assert "experimental relative pose held constant" in semantics["interpretation_limit"]
    availability = result["upstream_predicted_interface_availability"]
    assert availability["predicted_contact_present_pair_count"] == 1
    assert availability["predicted_contact_absent_pair_count"] == 0


def test_contactless_prediction_remains_in_upstream_availability_and_reliability():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["paired_predicted_contact_count_total"] = 0
    predicted["status"] = "failed"
    predicted["error"] = "No interface patch passed topology extraction"
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    availability = result["upstream_predicted_interface_availability"]
    assert availability["predicted_contact_present_pair_count"] == 0
    assert availability["predicted_contact_absent_pair_count"] == 1
    reliability = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert reliability["all_attempted_unusable_output"]["mean_paired_difference"] == 1.0


def test_predicted_complex_semantics_do_not_claim_pose_is_controlled():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["structure_type"] = "afdb"
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    semantics = result["predicted_cohort_semantics"]
    assert semantics["analysis_role"] == "matched predicted-complex external-validity analysis"
    assert "conformation and relative-pose effects are combined" in semantics["interpretation_limit"]


def test_nmr_sensitivity_uses_frozen_experimental_method_metadata():
    xray = _experimental("e1")
    xray_predicted = _predicted("p1", "e1")
    nmr = _experimental("e2")
    nmr["experimental_methods_json"] = '["SOLUTION NMR"]'
    nmr["experimental_method_group"] = "solution_nmr"
    nmr["experimental_method_contains_nmr"] = True
    nmr_predicted = _predicted("p2", "e2")
    nmr_predicted["experimental_methods_json"] = nmr["experimental_methods_json"]
    nmr_predicted["experimental_method_group"] = nmr["experimental_method_group"]
    nmr_predicted["experimental_method_contains_nmr"] = True
    pairs = ANALYZE.pair_rows(
        {"files": [xray, nmr]},
        {"files": [xray_predicted, nmr_predicted]},
    )

    result = ANALYZE.analyze_pairs(pairs, bootstrap_iterations=100, seed=7)

    attrition = result["sensitivity_cohorts"]["exclude_any_nmr_experimental_method"]["attrition"]
    assert attrition == {
        "source_pair_count": 2,
        "retained_pair_count": 1,
        "excluded_missing_filter_metadata_count": 0,
        "excluded_filter_mismatch_count": 1,
    }


def test_atom_weighted_confidence_is_not_accepted_for_the_plddt_cohort():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["confidence_preflight"] = {"summary_unit": "atom", "mean": 85.0}
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(pairs, bootstrap_iterations=100, seed=7)

    attrition = result["sensitivity_cohorts"]["mean_plddt_ge_70"]["attrition"]
    assert attrition["retained_pair_count"] == 0
    assert attrition["excluded_missing_filter_metadata_count"] == 1


def test_failed_prediction_remains_in_reliability_but_not_continuous_domain():
    experimental_row = _experimental("e1")
    predicted_row = _predicted("p1", "e1")
    predicted_row["status"] = "failed"
    predicted_row["error"] = "fixture failure"

    pairs = ANALYZE.pair_rows({"files": [experimental_row]}, {"files": [predicted_row]})
    result = ANALYZE.analyze_pairs(
        pairs,
        methods=("optcuts_automatic",),
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    method = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert method["finite_common_pair_count_by_endpoint"]["objective_weighted_fragmentation"] == 0
    reliability = method["all_attempted_unusable_output"]
    assert reliability["paired_structure_count"] == 1
    assert reliability["mean_paired_difference"] == 1.0


def test_continuous_domains_are_endpoint_specific():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    del predicted["independent_optcuts_arm_quality"]["optcuts_automatic"]["quality"]["seam"]
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        methods=("optcuts_automatic",),
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    method = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert method["finite_common_pair_count_by_endpoint"]["objective_weighted_fragmentation"] == 1
    assert method["finite_common_pair_count_by_endpoint"]["normalized_seam_length"] == 0
    assert method["all_attempted_unusable_output"]["mean_paired_difference"] == 1.0


def test_transport_rejects_legacy_top_level_method_payloads():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    del predicted["independent_optcuts_arm_quality"]
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        methods=("optcuts_automatic",),
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    transport = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert transport["finite_common_pair_count_by_endpoint"]["distortion_mean"] == 0
    assert transport["all_attempted_unusable_output"]["mean_paired_difference"] == 1.0


def test_noninjective_arm_remains_in_continuous_efficacy_and_reliability_denominator():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted_arm = predicted["independent_optcuts_arm_quality"]["optcuts_automatic"]
    predicted_arm["usable"] = False
    predicted_arm["globally_injective"] = False
    predicted_arm["quality"]["injectivity"]["all_patches_globally_injective"] = False
    pair_arm = predicted["residue_aware_pair_quality"]["arms"]["optcuts_automatic"]
    pair_arm["usable"] = False
    pair_arm["globally_injective"] = False
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        methods=("optcuts_automatic",),
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    method = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert method["finite_common_pair_count_by_endpoint"]["distortion_mean"] == 1
    assert method["all_attempted_unusable_output"]["mean_paired_difference"] == 1.0


def test_independent_arms_survive_unrelated_comparator_failure():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    experimental["status"] = "incomplete_comparison"
    predicted["status"] = "incomplete_comparison"
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    result = ANALYZE.analyze_pairs(
        pairs,
        bootstrap_iterations=100,
        seed=7,
        include_sensitivity_cohorts=False,
    )

    transport = result["main_all_frozen_test_pairs"]["method_transport"]["optcuts_automatic"]
    assert transport["finite_common_pair_count_by_endpoint"]["distortion_mean"] == 1
    efficacy = result["main_all_frozen_test_pairs"]["residue_aware_efficacy_by_structure_source"]
    assert efficacy["experimental"]["continuous_endpoints"]["distortion_mean"]["paired_structure_count"] == 1


def test_pairing_rejects_split_component_drift():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["analysis_split_component_id"] = "different"

    with pytest.raises(ValueError, match="analysis_split_component_id"):
        ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})


def test_pairing_rejects_experimental_method_metadata_drift():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["experimental_method_contains_nmr"] = True

    with pytest.raises(ValueError, match="inconsistent experimental_method_contains_nmr"):
        ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})


def test_pairing_rejects_mixed_predicted_cohort_semantics():
    first_experimental = _experimental("e1")
    second_experimental = _experimental("e2")
    first = _predicted("p1", "e1")
    second = _predicted("p2", "e2")
    second["structure_type"] = "afdb"

    with pytest.raises(ValueError, match="mixes predicted structure types"):
        ANALYZE.pair_rows(
            {"files": [first_experimental, second_experimental]},
            {"files": [first, second]},
        )


def test_pairing_rejects_duplicate_use_of_one_experimental_record():
    experimental = _experimental("e1")
    first = _predicted("p1", "e1")
    second = _predicted("p2", "e1")
    second["paired_record_id"] = "another-pair-id"

    with pytest.raises(ValueError, match="reuses an experimental record"):
        ANALYZE.pair_rows({"files": [experimental]}, {"files": [first, second]})


def test_pairing_rejects_a_mixed_analysis_split_report():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["analysis_split"] = "development"

    with pytest.raises(ValueError, match="test-only predicted report"):
        ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})


def test_missing_filter_metadata_is_not_counted_as_filter_failure():
    experimental = _experimental("e1")
    predicted = _predicted("p1", "e1")
    predicted["selected_residue_contact_fraction"] = ""
    pairs = ANALYZE.pair_rows({"files": [experimental]}, {"files": [predicted]})

    selected, attrition = ANALYZE._subset(
        pairs,
        [ANALYZE._numeric_filter("selected_residue_contact_fraction", lambda value: value >= 0.75)],
    )

    assert selected == []
    assert attrition["excluded_missing_filter_metadata_count"] == 1
    assert attrition["excluded_filter_mismatch_count"] == 0


def _formal_report_payload(root: Path) -> dict[str, object]:
    commit = "a" * 40
    config = {
        "formal_mode": True,
        "benchmark_purpose": "quality",
        "repetitions": 1,
        "warmup_runs": 0,
        "expected_git_commit": commit,
        "input_folder": str(root / "inputs"),
        "output_root": str(root),
        "manifest_path": str(root / "manifest.csv"),
        "coordinate_audit_path": str(root / "coordinate-audit.json"),
        "expected_coordinate_audit_sha256": "b" * 64,
        "optcuts": {"expected_binary_sha256": "c" * 64, "residue_fragmentation_weight": 5.0},
        **ANALYZE.ARTIFACT_FILENAMES,
        "artifact_checksums_filename": "benchmark_artifact_checksums.json",
    }
    return {
        "schema_version": "2.0",
        "topoppi_version": __version__,
        "config": config,
        "runtime": {
            "formal_mode": True,
            "config_fingerprint": "fingerprint",
            "coordinate_audit": {"status": "validated", "actual_sha256": "b" * 64},
            "environment": {
                "git_commit": commit,
                "git_worktree_dirty": False,
                "package_versions": {"topoppi": __version__},
            },
        },
        "metric_protocol": {"domain": "exact"},
        "preprocessing": {"accepted_files": 0, "integrity_error_count": 0, "accepted": []},
        "files": [],
    }


def _write_formal_bundle(root: Path) -> Path:
    payload = _formal_report_payload(root)
    report = root / "benchmark_report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    for filename in set(ANALYZE.ARTIFACT_FILENAMES.values()) - {report.name}:
        (root / filename).write_text(f"fixture:{filename}\n", encoding="utf-8")
    artifacts = []
    for filename in ANALYZE.ARTIFACT_FILENAMES.values():
        path = root / filename
        artifacts.append(
            {
                "filename": filename,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "benchmark_artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "config_fingerprint": "fingerprint",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return report


def test_read_report_validates_the_complete_evidence_bundle(tmp_path):
    report = _write_formal_bundle(tmp_path)

    assert ANALYZE.read_report(report)["schema_version"] == "2.0"
    (tmp_path / "benchmark_summary.csv").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum differs"):
        ANALYZE.read_report(report)


def test_paired_reports_require_the_same_scientific_protocol(tmp_path):
    experimental = _formal_report_payload(tmp_path / "experimental")
    predicted = copy.deepcopy(experimental)
    predicted["config"]["input_folder"] = "/different/input"
    predicted["config"]["output_root"] = "/different/output"
    predicted["config"]["manifest_path"] = "/different/manifest.csv"
    predicted["config"]["coordinate_audit_path"] = "/different/coordinate-audit.json"
    predicted["config"]["expected_coordinate_audit_sha256"] = "d" * 64
    predicted["runtime"]["coordinate_audit"]["actual_sha256"] = "d" * 64

    signature = ANALYZE.require_compatible_protocols(experimental, predicted)

    assert len(signature) == 64
    predicted["config"]["optcuts"]["residue_fragmentation_weight"] = 20.0
    with pytest.raises(ValueError, match="different scientific protocols"):
        ANALYZE.require_compatible_protocols(experimental, predicted)
