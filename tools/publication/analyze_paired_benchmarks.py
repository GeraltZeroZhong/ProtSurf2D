#!/usr/bin/env python3
"""Analyze experimental/predicted benchmarks on exact paired test records.

The script deliberately treats failures as outcomes for reliability while
restricting continuous efficacy estimates to prespecified finite common
domains.  Dependence metadata comes from the predicted cohort, where reused
AFDB source accessions have already been joined to experimental homology
clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from topoppi.benchmarking.evidence_bundle import (
    BENCHMARK_ARTIFACT_FILENAMES,
    read_json_object,
    validate_benchmark_evidence_bundle,
)
from topoppi.benchmarking.manifest_metadata import (
    INFERENCE_DEPENDENCY_BASIS,
    inference_family_id,
)
from topoppi.benchmarking.statistics import paired_method_comparison
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

DEFAULT_METHODS = ("optcuts_automatic", "residue_aware_optcuts")
STANDARD_METHOD = "optcuts_automatic"
RESIDUE_AWARE_METHOD = "residue_aware_optcuts"
PAIRED_CLUSTER_KEY = "inference_family_id"
ENDPOINTS: dict[str, tuple[tuple[str, ...], float]] = {
    "objective_weighted_fragmentation": (
        ("residue_footprint_fragmentation", "objective_weighted_fragmentation"),
        0.0,
    ),
    "symmetric_dirichlet_mean": (("symmetric_dirichlet", "mean"), 2.0),
    "distortion_mean": (("distortion", "mean"), 0.0),
    "angle_distortion_mean": (("angle_distortion", "mean"), 0.0),
    "area_distortion_mean": (("area_distortion", "mean"), 0.0),
    "normalized_seam_length": (("seam", "seam_length_3d_normalized"), 0.0),
}
PRIMARY_ENDPOINT = "objective_weighted_fragmentation"
ARTIFACT_FILENAMES = BENCHMARK_ARTIFACT_FILENAMES
NONCOMPARABLE_CONFIG_FIELDS = {
    "input_folder",
    "output_root",
    "manifest_path",
    "coordinate_audit_path",
    "expected_coordinate_audit_sha256",
}
PREDICTED_COHORT_SEMANTICS: dict[str, dict[str, object]] = {
    "afdb_monomer_replacement": {
        "structure_type": "afdb_monomer_replacement",
        "analysis_role": "controlled fixed-pose conformational sensitivity analysis",
        "pose_source": (
            "experimental relative partner pose after each AFDB monomer is independently "
            "superposed to its matched experimental chain"
        ),
        "variables_changed": [
            "predicted monomer coordinates and covered residue domains",
            "the resulting partner surfaces and interface contacts",
        ],
        "transport_estimand": (
            "change in each endpoint, and in the within-structure TopoPPI benefit, after "
            "substituting AFDB monomer geometry while retaining the experimental relative pose"
        ),
        "evidence_role": "controlled conformational and surface-geometry sensitivity evidence",
        "interpretation_limit": (
            "fixed-pose monomer perturbation with the experimental relative pose held constant; "
            "conclusions apply to conformational and surface-geometry sensitivity"
        ),
    },
    "afdb": {
        "structure_type": "afdb",
        "analysis_role": "matched predicted-complex external-validity analysis",
        "pose_source": "AlphaFold DB predicted complex assembly",
        "variables_changed": [
            "predicted partner conformations and covered residue domains",
            "predicted relative partner pose and the resulting interface contacts",
        ],
        "transport_estimand": (
            "change in each endpoint, and in the within-structure TopoPPI benefit, between "
            "the independently predicted AFDB complex and its matched experimental complex"
        ),
        "evidence_role": "secondary matched predicted-complex external-validity evidence",
        "interpretation_limit": (
            "conformation and relative-pose effects are combined, and AFDB complex availability "
            "defines a selected matched subset with limited deployment-population coverage"
        ),
    },
}


def read_report(path: Path) -> dict[str, object]:
    payload = read_json_object(path, "Benchmark report")
    if payload.get("schema_version") != "2.0" or not isinstance(payload.get("files"), list):
        raise ValueError(f"Benchmark report has no files list: {path}")
    config = payload.get("config") or {}
    runtime = payload.get("runtime") or {}
    coordinate_audit = runtime.get("coordinate_audit") if isinstance(runtime, dict) else None
    if not isinstance(config, dict) or not bool(config.get("formal_mode")):
        raise ValueError(f"Paired publication analysis requires a formal benchmark report: {path}")
    if not isinstance(runtime, dict) or not bool(runtime.get("formal_mode")):
        raise ValueError(f"Benchmark runtime is not marked formal: {path}")
    if not isinstance(coordinate_audit, dict) or coordinate_audit.get("status") != "validated":
        raise ValueError(f"Benchmark report lacks a validated coordinate audit: {path}")
    if _text(config.get("benchmark_purpose")).lower() != "quality":
        raise ValueError(f"Paired publication analysis requires a quality benchmark: {path}")
    if int(config.get("repetitions", -1)) != 1 or int(config.get("warmup_runs", -1)) != 0:
        raise ValueError(f"Formal paired quality reports require one measurement and no warm-up: {path}")
    environment = runtime.get("environment")
    if not isinstance(environment, Mapping) or environment.get("git_worktree_dirty") is not False:
        raise ValueError(f"Benchmark report lacks clean-worktree provenance: {path}")
    commit = _text(environment.get("git_commit")).lower()
    expected_commit = _text(config.get("expected_git_commit")).lower()
    if not commit or not expected_commit or commit != expected_commit:
        raise ValueError(f"Benchmark report Git revision differs from its frozen configuration: {path}")
    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError(f"Benchmark report lacks preprocessing evidence: {path}")
    files = payload["files"]
    try:
        accepted_count = int(preprocessing.get("accepted_files"))
        integrity_errors = int(preprocessing.get("integrity_error_count"))
    except (TypeError, ValueError):
        accepted_count = -1
        integrity_errors = -1
    if accepted_count != len(files) or integrity_errors != 0:
        raise ValueError(f"Benchmark report does not retain every accepted structure attempt: {path}")
    accepted = preprocessing.get("accepted")
    if not isinstance(accepted, list) or any(not isinstance(row, Mapping) for row in accepted):
        raise ValueError(f"Benchmark report lacks per-structure preprocessing evidence: {path}")

    def identity(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (
            _text(row.get("manifest_record_id")),
            _text(row.get("pdb")),
            _text(row.get("input_sha256")).lower(),
        )

    accepted_identities = [identity(row) for row in accepted]
    result_identities = [identity(row) for row in files if isinstance(row, Mapping)]
    if (
        len(result_identities) != len(files)
        or any(not all(values) for values in (*accepted_identities, *result_identities))
        or len(set(accepted_identities)) != len(accepted_identities)
        or len(set(result_identities)) != len(result_identities)
        or set(accepted_identities) != set(result_identities)
    ):
        raise ValueError(f"Benchmark result identities differ from accepted preprocessing jobs: {path}")
    validate_benchmark_evidence_bundle(path, payload)
    return payload


def _protocol_signature(report: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    config = report.get("config")
    runtime = report.get("runtime")
    if not isinstance(config, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("Benchmark report lacks protocol metadata.")
    environment = runtime.get("environment")
    coordinate_audit = runtime.get("coordinate_audit")
    if not isinstance(environment, Mapping) or not isinstance(coordinate_audit, Mapping):
        raise ValueError("Benchmark report lacks runtime provenance.")
    comparable_config = {key: value for key, value in config.items() if key not in NONCOMPARABLE_CONFIG_FIELDS}
    protocol = {
        "topoppi_version": report.get("topoppi_version"),
        "python": runtime.get("python"),
        "platform": runtime.get("platform"),
        "git_commit": environment.get("git_commit"),
        "package_versions": environment.get("package_versions"),
        "metric_protocol": report.get("metric_protocol"),
        "config": comparable_config,
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), protocol


def require_compatible_protocols(
    experimental: Mapping[str, object],
    predicted: Mapping[str, object],
) -> str:
    experimental_signature, _experimental_protocol = _protocol_signature(experimental)
    predicted_signature, _predicted_protocol = _protocol_signature(predicted)
    if experimental_signature != predicted_signature:
        raise ValueError("Experimental and predicted benchmark reports use different scientific protocols.")
    return experimental_signature


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _metadata_boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _text(value).lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def _experimental_method_metadata(
    row: Mapping[str, object],
    *,
    label: str,
) -> tuple[tuple[str, ...], str, bool]:
    raw_methods = _text(row.get("experimental_methods_json"))
    try:
        parsed = json.loads(raw_methods)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} has invalid experimental_methods_json metadata.") from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(method, str) or not method.strip() for method in parsed)
    ):
        raise ValueError(f"{label} has invalid experimental_methods_json metadata.")
    methods = tuple(method.strip().upper() for method in parsed)
    if len(set(methods)) != len(methods):
        raise ValueError(f"{label} has duplicate experimental methods.")
    group = _text(row.get("experimental_method_group")).lower()
    contains_nmr = _metadata_boolean(row.get("experimental_method_contains_nmr"))
    if not group or contains_nmr is None:
        raise ValueError(f"{label} lacks frozen experimental-method metadata.")
    if contains_nmr != any("NMR" in method for method in methods):
        raise ValueError(f"{label} has inconsistent experimental_method_contains_nmr metadata.")
    return methods, group, contains_nmr


def _cohort_semantics_for_structure_types(
    structure_types: Sequence[object],
    *,
    context: str,
) -> dict[str, object]:
    normalized = {_text(value).lower() for value in structure_types}
    if "" in normalized:
        raise ValueError(f"{context} contains a row without structure_type.")
    if len(normalized) != 1:
        raise ValueError(f"{context} mixes predicted structure types: {sorted(normalized)}")
    structure_type = next(iter(normalized))
    semantics = PREDICTED_COHORT_SEMANTICS.get(structure_type)
    if semantics is None:
        raise ValueError(f"{context} has unsupported predicted structure_type: {structure_type}")
    return dict(semantics)


def predicted_cohort_semantics(report: Mapping[str, object]) -> dict[str, object]:
    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Predicted report contains no attempted structures.")
    if any(not isinstance(row, Mapping) for row in files):
        raise ValueError("Predicted report contains a non-object file row.")
    return _cohort_semantics_for_structure_types(
        [row.get("structure_type") for row in files],
        context="Predicted report",
    )


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nested(payload: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _method_payload(row: Mapping[str, object], method: str) -> dict[str, object] | None:
    if _text(row.get("status")).lower() not in {"ok", "complete", "incomplete_comparison"} or row.get("error"):
        return None
    evidence = row.get("independent_optcuts_arm_quality")
    if not isinstance(evidence, Mapping):
        return None
    arm = evidence.get(method)
    if (
        not isinstance(arm, Mapping)
        or not bool(arm.get("domain_complete", False))
        or not bool(arm.get("metric_finite", False))
    ):
        return None
    payload = arm.get("quality")
    return payload if isinstance(payload, dict) else None


def _residue_aware_pair_payloads(
    row: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]] | None:
    if _text(row.get("status")).lower() not in {"ok", "complete", "incomplete_comparison"} or row.get("error"):
        return None
    pair = row.get("residue_aware_pair_quality")
    if not isinstance(pair, Mapping) or not bool(pair.get("complete", False)):
        return None
    arms = pair.get("arms")
    if not isinstance(arms, Mapping) or any(
        not isinstance(arms.get(method), Mapping)
        or not bool(arms[method].get("domain_complete", False))
        or not bool(arms[method].get("metric_finite", False))
        for method in (STANDARD_METHOD, RESIDUE_AWARE_METHOD)
    ):
        return None
    methods = pair.get("methods")
    if not isinstance(methods, Mapping):
        return None
    standard = methods.get(STANDARD_METHOD)
    topoppi = methods.get(RESIDUE_AWARE_METHOD)
    if not isinstance(standard, dict) or not isinstance(topoppi, dict):
        return None
    return standard, topoppi


def _finite_method_payload(
    row: Mapping[str, object],
    method: str,
    path: Sequence[str],
) -> dict[str, object] | None:
    payload = _method_payload(row, method)
    return payload if payload is not None and _finite(_nested(payload, path)) is not None else None


def _unusable_output(row: Mapping[str, object], method: str) -> float:
    payload = _method_payload(row, method)
    if payload is None:
        return 1.0
    if any(_finite(_nested(payload, path)) is None for path, _reference in ENDPOINTS.values()):
        return 1.0
    injectivity = payload.get("injectivity")
    if not isinstance(injectivity, Mapping):
        return 1.0
    return float(not bool(injectivity.get("all_patches_globally_injective", False)))


def _index_experimental(report: Mapping[str, object]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for raw in report["files"]:
        if not isinstance(raw, dict):
            raise ValueError("Experimental report contains a non-object file row.")
        record_id = _text(raw.get("manifest_record_id"))
        if not record_id:
            raise ValueError("Experimental report row lacks manifest_record_id.")
        if _text(raw.get("analysis_split")).lower() != "test":
            raise ValueError("Paired publication analysis requires a test-only experimental report.")
        if record_id in indexed:
            raise ValueError(f"Duplicate experimental manifest_record_id: {record_id}")
        indexed[record_id] = raw
    return indexed


def pair_rows(
    experimental_report: Mapping[str, object],
    predicted_report: Mapping[str, object],
) -> list[dict[str, object]]:
    """Join every attempted predicted test record to exactly one experiment."""

    experimental = _index_experimental(experimental_report)
    paired: list[dict[str, object]] = []
    predicted_ids: set[str] = set()
    reference_ids: set[str] = set()
    pair_ids: set[str] = set()
    for raw in predicted_report["files"]:
        if not isinstance(raw, dict):
            raise ValueError("Predicted report contains a non-object file row.")
        if _text(raw.get("analysis_split")).lower() != "test":
            raise ValueError("Paired publication analysis requires a test-only predicted report.")
        predicted_record_id = _text(raw.get("manifest_record_id"))
        reference_id = _text(raw.get("paired_experimental_record_id"))
        pair_id = _text(raw.get("paired_record_id"))
        if not predicted_record_id or not reference_id or not pair_id:
            raise ValueError("Predicted test row lacks manifest, pair, or paired experimental record ID.")
        if predicted_record_id in predicted_ids:
            raise ValueError(f"Duplicate predicted manifest_record_id: {predicted_record_id}")
        if reference_id in reference_ids:
            raise ValueError(f"Predicted report reuses an experimental record: {reference_id}")
        if pair_id in pair_ids:
            raise ValueError(f"Predicted report contains duplicate paired_record_id: {pair_id}")
        predicted_ids.add(predicted_record_id)
        reference_ids.add(reference_id)
        pair_ids.add(pair_id)
        reference = experimental.get(reference_id)
        if reference is None:
            raise ValueError(f"Predicted row references absent experiment: {reference_id}")
        if _text(reference.get("analysis_split")).lower() != "test":
            raise ValueError(f"Paired experiment is not in the frozen test split: {reference_id}")
        if _text(reference.get("structure_type")).lower() != "experimental":
            raise ValueError(f"Paired reference is not marked experimental: {reference_id}")
        _cohort_semantics_for_structure_types(
            [raw.get("structure_type")],
            context=f"Predicted row {predicted_record_id}",
        )
        for field in (
            "cluster_id",
            "family_id",
            "sequence_cluster_a",
            "sequence_cluster_b",
            "analysis_split_component_id",
            "analysis_split_basis",
        ):
            predicted_value = _text(raw.get(field))
            reference_value = _text(reference.get(field))
            if not predicted_value or not reference_value or predicted_value != reference_value:
                raise ValueError(
                    f"Paired identity metadata differs for {reference_id}: {field}="
                    f"{predicted_value!r} vs {reference_value!r}"
                )
        predicted_method_metadata = _experimental_method_metadata(
            raw,
            label=f"Predicted row {predicted_record_id}",
        )
        reference_method_metadata = _experimental_method_metadata(
            reference,
            label=f"Experimental row {reference_id}",
        )
        if predicted_method_metadata != reference_method_metadata:
            raise ValueError(f"Paired experimental-method metadata differs for {reference_id}.")
        inference_fields = (
            "inference_sequence_cluster_a",
            "inference_sequence_cluster_b",
            "inference_family_id",
            "inference_dependency_basis",
        )
        if any(not _text(raw.get(field)) for field in inference_fields):
            raise ValueError(f"Predicted paired row lacks frozen dependence metadata: {predicted_record_id}")
        for field in inference_fields:
            if _text(reference.get(field)) != _text(raw.get(field)):
                raise ValueError(f"Paired inference metadata differs for {reference_id}: {field}")
        inference_a = _text(raw.get("inference_sequence_cluster_a"))
        inference_b = _text(raw.get("inference_sequence_cluster_b"))
        if _text(raw.get("inference_family_id")) != inference_family_id(inference_a, inference_b):
            raise ValueError(f"Predicted paired row has an invalid inference family: {predicted_record_id}")
        if _text(raw.get("inference_dependency_basis")) != INFERENCE_DEPENDENCY_BASIS:
            raise ValueError(f"Predicted paired row has an invalid inference dependency basis: {predicted_record_id}")
        paired.append(
            {
                "predicted_record_id": predicted_record_id,
                "experimental_record_id": reference_id,
                "experimental": reference,
                "predicted": raw,
            }
        )
    if not paired:
        raise ValueError("Predicted report contains no attempted test records.")
    _cohort_semantics_for_structure_types(
        [pair["predicted"].get("structure_type") for pair in paired],
        context="Predicted paired cohort",
    )
    component_by_inference_dependency: dict[str, str] = {}
    for pair in paired:
        predicted = pair["predicted"]
        component = _text(predicted.get("analysis_split_component_id"))
        for field in (
            "inference_sequence_cluster_a",
            "inference_sequence_cluster_b",
            "inference_family_id",
        ):
            dependency = _text(predicted.get(field))
            previous = component_by_inference_dependency.setdefault(dependency, component)
            if previous != component:
                raise ValueError(f"Prediction dependency {dependency!r} is split across analysis components.")
    return paired


def _comparison_metadata(pair: Mapping[str, object]) -> dict[str, object]:
    predicted = pair["predicted"]
    if not isinstance(predicted, Mapping):
        raise ValueError("Invalid paired predicted row.")
    keys = (
        "family_id",
        "cluster_id",
        "sequence_cluster_a",
        "sequence_cluster_b",
        "inference_sequence_cluster_a",
        "inference_sequence_cluster_b",
        "inference_family_id",
        "inference_dependency_basis",
        "analysis_split",
        "analysis_split_component_id",
        "paired_geometry_stratum",
        "paired_contact_cutoff_angstrom",
        "paired_predicted_contact_count_total",
        "paired_alignment_a_optimal_correspondence_count",
        "paired_alignment_b_optimal_correspondence_count",
        "paired_alignment_a_selected_pair_consensus_fraction",
        "paired_alignment_b_selected_pair_consensus_fraction",
    )
    return {key: predicted.get(key) for key in keys}


def _predicted_interface_availability(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = []
    cutoffs = set()
    for pair in pairs:
        predicted = pair.get("predicted")
        if not isinstance(predicted, Mapping):
            raise ValueError("Invalid paired predicted row.")
        count = _finite(predicted.get("paired_predicted_contact_count_total"))
        cutoff = _finite(predicted.get("paired_contact_cutoff_angstrom"))
        if count is None or count < 0.0 or not count.is_integer():
            raise ValueError("Predicted paired row lacks a non-negative integer contact count.")
        if cutoff is None or cutoff <= 0.0:
            raise ValueError("Predicted paired row lacks a positive contact cutoff.")
        counts.append(int(count))
        cutoffs.add(float(cutoff))
    if len(cutoffs) != 1:
        raise ValueError("Predicted paired rows use inconsistent contact cutoffs.")
    present = sum(count > 0 for count in counts)
    return {
        "analysis_role": "upstream_predicted_interface_availability",
        "attempted_pair_count": len(counts),
        "contact_cutoff_angstrom": next(iter(cutoffs)),
        "predicted_contact_present_pair_count": present,
        "predicted_contact_absent_pair_count": len(counts) - present,
        "predicted_contact_present_fraction": present / len(counts),
        "definition": (
            "at least one cross-chain residue pair has heavy atoms within the frozen contact cutoff; "
            "contact absence remains in all-attempted reliability but is outside continuous map efficacy"
        ),
    }


def _paired_values(
    pairs: Sequence[Mapping[str, object]],
    method: str,
    path: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for pair in pairs:
        experimental = pair["experimental"]
        predicted = pair["predicted"]
        if not isinstance(experimental, Mapping) or not isinstance(predicted, Mapping):
            continue
        experimental_payload = _finite_method_payload(experimental, method, path)
        predicted_payload = _finite_method_payload(predicted, method, path)
        if experimental_payload is None or predicted_payload is None:
            continue
        rows.append(
            {
                **_comparison_metadata(pair),
                "experimental_method": experimental_payload,
                "predicted_method": predicted_payload,
            }
        )
    return rows


def _paired_reliability(
    pairs: Sequence[Mapping[str, object]],
    method: str,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    for pair in pairs:
        experimental = pair["experimental"]
        predicted = pair["predicted"]
        if not isinstance(experimental, Mapping) or not isinstance(predicted, Mapping):
            continue
        rows.append(
            {
                **_comparison_metadata(pair),
                "predicted_method": {"unusable": _unusable_output(predicted, method)},
                "experimental_method": {"unusable": _unusable_output(experimental, method)},
            }
        )
    comparison = paired_method_comparison(
        rows,
        baseline="predicted_method",
        treatment="experimental_method",
        metric_path=("unusable",),
        cluster_key=PAIRED_CLUSTER_KEY,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        binary_endpoint=True,
    )
    comparison.update(
        {
            "analysis_role": "all_attempted_unusable_output_transport",
            "coding": ("0=complete finite globally injective output; 1=incomplete, nonfinite, or noninjective output"),
            "signed_difference_definition": (
                "predicted minus experimental unusable-output indicator; positive values indicate "
                "lower reliability on the predicted structure"
            ),
        }
    )
    return comparison


def _method_transport(
    pairs: Sequence[Mapping[str, object]],
    method: str,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    comparisons = {}
    common_counts = {}
    for index, (name, (path, reference)) in enumerate(ENDPOINTS.items()):
        common = _paired_values(pairs, method, path)
        common_counts[name] = len(common)
        comparisons[name] = paired_method_comparison(
            common,
            baseline="predicted_method",
            treatment="experimental_method",
            metric_path=path,
            cluster_key=PAIRED_CLUSTER_KEY,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + index,
            relative_reference=reference,
        )
    return {
        "attempted_pair_count": len(pairs),
        "finite_common_pair_count_by_endpoint": common_counts,
        "excluded_from_continuous_efficacy_count_by_endpoint": {
            name: len(pairs) - count for name, count in common_counts.items()
        },
        "signed_difference_definition": (
            "predicted minus experimental; positive values indicate larger lower-is-better error "
            "on the predicted structure"
        ),
        "continuous_endpoints": comparisons,
        "all_attempted_unusable_output": _paired_reliability(
            pairs,
            method,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + 80,
        ),
    }


def _topoppi_effect_transport(
    pairs: Sequence[Mapping[str, object]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    endpoint_rows: dict[str, list[dict[str, object]]] = {name: [] for name in ENDPOINTS}
    for pair in pairs:
        experimental = pair["experimental"]
        predicted = pair["predicted"]
        if not isinstance(experimental, Mapping) or not isinstance(predicted, Mapping):
            continue
        experimental_pair = _residue_aware_pair_payloads(experimental)
        predicted_pair = _residue_aware_pair_payloads(predicted)
        if experimental_pair is None or predicted_pair is None:
            continue
        exp_standard, exp_topoppi = experimental_pair
        pred_standard, pred_topoppi = predicted_pair
        for name, (path, _reference) in ENDPOINTS.items():
            endpoint_payloads = (exp_standard, exp_topoppi, pred_standard, pred_topoppi)
            if any(_finite(_nested(payload, path)) is None for payload in endpoint_payloads):
                continue
            exp_effect = float(_nested(exp_standard, path)) - float(_nested(exp_topoppi, path))
            pred_effect = float(_nested(pred_standard, path)) - float(_nested(pred_topoppi, path))
            endpoint_rows[name].append(
                {
                    **_comparison_metadata(pair),
                    "predicted_effect": {"value": pred_effect},
                    "experimental_effect": {"value": exp_effect},
                }
            )
    comparisons = {
        name: paired_method_comparison(
            rows,
            baseline="predicted_effect",
            treatment="experimental_effect",
            metric_path=("value",),
            cluster_key=PAIRED_CLUSTER_KEY,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + index,
        )
        for index, (name, rows) in enumerate(endpoint_rows.items())
    }
    return {
        "attempted_pair_count": len(pairs),
        "four_method_payload_common_pair_count_by_endpoint": {name: len(rows) for name, rows in endpoint_rows.items()},
        "effect_definition": (
            "within-structure TopoPPI benefit is standard OptCuts minus TopoPPI for each lower-is-better endpoint"
        ),
        "signed_transport_difference_definition": ("predicted TopoPPI benefit minus experimental TopoPPI benefit"),
        "continuous_endpoints": comparisons,
    }


def _topoppi_efficacy(
    pairs: Sequence[Mapping[str, object]],
    source: str,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    if source not in {"experimental", "predicted"}:
        raise ValueError("TopoPPI efficacy source must be experimental or predicted.")
    endpoint_rows: dict[str, list[dict[str, object]]] = {name: [] for name in ENDPOINTS}
    for pair in pairs:
        structure = pair.get(source)
        if not isinstance(structure, Mapping):
            continue
        payloads = _residue_aware_pair_payloads(structure)
        if payloads is None:
            continue
        standard, topoppi = payloads
        for name, (path, _reference) in ENDPOINTS.items():
            if _finite(_nested(standard, path)) is None or _finite(_nested(topoppi, path)) is None:
                continue
            endpoint_rows[name].append(
                {
                    **_comparison_metadata(pair),
                    "standard": standard,
                    "topoppi": topoppi,
                }
            )
    comparisons = {
        name: paired_method_comparison(
            rows,
            baseline="standard",
            treatment="topoppi",
            metric_path=ENDPOINTS[name][0],
            cluster_key=PAIRED_CLUSTER_KEY,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + index,
            relative_reference=ENDPOINTS[name][1],
        )
        for index, (name, rows) in enumerate(endpoint_rows.items())
    }
    return {
        "structure_source": source,
        "attempted_pair_count": len(pairs),
        "exact_standard_topoppi_common_pair_count_by_endpoint": {
            name: len(rows) for name, rows in endpoint_rows.items()
        },
        "effect_definition": (
            "standard OptCuts minus TopoPPI on the exact residue-aware pair domain; "
            "positive values favor TopoPPI for every lower-is-better endpoint"
        ),
        "continuous_endpoints": comparisons,
    }


def _numeric_filter(
    field: str,
    predicate: Callable[[float], bool],
) -> Callable[[Mapping[str, object]], tuple[bool, bool]]:
    def evaluate(pair: Mapping[str, object]) -> tuple[bool, bool]:
        predicted = pair.get("predicted")
        value = _finite(predicted.get(field)) if isinstance(predicted, Mapping) else None
        return (False, True) if value is None else (bool(predicate(value)), False)

    return evaluate


def _text_filter(
    field: str,
    accepted: set[str],
) -> Callable[[Mapping[str, object]], tuple[bool, bool]]:
    def evaluate(pair: Mapping[str, object]) -> tuple[bool, bool]:
        predicted = pair.get("predicted")
        value = _text(predicted.get(field)).lower() if isinstance(predicted, Mapping) else ""
        return (False, True) if not value else (value in accepted, False)

    return evaluate


def _boolean_filter(
    field: str,
    expected: bool,
    *,
    source: str = "predicted",
) -> Callable[[Mapping[str, object]], tuple[bool, bool]]:
    if source not in {"experimental", "predicted"}:
        raise ValueError("Boolean cohort-filter source must be experimental or predicted.")

    def evaluate(pair: Mapping[str, object]) -> tuple[bool, bool]:
        row = pair.get(source)
        value = _metadata_boolean(row.get(field)) if isinstance(row, Mapping) else None
        return (False, True) if value is None else (value is expected, False)

    return evaluate


def _confidence_mean_filter(
    threshold: float,
) -> Callable[[Mapping[str, object]], tuple[bool, bool]]:
    def evaluate(pair: Mapping[str, object]) -> tuple[bool, bool]:
        predicted = pair.get("predicted")
        preflight = predicted.get("confidence_preflight") if isinstance(predicted, Mapping) else None
        if not isinstance(preflight, Mapping) or _text(preflight.get("summary_unit")) != "residue":
            return False, True
        value = _finite(preflight.get("mean")) if isinstance(preflight, Mapping) else None
        return (False, True) if value is None else (value >= threshold, False)

    return evaluate


def _subset(
    pairs: Sequence[Mapping[str, object]],
    filters: Sequence[Callable[[Mapping[str, object]], tuple[bool, bool]]],
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    retained = []
    missing = 0
    failed = 0
    for pair in pairs:
        outcomes = [condition(pair) for condition in filters]
        if any(is_missing for _matches, is_missing in outcomes):
            missing += 1
        elif all(matches for matches, _is_missing in outcomes):
            retained.append(pair)
        else:
            failed += 1
    return retained, {
        "source_pair_count": len(pairs),
        "retained_pair_count": len(retained),
        "excluded_missing_filter_metadata_count": missing,
        "excluded_filter_mismatch_count": failed,
    }


def _cohort_definitions():
    single = _numeric_filter("candidate_chain_pair_count", lambda value: value == 1.0)
    dominant = _numeric_filter("selected_residue_contact_fraction", lambda value: value >= 0.75)
    partner_length = _numeric_filter("chain_b_residue_count", lambda value: value >= 10.0)
    unique_alignment_a = _numeric_filter(
        "paired_alignment_a_optimal_correspondence_count",
        lambda value: value == 1.0,
    )
    unique_alignment_b = _numeric_filter(
        "paired_alignment_b_optimal_correspondence_count",
        lambda value: value == 1.0,
    )
    consensus_a = _numeric_filter(
        "paired_alignment_a_selected_pair_consensus_fraction",
        lambda value: value >= 0.95,
    )
    consensus_b = _numeric_filter(
        "paired_alignment_b_selected_pair_consensus_fraction",
        lambda value: value >= 0.95,
    )
    return {
        "all_frozen_test_pairs": [],
        "single_candidate_chain_pair": [single],
        "dominant_interface_residue_fraction_ge_0_75": [dominant],
        "partner_chain_b_residues_ge_10": [partner_length],
        "strict_binary_interface": [single, dominant, partner_length],
        "geometry_high_fidelity": [_text_filter("paired_geometry_stratum", {"high_fidelity"})],
        "geometry_high_or_moderate_fidelity": [
            _text_filter(
                "paired_geometry_stratum",
                {"high_fidelity", "moderate_fidelity"},
            )
        ],
        "mean_plddt_ge_70": [_confidence_mean_filter(70.0)],
        "unique_sequence_correspondence": [unique_alignment_a, unique_alignment_b],
        "sequence_correspondence_consensus_ge_0_95": [consensus_a, consensus_b],
        "exclude_any_nmr_experimental_method": [
            _boolean_filter(
                "experimental_method_contains_nmr",
                False,
                source="experimental",
            )
        ],
    }


def analyze_pairs(
    pairs: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    bootstrap_iterations: int = 10000,
    seed: int = 20260817,
    include_sensitivity_cohorts: bool = True,
) -> dict[str, object]:
    methods = tuple(dict.fromkeys(str(method).strip() for method in methods if str(method).strip()))
    if not methods:
        raise ValueError("At least one method is required.")
    unsupported = sorted(set(methods) - set(DEFAULT_METHODS))
    if unsupported:
        raise ValueError("Unsupported paired-analysis methods: " + ", ".join(unsupported))
    cohort_semantics = _cohort_semantics_for_structure_types(
        [pair.get("predicted", {}).get("structure_type") for pair in pairs],
        context="Predicted paired cohort",
    )

    def one_cohort(rows: Sequence[Mapping[str, object]], offset: int) -> dict[str, object]:
        residue_aware_requested = {STANDARD_METHOD, RESIDUE_AWARE_METHOD}.issubset(methods)
        return {
            "method_transport": {
                method: _method_transport(
                    rows,
                    method,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + offset + method_index * 100,
                )
                for method_index, method in enumerate(methods)
            },
            "residue_aware_efficacy_by_structure_source": (
                {
                    "experimental": _topoppi_efficacy(
                        rows,
                        "experimental",
                        bootstrap_iterations=bootstrap_iterations,
                        seed=seed + offset + 700,
                    ),
                    "predicted": _topoppi_efficacy(
                        rows,
                        "predicted",
                        bootstrap_iterations=bootstrap_iterations,
                        seed=seed + offset + 800,
                    ),
                }
                if residue_aware_requested
                else {"status": "not_requested"}
            ),
            "residue_aware_effect_transport": (
                _topoppi_effect_transport(
                    rows,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + offset + 900,
                )
                if residue_aware_requested
                else {"status": "not_requested"}
            ),
        }

    main = one_cohort(pairs, 0)
    sensitivities = {}
    if include_sensitivity_cohorts:
        for cohort_index, (name, filters) in enumerate(_cohort_definitions().items()):
            if not filters:
                continue
            selected, attrition = _subset(pairs, filters)
            sensitivities[name] = {
                "attrition": attrition,
                "analysis": one_cohort(selected, (cohort_index + 1) * 2000),
            }
    return {
        "attempted_test_pair_count": len(pairs),
        "upstream_predicted_interface_availability": _predicted_interface_availability(pairs),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_hypothesis": (
            "TopoPPI reduces objective-weighted residue-footprint fragmentation relative "
            "to standard OptCuts on each exact within-structure pair domain"
        ),
        "confirmatory_scope": (
            "confirmatory inference uses the full frozen experimental test benchmark; efficacy "
            f"within this matched predicted cohort is {cohort_semantics['evidence_role']}"
        ),
        "predicted_cohort_semantics": cohort_semantics,
        "transport_estimand": cohort_semantics["transport_estimand"],
        "transport_interpretation_limit": cohort_semantics["interpretation_limit"],
        "multiplicity_policy": (
            "the endpoint was prespecified, but all analyses in this paired transport artifact are "
            "external-validity or sensitivity estimates; transport, supporting endpoints, and "
            "filtered cohorts are interpreted by effect size and confidence interval"
        ),
        "main_all_frozen_test_pairs": main,
        "sensitivity_cohorts": sensitivities,
    }


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Predicted reports must use LABEL=/path/to/report.json.")
    return label.strip(), Path(raw_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired experimental and predicted benchmark reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experimental-report",
        required=True,
        type=Path,
        help="Experimental benchmark_report.json used as the pairing reference.",
    )
    parser.add_argument(
        "--predicted-report",
        action="append",
        required=True,
        type=_parse_labeled_path,
        help="Repeat as LABEL=/path/to/benchmark_report.json.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path for the paired analysis JSON.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Benchmark methods to include in each paired comparison.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Bootstrap samples per paired endpoint.",
    )
    parser.add_argument("--seed", type=int, default=20260817, help="Random seed for bootstrap sampling.")
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("Publication paired analysis requires at least 1000 bootstrap iterations.")

    labels = [label for label, _path in args.predicted_report]
    if len(labels) != len(set(labels)):
        raise ValueError("Predicted report labels must be unique.")
    experimental_path = args.experimental_report.expanduser().resolve()
    experimental = read_report(experimental_path)
    analyses = {}
    predicted_artifacts = {}
    predicted_reports = {}
    protocol_signatures = {}
    predicted_semantics = {}
    for label, path in args.predicted_report:
        predicted = read_report(path)
        protocol_signatures[label] = require_compatible_protocols(experimental, predicted)
        predicted_semantics[label] = predicted_cohort_semantics(predicted)
        pairs = pair_rows(experimental, predicted)
        analyses[label] = analyze_pairs(
            pairs,
            methods=args.methods,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        predicted_artifacts[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        predicted_reports[label] = predicted

    output = {
        "schema_version": "2.0",
        "analysis_population": "frozen test split only",
        "pairing_rule": "predicted.paired_experimental_record_id equals experimental.manifest_record_id",
        "continuous_missingness_rule": (
            "continuous effects use finite exact-pair common domains; all attempted pairs remain "
            "in the prespecified unusable-output endpoint"
        ),
        "dependence_rule": (
            "resolved prediction-dependency family means drive the primary bootstrap; "
            "a heterotypic-family dyadic-robust sandwich interval and node-influence diagnostics "
            "are retained as the shared-protein dependency sensitivity"
        ),
        "experimental_report": {
            "path": str(experimental_path),
            "sha256": sha256_file(experimental_path),
        },
        "predicted_reports": predicted_artifacts,
        "predicted_cohort_semantics": predicted_semantics,
        "scientific_protocol_signature_by_predicted_report": protocol_signatures,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "random_seed": int(args.seed),
        "analyses": analyses,
        "predicted_test_attempt_counts": {
            label: int(analysis["attempted_test_pair_count"]) for label, analysis in analyses.items()
        },
        "predicted_geometry_strata": {
            label: dict(
                sorted(
                    Counter(
                        _text(row.get("paired_geometry_stratum")) or "missing"
                        for row in report["files"]
                        if isinstance(row, dict) and _text(row.get("analysis_split")).lower() == "test"
                    ).items()
                )
            )
            for label, report in predicted_reports.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(output, args.output)


if __name__ == "__main__":
    main()
