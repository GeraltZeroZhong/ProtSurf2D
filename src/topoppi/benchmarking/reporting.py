"""Structure-level benchmark aggregation and flat evidence tables."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import numpy as np

from topoppi.benchmark_methods import (
    DEFAULT_STANDARD_METHODS,
    OPTCUTS_VARIANTS,
    PARAMETERIZATION_METHODS,
    RESIDUE_AWARE_BASELINE,
    RESIDUE_AWARE_OPTCUTS_METHODS,
)
from topoppi.benchmarking.manifest_metadata import PREDICTED_STRUCTURE_TYPES
from topoppi.benchmarking.statistics import paired_method_comparison

ALL_METHODS = (*PARAMETERIZATION_METHODS, *OPTCUTS_VARIANTS)
RESIDUE_AWARE_PAIR_METHODS = ("optcuts_automatic", "residue_aware_optcuts")

METHOD_ROLES = {
    "lscm": "free_boundary_conformal_baseline",
    "harmonic": "uniform_weight_tutte_bijective_baseline",
    "slim": "boundary_constrained_symmetric_dirichlet_optimization_baseline",
    "spherical": "legacy_projective_comparator",
    "cylindrical": "legacy_projective_comparator",
    "optcuts_automatic": "automatic_optcuts_reference",
    "optcuts_lscm_initialized": "feasibility_limited_initialization_diagnostic",
    "residue_aware_optcuts": "topoppi_method",
}


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _runtime_observation(measurement: Dict[str, object]) -> float:
    """Return completed runtime or the last rigorous right-censoring lower bound."""

    explicit = _as_float(measurement.get("runtime_observation_sec"))
    if np.isfinite(explicit):
        return explicit
    if bool(measurement.get("right_censored", False)):
        event = _as_float(measurement.get("censoring_event_elapsed_sec"))
        if np.isfinite(event):
            return event
        threshold = _as_float(measurement.get("censoring_threshold_sec"))
        if np.isfinite(threshold):
            return threshold
    return _as_float(measurement.get("wall_sec"))


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _metadata_boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _nested(record: Dict[str, object], path: Sequence[str], default=float("nan")) -> float:
    current: object = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return float(default)
        current = current[key]
    return _as_float(current, default=float(default))


def _distribution(values) -> Dict[str, float | int]:
    numeric = np.asarray([_as_float(value) for value in values], dtype=np.float64)
    array = numeric[np.isfinite(numeric)]
    if not len(array):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _residue_aware_pair_row(row: Dict[str, object]) -> tuple[Dict[str, object], bool]:
    """Expose exact TopoPPI-pair values at method keys for generic statistics."""

    pair_quality = row.get("residue_aware_pair_quality") or {}
    pair_methods = pair_quality.get("methods", {}) if isinstance(pair_quality, dict) else {}
    if not isinstance(pair_methods, dict) or not pair_methods:
        return row, False
    paired_row = dict(row)
    paired_row.update(pair_methods)
    return paired_row, True


def _initialization_pair_row(row: Dict[str, object]) -> tuple[Dict[str, object], bool]:
    pair_quality = row.get("optcuts_initialization_pair_quality") or {}
    pair_methods = pair_quality.get("methods", {}) if isinstance(pair_quality, dict) else {}
    if not isinstance(pair_methods, dict) or not pair_methods:
        return row, False
    paired_row = dict(row)
    paired_row.update(pair_methods)
    return paired_row, True


def _topology_pair_row(row: Dict[str, object]) -> tuple[Dict[str, object], bool]:
    pair_quality = row.get("topology_preprocessing_pair_quality") or {}
    pair_methods = pair_quality.get("methods", {}) if isinstance(pair_quality, dict) else {}
    if not isinstance(pair_methods, dict) or not pair_methods:
        return row, False
    paired_row = dict(row)
    paired_row.update(pair_methods)
    return paired_row, True


def _topology_arm_completion(row: Dict[str, object]) -> tuple[bool, bool, bool]:
    pair_quality = row.get("topology_preprocessing_pair_quality") or {}
    if not isinstance(pair_quality, dict):
        return False, False, False
    reached = str(pair_quality.get("status") or "") not in {"", "disabled", "not_evaluated"}
    expected = _as_int(pair_quality.get("expected_patch_count", 0))
    unique = bool(pair_quality.get("unique_patch_ids", False))
    without_preparation = bool(
        reached and unique and expected > 0 and _as_int(pair_quality.get("raw_success_patch_count", 0)) == expected
    )
    with_preparation = bool(
        reached and unique and expected > 0 and _as_int(pair_quality.get("prepared_success_patch_count", 0)) == expected
    )
    return reached, without_preparation, with_preparation


def _standard_pair_row(
    row: Dict[str, object],
    baseline: str,
    treatment: str = "optcuts_automatic",
) -> tuple[Dict[str, object], bool]:
    pairs = row.get("standard_method_pair_quality") or {}
    pair = pairs.get(f"{baseline}_vs_{treatment}", {}) if isinstance(pairs, dict) else {}
    methods = pair.get("methods", {}) if isinstance(pair, dict) else {}
    complete = bool(pair.get("complete", False)) if isinstance(pair, dict) else False
    if not complete or not isinstance(methods, dict) or not all(method in methods for method in (baseline, treatment)):
        return row, False
    paired_row = dict(row)
    paired_row.update(methods)
    return paired_row, True


def _has_residue_aware_methods(row: Dict[str, object], methods: Sequence[str]) -> bool:
    pair_quality = row.get("residue_aware_pair_quality") or {}
    pair_methods = pair_quality.get("methods", {}) if isinstance(pair_quality, dict) else {}
    source = pair_methods if isinstance(pair_methods, dict) and pair_methods else row
    return all(method in source for method in methods)


def _arm_has_valid_geometry(arm: object) -> bool:
    """Keep geometry QC separate from finite-output analysis eligibility."""

    return bool(
        isinstance(arm, dict)
        and bool(arm.get("domain_complete", False))
        and bool(arm.get("metric_finite", False))
        and bool(arm.get("globally_injective", False))
    )


def _atlas_reference_method(row: Dict[str, object]) -> str:
    atlas_trainability = row.get("atlas_trainability") or {}
    if isinstance(atlas_trainability, dict):
        declared = str(atlas_trainability.get("reference_method") or "").strip()
        if declared and isinstance(row.get(declared), dict):
            return declared
    for method in ("optcuts_automatic", "harmonic", "lscm"):
        if isinstance(row.get(method), dict):
            return method
    return ""


def _atlas_metric(row: Dict[str, object], metric: str) -> float:
    method = _atlas_reference_method(row)
    return _nested(row, (method, "atlas", metric)) if method else float("nan")


def _inferential_cluster_key(rows: Sequence[Dict[str, object]]) -> str:
    """Resolve the frozen dependence unit for a homogeneous or mixed cohort."""

    structure_types = {str(row.get("structure_type") or "experimental").strip().lower() for row in rows}
    predicted = structure_types & set(PREDICTED_STRUCTURE_TYPES)
    if structure_types and structure_types <= set(PREDICTED_STRUCTURE_TYPES):
        return "inference_family_id"
    if predicted:
        return "analysis_split_component_id"
    return "family_id"


def _apply_benjamini_hochberg(
    comparisons: Dict[str, Dict[str, object]],
    *,
    family: str,
) -> None:
    finite = [
        (name, float(block.get("wilcoxon_p_value", float("nan"))))
        for name, block in comparisons.items()
        if np.isfinite(float(block.get("wilcoxon_p_value", float("nan"))))
    ]
    if not finite:
        return
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        rank = reverse_index + 1
        candidate = ordered[reverse_index][1] * len(ordered) / rank
        running = min(running, candidate)
        adjusted[reverse_index] = min(1.0, running)
    for (name, _p_value), q_value in zip(ordered, adjusted, strict=True):
        comparisons[name]["wilcoxon_q_value_bh"] = float(q_value)
        comparisons[name]["multiple_testing_family"] = family


def aggregate_results(
    rows: List[Dict[str, object]],
    *,
    methods: Sequence[str] | None = None,
    bootstrap_iterations: int = 2000,
    random_seed: int = 20260817,
    _include_method_sensitivity: bool = True,
) -> Dict[str, object]:
    if methods is None:
        present = tuple(method for method in ALL_METHODS if any(method in row for row in rows))
        methods = present or DEFAULT_STANDARD_METHODS
    methods = tuple(dict.fromkeys(str(method) for method in methods))
    standard_methods = tuple(method for method in methods if method not in RESIDUE_AWARE_OPTCUTS_METHODS)
    residue_aware_methods = tuple(method for method in methods if method in RESIDUE_AWARE_OPTCUTS_METHODS)
    test_rows = [row for row in rows if str(row.get("analysis_split") or "test").strip().lower() == "test"]
    inferential_cluster_key = _inferential_cluster_key(test_rows)
    nonerror = [row for row in rows if "error" not in row]
    valid = [row for row in nonerror if _as_int((row.get("comparison_domain") or {}).get("common_patch_count", 0)) > 0]

    complete = [row for row in valid if bool((row.get("comparison_domain") or {}).get("complete", False))]
    primary = [row for row in complete if str(row.get("analysis_split") or "test").strip().lower() == "test"]
    development_complete = [
        row for row in complete if str(row.get("analysis_split") or "test").strip().lower() == "development"
    ]
    exploratory_complete = [
        row for row in complete if str(row.get("analysis_split") or "test").strip().lower() == "exploratory"
    ]
    initialization_complete = [
        row
        for row in nonerror
        if bool((row.get("initialization_comparison_domain") or {}).get("complete", False))
        and _as_int((row.get("initialization_comparison_domain") or {}).get("common_patch_count", 0)) > 0
    ]
    initialization_primary_raw = [
        row for row in initialization_complete if str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    initialization_pair_rows = [_initialization_pair_row(row) for row in initialization_primary_raw]
    initialization_primary = [row for row, exact in initialization_pair_rows if exact]
    initialization_exact_pair_count = int(sum(exact for _row, exact in initialization_pair_rows))
    residue_aware_complete = [
        row
        for row in nonerror
        if bool((row.get("residue_aware_comparison_domain") or {}).get("complete", False))
        and _as_int((row.get("residue_aware_comparison_domain") or {}).get("common_patch_count", 0)) > 0
        and _has_residue_aware_methods(row, residue_aware_methods)
    ]
    residue_aware_primary_raw = [
        row for row in residue_aware_complete if str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    residue_aware_development_complete = [
        row
        for row in residue_aware_complete
        if str(row.get("analysis_split") or "test").strip().lower() == "development"
    ]
    residue_aware_exploratory_complete = [
        row
        for row in residue_aware_complete
        if str(row.get("analysis_split") or "test").strip().lower() == "exploratory"
    ]
    residue_aware_pair_rows = [_residue_aware_pair_row(row) for row in residue_aware_primary_raw]
    residue_aware_primary = [row for row, exact in residue_aware_pair_rows if exact]
    residue_aware_exact_pair_count = int(sum(exact for _row, exact in residue_aware_pair_rows))
    topology_attempted = [
        row
        for row in rows
        if str((row.get("topology_preprocessing_pair_quality") or {}).get("status") or "")
        not in {"", "disabled", "not_evaluated"}
    ]
    topology_attempted_test = [
        row for row in topology_attempted if str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    topology_configuration_is_recorded = any("topology_ablation_configured" in row for row in rows)
    topology_configured = (
        [row for row in rows if bool(row.get("topology_ablation_configured", False))]
        if topology_configuration_is_recorded
        else (list(rows) if topology_attempted else [])
    )
    topology_configured_test = [
        row for row in topology_configured if str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    topology_complete_raw = [
        row
        for row in nonerror
        if bool((row.get("topology_preprocessing_pair_quality") or {}).get("complete", False))
        and str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    topology_pair_rows = [_topology_pair_row(row) for row in topology_complete_raw]
    topology_primary = [row for row, exact in topology_pair_rows if exact]
    method_distributions = {}
    method_execution = {}
    fragmentation_distributions = {}
    for method in methods:
        if method == "optcuts_lscm_initialized":
            analysis_rows = initialization_primary
        else:
            analysis_rows = residue_aware_primary if method in residue_aware_methods else primary
        method_distributions[method] = {
            "method_role": METHOD_ROLES.get(method, "configured_method"),
            "analysis_scope": (
                "complete exact geometry-only/TopoPPI domains with finite metrics; geometry QC is not an inclusion gate"
                if method in residue_aware_methods
                else "complete same-domain finite outputs; global injectivity is a separate validity endpoint"
            ),
            "analysis_structure_count": int(len(analysis_rows)),
            "distortion_mean": _distribution(_nested(row, (method, "distortion", "mean")) for row in analysis_rows),
            "symmetric_dirichlet_mean": _distribution(
                _nested(row, (method, "symmetric_dirichlet", "mean")) for row in analysis_rows
            ),
            "angle_distortion_mean": _distribution(
                _nested(row, (method, "angle_distortion", "mean")) for row in analysis_rows
            ),
            "area_distortion_mean": _distribution(
                _nested(row, (method, "area_distortion", "mean")) for row in analysis_rows
            ),
            "flip_rate": _distribution(_nested(row, (method, "flip_rate")) for row in analysis_rows),
            "globally_injective_patch_rate": _distribution(
                _nested(row, (method, "injectivity", "globally_injective_patch_rate")) for row in analysis_rows
            ),
            "globally_injective_structure_count": int(
                sum(
                    bool((row.get(method) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                    for row in analysis_rows
                )
            ),
        }
        method_distributions[method]["globally_injective_structure_rate"] = (
            float(method_distributions[method]["globally_injective_structure_count"] / len(analysis_rows))
            if analysis_rows
            else float("nan")
        )
        attempted_patch_count = int(
            sum(_as_int(row.get("method_execution", {}).get(method, {}).get("attempted", 0)) for row in test_rows)
        )
        success_patch_count = int(
            sum(_as_int(row.get("method_execution", {}).get(method, {}).get("success", 0)) for row in test_rows)
        )
        attempted_patch_count_all_splits = int(
            sum(_as_int(row.get("method_execution", {}).get(method, {}).get("attempted", 0)) for row in rows)
        )
        success_patch_count_all_splits = int(
            sum(_as_int(row.get("method_execution", {}).get(method, {}).get("success", 0)) for row in rows)
        )
        method_execution[method] = {
            "analysis_split": "test",
            "attempted_patch_count": attempted_patch_count,
            "successful_patch_count": success_patch_count,
            "failed_patch_count": int(max(0, attempted_patch_count - success_patch_count)),
            "all_attempted_failure_rate": float((attempted_patch_count - success_patch_count) / attempted_patch_count)
            if attempted_patch_count
            else float("nan"),
            "attempted_patch_count_all_splits": attempted_patch_count_all_splits,
            "successful_patch_count_all_splits": success_patch_count_all_splits,
            "failed_patch_count_all_splits": int(
                max(0, attempted_patch_count_all_splits - success_patch_count_all_splits)
            ),
        }
        fragmentation_distributions[method] = {
            "mean_fragmentation": _distribution(
                _nested(
                    row,
                    ("residue_footprint_fragmentation", "methods", method, "mean_fragmentation"),
                )
                for row in analysis_rows
            ),
            "area_weighted_fragmentation": _distribution(
                _nested(
                    row,
                    (
                        "residue_footprint_fragmentation",
                        "methods",
                        method,
                        "area_weighted_fragmentation",
                    ),
                )
                for row in analysis_rows
            ),
            "interaction_weighted_fragmentation": _distribution(
                _nested(
                    row,
                    (
                        "residue_footprint_fragmentation",
                        "methods",
                        method,
                        "interaction_weighted_fragmentation",
                    ),
                )
                for row in analysis_rows
            ),
            "objective_weighted_fragmentation": _distribution(
                _nested(
                    row,
                    (
                        "residue_footprint_fragmentation",
                        "methods",
                        method,
                        "objective_weighted_fragmentation",
                    ),
                )
                for row in analysis_rows
            ),
            "nonseparating_seam_crossing_edge_count": _distribution(
                _nested(
                    row,
                    (
                        "residue_footprint_fragmentation",
                        "methods",
                        method,
                        "nonlocality_audit",
                        "nonseparating_seam_crossing_edge_count",
                    ),
                )
                for row in analysis_rows
            ),
        }

    paired = {}
    paired_symmetric_dirichlet = {}
    paired_validity = {}
    paired_injective_only = {}
    if "optcuts_automatic" in standard_methods:
        baselines = tuple(method for method in PARAMETERIZATION_METHODS if method in standard_methods)
        for comparison_index, baseline in enumerate(baselines):
            comparison_name = f"{baseline}_vs_optcuts_automatic"
            attempted_pair_rows = []
            exact_pair_rows = []
            invalidity_rows = []
            injective_rows = []
            for row in test_rows:
                attempted_pair_rows.append(row)
                pairs = row.get("standard_method_pair_quality") or {}
                pair = pairs.get(comparison_name, {}) if isinstance(pairs, dict) else {}
                if not isinstance(pair, dict) or not pair:
                    validity_row = dict(row)
                    validity_row[baseline] = {"structure_unusable": 1.0}
                    validity_row["optcuts_automatic"] = {"structure_unusable": 1.0}
                    invalidity_rows.append(validity_row)
                    continue
                paired_row, exact = _standard_pair_row(row, baseline)
                if exact:
                    exact_pair_rows.append(paired_row)
                arms = pair.get("arms") or {}
                baseline_arm = arms.get(baseline) if isinstance(arms, dict) else None
                treatment_arm = arms.get("optcuts_automatic") if isinstance(arms, dict) else None
                baseline_invalid = not _arm_has_valid_geometry(baseline_arm)
                treatment_invalid = not _arm_has_valid_geometry(treatment_arm)
                validity_row = dict(row)
                validity_row[baseline] = {"structure_unusable": float(baseline_invalid)}
                validity_row["optcuts_automatic"] = {"structure_unusable": float(treatment_invalid)}
                invalidity_rows.append(validity_row)
                if exact:
                    baseline_valid = bool(
                        (paired_row.get(baseline) or {})
                        .get("injectivity", {})
                        .get("all_patches_globally_injective", False)
                    )
                    treatment_valid = bool(
                        (paired_row.get("optcuts_automatic") or {})
                        .get("injectivity", {})
                        .get("all_patches_globally_injective", False)
                    )
                    if baseline_valid and treatment_valid:
                        injective_rows.append(paired_row)
            paired[f"{baseline}_vs_optcuts_automatic"] = paired_method_comparison(
                exact_pair_rows,
                baseline=baseline,
                treatment="optcuts_automatic",
                metric_path=("distortion", "mean"),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + comparison_index,
            )
            paired_symmetric_dirichlet[f"{baseline}_vs_optcuts_automatic"] = paired_method_comparison(
                exact_pair_rows,
                baseline=baseline,
                treatment="optcuts_automatic",
                metric_path=("symmetric_dirichlet", "mean"),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + len(baselines) + comparison_index,
                relative_reference=2.0,
            )
            for comparison in (paired[comparison_name], paired_symmetric_dirichlet[comparison_name]):
                comparison.update(
                    {
                        "analysis_role": "all_finite_output_distortion_diagnostic",
                        "method_role": METHOD_ROLES.get(baseline, "configured_method"),
                        "attempted_pair_structure_count": int(len(attempted_pair_rows)),
                        "complete_exact_pair_structure_count": int(len(exact_pair_rows)),
                        "excluded_incomplete_pair_structure_count": int(
                            len(attempted_pair_rows) - len(exact_pair_rows)
                        ),
                        "validity_caveat": ("finite distortion does not imply a globally injective usable map"),
                    }
                )
            paired_validity[comparison_name] = paired_method_comparison(
                invalidity_rows,
                baseline=baseline,
                treatment="optcuts_automatic",
                metric_path=("structure_unusable",),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + 2 * len(baselines) + comparison_index,
                binary_endpoint=True,
            )
            paired_validity[comparison_name].update(
                {
                    "analysis_role": "all_attempted_unusable_output_endpoint",
                    "coding": (
                        "0=complete finite globally injective output; 1=incomplete, nonfinite, or noninjective output"
                    ),
                    "direction": "positive baseline-minus-treatment difference favors automatic OptCuts",
                    "attempted_pair_structure_count": int(len(attempted_pair_rows)),
                    "reliability_pair_structure_count": int(len(invalidity_rows)),
                    "domain_rule": (
                        "an arm is unusable if it fails to return every exact expected patch, "
                        "has nonfinite distortion, or is not globally injective"
                    ),
                }
            )
            paired_injective_only[comparison_name] = paired_method_comparison(
                injective_rows,
                baseline=baseline,
                treatment="optcuts_automatic",
                metric_path=("symmetric_dirichlet", "mean"),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + 3 * len(baselines) + comparison_index,
                relative_reference=2.0,
            )
            paired_injective_only[comparison_name].update(
                {
                    "analysis_role": "secondary_distortion_among_jointly_valid_maps",
                    "eligible_structure_count": int(len(injective_rows)),
                    "complete_same_domain_structure_count": int(len(exact_pair_rows)),
                    "conditioning_warning": (
                        "conditioning on post-method validity can select a nonrepresentative subset"
                    ),
                }
            )
        _apply_benjamini_hochberg(
            paired,
            family=f"{len(paired)} secondary log-stretch distortion comparisons",
        )
        _apply_benjamini_hochberg(
            paired_symmetric_dirichlet,
            family=f"{len(paired_symmetric_dirichlet)} symmetric-Dirichlet comparisons",
        )
        _apply_benjamini_hochberg(
            paired_validity,
            family=f"{len(paired_validity)} unusable-output comparisons",
        )
        _apply_benjamini_hochberg(
            paired_injective_only,
            family=f"{len(paired_injective_only)} jointly-injective symmetric-Dirichlet comparisons",
        )

    initialization_diagnostic: Dict[str, object] = {
        "status": (
            "no_exact_pair_quality_rows"
            if initialization_primary_raw and "optcuts_lscm_initialized" in standard_methods
            else "not_evaluated"
        ),
        "complete_domain_test_structure_count": int(len(initialization_primary_raw)),
        "complete_test_structure_count": int(len(initialization_primary)),
        "exact_pair_quality_structure_count": initialization_exact_pair_count,
        "excluded_without_exact_pair_quality_count": int(
            len(initialization_primary_raw) - initialization_exact_pair_count
        ),
        "comparisons": {},
    }
    if initialization_primary and "optcuts_lscm_initialized" in standard_methods:
        initialization_comparisons = {
            "lscm_vs_optcuts_lscm_initialized": paired_method_comparison(
                initialization_primary,
                baseline="lscm",
                treatment="optcuts_lscm_initialized",
                metric_path=("symmetric_dirichlet", "mean"),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + 2 * len(paired),
                relative_reference=2.0,
            ),
            "optcuts_automatic_vs_optcuts_lscm_initialized": paired_method_comparison(
                initialization_primary,
                baseline="optcuts_automatic",
                treatment="optcuts_lscm_initialized",
                metric_path=("symmetric_dirichlet", "mean"),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + 2 * len(paired) + 1,
                relative_reference=2.0,
            ),
        }
        initialization_diagnostic = {
            "status": "evaluated",
            "scope": "structures whose every free-boundary LSCM patch is globally injective",
            "complete_domain_test_structure_count": int(len(initialization_primary_raw)),
            "complete_test_structure_count": int(len(initialization_primary)),
            "exact_pair_quality_structure_count": initialization_exact_pair_count,
            "excluded_without_exact_pair_quality_count": int(
                len(initialization_primary_raw) - initialization_exact_pair_count
            ),
            "comparisons": initialization_comparisons,
        }

    residue_aware_summary: Dict[str, object] = {
        "status": "no_exact_pair_quality_rows"
        if residue_aware_primary_raw and residue_aware_methods
        else "not_evaluated",
        "complete_domain_test_structure_count": int(len(residue_aware_primary_raw)),
        "complete_test_structure_count": int(len(residue_aware_primary)),
        "exact_pair_quality_structure_count": residue_aware_exact_pair_count,
        "excluded_without_exact_pair_quality_count": int(
            len(residue_aware_primary_raw) - residue_aware_exact_pair_count
        ),
        "comparisons": {},
        "paired_reliability_comparisons": {},
    }
    residue_aware_reliability_comparisons: Dict[str, Dict[str, object]] = {}
    residue_aware_attempted_pair_counts: Dict[str, int] = {}
    for comparison_index, treatment in enumerate(residue_aware_methods):
        baseline = RESIDUE_AWARE_BASELINE[treatment]
        invalidity_rows = []
        attempted_pair_count = 0
        for row in test_rows:
            attempted_pair_count += 1
            pair = row.get("residue_aware_pair_quality") or {}
            arms = pair.get("arms", {}) if isinstance(pair, dict) else {}
            if not isinstance(arms, dict) or not all(method in arms for method in (baseline, treatment)):
                validity_row = dict(row)
                validity_row[baseline] = {"structure_unusable": 1.0}
                validity_row[treatment] = {"structure_unusable": 1.0}
                invalidity_rows.append(validity_row)
                continue
            baseline_arm = arms[baseline]
            treatment_arm = arms[treatment]
            validity_row = dict(row)
            validity_row[baseline] = {"structure_unusable": float(not _arm_has_valid_geometry(baseline_arm))}
            validity_row[treatment] = {"structure_unusable": float(not _arm_has_valid_geometry(treatment_arm))}
            invalidity_rows.append(validity_row)
        key = "automatic_unusable_output"
        comparison = paired_method_comparison(
            invalidity_rows,
            baseline=baseline,
            treatment=treatment,
            metric_path=("structure_unusable",),
            cluster_key=inferential_cluster_key,
            bootstrap_iterations=int(bootstrap_iterations),
            seed=int(random_seed) + 80 + comparison_index,
            binary_endpoint=True,
        )
        comparison.update(
            {
                "analysis_role": "prespecified_paired_reliability_endpoint",
                "coding": (
                    "0=complete finite globally injective output; 1=incomplete, nonfinite, or noninjective output"
                ),
                "direction": "positive baseline-minus-treatment difference favors TopoPPI",
                "attempted_pair_structure_count": int(attempted_pair_count),
                "reliability_pair_structure_count": int(len(invalidity_rows)),
                "domain_rule": (
                    "each arm must return every expected patch on exact source-face geometry, "
                    "with finite distortion and global injectivity"
                ),
            }
        )
        residue_aware_reliability_comparisons[key] = comparison
        residue_aware_attempted_pair_counts[treatment] = attempted_pair_count
    if residue_aware_primary and residue_aware_methods:
        residue_aware_paired = {}
        comparison_seed = int(random_seed) + len(paired)
        for treatment in residue_aware_methods:
            baseline = RESIDUE_AWARE_BASELINE[treatment]
            initialization = "automatic"
            for metric_name, metric_path in (
                ("distortion_mean", ("distortion", "mean")),
                ("symmetric_dirichlet_mean", ("symmetric_dirichlet", "mean")),
                (
                    "objective_weighted_fragmentation",
                    (
                        "residue_footprint_fragmentation",
                        "objective_weighted_fragmentation",
                    ),
                ),
                (
                    "normalized_seam_length",
                    ("seam", "seam_length_3d_normalized"),
                ),
            ):
                key = f"{initialization}_{metric_name}"
                residue_aware_paired[key] = paired_method_comparison(
                    residue_aware_primary,
                    baseline=baseline,
                    treatment=treatment,
                    metric_path=metric_path,
                    cluster_key=inferential_cluster_key,
                    bootstrap_iterations=int(bootstrap_iterations),
                    seed=comparison_seed,
                    relative_reference=(2.0 if metric_name == "symmetric_dirichlet_mean" else 0.0),
                )
                comparison_seed += 1
        primary_key = "automatic_objective_weighted_fragmentation"
        supporting_keys = [
            "automatic_distortion_mean",
            "automatic_symmetric_dirichlet_mean",
            "automatic_normalized_seam_length",
        ]
        for key, comparison in residue_aware_paired.items():
            if key == primary_key:
                comparison["analysis_role"] = "prespecified_primary_efficacy"
            else:
                comparison["analysis_role"] = "supporting_tradeoff_or_constraint"
        _apply_benjamini_hochberg(
            {key: residue_aware_paired[key] for key in supporting_keys},
            family=f"{len(supporting_keys)} supporting automatic-initialization tradeoff comparisons",
        )
        residue_aware_summary = {
            "status": "evaluated",
            "complete_domain_test_structure_count": int(len(residue_aware_primary_raw)),
            "complete_test_structure_count": int(len(residue_aware_primary)),
            "exact_pair_quality_structure_count": residue_aware_exact_pair_count,
            "excluded_without_exact_pair_quality_count": int(
                len(residue_aware_primary_raw) - residue_aware_exact_pair_count
            ),
            "development_rows_excluded": True,
            "primary_comparison": primary_key,
            "primary_multiplicity_rule": "single prespecified primary endpoint; no multiplicity adjustment",
            "supporting_comparisons": supporting_keys,
            "sensitivity_comparisons": [],
            "comparisons": residue_aware_paired,
            "paired_reliability_comparisons": residue_aware_reliability_comparisons,
            "attempted_pair_structure_count_by_treatment": residue_aware_attempted_pair_counts,
            "complete_case_caveat": (
                "efficacy endpoints use complete finite exact pairs without conditioning on flips or overlap; "
                "the all-attempted geometry/reliability endpoint must be interpreted jointly"
            ),
        }
    elif residue_aware_reliability_comparisons:
        residue_aware_summary["paired_reliability_comparisons"] = residue_aware_reliability_comparisons
        residue_aware_summary["attempted_pair_structure_count_by_treatment"] = residue_aware_attempted_pair_counts

    topology_preprocessing_summary: Dict[str, object] = {
        "status": "not_evaluated",
        "configured_test_structure_count": int(len(topology_configured_test)),
        "ablation_reached_test_structure_count": int(len(topology_attempted_test)),
        "attempted_structure_count": int(len(topology_configured_test)),
        "complete_test_structure_count": 0,
        "comparisons": {},
    }
    if topology_configured or topology_attempted:
        baseline = "optcuts_without_topology_preparation"
        treatment = "optcuts_with_topology_preparation"
        comparisons = {}
        for comparison_index, (metric_name, metric_path) in enumerate(
            (
                ("distortion_mean", ("distortion", "mean")),
                ("symmetric_dirichlet_mean", ("symmetric_dirichlet", "mean")),
                ("normalized_seam_length", ("seam", "seam_length_3d_normalized")),
                ("runtime_wall_sec", ("runtime", "wall_sec")),
            )
        ):
            comparisons[metric_name] = paired_method_comparison(
                topology_primary,
                baseline=baseline,
                treatment=treatment,
                metric_path=metric_path,
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(random_seed) + 100 + comparison_index,
                relative_reference=(2.0 if metric_name == "symmetric_dirichlet_mean" else 0.0),
            )
            comparisons[metric_name]["analysis_role"] = "prespecified_topology_ablation_endpoint"
        _apply_benjamini_hochberg(
            comparisons,
            family=f"{len(comparisons)} topology-preparation ablation endpoints",
        )
        without_valid = int(
            sum(
                bool((row.get(baseline) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                for row in topology_primary
            )
        )
        with_valid = int(
            sum(
                bool((row.get(treatment) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                for row in topology_primary
            )
        )
        gained_validity = int(
            sum(
                not bool((row.get(baseline) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                and bool((row.get(treatment) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                for row in topology_primary
            )
        )
        lost_validity = int(
            sum(
                bool((row.get(baseline) or {}).get("injectivity", {}).get("all_patches_globally_injective", False))
                and not bool(
                    (row.get(treatment) or {}).get("injectivity", {}).get("all_patches_globally_injective", False)
                )
                for row in topology_primary
            )
        )
        arm_completion = [_topology_arm_completion(row) for row in topology_configured_test]
        reached_count = int(sum(reached for reached, _without, _with in arm_completion))
        without_complete_count = int(sum(without for _reached, without, _with in arm_completion))
        with_complete_count = int(sum(with_ for _reached, _without, with_ in arm_completion))
        both_complete_count = int(sum(without and with_ for _reached, without, with_ in arm_completion))
        gained_complete_count = int(sum((not without) and with_ for _reached, without, with_ in arm_completion))
        lost_complete_count = int(sum(without and (not with_) for _reached, without, with_ in arm_completion))
        neither_complete_count = int(
            len(arm_completion) - both_complete_count - gained_complete_count - lost_complete_count
        )
        topology_preprocessing_summary = {
            "status": "evaluated" if topology_primary else "no_complete_test_pairs",
            "configured_test_structure_count": int(len(topology_configured_test)),
            "ablation_reached_test_structure_count": reached_count,
            "not_reached_test_structure_count": int(len(topology_configured_test) - reached_count),
            "attempted_structure_count": int(len(topology_configured_test)),
            "attempted_all_splits_structure_count": int(len(topology_attempted)),
            "complete_test_structure_count": int(len(topology_primary)),
            "ineligible_or_incomplete_structure_count": int(len(topology_configured_test) - len(topology_complete_raw)),
            "exact_pair_quality_structure_count": int(sum(exact for _row, exact in topology_pair_rows)),
            "eligibility_rule": ("all extracted patches must succeed in both arms on identical source-face geometry"),
            "all_configured_structure_completion": {
                "without_topology_preparation_count": without_complete_count,
                "with_topology_preparation_count": with_complete_count,
                "both_complete_count": both_complete_count,
                "gained_after_preparation_count": gained_complete_count,
                "lost_after_preparation_count": lost_complete_count,
                "neither_complete_count": neither_complete_count,
                "rule": (
                    "an arm is complete only when it emits one uniquely identified output for every "
                    "extracted patch; failures before the ablation are retained as neither complete"
                ),
            },
            "global_injectivity": {
                "without_topology_preparation_count": without_valid,
                "with_topology_preparation_count": with_valid,
                "gained_after_preparation_count": gained_validity,
                "lost_after_preparation_count": lost_validity,
            },
            "comparisons": comparisons,
        }

    multi_patch = [
        row for row in primary if _as_int((row.get("comparison_domain") or {}).get("common_patch_count", 0)) > 1
    ]
    atlas_multi = {
        "structure_count": int(len(multi_patch)),
        "reference_methods": sorted({_atlas_reference_method(row) for row in multi_patch} - {""}),
        "utilization": _distribution(_atlas_metric(row, "utilization") for row in multi_patch),
        "overlap_ratio": _distribution(_atlas_metric(row, "overlap_ratio") for row in multi_patch),
        "overdraw_ratio": _distribution(_atlas_metric(row, "overdraw_ratio") for row in multi_patch),
        "min_chart_gap": _distribution(_atlas_metric(row, "min_chart_gap") for row in multi_patch),
        "padding_violations": _distribution(_atlas_metric(row, "padding_violations") for row in multi_patch),
        "scope": "multi_patch_structures_only",
    }

    retention_names = (
        "face_retention_ratio",
        "source_vertex_retention_ratio",
        "area_retention_ratio",
        "source_atom_retention_ratio",
        "topology_face_retention_ratio",
        "topology_source_vertex_retention_ratio",
        "topology_area_retention_ratio",
        "topology_source_atom_retention_ratio",
        "parameterization_face_retention_ratio",
        "parameterization_source_vertex_retention_ratio",
        "parameterization_area_retention_ratio",
        "parameterization_source_atom_retention_ratio",
        "residue_retention_ratio",
        "topology_residue_retention_ratio",
        "parameterization_residue_retention_ratio",
        "geometric_contact_pair_retention_ratio",
        "topology_geometric_contact_pair_retention_ratio",
        "parameterization_geometric_contact_pair_retention_ratio",
        "declared_hotspot_retention_ratio",
        "topology_declared_hotspot_retention_ratio",
        "parameterization_declared_hotspot_retention_ratio",
        "declared_interaction_retention_ratio",
        "topology_declared_interaction_retention_ratio",
        "parameterization_declared_interaction_retention_ratio",
        "confidence_atom_retention_ratio",
        "topology_confidence_atom_retention_ratio",
        "parameterization_confidence_atom_retention_ratio",
    )
    all_patch_records = [record for row in rows for record in row.get("patch_records", [])]
    complete_patch_records = [record for row in primary for record in row.get("patch_records", [])]
    retention = {
        name: _distribution(_as_float(record.get(name)) for record in all_patch_records) for name in retention_names
    }
    complete_retention = {
        name: _distribution(_as_float(record.get(name)) for record in complete_patch_records)
        for name in retention_names
    }
    mesh_cardinality_names = (
        "materialized_vertex_count_ratio",
        "topology_materialized_vertex_count_ratio",
        "parameterization_materialized_vertex_count_ratio",
    )
    mesh_cardinality_change = {
        name: _distribution(_as_float(record.get(name)) for record in all_patch_records)
        for name in mesh_cardinality_names
    }
    complete_mesh_cardinality_change = {
        name: _distribution(_as_float(record.get(name)) for record in complete_patch_records)
        for name in mesh_cardinality_names
    }

    count_fields = {
        "face": ("face_count_before", "face_count_after_topology_sanitation", "face_count_after"),
        "source_vertex": (
            "source_vertex_count_before",
            "source_vertex_count_after_topology_sanitation",
            "source_vertex_count_after",
        ),
        "area": ("area_before", "area_after_topology_sanitation", "area_after"),
        "source_atom": (
            "source_atom_count_before",
            "source_atom_count_after_topology_sanitation",
            "source_atom_count_after",
        ),
        "residue": (
            "residue_count_before",
            "residue_count_after_topology_sanitation",
            "residue_count_after",
        ),
        "geometric_contact_pair": (
            "geometric_contact_pair_count_before",
            "geometric_contact_pair_count_after_topology_sanitation",
            "geometric_contact_pair_count_after",
        ),
        "declared_hotspot": (
            "declared_hotspot_count_on_patch_before",
            "declared_hotspot_count_after_topology_sanitation",
            "declared_hotspot_count_after",
        ),
        "declared_interaction": (
            "declared_interaction_count_on_patch_before",
            "declared_interaction_count_after_topology_sanitation",
            "declared_interaction_count_after",
        ),
        "confidence_atom": (
            "confidence_atom_count_before",
            "confidence_atom_count_after_topology_sanitation",
            "confidence_atom_count_after",
        ),
    }

    def _pooled_ratio(records, before_name: str, after_name: str) -> Dict[str, float]:
        pairs = [(_as_float(record.get(before_name)), _as_float(record.get(after_name))) for record in records]
        valid_pairs = [
            (before, after)
            for before, after in pairs
            if np.isfinite(before) and np.isfinite(after) and before > 0.0 and after >= 0.0
        ]
        before_total = float(sum(before for before, _after in valid_pairs))
        after_total = float(sum(after for _before, after in valid_pairs))
        return {
            "denominator_total": before_total,
            "retained_total": after_total,
            "retention_ratio": float(after_total / before_total) if before_total > 0.0 else float("nan"),
        }

    def _pooled_retention(records) -> Dict[str, object]:
        pooled = {}
        for name, (before, after_topology, after) in count_fields.items():
            # A component with no valid original incidence cannot acquire that
            # entity later. Exclude these rows from every stage so each ratio
            # uses the same valid population.
            eligible_records = [
                record
                for record in records
                if np.isfinite(_as_float(record.get(before))) and _as_float(record.get(before)) > 0.0
            ]
            pooled[name] = {
                "overall": _pooled_ratio(eligible_records, before, after),
                "topology_sanitation": _pooled_ratio(eligible_records, before, after_topology),
                "parameterization": _pooled_ratio(eligible_records, after_topology, after),
            }
        return pooled

    pooled_retention = _pooled_retention(all_patch_records)
    complete_pooled_retention = _pooled_retention(complete_patch_records)
    benchmark_purposes = sorted({str(row.get("benchmark_purpose") or "performance") for row in rows})
    performance_attempt_rows = [
        row
        for row in rows
        if str(row.get("benchmark_purpose") or "performance") == "performance"
        and str(row.get("analysis_split") or "test").strip().lower() == "test"
    ]
    performance_rows = [
        row
        for row in performance_attempt_rows
        if str(row.get("status") or "").strip().lower() == "ok" and not row.get("error")
    ]
    operational_attempt_rows = [
        row
        for row in performance_attempt_rows
        if str(row.get("execution_profile") or "").strip().lower() == "operational_optcuts"
    ]
    operational_usable_rows = [
        row
        for row in operational_attempt_rows
        if bool((row.get("execution_certificate") or {}).get("scientifically_usable", False))
    ]
    wall_values = [_nested(row, ("timing", "isolated_repetitions", "wall_sec_median")) for row in performance_rows]
    memory_values = [_nested(row, ("memory", "peak_rss_mb")) for row in performance_rows]
    performance_run_measurements = [
        measurement
        for row in performance_attempt_rows
        for measurement in row.get("worker_measurements", [])
        if isinstance(measurement, dict) and not bool(measurement.get("warmup", False))
    ]
    observed_run_runtime_values = [_runtime_observation(measurement) for measurement in performance_run_measurements]
    supervisor_wall_values = [_as_float(measurement.get("wall_sec")) for measurement in performance_run_measurements]
    observed_run_memory_values = [
        _as_float(measurement.get("peak_rss_mb")) for measurement in performance_run_measurements
    ]
    right_censored_measurements = [
        measurement for measurement in performance_run_measurements if bool(measurement.get("right_censored", False))
    ]
    termination_reason_counts: Dict[str, int] = {}
    for measurement in performance_run_measurements:
        reason = str(measurement.get("termination_reason") or "completed")
        termination_reason_counts[reason] = termination_reason_counts.get(reason, 0) + 1
    cycle_ranks = [
        _nested(
            row,
            (
                "residue_footprint_fragmentation",
                "methods",
                "lscm",
                "nonlocality_audit",
                "cycle_rank",
            ),
        )
        for row in primary
    ]
    cyclic_residue_ratios = [
        _nested(
            row,
            (
                "residue_footprint_fragmentation",
                "methods",
                "lscm",
                "nonlocality_audit",
                "cyclic_residue_ratio",
            ),
        )
        for row in primary
    ]
    nonlocal_structure_count = int(sum(np.isfinite(value) and value > 0.0 for value in cycle_ranks))

    def _strata(key: str) -> Dict[str, object]:
        labels = sorted({str(row.get(key) or "not_declared") for row in test_rows})
        blocks = {}
        for label in labels:
            attempted = [row for row in test_rows if str(row.get(key) or "not_declared") == label]
            attempted_all_splits = [row for row in rows if str(row.get(key) or "not_declared") == label]
            completed = [row for row in primary if str(row.get(key) or "not_declared") == label]
            reference_method = next(
                (
                    method
                    for method in ("optcuts_automatic", "harmonic", "lscm")
                    if any(isinstance(row.get(method), dict) for row in completed)
                ),
                "",
            )
            blocks[label] = {
                "attempted_structure_count": int(len(attempted)),
                "attempted_all_splits_structure_count": int(len(attempted_all_splits)),
                "complete_comparison_structure_count": int(len(completed)),
                "failed_structure_count": int(sum(bool(row.get("error")) for row in attempted)),
                "reference_method": reference_method or None,
                "reference_method_distortion_mean": _distribution(
                    _nested(row, (reference_method, "distortion", "mean")) for row in completed
                ),
                "reference_method_area_distortion_mean": _distribution(
                    _nested(row, (reference_method, "area_distortion", "mean")) for row in completed
                ),
            }
        return blocks

    summary = {
        "valid_structure_count": int(len(valid)),
        "attempted_structure_count": int(len(rows)),
        "complete_comparison_structure_count": int(len(complete)),
        "complete_test_structure_count": int(len(primary)),
        "complete_development_structure_count": int(len(development_complete)),
        "complete_exploratory_structure_count": int(len(exploratory_complete)),
        "complete_residue_aware_test_structure_count": int(len(residue_aware_primary)),
        "complete_residue_aware_development_structure_count": int(len(residue_aware_development_complete)),
        "complete_residue_aware_exploratory_structure_count": int(len(residue_aware_exploratory_complete)),
        "complete_lscm_initialization_test_structure_count": int(len(initialization_primary)),
        "incomplete_comparison_structure_count": int(len(valid) - len(complete)),
        "failed_structure_count": int(sum(bool(row.get("error")) for row in rows)),
        "excluded_without_common_domain_count": int(
            sum(
                not row.get("error")
                and str(row.get("execution_profile") or "comparative").strip().lower() != "operational_optcuts"
                and _as_int((row.get("comparison_domain") or {}).get("common_patch_count", 0)) == 0
                for row in rows
            )
        ),
        "primary_analysis_rule": (
            "standard-method diagnostics use complete same-domain test structures; the prespecified "
            "TopoPPI efficacy endpoint additionally requires an exact, finite geometry-only/TopoPPI "
            "pair but does not condition on local flips or global overlap"
        ),
        "inferential_cluster_key": inferential_cluster_key,
        "inferential_cluster_rule": (
            "experimental-only cohorts use interaction family_id; predicted-only cohorts use "
            "prediction-dependency inference_family_id; mixed cohorts use the frozen "
            "analysis_split_component_id dependency component"
        ),
        "configured_methods": list(methods),
        "benchmark_purposes": benchmark_purposes,
        "performance_timing_structure_count": int(len(performance_rows)),
        "performance_attempted_structure_count": int(len(performance_attempt_rows)),
        "operational_attempted_structure_count": int(len(operational_attempt_rows)),
        "operational_scientifically_usable_structure_count": int(len(operational_usable_rows)),
        "operational_scientifically_unusable_structure_count": int(
            len(operational_attempt_rows) - len(operational_usable_rows)
        ),
        "operational_scientifically_usable_structure_rate": (
            float(len(operational_usable_rows) / len(operational_attempt_rows))
            if operational_attempt_rows
            else float("nan")
        ),
        "performance_observed_run_count": int(len(performance_run_measurements)),
        "performance_right_censored_run_count": int(len(right_censored_measurements)),
        "performance_termination_reason_counts": termination_reason_counts,
        "performance_timing_rule": (
            "successful test-split performance-purpose rows only; operational_optcuts excludes "
            "comparison metrics and ablations"
        ),
        "performance_all_observed_run_rule": (
            "all non-warmup supervisor observations, including method-budget, worker-timeout, "
            "or memory-limit censoring; "
            "for a right-censored run, runtime_observation_sec is elapsed time at the censoring "
            "event and excludes subsequent process-termination overhead"
        ),
        "performance_execution_profiles": sorted(
            {str(row.get("execution_profile") or "comparative") for row in performance_rows}
        ),
        "performance_operational_methods": sorted(
            {
                str(row.get("operational_method"))
                for row in performance_rows
                if str(row.get("operational_method") or "").strip()
            }
        ),
        "method_distributions": method_distributions,
        "method_execution_all_attempted": method_execution,
        "residue_footprint_fragmentation": {
            "method_distributions": fragmentation_distributions,
            "domain_nonlocality": {
                "eligibility": (
                    "nonlocal_structure_present" if nonlocal_structure_count else "not_observed_in_complete_domains"
                ),
                "complete_structure_count": int(len(primary)),
                "structure_with_nonlocal_footprints_count": nonlocal_structure_count,
                "structure_with_nonlocal_footprints_ratio": float(nonlocal_structure_count / len(primary))
                if primary
                else float("nan"),
                "cycle_rank": _distribution(cycle_ranks),
                "cyclic_residue_ratio": _distribution(cyclic_residue_ratios),
            },
        },
        "paired_cluster_aware_comparisons": paired,
        "paired_cluster_aware_symmetric_dirichlet_comparisons": paired_symmetric_dirichlet,
        "paired_unusable_output_comparisons": paired_validity,
        "paired_jointly_injective_symmetric_dirichlet_comparisons": paired_injective_only,
        "lscm_initialization_diagnostic": initialization_diagnostic,
        "residue_aware_optcuts_comparisons": residue_aware_summary,
        "topology_preprocessing_ablation": topology_preprocessing_summary,
        "multi_patch_atlas": atlas_multi,
        "topology_biological_retention": retention,
        "topology_biological_retention_scope": "all_attempted_interface_components",
        "topology_biological_retention_incidence_definition": (
            "a source vertex, nearest-atom provenance label, residue, or contact is retained when "
            "at least one surviving mesh vertex carries its source identity"
        ),
        "topology_biological_retention_pooled_component_incidence": pooled_retention,
        "topology_biological_retention_complete_comparisons": complete_retention,
        "topology_biological_retention_complete_comparisons_pooled_component_incidence": (complete_pooled_retention),
        "topology_mesh_cardinality_change": mesh_cardinality_change,
        "topology_mesh_cardinality_change_complete_comparisons": complete_mesh_cardinality_change,
        "isolated_end_to_end_wall_sec": _distribution(wall_values),
        "isolated_peak_rss_mb": _distribution(memory_values),
        "isolated_runtime_observation_sec_including_censored_lower_bounds": _distribution(observed_run_runtime_values),
        "isolated_supervisor_wall_sec_all_observed_runs": _distribution(supervisor_wall_values),
        "isolated_observed_run_peak_rss_mb": _distribution(observed_run_memory_values),
        "structure_type_strata": _strata("structure_type"),
        "experimental_method_group_strata": _strata("experimental_method_group"),
        "confidence_strata": _strata("confidence_stratum"),
        "afdb_ipsae_strata": _strata("afdb_ipsae_stratum"),
        "paired_geometry_strata": _strata("paired_geometry_stratum"),
    }
    if _include_method_sensitivity:
        nmr_flags = [_metadata_boolean(row.get("experimental_method_contains_nmr")) for row in rows]
        if rows and all(flag is not None for flag in nmr_flags):
            non_nmr_rows = [row for row, contains_nmr in zip(rows, nmr_flags, strict=True) if not contains_nmr]
            summary["experimental_method_sensitivity"] = {
                "status": "evaluated",
                "analysis_role": "prespecified_sensitivity_excluding_any_NMR_entry",
                "primary_analysis_retains_nmr": True,
                "source_field": "experimental_method_contains_nmr",
                "all_splits_input_structure_count": int(len(rows)),
                "all_splits_excluded_nmr_structure_count": int(len(rows) - len(non_nmr_rows)),
                "filtered_summary": aggregate_results(
                    non_nmr_rows,
                    methods=methods,
                    bootstrap_iterations=int(bootstrap_iterations),
                    random_seed=int(random_seed) + 10_000,
                    _include_method_sensitivity=False,
                ),
            }
        else:
            summary["experimental_method_sensitivity"] = {
                "status": "not_evaluated_missing_frozen_experiment_metadata",
                "analysis_role": "prespecified_sensitivity_excluding_any_NMR_entry",
            }
    else:
        summary["experimental_method_sensitivity"] = {"status": "not_repeated_inside_filtered_summary"}
    return summary


def write_csv(rows: List[Dict[str, object]], output_root: str, filename: str = "benchmark_summary.csv") -> None:
    path = os.path.join(output_root, filename)
    header = [
        "pdb",
        "manifest_record_id",
        "input_sha256",
        "status",
        "chain_a",
        "chain_b",
        "chain_selection_mode",
        "cluster_id",
        "family_id",
        "sequence_cluster_a",
        "sequence_cluster_b",
        "inference_sequence_cluster_a",
        "inference_sequence_cluster_b",
        "inference_family_id",
        "inference_dependency_basis",
        "analysis_split",
        "analysis_split_component_id",
        "analysis_split_basis",
        "chain_a_residue_count",
        "chain_b_residue_count",
        "candidate_chain_pair_count",
        "selected_atom_contact_fraction",
        "selected_residue_contact_fraction",
        "benchmark_purpose",
        "structure_type",
        "structure_method",
        "resolution_angstrom",
        "experimental_methods_json",
        "experimental_method_group",
        "experimental_method_contains_nmr",
        "pdbbind_index_resolution_angstrom",
        "rcsb_resolution_combined_angstrom_json",
        "rcsb_experiment_metadata_source",
        "confidence_metric",
        "confidence_stratum",
        "afdb_ipsae_stratum",
        "afdb_model_id",
        "afdb_iptm",
        "afdb_ipsae",
        "afdb_pdockq",
        "afdb_pdockq2",
        "afdb_lis",
        "paired_record_id",
        "paired_experimental_record_id",
        "paired_geometry_stratum",
        "paired_contact_cutoff_angstrom",
        "paired_predicted_contact_count_total",
        "paired_contact_recall_fnat",
        "paired_contact_precision",
        "paired_contact_jaccard",
        "paired_experimental_contact_mapping_coverage",
        "paired_interface_residue_a_mapping_coverage",
        "paired_interface_residue_b_mapping_coverage",
        "paired_interface_ligand_ca_mapping_coverage",
        "paired_interface_ligand_ca_rmsd_angstrom",
        "paired_cross_chain_clash_atom_fraction",
        "paired_alignment_a_optimal_correspondence_count",
        "paired_alignment_b_optimal_correspondence_count",
        "paired_alignment_a_selected_pair_consensus_fraction",
        "paired_alignment_b_selected_pair_consensus_fraction",
        "raw_patch_count",
        "prepared_patch_count",
        "common_patch_count",
        "comparison_complete",
        "comparison_signature",
        "initialization_common_patch_count",
        "initialization_comparison_complete",
        "initialization_comparison_signature",
        "residue_aware_common_patch_count",
        "residue_aware_comparison_complete",
        "residue_aware_comparison_signature",
        "residue_aware_pair_status",
        "residue_aware_pair_complete",
        "residue_aware_pair_expected_patch_count",
        "residue_aware_pair_common_patch_count",
        "residue_aware_pair_domain_signature",
    ]
    for method in ALL_METHODS:
        header.extend(
            [
                f"{method}_metric_domain",
                f"{method}_metric_domain_complete",
                f"{method}_metric_domain_signature",
                f"{method}_distortion_mean",
                f"{method}_symmetric_dirichlet_mean",
                f"{method}_angle_mean_rad",
                f"{method}_area_mean",
                f"{method}_flip_rate",
                f"{method}_globally_injective_patch_rate",
                f"{method}_all_patches_globally_injective",
                f"{method}_atlas_overdraw_ratio",
                f"{method}_seam_edge_count",
                f"{method}_seam_length_3d_normalized",
                f"{method}_residue_fragmentation_mean",
                f"{method}_residue_fragmentation_area_weighted",
                f"{method}_residue_fragmentation_interaction_weighted",
                f"{method}_residue_fragmentation_objective_weighted",
                f"{method}_residue_footprint_cycle_rank",
                f"{method}_nonseparating_seam_crossing_edge_count",
            ]
        )
    for method in RESIDUE_AWARE_PAIR_METHODS:
        header.extend(
            [
                f"residue_aware_pair_{method}_domain_complete",
                f"residue_aware_pair_{method}_metric_finite",
                f"residue_aware_pair_{method}_globally_injective",
                f"residue_aware_pair_{method}_usable",
                f"residue_aware_pair_{method}_distortion_mean",
                f"residue_aware_pair_{method}_symmetric_dirichlet_mean",
                f"residue_aware_pair_{method}_angle_mean_rad",
                f"residue_aware_pair_{method}_area_mean",
                f"residue_aware_pair_{method}_flip_rate",
                f"residue_aware_pair_{method}_seam_length_3d_normalized",
                f"residue_aware_pair_{method}_residue_fragmentation_objective_weighted",
            ]
        )
    header.extend(
        [
            "atlas_reference_method",
            "atlas_utilization_polygonal",
            "atlas_overlap_ratio_polygonal",
            "atlas_overdraw_ratio",
            "atlas_min_chart_gap_polygonal",
            "atlas_padding_violations_polygonal",
            "end_to_end_wall_sec_median",
            "end_to_end_cpu_sec_median",
            "peak_rss_mb",
            "worker_runtime_observation_count",
            "worker_runtime_observation_sec_last",
            "worker_supervisor_wall_sec_last",
            "worker_right_censored",
            "worker_termination_reason",
            "worker_censoring_threshold_sec",
            "worker_censoring_event_elapsed_sec",
            "error",
        ]
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for result in rows:
            selection = result.get("chain_selection", {})
            domain = result.get("comparison_domain", {})
            initialization_domain = result.get("initialization_comparison_domain", {})
            residue_aware_domain = result.get("residue_aware_comparison_domain", {})
            residue_aware_pair = result.get("residue_aware_pair_quality", {})
            if not isinstance(residue_aware_pair, dict):
                residue_aware_pair = {}
            runtime_observations = [
                measurement
                for measurement in result.get("worker_measurements", [])
                if isinstance(measurement, dict) and not bool(measurement.get("warmup", False))
            ]
            last_runtime_observation = runtime_observations[-1] if runtime_observations else {}
            flat = {
                "pdb": result.get("pdb"),
                "manifest_record_id": result.get("manifest_record_id") or "",
                "input_sha256": result.get("input_sha256", ""),
                "status": result.get("status", "failed" if result.get("error") else "unknown"),
                "chain_a": selection.get("chain_a", ""),
                "chain_b": selection.get("chain_b", ""),
                "chain_selection_mode": selection.get("mode", ""),
                "cluster_id": result.get("cluster_id") or "",
                "family_id": result.get("family_id") or "",
                "sequence_cluster_a": result.get("sequence_cluster_a") or "",
                "sequence_cluster_b": result.get("sequence_cluster_b") or "",
                "inference_sequence_cluster_a": result.get("inference_sequence_cluster_a") or "",
                "inference_sequence_cluster_b": result.get("inference_sequence_cluster_b") or "",
                "inference_family_id": result.get("inference_family_id") or "",
                "inference_dependency_basis": result.get("inference_dependency_basis") or "",
                "analysis_split": result.get("analysis_split") or "",
                "analysis_split_component_id": result.get("analysis_split_component_id") or "",
                "analysis_split_basis": result.get("analysis_split_basis") or "",
                "chain_a_residue_count": result.get("chain_a_residue_count")
                if result.get("chain_a_residue_count") is not None
                else "",
                "chain_b_residue_count": result.get("chain_b_residue_count")
                if result.get("chain_b_residue_count") is not None
                else "",
                "candidate_chain_pair_count": result.get("candidate_chain_pair_count")
                if result.get("candidate_chain_pair_count") is not None
                else "",
                "selected_atom_contact_fraction": result.get("selected_atom_contact_fraction")
                if result.get("selected_atom_contact_fraction") is not None
                else "",
                "selected_residue_contact_fraction": result.get("selected_residue_contact_fraction")
                if result.get("selected_residue_contact_fraction") is not None
                else "",
                "benchmark_purpose": result.get("benchmark_purpose") or "",
                "structure_type": result.get("structure_type") or "",
                "structure_method": result.get("structure_method") or "",
                "resolution_angstrom": result.get("resolution_angstrom") or "",
                "experimental_methods_json": result.get("experimental_methods_json") or "",
                "experimental_method_group": result.get("experimental_method_group") or "",
                "experimental_method_contains_nmr": (
                    result.get("experimental_method_contains_nmr")
                    if result.get("experimental_method_contains_nmr") is not None
                    else ""
                ),
                "pdbbind_index_resolution_angstrom": (result.get("pdbbind_index_resolution_angstrom") or ""),
                "rcsb_resolution_combined_angstrom_json": (result.get("rcsb_resolution_combined_angstrom_json") or ""),
                "rcsb_experiment_metadata_source": (result.get("rcsb_experiment_metadata_source") or ""),
                "confidence_metric": result.get("confidence_metric") or "",
                "confidence_stratum": result.get("confidence_stratum") or "",
                "afdb_ipsae_stratum": result.get("afdb_ipsae_stratum") or "",
                "paired_record_id": result.get("paired_record_id") or "",
                "paired_experimental_record_id": result.get("paired_experimental_record_id") or "",
                "paired_geometry_stratum": result.get("paired_geometry_stratum") or "",
                "paired_contact_cutoff_angstrom": (
                    result.get("paired_contact_cutoff_angstrom")
                    if result.get("paired_contact_cutoff_angstrom") is not None
                    else ""
                ),
                "paired_predicted_contact_count_total": (
                    result.get("paired_predicted_contact_count_total")
                    if result.get("paired_predicted_contact_count_total") is not None
                    else ""
                ),
                "paired_alignment_a_optimal_correspondence_count": result.get(
                    "paired_alignment_a_optimal_correspondence_count"
                )
                if result.get("paired_alignment_a_optimal_correspondence_count") is not None
                else "",
                "paired_alignment_b_optimal_correspondence_count": result.get(
                    "paired_alignment_b_optimal_correspondence_count"
                )
                if result.get("paired_alignment_b_optimal_correspondence_count") is not None
                else "",
                "paired_alignment_a_selected_pair_consensus_fraction": result.get(
                    "paired_alignment_a_selected_pair_consensus_fraction"
                )
                if result.get("paired_alignment_a_selected_pair_consensus_fraction") is not None
                else "",
                "paired_alignment_b_selected_pair_consensus_fraction": result.get(
                    "paired_alignment_b_selected_pair_consensus_fraction"
                )
                if result.get("paired_alignment_b_selected_pair_consensus_fraction") is not None
                else "",
                "raw_patch_count": result.get("patch_count", 0),
                "prepared_patch_count": result.get("prepared_patch_count", 0),
                "common_patch_count": domain.get("common_patch_count", 0),
                "comparison_complete": domain.get("complete", False),
                "comparison_signature": domain.get("signature", ""),
                "initialization_common_patch_count": initialization_domain.get("common_patch_count", 0),
                "initialization_comparison_complete": initialization_domain.get("complete", False),
                "initialization_comparison_signature": initialization_domain.get("signature", ""),
                "residue_aware_common_patch_count": residue_aware_domain.get("common_patch_count", 0),
                "residue_aware_comparison_complete": residue_aware_domain.get("complete", False),
                "residue_aware_comparison_signature": residue_aware_domain.get("signature", ""),
                "residue_aware_pair_status": residue_aware_pair.get("status", ""),
                "residue_aware_pair_complete": residue_aware_pair.get("complete", False),
                "residue_aware_pair_expected_patch_count": residue_aware_pair.get("expected_patch_count", 0),
                "residue_aware_pair_common_patch_count": residue_aware_pair.get("common_patch_count", 0),
                "residue_aware_pair_domain_signature": residue_aware_pair.get("domain_signature", ""),
                "error": result.get("error", ""),
            }
            afdb = result.get("afdb_complex_confidence", {})
            if isinstance(afdb, dict):
                flat.update(
                    {
                        "afdb_model_id": afdb.get("model_id") or "",
                        "afdb_iptm": afdb.get("iptm") if afdb.get("iptm") is not None else "",
                        "afdb_ipsae": afdb.get("ipsae") if afdb.get("ipsae") is not None else "",
                        "afdb_pdockq": afdb.get("pdockq") if afdb.get("pdockq") is not None else "",
                        "afdb_pdockq2": afdb.get("pdockq2") if afdb.get("pdockq2") is not None else "",
                        "afdb_lis": afdb.get("lis") if afdb.get("lis") is not None else "",
                    }
                )
            paired_qc = result.get("paired_geometry_qc", {})
            if isinstance(paired_qc, dict):
                paired_fields = {
                    "paired_contact_cutoff_angstrom": "contact_cutoff_angstrom",
                    "paired_predicted_contact_count_total": "predicted_contact_count_total",
                    "paired_contact_recall_fnat": "contact_recall_fnat",
                    "paired_contact_precision": "contact_precision",
                    "paired_contact_jaccard": "contact_jaccard",
                    "paired_experimental_contact_mapping_coverage": ("experimental_contact_mapping_coverage"),
                    "paired_interface_residue_a_mapping_coverage": ("interface_residue_a_mapping_coverage"),
                    "paired_interface_residue_b_mapping_coverage": ("interface_residue_b_mapping_coverage"),
                    "paired_interface_ligand_ca_mapping_coverage": ("interface_ligand_ca_mapping_coverage"),
                    "paired_interface_ligand_ca_rmsd_angstrom": ("interface_ligand_ca_rmsd_angstrom"),
                    "paired_cross_chain_clash_atom_fraction": ("cross_chain_clash_atom_fraction"),
                }
                for output_name, source_name in paired_fields.items():
                    value = paired_qc.get(source_name)
                    flat[output_name] = value if value is not None else ""
            for method in ALL_METHODS:
                if method == "optcuts_lscm_initialized":
                    method_domain = initialization_domain
                    method_domain_name = "initialization_comparison_domain"
                elif method in RESIDUE_AWARE_OPTCUTS_METHODS:
                    method_domain = residue_aware_domain
                    method_domain_name = "residue_aware_comparison_domain"
                else:
                    method_domain = domain
                    method_domain_name = "comparison_domain"
                flat[f"{method}_metric_domain"] = method_domain_name
                flat[f"{method}_metric_domain_complete"] = method_domain.get("complete", False)
                flat[f"{method}_metric_domain_signature"] = method_domain.get("signature", "")
                flat[f"{method}_distortion_mean"] = _nested(result, (method, "distortion", "mean"))
                flat[f"{method}_symmetric_dirichlet_mean"] = _nested(
                    result,
                    (method, "symmetric_dirichlet", "mean"),
                )
                flat[f"{method}_angle_mean_rad"] = _nested(result, (method, "angle_distortion", "mean"))
                flat[f"{method}_area_mean"] = _nested(result, (method, "area_distortion", "mean"))
                flat[f"{method}_flip_rate"] = _nested(result, (method, "flip_rate"))
                flat[f"{method}_globally_injective_patch_rate"] = _nested(
                    result,
                    (method, "injectivity", "globally_injective_patch_rate"),
                )
                method_quality = result.get(method, {})
                method_injectivity = method_quality.get("injectivity", {}) if isinstance(method_quality, dict) else {}
                flat[f"{method}_all_patches_globally_injective"] = bool(
                    method_injectivity.get("all_patches_globally_injective", False)
                )
                flat[f"{method}_atlas_overdraw_ratio"] = _nested(
                    result,
                    (method, "atlas", "overdraw_ratio"),
                )
                flat[f"{method}_seam_edge_count"] = _nested(result, (method, "seam", "seam_edge_count"))
                flat[f"{method}_seam_length_3d_normalized"] = _nested(
                    result,
                    (method, "seam", "seam_length_3d_normalized"),
                )
                fragmentation_path = ("residue_footprint_fragmentation", "methods", method)
                flat[f"{method}_residue_fragmentation_mean"] = _nested(
                    result,
                    (*fragmentation_path, "mean_fragmentation"),
                )
                flat[f"{method}_residue_fragmentation_area_weighted"] = _nested(
                    result,
                    (*fragmentation_path, "area_weighted_fragmentation"),
                )
                flat[f"{method}_residue_fragmentation_interaction_weighted"] = _nested(
                    result,
                    (*fragmentation_path, "interaction_weighted_fragmentation"),
                )
                flat[f"{method}_residue_fragmentation_objective_weighted"] = _nested(
                    result,
                    (*fragmentation_path, "objective_weighted_fragmentation"),
                )
                flat[f"{method}_residue_footprint_cycle_rank"] = _nested(
                    result,
                    (*fragmentation_path, "nonlocality_audit", "cycle_rank"),
                )
                flat[f"{method}_nonseparating_seam_crossing_edge_count"] = _nested(
                    result,
                    (
                        *fragmentation_path,
                        "nonlocality_audit",
                        "nonseparating_seam_crossing_edge_count",
                    ),
                )
            residue_aware_pair_methods = residue_aware_pair.get("methods", {})
            residue_aware_pair_arms = residue_aware_pair.get("arms", {})
            if not isinstance(residue_aware_pair_methods, dict):
                residue_aware_pair_methods = {}
            if not isinstance(residue_aware_pair_arms, dict):
                residue_aware_pair_arms = {}
            for method in RESIDUE_AWARE_PAIR_METHODS:
                arm = residue_aware_pair_arms.get(method, {})
                quality = residue_aware_pair_methods.get(method, {})
                if not isinstance(arm, dict):
                    arm = {}
                if not isinstance(quality, dict):
                    quality = {}
                prefix = f"residue_aware_pair_{method}"
                flat[f"{prefix}_domain_complete"] = arm.get("domain_complete", False)
                flat[f"{prefix}_metric_finite"] = arm.get("metric_finite", False)
                flat[f"{prefix}_globally_injective"] = arm.get("globally_injective", False)
                flat[f"{prefix}_usable"] = arm.get("usable", False)
                flat[f"{prefix}_distortion_mean"] = _nested(quality, ("distortion", "mean"))
                flat[f"{prefix}_symmetric_dirichlet_mean"] = _nested(
                    quality,
                    ("symmetric_dirichlet", "mean"),
                )
                flat[f"{prefix}_angle_mean_rad"] = _nested(quality, ("angle_distortion", "mean"))
                flat[f"{prefix}_area_mean"] = _nested(quality, ("area_distortion", "mean"))
                flat[f"{prefix}_flip_rate"] = _nested(quality, ("flip_rate",))
                flat[f"{prefix}_seam_length_3d_normalized"] = _nested(
                    quality,
                    ("seam", "seam_length_3d_normalized"),
                )
                flat[f"{prefix}_residue_fragmentation_objective_weighted"] = _nested(
                    quality,
                    ("residue_footprint_fragmentation", "objective_weighted_fragmentation"),
                )
            flat.update(
                {
                    "atlas_reference_method": _atlas_reference_method(result),
                    "atlas_utilization_polygonal": _atlas_metric(result, "utilization"),
                    "atlas_overlap_ratio_polygonal": _atlas_metric(result, "overlap_ratio"),
                    "atlas_overdraw_ratio": _atlas_metric(result, "overdraw_ratio"),
                    "atlas_min_chart_gap_polygonal": _atlas_metric(result, "min_chart_gap"),
                    "atlas_padding_violations_polygonal": _atlas_metric(result, "padding_violations"),
                    "end_to_end_wall_sec_median": _nested(
                        result,
                        ("timing", "isolated_repetitions", "wall_sec_median"),
                    ),
                    "end_to_end_cpu_sec_median": _nested(
                        result,
                        ("timing", "isolated_repetitions", "cpu_sec_median"),
                    ),
                    "peak_rss_mb": _nested(result, ("memory", "peak_rss_mb")),
                    "worker_runtime_observation_count": int(len(runtime_observations)),
                    "worker_runtime_observation_sec_last": (
                        _runtime_observation(last_runtime_observation) if last_runtime_observation else ""
                    ),
                    "worker_supervisor_wall_sec_last": last_runtime_observation.get("wall_sec", ""),
                    "worker_right_censored": bool(last_runtime_observation.get("right_censored", False)),
                    "worker_termination_reason": last_runtime_observation.get("termination_reason") or "",
                    "worker_censoring_threshold_sec": (
                        last_runtime_observation.get("censoring_threshold_sec")
                        if last_runtime_observation.get("censoring_threshold_sec") is not None
                        else ""
                    ),
                    "worker_censoring_event_elapsed_sec": (
                        last_runtime_observation.get("censoring_event_elapsed_sec")
                        if last_runtime_observation.get("censoring_event_elapsed_sec") is not None
                        else ""
                    ),
                }
            )
            writer.writerow(flat)
