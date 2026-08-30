"""Paired, cluster-aware benchmark summaries with finite p-value reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon
from scipy.stats import t as student_t


def _nested_value(record: Dict[str, object], path: Sequence[str]) -> float:
    current: object = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return float("nan")
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )
    estimates = np.mean(sampled, axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _dyadic_robust_mean_sensitivity(
    values: np.ndarray,
    partner_a: np.ndarray,
    partner_b: np.ndarray,
) -> Dict[str, object]:
    """Estimate shared-member dependence for heterotypic family means.

    The intercept-only dyadic-robust sandwich meat is the sum of residual
    cross-products over pairs of observed dyads that share a dependency node.
    The calculation accepts an incomplete, sparse graph.  Self-pairs
    (homotypic interaction families) are excluded because the published dyadic
    theory assumes two distinct members per observation; their count remains
    explicit in the returned audit record.

    This is a dependence sensitivity, not the primary interval.  The normal
    interval uses the asymptotic dyadic-robust standard error.  A second
    interval uses the finite-degree t critical-value heuristic proposed by
    Tabord-Meehan, kappa = G * median(degree) / max(degree), to expose highly
    unbalanced dependency graphs.
    """

    values = np.asarray(values, dtype=np.float64)
    partner_a = np.asarray(partner_a, dtype=object)
    partner_b = np.asarray(partner_b, dtype=object)
    base: Dict[str, object] = {
        "status": "unavailable",
        "role": "shared_protein_dependency_sensitivity_not_primary",
        "scope": "heterotypic_interaction_families_only",
        "method": "intercept_only_dyadic_robust_sandwich",
        "family_count_total": int(len(values)),
        "heterotypic_family_count": 0,
        "homotypic_family_count_excluded": 0,
        "protein_cluster_count": 0,
        "reference": {
            "estimator": "Aronow, Samii & Assenova (2015), Eq. 3; sparse-graph conditions from Tabord-Meehan (2019)",
            "degrees_of_freedom_heuristic": "Tabord-Meehan (2019), kappa=G*median_degree/max_degree",
        },
    }
    if len(values) != len(partner_a) or len(values) != len(partner_b):
        base["reason"] = "partner_array_length_mismatch"
        return base
    finite = np.isfinite(values)
    nonempty = np.asarray(
        [
            isinstance(a, str) and bool(a.strip()) and isinstance(b, str) and bool(b.strip())
            for a, b in zip(partner_a, partner_b, strict=True)
        ],
        dtype=bool,
    )
    if not np.all(finite & nonempty):
        base["reason"] = "nonfinite_value_or_missing_partner"
        return base

    homotypic = partner_a == partner_b
    heterotypic = ~homotypic
    base["heterotypic_family_count"] = int(np.count_nonzero(heterotypic))
    base["homotypic_family_count_excluded"] = int(np.count_nonzero(homotypic))
    if np.count_nonzero(heterotypic) < 2:
        base["reason"] = "fewer_than_two_heterotypic_families"
        return base

    y = values[heterotypic]
    left = partner_a[heterotypic]
    right = partner_b[heterotypic]
    protein_clusters, inverse = np.unique(np.concatenate((left, right)), return_inverse=True)
    if len(protein_clusters) < 3:
        base["reason"] = "fewer_than_three_distinct_protein_clusters"
        return base
    index_left = inverse[: len(y)]
    index_right = inverse[len(y) :]
    degree = np.bincount(np.concatenate((index_left, index_right)), minlength=len(protein_clusters))
    mean = float(np.mean(y))
    residual = y - mean
    node_scores = np.bincount(index_left, weights=residual, minlength=len(protein_clusters))
    node_scores += np.bincount(index_right, weights=residual, minlength=len(protein_clusters))
    sandwich_meat = float(np.dot(node_scores, node_scores) - np.dot(residual, residual))
    raw_variance = float(sandwich_meat / (len(y) ** 2))
    maximum_degree = int(np.max(degree))
    median_degree = float(np.median(degree))
    kappa = float(len(protein_clusters) * median_degree / maximum_degree)

    incident_sum = np.bincount(index_left, weights=y, minlength=len(protein_clusters))
    incident_sum += np.bincount(index_right, weights=y, minlength=len(protein_clusters))
    remaining_count = len(y) - degree
    leave_one_node_out = np.divide(
        float(np.sum(y)) - incident_sum,
        remaining_count,
        out=np.full(len(protein_clusters), np.nan, dtype=np.float64),
        where=remaining_count > 0,
    )
    shifts = leave_one_node_out - mean
    finite_shift = np.isfinite(shifts)
    influential_index = int(np.nanargmax(np.abs(shifts))) if np.any(finite_shift) else None

    base.update(
        {
            "protein_cluster_count": int(len(protein_clusters)),
            "median_observed_degree": median_degree,
            "maximum_observed_degree": maximum_degree,
            "maximum_degree_family_fraction": float(maximum_degree / len(y)),
            "effective_degrees_of_freedom_heuristic": kappa,
            "mean": mean,
            "raw_sandwich_meat": sandwich_meat,
            "raw_variance": raw_variance,
            "variance_nonnegative": bool(raw_variance >= 0.0),
            "maximum_absolute_leave_one_protein_cluster_out_mean_shift": (
                float(abs(shifts[influential_index])) if influential_index is not None else float("nan")
            ),
            "most_influential_protein_cluster": (
                str(protein_clusters[influential_index]) if influential_index is not None else ""
            ),
            "most_influential_protein_cluster_degree": (
                int(degree[influential_index]) if influential_index is not None else 0
            ),
        }
    )
    if not np.isfinite(raw_variance) or raw_variance <= 0.0:
        base["reason"] = "nonpositive_finite_sample_sandwich_variance"
        return base

    standard_error = float(np.sqrt(raw_variance))
    normal_critical = 1.959963984540054
    t_critical = float(student_t.ppf(0.975, df=kappa))
    base.update(
        {
            "status": "evaluated",
            "reason": "",
            "standard_error": standard_error,
            "normal_ci95": [
                mean - normal_critical * standard_error,
                mean + normal_critical * standard_error,
            ],
            "finite_degree_t_ci95": [
                mean - t_critical * standard_error,
                mean + t_critical * standard_error,
            ],
            "finite_degree_t_critical_value": t_critical,
        }
    )
    return base


def _bootstrap_ratio_of_means_ci(
    baseline: np.ndarray,
    treatment: np.ndarray,
    *,
    reference: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(baseline) < 2 or len(treatment) != len(baseline):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(iterations, len(baseline)))
    sampled_baseline = np.mean(baseline[indices], axis=1)
    sampled_treatment = np.mean(treatment[indices], axis=1)
    denominator = sampled_baseline - float(reference)
    valid = np.abs(denominator) > 1e-12
    estimates = np.divide(
        sampled_baseline - sampled_treatment,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=valid,
    )
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def paired_method_comparison(
    rows: Iterable[Dict[str, object]],
    *,
    baseline: str,
    treatment: str,
    metric_path: Sequence[str] = ("distortion", "mean"),
    cluster_key: str = "family_id",
    bootstrap_iterations: int = 2000,
    seed: int = 20260817,
    relative_reference: float = 0.0,
    binary_endpoint: bool = False,
) -> Dict[str, object]:
    """Compare two methods using paired structure-level values.

    Differences are ``baseline - treatment`` so positive values favor the
    treatment for lower-is-better metrics.
    """

    rows = list(rows)
    baseline_values = []
    treatment_values = []
    clusters = []
    cluster_sources = []
    partner_pairs: list[tuple[str, str] | None] = []
    partner_pair_sources = []
    for row_index, row in enumerate(rows):
        baseline_value = _nested_value(row, (baseline, *metric_path))
        treatment_value = _nested_value(row, (treatment, *metric_path))
        if not np.isfinite(baseline_value) or not np.isfinite(treatment_value):
            continue
        baseline_values.append(baseline_value)
        treatment_values.append(treatment_value)
        cluster = str(row.get(cluster_key) or "").strip()
        cluster_source = cluster_key if cluster else ""
        if not cluster and cluster_key != "cluster_id":
            cluster = str(row.get("cluster_id") or "").strip()
            cluster_source = "cluster_id" if cluster else ""
        if not cluster:
            cluster = f"structure:{row_index}"
            cluster_source = "structure"
        clusters.append(cluster)
        cluster_sources.append(cluster_source)
        partner_key_prefix = {
            "family_id": "sequence_cluster",
            "inference_family_id": "inference_sequence_cluster",
        }.get(cluster_key)
        partner_a = (
            str(row.get(f"{partner_key_prefix}_a") or "").strip()
            if partner_key_prefix is not None and cluster_source == cluster_key
            else ""
        )
        partner_b = (
            str(row.get(f"{partner_key_prefix}_b") or "").strip()
            if partner_key_prefix is not None and cluster_source == cluster_key
            else ""
        )
        partner_pairs.append(tuple(sorted((partner_a, partner_b))) if partner_a and partner_b else None)
        partner_pair_sources.append(
            partner_key_prefix if partner_key_prefix is not None and partner_a and partner_b else "unavailable"
        )
    if not baseline_values:
        return {
            "baseline": baseline,
            "treatment": treatment,
            "metric_path": list(metric_path),
            "paired_structure_count": 0,
        }

    baseline_array = np.asarray(baseline_values, dtype=np.float64)
    treatment_array = np.asarray(treatment_values, dtype=np.float64)
    differences = baseline_array - treatment_array
    relative_denominator = baseline_array - float(relative_reference)
    relative = np.divide(
        differences,
        relative_denominator,
        out=np.full_like(differences, np.nan),
        where=np.abs(relative_denominator) > 1e-12,
    )
    cluster_array = np.asarray(clusters, dtype=object)
    unique_clusters, cluster_indices = np.unique(cluster_array, return_inverse=True)
    cluster_differences = np.bincount(cluster_indices, weights=differences) / np.bincount(cluster_indices)
    cluster_baseline = np.bincount(cluster_indices, weights=baseline_array) / np.bincount(cluster_indices)
    cluster_treatment = np.bincount(cluster_indices, weights=treatment_array) / np.bincount(cluster_indices)
    finite_relative = np.isfinite(relative)
    cluster_relative_sum = np.bincount(
        cluster_indices[finite_relative],
        weights=relative[finite_relative],
        minlength=len(unique_clusters),
    )
    cluster_relative_count = np.bincount(
        cluster_indices[finite_relative],
        minlength=len(unique_clusters),
    )
    cluster_relative = np.divide(
        cluster_relative_sum,
        cluster_relative_count,
        out=np.full(len(unique_clusters), np.nan),
        where=cluster_relative_count > 0,
    )
    cluster_ci = _bootstrap_mean_ci(
        cluster_differences,
        iterations=int(bootstrap_iterations),
        seed=int(seed),
    )
    relative_cluster_ci = _bootstrap_mean_ci(
        cluster_relative[np.isfinite(cluster_relative)],
        iterations=int(bootstrap_iterations),
        seed=int(seed) + 1,
    )
    relative_of_cluster_means = (
        float(
            (np.mean(cluster_baseline) - np.mean(cluster_treatment))
            / (np.mean(cluster_baseline) - float(relative_reference))
        )
        if abs(float(np.mean(cluster_baseline)) - float(relative_reference)) > 1e-12
        else float("nan")
    )
    relative_of_cluster_means_ci = _bootstrap_ratio_of_means_ci(
        cluster_baseline,
        cluster_treatment,
        reference=float(relative_reference),
        iterations=int(bootstrap_iterations),
        seed=int(seed) + 4,
    )

    partners_by_cluster: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for cluster, pair in zip(clusters, partner_pairs, strict=True):
        if pair is not None:
            partners_by_cluster[cluster].add(pair)
    complete_partner_clusters = all(len(partners_by_cluster[str(cluster)]) == 1 for cluster in unique_clusters)
    if complete_partner_clusters:
        grouped_pairs = [next(iter(partners_by_cluster[str(cluster)])) for cluster in unique_clusters]
        grouped_partner_a = np.asarray([pair[0] for pair in grouped_pairs], dtype=object)
        grouped_partner_b = np.asarray([pair[1] for pair in grouped_pairs], dtype=object)
        shared_protein_difference_sensitivity = _dyadic_robust_mean_sensitivity(
            cluster_differences,
            grouped_partner_a,
            grouped_partner_b,
        )
        relative_mask = np.isfinite(cluster_relative)
        shared_protein_relative_sensitivity = _dyadic_robust_mean_sensitivity(
            cluster_relative[relative_mask],
            grouped_partner_a[relative_mask],
            grouped_partner_b[relative_mask],
        )
    else:
        grouped_partner_a = np.asarray([], dtype=object)
        grouped_partner_b = np.asarray([], dtype=object)
        unavailable = {
            "status": "unavailable",
            "reason": "cluster_to_partner_mapping_missing_or_inconsistent",
            "role": "shared_protein_dependency_sensitivity_not_primary",
            "scope": "heterotypic_interaction_families_only",
            "method": "intercept_only_dyadic_robust_sandwich",
            "family_count_total": int(len(unique_clusters)),
            "heterotypic_family_count": 0,
            "homotypic_family_count_excluded": 0,
            "protein_cluster_count": 0,
        }
        shared_protein_difference_sensitivity = dict(unavailable)
        shared_protein_relative_sensitivity = dict(unavailable)
    primary_ci = cluster_ci
    if len(cluster_differences) < 2:
        p_value = float("nan")
    elif np.all(cluster_differences == 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(
                wilcoxon(
                    cluster_differences,
                    zero_method="pratt",
                    alternative="two-sided",
                ).pvalue
            )
        except ValueError:
            p_value = float("nan")
    p_censored = bool(np.isfinite(p_value) and p_value == 0.0)
    if p_censored:
        p_value = float(np.finfo(np.float64).tiny)

    std = float(np.std(cluster_differences, ddof=1)) if len(cluster_differences) > 1 else float("nan")
    standard_error = float(std / np.sqrt(len(cluster_differences))) if np.isfinite(std) else float("nan")
    primary_standard_error = standard_error
    finite_relative_values = relative[np.isfinite(relative)]
    source_counts = {
        source: int(sum(value == source for value in cluster_sources)) for source in sorted(set(cluster_sources))
    }
    all_requested_cluster_keys_available = bool(cluster_sources) and all(
        source in {cluster_key, "inference_family_id"} for source in cluster_sources
    )
    unique_cluster_sources = set(cluster_sources)
    if unique_cluster_sources == {"inference_family_id"}:
        inferential_unit = "prediction_dependency_family_mean"
    elif unique_cluster_sources == {"family_id"}:
        inferential_unit = "interaction_family_mean"
    elif len(unique_cluster_sources) == 1:
        inferential_unit = f"{next(iter(unique_cluster_sources))}_mean"
    else:
        inferential_unit = "resolved_cluster_mean"
    result = {
        "baseline": baseline,
        "treatment": treatment,
        "metric_path": list(metric_path),
        "paired_structure_count": int(len(differences)),
        "cluster_count": int(len(unique_clusters)),
        "cluster_key": cluster_key,
        "cluster_fallback": "requested_key_then_cluster_id_then_structure",
        "mean_baseline": float(np.mean(baseline_array)),
        "mean_treatment": float(np.mean(treatment_array)),
        "mean_cluster_baseline": float(np.mean(cluster_baseline)),
        "mean_cluster_treatment": float(np.mean(cluster_treatment)),
        "mean_paired_difference": float(np.mean(differences)),
        "median_paired_difference": float(np.median(differences)),
        "mean_cluster_difference": float(np.mean(cluster_differences)),
        "median_cluster_difference": float(np.median(cluster_differences)),
        "standard_error_cluster_difference": standard_error,
        "primary_standard_error_difference": primary_standard_error,
        "mean_relative_improvement": (
            float(np.mean(finite_relative_values)) if len(finite_relative_values) else float("nan")
        ),
        "mean_cluster_relative_improvement": (
            float(np.nanmean(cluster_relative)) if np.any(np.isfinite(cluster_relative)) else float("nan")
        ),
        "relative_improvement_reference": float(relative_reference),
        "relative_improvement_definition": (
            "(baseline-treatment)/(baseline-reference); paired-relative summaries average "
            "unit-level ratios, while relative_improvement_of_cluster_means is the ratio of "
            "equally weighted cluster means"
        ),
        "relative_improvement_of_cluster_means": relative_of_cluster_means,
        "cluster_bootstrap_relative_improvement_of_means_ci95": list(relative_of_cluster_means_ci),
        "primary_mean_difference_ci95": list(primary_ci),
        "resolved_cluster_bootstrap_mean_difference_ci95": list(cluster_ci),
        "resolved_cluster_bootstrap_mean_relative_improvement_ci95": list(relative_cluster_ci),
        "shared_protein_dependency_sensitivity": {
            "mean_difference": shared_protein_difference_sensitivity,
            "mean_relative_improvement": shared_protein_relative_sensitivity,
        },
        "paired_standardized_effect_dz_cluster": float(np.mean(cluster_differences) / std)
        if std > 0.0
        else float("nan"),
        "wilcoxon_p_value": p_value,
        "wilcoxon_p_value_censored_from_zero": p_censored,
        "bootstrap_iterations": int(bootstrap_iterations),
        "random_seed": int(seed),
        "descriptive_unit": "structure",
        "inferential_unit": inferential_unit,
        "cluster_source_counts": source_counts,
        "partner_cluster_source_counts": {
            source: int(sum(value == source for value in partner_pair_sources))
            for source in sorted(set(partner_pair_sources))
        },
        "all_requested_cluster_keys_available": all_requested_cluster_keys_available,
        "primary_interval_method": "resolved_cluster_bootstrap",
        "wilcoxon_role": "secondary two-sided signed-rank test on resolved cluster means",
    }
    if binary_endpoint:
        if not np.all(np.isin(baseline_array, (0.0, 1.0))) or not np.all(np.isin(treatment_array, (0.0, 1.0))):
            raise ValueError("binary_endpoint requires values coded exactly as 0 or 1.")
        baseline_event_treatment_nonevent = int(np.count_nonzero((baseline_array == 1.0) & (treatment_array == 0.0)))
        baseline_nonevent_treatment_event = int(np.count_nonzero((baseline_array == 0.0) & (treatment_array == 1.0)))
        discordant = baseline_event_treatment_nonevent + baseline_nonevent_treatment_event
        exact_p = (
            float(
                binomtest(
                    baseline_event_treatment_nonevent,
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        result.update(
            {
                "baseline_event_treatment_nonevent_count": baseline_event_treatment_nonevent,
                "baseline_nonevent_treatment_event_count": baseline_nonevent_treatment_event,
                "discordant_pair_count": int(discordant),
                "concordant_pair_count": int(len(differences) - discordant),
                "mcnemar_exact_p_value": exact_p,
                "mcnemar_role": (
                    "unclustered exact discordant-pair diagnostic; clustered bootstrap and "
                    "resolved-cluster signed-rank summaries remain the dependence-aware analyses"
                ),
            }
        )
    return result


__all__ = ["paired_method_comparison"]
