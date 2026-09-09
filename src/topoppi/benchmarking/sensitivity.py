"""Frozen sensitivity-study plans built on the auditable benchmark runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from itertools import product
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from topoppi.benchmarking.manifest_metadata import PREDICTED_STRUCTURE_TYPES
from topoppi.benchmarking.runner import BenchmarkRunner
from topoppi.benchmarking.statistics import paired_method_comparison
from topoppi.config import BenchmarkConfig, benchmark_config_from_dict
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

SENSITIVITY_AXES = {
    "topology.distance_cutoff": ("topology", "distance_cutoff"),
    "surface.grid_resolution": ("surface", "grid_resolution"),
    "surface.sigma": ("surface", "sigma"),
    "surface.level": ("surface", "level"),
    "optcuts.optcuts_lambda_init": ("optcuts", "optcuts_lambda_init"),
    "optcuts.optcuts_distortion_bound": ("optcuts", "optcuts_distortion_bound"),
}

SENSITIVITY_METHODS = ("optcuts_automatic", "residue_aware_optcuts")
SENSITIVITY_STANDARD_METHOD = "optcuts_automatic"
SENSITIVITY_RESIDUE_AWARE_METHOD = "residue_aware_optcuts"
SENSITIVITY_ENDPOINTS = {
    "distortion_mean": (("distortion", "mean"), 0.0),
    "symmetric_dirichlet_mean": (("symmetric_dirichlet", "mean"), 2.0),
    "angle_distortion_mean": (("angle_distortion", "mean"), 0.0),
    "area_distortion_mean": (("area_distortion", "mean"), 0.0),
    "normalized_seam_length": (("seam", "seam_length_3d_normalized"), 0.0),
    "objective_weighted_fragmentation": (
        ("residue_footprint_fragmentation", "objective_weighted_fragmentation"),
        0.0,
    ),
}

_AXIS_ALIASES = {
    "distance_cutoff": "topology.distance_cutoff",
    "cutoff": "topology.distance_cutoff",
    "grid_resolution": "surface.grid_resolution",
    "grid_spacing": "surface.grid_resolution",
    "sigma": "surface.sigma",
    "isovalue": "surface.level",
    "level": "surface.level",
    "lambda": "optcuts.optcuts_lambda_init",
    "lambda_init": "optcuts.optcuts_lambda_init",
    "distortion_bound": "optcuts.optcuts_distortion_bound",
}


def _require_sensitivity_baseline(config: BenchmarkConfig) -> None:
    if SENSITIVITY_STANDARD_METHOD not in config.resolved_optcuts_variants():
        raise ValueError(
            "Sensitivity studies require optcuts_automatic as the baseline arm. "
            "Add it to optcuts_variants and create a new plan."
        )


@dataclass(frozen=True)
class SensitivityScenario:
    scenario_id: str
    changes: Dict[str, float]
    config: BenchmarkConfig


def _canonical_axis(name: str) -> str:
    normalized = str(name).strip().lower()
    canonical = _AXIS_ALIASES.get(normalized, normalized)
    if canonical not in SENSITIVITY_AXES:
        supported = ", ".join(sorted(SENSITIVITY_AXES))
        raise ValueError(f"Unsupported sensitivity axis '{name}'. Supported axes: {supported}.")
    return canonical


def normalize_sensitivity_axes(payload: Mapping[str, object]) -> Dict[str, list[float]]:
    """Normalize aliases, reject non-finite values, and preserve stable order."""

    raw_axes = payload.get("axes", payload)
    if not isinstance(raw_axes, Mapping) or not raw_axes:
        raise ValueError("Sensitivity axes must be a non-empty mapping of axis names to value lists.")
    axes: Dict[str, list[float]] = {}
    for raw_name, raw_values in raw_axes.items():
        name = _canonical_axis(str(raw_name))
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Iterable):
            raise ValueError(f"Sensitivity axis '{raw_name}' must contain a list of numeric values.")
        values = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"Sensitivity axis '{raw_name}' contains a non-numeric value.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Sensitivity axis '{raw_name}' contains a non-finite value.")
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError(f"Sensitivity axis '{raw_name}' has no values.")
        axes[name] = values
    return axes


def _axis_value(config: BenchmarkConfig, axis: str) -> float:
    section_name, field_name = SENSITIVITY_AXES[axis]
    return float(getattr(getattr(config, section_name), field_name))


def _replace_axis(config: BenchmarkConfig, axis: str, value: float) -> BenchmarkConfig:
    section_name, field_name = SENSITIVITY_AXES[axis]
    section = getattr(config, section_name)
    return replace(config, **{section_name: replace(section, **{field_name: value})})


def _scenario_id(changes: Mapping[str, float]) -> str:
    if not changes:
        return "baseline"
    payload = json.dumps(dict(sorted(changes.items())), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    label = "__".join(name.split(".")[-1] for name in sorted(changes))
    return f"{label}__{digest}"


def build_sensitivity_scenarios(
    base_config: BenchmarkConfig,
    axes: Mapping[str, object],
    *,
    design: str = "one_factor",
    output_root: str | None = None,
) -> list[SensitivityScenario]:
    """Build deterministic OFAT or factorial scenarios without executing them."""

    _require_sensitivity_baseline(base_config)
    normalized = normalize_sensitivity_axes(axes)
    mode = str(design).strip().lower()
    if mode not in {"one_factor", "factorial"}:
        raise ValueError("Sensitivity design must be 'one_factor' or 'factorial'.")
    root = Path(output_root or base_config.output_root).resolve()
    change_sets: list[Dict[str, float]] = []
    if mode == "one_factor":
        change_sets.append({})
        for axis, values in normalized.items():
            baseline = _axis_value(base_config, axis)
            change_sets.extend(
                {axis: value} for value in values if not math.isclose(value, baseline, rel_tol=0.0, abs_tol=1e-15)
            )
    else:
        names = list(normalized)
        for values in product(*(normalized[name] for name in names)):
            changes = {
                name: float(value)
                for name, value in zip(names, values, strict=True)
                if not math.isclose(value, _axis_value(base_config, name), rel_tol=0.0, abs_tol=1e-15)
            }
            change_sets.append(changes)
        if not any(not changes for changes in change_sets):
            change_sets.insert(0, {})

    scenarios = []
    for changes in change_sets:
        scenario_id = _scenario_id(changes)
        config = base_config
        for axis, value in sorted(changes.items()):
            config = _replace_axis(config, axis, value)
        config = replace(config, output_root=str(root / scenario_id), resume=True)
        config.validate()
        scenarios.append(SensitivityScenario(scenario_id, dict(changes), config))
    if not scenarios:
        raise ValueError("Sensitivity design produced no scenarios.")
    return scenarios


def write_sensitivity_plan(
    base_config: BenchmarkConfig,
    axes: Mapping[str, object],
    plan_root: str,
    *,
    design: str = "one_factor",
) -> str:
    """Write immutable scenario configs and a plan; benchmark jobs are not run."""

    target_root = Path(plan_root).resolve()
    scenarios = build_sensitivity_scenarios(
        base_config,
        axes,
        design=design,
        output_root=str(target_root / "results"),
    )
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in scenarios:
        config_name = f"config.{scenario.scenario_id}.json"
        config_path = target_root / config_name
        dump_json_atomic(asdict(scenario.config), config_path)
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "changes": scenario.changes,
                "config_file": config_name,
                "config_sha256": sha256_file(config_path),
            }
        )
    normalized = normalize_sensitivity_axes(axes)
    plan = {
        "schema_version": "1.0",
        "design": str(design).strip().lower(),
        "axis_policy": "supported_axes_only",
        "axes": normalized,
        "scenario_count": int(len(records)),
        "baseline_config": asdict(base_config),
        "invariants": {
            "input_folder": str(Path(base_config.input_folder).resolve()),
            "manifest_path": str(Path(base_config.manifest_path).resolve()) if base_config.manifest_path else None,
            "random_seed": int(base_config.random_seed),
            "chain_selection_mode": base_config.chain_selection_mode,
            "metric_protocol": "defined by benchmark_report.schema_version=2.0",
        },
        "scenarios": records,
        "execution_guard": (
            "Run the scenarios with topoppi-benchmark run-sensitivity. "
            "Include --confirm-formal-benchmark for formal scenarios."
        ),
    }
    plan_path = target_root / "sensitivity_plan.json"
    dump_json_atomic(plan, plan_path)
    return str(plan_path)


def _load_sensitivity_scenario(
    plan_path: Path,
    baseline_config: BenchmarkConfig,
    record: Mapping[str, object],
) -> SensitivityScenario:
    scenario_id = str(record.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError("Sensitivity plan has a missing scenario_id.")
    changes = _scenario_changes(scenario_id, record.get("changes", {}))
    config = _scenario_config(plan_path, scenario_id, record)

    expected_config = baseline_config
    for axis, value in sorted(changes.items()):
        expected_config = _replace_axis(expected_config, axis, value)
    expected_output_root = str(plan_path.parent / "results" / scenario_id)
    expected_config = replace(expected_config, output_root=expected_output_root, resume=True)
    if config != expected_config:
        raise ValueError(f"Sensitivity scenario '{scenario_id}' changes settings outside its declared axes.")
    return SensitivityScenario(scenario_id=scenario_id, changes=changes, config=config)


def _scenario_changes(scenario_id: str, raw_changes: object) -> Dict[str, float]:
    if not isinstance(raw_changes, Mapping):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' has invalid changes.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_changes.values()):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' has a non-numeric change value.")
    changes = {str(key): float(value) for key, value in raw_changes.items()}
    if any(name not in SENSITIVITY_AXES for name in changes):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' changes an unsupported axis.")
    if any(not math.isfinite(value) for value in changes.values()):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' has a non-finite change value.")
    if scenario_id != _scenario_id(changes):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' does not match its declared changes.")
    return changes


def _scenario_config(
    plan_path: Path,
    scenario_id: str,
    record: Mapping[str, object],
) -> BenchmarkConfig:
    config_file = plan_path.parent / str(record.get("config_file") or "")
    expected_sha256 = str(record.get("config_sha256") or "").strip().lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' is missing a valid config SHA-256.")
    if not config_file.is_file():
        raise ValueError(f"Sensitivity scenario config is missing: {config_file.name}.")
    if sha256_file(config_file).lower() != expected_sha256:
        raise ValueError(f"Sensitivity scenario config checksum mismatch: {config_file.name}.")
    with config_file.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    if not isinstance(config_payload, dict):
        raise ValueError(f"Sensitivity scenario '{scenario_id}' has no valid config.")
    config = benchmark_config_from_dict(config_payload)
    config.validate()
    return config


def load_sensitivity_plan(path: str) -> tuple[Dict[str, object], list[SensitivityScenario]]:
    plan_path = Path(path).resolve()
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict) or str(plan.get("schema_version") or "") != "1.0":
        raise ValueError("Unsupported or missing sensitivity-plan schema_version.")
    baseline_payload = plan.get("baseline_config")
    if not isinstance(baseline_payload, dict):
        raise ValueError("Sensitivity plan is missing a valid baseline_config.")
    baseline_config = benchmark_config_from_dict(baseline_payload)
    baseline_config.validate()
    _require_sensitivity_baseline(baseline_config)
    raw_scenarios = plan.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("Sensitivity plan is missing a valid scenarios list.")
    declared_scenario_count = plan.get("scenario_count")
    if isinstance(declared_scenario_count, bool) or not isinstance(declared_scenario_count, int):
        raise ValueError("Sensitivity plan scenario_count must be an integer.")

    scenarios = []
    seen_ids = set()
    for record in raw_scenarios:
        if not isinstance(record, Mapping):
            raise ValueError("Sensitivity plan contains a non-object scenario record.")
        scenario = _load_sensitivity_scenario(plan_path, baseline_config, record)
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"Sensitivity plan has a duplicate scenario_id: {scenario.scenario_id!r}.")
        seen_ids.add(scenario.scenario_id)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError("Sensitivity plan contains no scenarios.")
    expected = build_sensitivity_scenarios(
        baseline_config,
        plan.get("axes", {}),
        design=str(plan.get("design") or ""),
        output_root=str(plan_path.parent / "results"),
    )
    expected_ids = {scenario.scenario_id for scenario in expected}
    if seen_ids != expected_ids or declared_scenario_count != len(scenarios):
        raise ValueError("Sensitivity plan scenario set does not match its frozen axes/design.")
    return plan, scenarios


class SensitivityBenchmarkRunner:
    """Preflight or explicitly execute a frozen set of benchmark scenarios."""

    def __init__(self, plan_path: str, log_fn=None):
        self.plan_path = str(Path(plan_path).resolve())
        self.plan, self.scenarios = load_sensitivity_plan(self.plan_path)
        self.log = log_fn or (lambda _message: None)
        self._scenario_runners = {
            scenario.scenario_id: BenchmarkRunner(scenario.config, log_fn=self.log) for scenario in self.scenarios
        }

    def preflight(self) -> Dict[str, object]:
        checks = []
        for scenario in self.scenarios:
            check = self._scenario_runners[scenario.scenario_id].preflight()
            blockers = list(check["blockers"])
            checks.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "changes": scenario.changes,
                    "ready": bool(check["ready"]) and not blockers,
                    "blockers": blockers,
                    "config_fingerprint": check["config_fingerprint"],
                    "planned_worker_process_count": check["planned_worker_process_count"],
                }
            )
        return {
            "ready": all(item["ready"] for item in checks),
            "scenario_count": int(len(checks)),
            "planned_worker_process_count": int(sum(int(item["planned_worker_process_count"]) for item in checks)),
            "scenarios": checks,
            "note": "Sensitivity preflight is read-only and starts no worker processes.",
        }

    def run(self, *, confirm_formal: bool = False) -> Dict[str, object]:
        if any(scenario.config.formal_mode for scenario in self.scenarios) and not confirm_formal:
            raise RuntimeError("Formal sensitivity execution requires confirm_formal=True.")
        preflight = self.preflight()
        if not preflight["ready"]:
            blockers = [
                f"{item['scenario_id']}: {blocker}" for item in preflight["scenarios"] for blocker in item["blockers"]
            ]
            detail = "; ".join(blockers) or "review the scenario preflight records"
            raise RuntimeError(f"Sensitivity preflight failed: {detail}")
        records = []
        reports: dict[str, Mapping[str, object]] = {}
        for scenario in self.scenarios:
            self.log(f"[Sensitivity] Starting {scenario.scenario_id}: {scenario.changes or 'baseline'}")
            try:
                report = self._scenario_runners[scenario.scenario_id].run()
                record = _sensitivity_summary_row(scenario, report)
                reports[scenario.scenario_id] = report
            except (OSError, RuntimeError, ValueError) as exc:
                record = {
                    "scenario_id": scenario.scenario_id,
                    "changes": scenario.changes,
                    "status": "failed",
                    "error": str(exc),
                    "report_path": str(Path(scenario.config.output_root) / scenario.config.report_filename),
                }
            records.append(record)

        paired_analysis = _paired_sensitivity_analysis(
            self.scenarios,
            reports,
            bootstrap_iterations=max(int(scenario.config.bootstrap_iterations) for scenario in self.scenarios),
            seed=int(self.scenarios[0].config.random_seed),
        )
        record_by_id = {str(record["scenario_id"]): record for record in records}
        for scenario_id, comparison in paired_analysis.get("scenario_vs_baseline", {}).items():
            if scenario_id in record_by_id:
                record_by_id[scenario_id]["paired_analysis"] = comparison

        root = Path(self.plan_path).parent
        output = {
            "schema_version": "1.0",
            "plan_file": self.plan_path,
            "plan_sha256": sha256_file(self.plan_path),
            "scenario_count": int(len(records)),
            "scenarios": records,
            "paired_analysis": paired_analysis,
        }
        dump_json_atomic(output, root / "sensitivity_results.json")
        _write_sensitivity_csv(records, root / "sensitivity_summary.csv")
        if paired_analysis.get("status") != "evaluated":
            raise RuntimeError("Sensitivity study is incomplete; inspect sensitivity_results.json before analysis.")
        return output


def _nested(record: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _attempted_identity(row: Mapping[str, object]) -> tuple[str, ...]:
    selection = row.get("chain_selection") or {}
    if not isinstance(selection, Mapping):
        selection = {}
    return tuple(
        str(value or "")
        for value in (
            row.get("manifest_record_id"),
            row.get("pdb"),
            row.get("input_sha256"),
            row.get("interaction_sha256"),
            selection.get("chain_a"),
            selection.get("chain_b"),
            row.get("structure_type"),
            row.get("family_id"),
            row.get("sequence_cluster_a"),
            row.get("sequence_cluster_b"),
            row.get("inference_family_id"),
            row.get("inference_sequence_cluster_a"),
            row.get("inference_sequence_cluster_b"),
            row.get("inference_dependency_basis"),
            row.get("analysis_split"),
            row.get("analysis_split_component_id"),
            row.get("analysis_split_basis"),
            row.get("paired_record_id"),
            row.get("paired_experimental_record_id"),
        )
    )


def _indexed_attempts(report: Mapping[str, object]) -> dict[tuple[str, ...], Dict[str, object]]:
    indexed = {}
    files = report.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Sensitivity report has no files list.")
    for raw in files:
        if not isinstance(raw, dict):
            raise ValueError("Sensitivity report contains a non-object file row.")
        identity = _attempted_identity(raw)
        if not identity[1] or not identity[2]:
            raise ValueError("Sensitivity row lacks a stable file/checksum identity.")
        if identity in indexed:
            raise ValueError(f"Sensitivity report has a duplicate attempted identity: {identity}")
        indexed[identity] = raw
    if not indexed:
        raise ValueError("Sensitivity report contains no attempted structures.")
    return indexed


def _inferential_cluster_key(rows: Sequence[Mapping[str, object]]) -> str:
    """Resolve the frozen dependency unit for one sensitivity cohort."""

    structure_types = {
        str(row.get("structure_type") or "").strip().lower()
        for row in rows
        if str(row.get("structure_type") or "").strip()
    }
    if structure_types and structure_types.issubset(PREDICTED_STRUCTURE_TYPES):
        return "inference_family_id"
    if structure_types & PREDICTED_STRUCTURE_TYPES:
        return "analysis_split_component_id"
    return "family_id"


def _finite_method_payload(
    row: Mapping[str, object],
    method: str,
) -> Dict[str, object] | None:
    arm_evidence = row.get("independent_optcuts_arm_quality")
    if not isinstance(arm_evidence, Mapping):
        return None
    arm = arm_evidence.get(method)
    required_arm_flags = ("domain_complete", "metric_finite")
    if not isinstance(arm, Mapping) or not all(bool(arm.get(flag, False)) for flag in required_arm_flags):
        return None
    payload = arm.get("quality")
    if not isinstance(payload, dict):
        return None
    for path, _reference in SENSITIVITY_ENDPOINTS.values():
        try:
            value = float(_nested(payload, path))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
    return payload


def _method_domain_signature(row: Mapping[str, object], method: str) -> str:
    arms = row.get("independent_optcuts_arm_quality")
    arm = arms.get(method) if isinstance(arms, Mapping) else None
    return str(arm.get("domain_signature") or "").strip() if isinstance(arm, Mapping) else ""


def _paired_sensitivity_analysis(
    scenarios: Sequence[SensitivityScenario],
    reports: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> Dict[str, object]:
    successful_ids = [scenario.scenario_id for scenario in scenarios if scenario.scenario_id in reports]
    if "baseline" not in successful_ids:
        return {
            "status": "unavailable",
            "reason": "baseline scenario did not complete",
            "successful_scenario_count": len(successful_ids),
        }
    indexed = {scenario_id: _indexed_attempts(reports[scenario_id]) for scenario_id in successful_ids}
    inferential_cluster_key = _inferential_cluster_key(list(indexed["baseline"].values()))
    baseline_identities = set(indexed["baseline"])
    mismatched = {
        scenario_id: {
            "missing_vs_baseline": len(baseline_identities - set(rows)),
            "extra_vs_baseline": len(set(rows) - baseline_identities),
        }
        for scenario_id, rows in indexed.items()
        if set(rows) != baseline_identities
    }
    if mismatched:
        raise ValueError(
            "Sensitivity scenarios were not evaluated on identical attempted structure identities: "
            + json.dumps(mismatched, sort_keys=True)
        )

    configured_methods = [
        method
        for method in SENSITIVITY_METHODS
        if all(method in scenario.config.resolved_optcuts_variants() for scenario in scenarios)
    ]
    complete_ids = {
        method: {
            scenario_id: {identity for identity, row in rows.items() if _finite_method_payload(row, method) is not None}
            for scenario_id, rows in indexed.items()
        }
        for method in configured_methods
    }
    all_scenario_common = {
        method: set.intersection(*(set(values) for values in method_ids.values()))
        for method, method_ids in complete_ids.items()
    }
    scenario_comparisons = {}
    ordered_scenarios = [scenario.scenario_id for scenario in scenarios if scenario.scenario_id in indexed]
    comparison_scenarios = [scenario_id for scenario_id in ordered_scenarios if scenario_id != "baseline"]
    for scenario_index, scenario_id in enumerate(comparison_scenarios):
        method_comparisons = {}
        for method_index, method in enumerate(configured_methods):
            pairwise_common = complete_ids[method]["baseline"] & complete_ids[method][scenario_id]

            def comparison_rows(
                identities: set[tuple[str, ...]],
                scenario_key: str = scenario_id,
                selected_method: str = method,
            ) -> list[Dict[str, object]]:
                rows = []
                for identity in sorted(identities):
                    baseline_row = indexed["baseline"][identity]
                    scenario_row = indexed[scenario_key][identity]
                    combined = dict(baseline_row)
                    combined["baseline_scenario"] = _finite_method_payload(baseline_row, selected_method)
                    combined["sensitivity_scenario"] = _finite_method_payload(scenario_row, selected_method)
                    rows.append(combined)
                return rows

            pairwise_rows = comparison_rows(pairwise_common)
            all_common_rows = comparison_rows(all_scenario_common[method])
            pairwise_same_domain_count = sum(
                bool(_method_domain_signature(indexed["baseline"][identity], method))
                and _method_domain_signature(indexed["baseline"][identity], method)
                == _method_domain_signature(indexed[scenario_id][identity], method)
                for identity in pairwise_common
            )
            all_scenario_same_domain_count = sum(
                bool(_method_domain_signature(indexed["baseline"][identity], method))
                and len({_method_domain_signature(indexed[key][identity], method) for key in ordered_scenarios}) == 1
                for identity in all_scenario_common[method]
            )
            endpoint_comparisons = {}
            all_common_endpoint_comparisons = {}
            for endpoint_index, (endpoint, (path, reference)) in enumerate(SENSITIVITY_ENDPOINTS.items()):
                endpoint_comparisons[endpoint] = paired_method_comparison(
                    pairwise_rows,
                    baseline="baseline_scenario",
                    treatment="sensitivity_scenario",
                    metric_path=path,
                    cluster_key=inferential_cluster_key,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + scenario_index * 1000 + method_index * 200 + endpoint_index,
                    relative_reference=reference,
                )
                all_common_endpoint_comparisons[endpoint] = paired_method_comparison(
                    all_common_rows,
                    baseline="baseline_scenario",
                    treatment="sensitivity_scenario",
                    metric_path=path,
                    cluster_key=inferential_cluster_key,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=(seed + scenario_index * 1000 + method_index * 200 + 50 + endpoint_index),
                    relative_reference=reference,
                )

            reliability_rows = []
            for identity in sorted(baseline_identities):
                baseline_row = indexed["baseline"][identity]
                scenario_row = indexed[scenario_id][identity]
                baseline_payload = _finite_method_payload(baseline_row, method)
                scenario_payload = _finite_method_payload(scenario_row, method)

                def unusable(payload: Dict[str, object] | None) -> float:
                    if payload is None:
                        return 1.0
                    injectivity = payload.get("injectivity") or {}
                    return float(
                        not isinstance(injectivity, Mapping)
                        or not bool(injectivity.get("all_patches_globally_injective", False))
                    )

                reliability = dict(baseline_row)
                reliability["baseline_scenario"] = {"structure_unusable": unusable(baseline_payload)}
                reliability["sensitivity_scenario"] = {"structure_unusable": unusable(scenario_payload)}
                reliability_rows.append(reliability)
            reliability_comparison = paired_method_comparison(
                reliability_rows,
                baseline="baseline_scenario",
                treatment="sensitivity_scenario",
                metric_path=("structure_unusable",),
                cluster_key=inferential_cluster_key,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + scenario_index * 1000 + method_index * 200 + 90,
                binary_endpoint=True,
            )
            reliability_comparison.update(
                {
                    "analysis_role": "all_attempted_unusable_output_sensitivity",
                    "coding": (
                        "0=complete finite globally injective output; 1=incomplete, nonfinite, or noninjective output"
                    ),
                }
            )
            method_comparisons[method] = {
                "attempted_structure_count": len(baseline_identities),
                "baseline_complete_structure_count": len(complete_ids[method]["baseline"]),
                "scenario_complete_structure_count": len(complete_ids[method][scenario_id]),
                "pairwise_common_complete_structure_count": len(pairwise_common),
                "all_scenario_common_complete_structure_count": len(all_scenario_common[method]),
                "pairwise_identical_source_domain_structure_count": pairwise_same_domain_count,
                "pairwise_changed_or_unknown_source_domain_structure_count": (
                    len(pairwise_common) - pairwise_same_domain_count
                ),
                "all_scenario_identical_source_domain_structure_count": all_scenario_same_domain_count,
                "excluded_from_pairwise_efficacy_count": len(baseline_identities) - len(pairwise_common),
                "pairwise_common_complete_structure_comparisons": endpoint_comparisons,
                "all_scenario_common_complete_structure_comparisons": all_common_endpoint_comparisons,
                "continuous_estimand": (
                    "whole-pipeline endpoint change among structures with complete finite outputs in both "
                    "scenarios; source-face domains may change with a surface or topology axis, and "
                    "geometry validity remains part of the reported outcome"
                ),
                "all_attempted_unusable_output_comparison": reliability_comparison,
            }

        residue_aware_effect_stability: Dict[str, object] = {"status": "not_configured"}
        if {SENSITIVITY_STANDARD_METHOD, SENSITIVITY_RESIDUE_AWARE_METHOD}.issubset(configured_methods):
            baseline_effect_complete = (
                complete_ids[SENSITIVITY_STANDARD_METHOD]["baseline"]
                & complete_ids[SENSITIVITY_RESIDUE_AWARE_METHOD]["baseline"]
            )
            scenario_effect_complete = (
                complete_ids[SENSITIVITY_STANDARD_METHOD][scenario_id]
                & complete_ids[SENSITIVITY_RESIDUE_AWARE_METHOD][scenario_id]
            )
            pairwise_effect_common = baseline_effect_complete & scenario_effect_complete
            all_effect_common = set.intersection(
                *(
                    complete_ids[SENSITIVITY_STANDARD_METHOD][key] & complete_ids[SENSITIVITY_RESIDUE_AWARE_METHOD][key]
                    for key in ordered_scenarios
                )
            )
            pairwise_effect_same_domain_count = sum(
                bool(
                    _method_domain_signature(
                        indexed["baseline"][identity],
                        SENSITIVITY_STANDARD_METHOD,
                    )
                )
                and _method_domain_signature(
                    indexed["baseline"][identity],
                    SENSITIVITY_STANDARD_METHOD,
                )
                == _method_domain_signature(
                    indexed[scenario_id][identity],
                    SENSITIVITY_STANDARD_METHOD,
                )
                for identity in pairwise_effect_common
            )
            all_effect_same_domain_count = sum(
                bool(
                    _method_domain_signature(
                        indexed["baseline"][identity],
                        SENSITIVITY_STANDARD_METHOD,
                    )
                )
                and len(
                    {
                        _method_domain_signature(
                            indexed[key][identity],
                            SENSITIVITY_STANDARD_METHOD,
                        )
                        for key in ordered_scenarios
                    }
                )
                == 1
                for identity in all_effect_common
            )

            def effect_rows(
                identities: set[tuple[str, ...]],
                endpoint_path: Sequence[str],
                scenario_key: str = scenario_id,
            ) -> list[Dict[str, object]]:
                rows = []
                for identity in sorted(identities):
                    baseline_row = indexed["baseline"][identity]
                    scenario_row = indexed[scenario_key][identity]
                    baseline_standard = _finite_method_payload(baseline_row, SENSITIVITY_STANDARD_METHOD)
                    baseline_topoppi = _finite_method_payload(baseline_row, SENSITIVITY_RESIDUE_AWARE_METHOD)
                    scenario_standard = _finite_method_payload(scenario_row, SENSITIVITY_STANDARD_METHOD)
                    scenario_topoppi = _finite_method_payload(scenario_row, SENSITIVITY_RESIDUE_AWARE_METHOD)
                    combined = dict(baseline_row)
                    combined["sensitivity_scenario_topoppi_effect"] = {
                        "value": float(_nested(scenario_standard, endpoint_path))
                        - float(_nested(scenario_topoppi, endpoint_path))
                    }
                    combined["baseline_scenario_topoppi_effect"] = {
                        "value": float(_nested(baseline_standard, endpoint_path))
                        - float(_nested(baseline_topoppi, endpoint_path))
                    }
                    rows.append(combined)
                return rows

            pairwise_effect_comparisons = {}
            all_common_effect_comparisons = {}
            for endpoint_index, (endpoint, (path, _reference)) in enumerate(SENSITIVITY_ENDPOINTS.items()):
                pairwise_effect_comparisons[endpoint] = paired_method_comparison(
                    effect_rows(pairwise_effect_common, path),
                    baseline="sensitivity_scenario_topoppi_effect",
                    treatment="baseline_scenario_topoppi_effect",
                    metric_path=("value",),
                    cluster_key=inferential_cluster_key,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + scenario_index * 1000 + 700 + endpoint_index,
                )
                all_common_effect_comparisons[endpoint] = paired_method_comparison(
                    effect_rows(all_effect_common, path),
                    baseline="sensitivity_scenario_topoppi_effect",
                    treatment="baseline_scenario_topoppi_effect",
                    metric_path=("value",),
                    cluster_key=inferential_cluster_key,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed + scenario_index * 1000 + 750 + endpoint_index,
                )
            residue_aware_effect_stability = {
                "status": "evaluated",
                "effect_definition": (
                    "within-scenario lower-is-better benefit equals optcuts_automatic minus residue_aware_optcuts"
                ),
                "signed_difference_definition": (
                    "sensitivity-scenario TopoPPI benefit minus baseline-scenario benefit"
                ),
                "pairwise_common_complete_structure_count": len(pairwise_effect_common),
                "all_scenario_common_complete_structure_count": len(all_effect_common),
                "pairwise_identical_source_domain_structure_count": pairwise_effect_same_domain_count,
                "all_scenario_identical_source_domain_structure_count": all_effect_same_domain_count,
                "pairwise_common_complete_structure_comparisons": pairwise_effect_comparisons,
                "all_scenario_common_complete_structure_comparisons": all_common_effect_comparisons,
                "transport_estimand": (
                    "change in the within-scenario exact-domain TopoPPI benefit across complete "
                    "structures; the source-face domain may itself change between scenarios"
                ),
            }
        scenario_comparisons[scenario_id] = {
            "attempted_structure_count": len(baseline_identities),
            "methods": method_comparisons,
            "residue_aware_treatment_effect_stability": residue_aware_effect_stability,
        }
    return {
        "status": "evaluated" if len(successful_ids) == len(scenarios) else "incomplete",
        "reason": (
            None
            if len(successful_ids) == len(scenarios)
            else "one or more planned scenarios did not produce a benchmark report"
        ),
        "methods": configured_methods,
        "inferential_cluster_key": inferential_cluster_key,
        "inferential_cluster_rule": (
            "experimental-only cohorts use family_id; predicted-only cohorts use "
            "inference_family_id; mixed cohorts use analysis_split_component_id"
        ),
        "attempted_identity_rule": (
            "manifest record, coordinate and interaction checksums, selected chain pair, structure type, "
            "experimental and prediction-dependency family/partner clusters, dependency basis, split component "
            "and basis, and paired-reference identities must match exactly across every completed scenario"
        ),
        "efficacy_completion_rule": (
            "the independent method arm must cover the complete source-face domain, have finite prespecified "
            "continuous endpoints; flips and nonlocal overlap remain separate all-attempted geometry QC"
        ),
        "sensitivity_estimand": (
            "whole-pipeline response to each frozen setting change; comparisons pair structure "
            "identities, allow source-face domains to differ, and report domain-signature agreement"
        ),
        "multiplicity_policy": (
            "sensitivity contrasts are robustness estimates; interpret effect sizes and confidence "
            "intervals, with signed-rank p-values descriptive only"
        ),
        "attempted_structure_count": len(baseline_identities),
        "successful_scenario_count": len(successful_ids),
        "planned_scenario_count": len(scenarios),
        "all_scenario_common_complete_structure_counts": {
            method: len(values) for method, values in all_scenario_common.items()
        },
        "scenario_complete_structure_counts": {
            method: {scenario_id: len(complete_ids[method][scenario_id]) for scenario_id in ordered_scenarios}
            for method in configured_methods
        },
        "scenario_vs_baseline": scenario_comparisons,
    }


def _sensitivity_summary_row(
    scenario: SensitivityScenario,
    report: Mapping[str, object],
) -> Dict[str, object]:
    summary = dict(report.get("summary", {}))
    treatment = "optcuts_automatic"
    report_path = Path(scenario.config.output_root) / scenario.config.report_filename
    return {
        "scenario_id": scenario.scenario_id,
        "changes": scenario.changes,
        "status": "ok",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path) if report_path.is_file() else None,
        "attempted_structure_count": summary.get("attempted_structure_count"),
        "complete_comparison_structure_count": summary.get("complete_comparison_structure_count"),
        "distortion_mean": _nested(summary, ("method_distributions", treatment, "distortion_mean", "mean")),
        "angle_distortion_mean_rad": _nested(
            summary,
            ("method_distributions", treatment, "angle_distortion_mean", "mean"),
        ),
        "area_distortion_mean": _nested(
            summary,
            ("method_distributions", treatment, "area_distortion_mean", "mean"),
        ),
        "flip_rate": _nested(summary, ("method_distributions", treatment, "flip_rate", "mean")),
        "all_attempted_failure_rate": _nested(
            summary,
            ("method_execution_all_attempted", treatment, "all_attempted_failure_rate"),
        ),
        "face_retention_ratio": _nested(
            summary,
            (
                "topology_biological_retention_pooled_component_incidence",
                "face",
                "overall",
                "retention_ratio",
            ),
        ),
        "residue_retention_ratio": _nested(
            summary,
            (
                "topology_biological_retention_pooled_component_incidence",
                "residue",
                "overall",
                "retention_ratio",
            ),
        ),
        "geometric_contact_pair_retention_ratio": _nested(
            summary,
            (
                "topology_biological_retention_pooled_component_incidence",
                "geometric_contact_pair",
                "overall",
                "retention_ratio",
            ),
        ),
        "end_to_end_wall_sec": _nested(summary, ("isolated_end_to_end_wall_sec", "median")),
        "peak_rss_mb": _nested(summary, ("isolated_peak_rss_mb", "median")),
        "multi_patch_atlas_overlap_ratio": _nested(summary, ("multi_patch_atlas", "overlap_ratio", "mean")),
        "multi_patch_atlas_utilization": _nested(summary, ("multi_patch_atlas", "utilization", "mean")),
    }


def _write_sensitivity_csv(records: Sequence[Mapping[str, object]], path: Path) -> None:
    scalar_fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key != "changes" and not isinstance(value, (dict, list, tuple))
        }
    )
    fields = ["scenario_id", "changes_json", *[field for field in scalar_fields if field != "scenario_id"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "changes_json": json.dumps(record.get("changes", {}), sort_keys=True, separators=(",", ":")),
                }
            )


__all__ = [
    "SENSITIVITY_AXES",
    "SensitivityBenchmarkRunner",
    "SensitivityScenario",
    "build_sensitivity_scenarios",
    "load_sensitivity_plan",
    "normalize_sensitivity_axes",
    "write_sensitivity_plan",
]
