#!/usr/bin/env python3
"""Apply the frozen development-only TopoPPI weight-selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from topoppi.benchmarking.coordinate_audit import require_validated_coordinate_audit
from topoppi.benchmarking.evidence_bundle import validate_benchmark_evidence_bundle
from topoppi.benchmarking.statistics import paired_method_comparison
from topoppi.config import benchmark_config_from_dict
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

BASELINE = "optcuts_automatic"
TREATMENT = "residue_aware_optcuts"
SELECTION_ENDPOINTS = (
    ("residue_footprint_fragmentation", "objective_weighted_fragmentation"),
    ("symmetric_dirichlet", "mean"),
    ("seam", "seam_length_3d_normalized"),
)


def normalized_config(payload: dict[str, object]) -> dict[str, object]:
    resolved = asdict(benchmark_config_from_dict(payload))
    return json.loads(json.dumps(resolved, sort_keys=True))


def current_git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, bool(status)


def attempted_identity(raw: dict[str, object]) -> tuple[object, ...]:
    chain = raw.get("chain_selection") or {}
    return (
        raw.get("manifest_record_id"),
        raw.get("pdb"),
        raw.get("input_sha256"),
        raw.get("interaction_sha256"),
        chain.get("chain_a") if isinstance(chain, dict) else None,
        chain.get("chain_b") if isinstance(chain, dict) else None,
        raw.get("structure_type"),
        raw.get("family_id"),
        raw.get("sequence_cluster_a"),
        raw.get("sequence_cluster_b"),
        raw.get("inference_family_id"),
        raw.get("inference_sequence_cluster_a"),
        raw.get("inference_sequence_cluster_b"),
        raw.get("inference_dependency_basis"),
        raw.get("analysis_split"),
        raw.get("analysis_split_component_id"),
        raw.get("analysis_split_basis"),
    )


def _nested_finite(payload: dict[str, object], path: tuple[str, ...]) -> float:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing weight-selection endpoint: {'.'.join(path)}")
        current = current[key]
    try:
        value = float(current)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric weight-selection endpoint: {'.'.join(path)}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite weight-selection endpoint: {'.'.join(path)}")
    return value


def indexed_pair_rows(
    report: dict[str, object],
) -> tuple[dict[tuple[object, ...], dict[str, object]], dict[tuple[object, ...], dict[str, object]]]:
    attempted_rows = {}
    complete_rows = {}
    for raw in report.get("files", []):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("analysis_split") or "").lower() != "development":
            raise ValueError("Weight selection received a non-development benchmark row.")
        identity = attempted_identity(raw)
        if identity in attempted_rows:
            raise ValueError(f"Duplicate attempted structure identity: {identity}")
        attempted_rows[identity] = raw
        domain = raw.get("residue_aware_comparison_domain") or {}
        pair = raw.get("residue_aware_pair_quality") or {}
        methods = pair.get("methods", {}) if isinstance(pair, dict) else {}
        if (
            not bool(domain.get("complete"))
            or not bool(pair.get("complete"))
            or not all(method in methods for method in (BASELINE, TREATMENT))
        ):
            continue
        if not all(
            str(raw.get(key) or "").strip() for key in ("family_id", "sequence_cluster_a", "sequence_cluster_b")
        ):
            raise ValueError("A complete development pair lacks family or partner-cluster provenance.")
        for method in (BASELINE, TREATMENT):
            method_payload = methods[method]
            if not isinstance(method_payload, dict):
                raise ValueError(f"Invalid exact-pair payload for {method}: {identity}")
            for path in SELECTION_ENDPOINTS:
                _nested_finite(method_payload, path)
        row = dict(raw)
        row.update(methods)
        complete_rows[identity] = row
    return attempted_rows, complete_rows


def baseline_fingerprint(row: dict[str, object]) -> str:
    pair = row.get("residue_aware_pair_quality") or {}
    baseline = row.get(BASELINE)
    if not isinstance(pair, dict) or not isinstance(baseline, dict):
        raise ValueError("Cannot fingerprint a missing exact-pair baseline.")
    payload = {
        "domain_signature": pair.get("domain_signature"),
        "domain_hashes": baseline.get("domain_hashes"),
        "endpoints": {"/".join(path): _nested_finite(baseline, path) for path in SELECTION_ENDPOINTS},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def pair_is_usable(row: dict[str, object]) -> bool:
    pair = row.get("residue_aware_pair_quality") or {}
    arms = pair.get("arms") if isinstance(pair, dict) else None
    return bool(
        isinstance(arms, dict)
        and all(
            isinstance(arms.get(method), dict) and bool(arms[method].get("usable", False))
            for method in (BASELINE, TREATMENT)
        )
    )


def pair_is_complete_finite(row: dict[str, object]) -> bool:
    pair = row.get("residue_aware_pair_quality") or {}
    arms = pair.get("arms") if isinstance(pair, dict) else None
    return bool(
        isinstance(arms, dict)
        and all(
            isinstance(arms.get(method), dict)
            and bool(arms[method].get("domain_complete", False))
            and bool(arms[method].get("metric_finite", False))
            for method in (BASELINE, TREATMENT)
        )
    )


def comparison(
    rows: list[dict[str, object]],
    path: tuple[str, ...],
    seed: int,
    *,
    relative_reference: float = 0.0,
) -> dict[str, object]:
    return paired_method_comparison(
        rows,
        baseline=BASELINE,
        treatment=TREATMENT,
        metric_path=path,
        bootstrap_iterations=5000,
        seed=seed,
        relative_reference=relative_reference,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen development rule to select an OptCuts weight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protocol",
        required=True,
        type=Path,
        help="Weight-study protocol JSON from the preparation script.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path for the selected-weight JSON.")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if int(protocol.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported weight-selection protocol schema.")
    repo_root = Path(__file__).resolve().parents[2]
    git_commit, git_dirty = current_git_state(repo_root)
    if git_dirty or git_commit != str(protocol.get("git_commit") or ""):
        raise RuntimeError("Weight selection must run from the clean revision frozen in the protocol.")
    manifest_path = Path(protocol["input_manifest"])
    binary_path = Path(protocol["binary"])
    if sha256_file(manifest_path) != str(protocol["input_manifest_sha256"]):
        raise ValueError("Development manifest checksum no longer matches the frozen protocol.")
    if sha256_file(binary_path) != str(protocol["binary_sha256"]):
        raise ValueError("OptCuts binary checksum no longer matches the frozen protocol.")
    coordinate_audit_path = Path(protocol["coordinate_audit"])
    coordinate_audit_sha256, _coordinate_audit_validations = require_validated_coordinate_audit(
        coordinate_audit_path,
        [manifest_path],
    )
    if coordinate_audit_sha256 != str(protocol["coordinate_audit_sha256"]):
        raise ValueError("Coordinate-audit checksum no longer matches the frozen protocol.")
    candidate_runs = []
    reference_identities = None
    for index, item in enumerate(protocol["configs"]):
        config_path = Path(item["config"])
        if sha256_file(config_path) != str(item["config_sha256"]):
            raise ValueError(f"Configuration checksum mismatch: {config_path}")
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if config_payload.get("formal_mode") is not True:
            raise ValueError(f"Weight-study configuration is not formal: {config_path}")
        report_path = Path(item["output_root"]) / "benchmark_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"Benchmark report is not a JSON object: {report_path}")
        validate_benchmark_evidence_bundle(report_path, report)
        if normalized_config(config_payload) != report.get("config"):
            raise ValueError(f"Benchmark report configuration mismatch: {report_path}")
        environment = (report.get("runtime") or {}).get("environment") or {}
        if (
            not isinstance(environment, dict)
            or environment.get("git_commit") != git_commit
            or environment.get("git_worktree_dirty") is not False
        ):
            raise ValueError(f"Benchmark report was not produced from the frozen clean revision: {report_path}")
        coordinate_audit = (report.get("runtime") or {}).get("coordinate_audit") or {}
        if (
            not isinstance(coordinate_audit, dict)
            or coordinate_audit.get("status") != "validated"
            or str(coordinate_audit.get("actual_sha256") or "") != str(protocol["coordinate_audit_sha256"])
        ):
            raise ValueError(f"Benchmark report lacks the frozen passing coordinate audit: {report_path}")
        attempted_rows, complete_rows = indexed_pair_rows(report)
        identities = tuple(
            sorted(
                attempted_rows,
                key=lambda identity: tuple("" if value is None else str(value) for value in identity),
            )
        )
        if not identities:
            raise ValueError(f"Benchmark report contains no attempted structures: {report_path}")
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError("Candidate weights were not evaluated on identical structure identities.")
        configured_weight = float(report["config"]["optcuts"]["residue_fragmentation_weight"])
        if configured_weight != float(item["weight"]):
            raise ValueError(f"Weight mismatch in {report_path}")
        if not complete_rows:
            raise ValueError(f"Candidate has no complete exact TopoPPI pair rows: {report_path}")
        candidate_runs.append(
            {
                "index": index,
                "weight": configured_weight,
                "report_path": report_path,
                "attempted": len(attempted_rows),
                "complete_finite": sum(pair_is_complete_finite(row) for row in attempted_rows.values()),
                "usable": sum(pair_is_usable(row) for row in attempted_rows.values()),
                "complete_rows": complete_rows,
            }
        )

    if not candidate_runs:
        raise ValueError("Weight-selection protocol contains no candidate configurations.")
    common_identities = set.intersection(*(set(run["complete_rows"]) for run in candidate_runs))
    if not common_identities:
        raise RuntimeError("Candidate weights have no shared complete exact-pair efficacy domain.")
    ordered_common_identities = sorted(
        common_identities,
        key=lambda identity: tuple("" if value is None else str(value) for value in identity),
    )
    reference_baselines = {
        identity: baseline_fingerprint(candidate_runs[0]["complete_rows"][identity])
        for identity in ordered_common_identities
    }
    for run in candidate_runs[1:]:
        for identity in ordered_common_identities:
            if baseline_fingerprint(run["complete_rows"][identity]) != reference_baselines[identity]:
                raise ValueError(f"Matched standard OptCuts baseline changed across candidate-weight runs: {identity}")

    records = []
    for run in candidate_runs:
        index = int(run["index"])
        configured_weight = float(run["weight"])
        report_path = Path(run["report_path"])
        attempted = int(run["attempted"])
        complete_finite = int(run["complete_finite"])
        usable = int(run["usable"])
        complete_rows = run["complete_rows"]
        rows = [complete_rows[identity] for identity in ordered_common_identities]
        primary = comparison(
            rows,
            ("residue_footprint_fragmentation", "objective_weighted_fragmentation"),
            20260817 + index * 10,
        )
        symmetric_dirichlet = comparison(
            rows,
            ("symmetric_dirichlet", "mean"),
            20260818 + index * 10,
            relative_reference=2.0,
        )
        seam = comparison(
            rows,
            ("seam", "seam_length_3d_normalized"),
            20260819 + index * 10,
        )
        records.append(
            {
                "weight": configured_weight,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(str(report_path)),
                "attempted_structure_count": attempted,
                "candidate_complete_finite_pair_structure_count": complete_finite,
                "pair_completion_rate": complete_finite / attempted if attempted else float("nan"),
                "candidate_complete_exact_pair_structure_count": len(complete_rows),
                "globally_injective_usable_pair_count": usable,
                "globally_injective_usable_pair_rate": usable / attempted if attempted else float("nan"),
                "shared_complete_pair_structure_count": len(rows),
                "primary": primary,
                "symmetric_dirichlet": symmetric_dirichlet,
                "normalized_seam_length": seam,
            }
        )

    best_completion = max(float(record["pair_completion_rate"]) for record in records)
    best_usable = max(float(record["globally_injective_usable_pair_rate"]) for record in records)
    if not np.isfinite(best_completion) or not np.isfinite(best_usable):
        raise RuntimeError("Candidate completion or usable-pair rates are not finite.")
    eligible = []
    for record in records:
        checks = {
            "completion": float(record["pair_completion_rate"]) >= best_completion - 0.02,
            "global_injectivity": float(record["globally_injective_usable_pair_rate"]) >= best_usable - 0.02,
            "symmetric_dirichlet": float(record["symmetric_dirichlet"]["relative_improvement_of_cluster_means"])
            >= -0.02,
            "normalized_seam": float(record["normalized_seam_length"]["relative_improvement_of_cluster_means"])
            >= -0.05,
        }
        record["eligibility_checks"] = checks
        record["eligible"] = all(checks.values())
        if record["eligible"]:
            eligible.append(record)
    if not eligible:
        raise RuntimeError("No candidate satisfies the frozen eligibility constraints.")

    if any(
        not np.isfinite(float(record["primary"].get("mean_cluster_difference", float("nan")))) for record in eligible
    ):
        raise RuntimeError("An eligible candidate has a non-finite primary effect.")
    best = max(eligible, key=lambda record: float(record["primary"]["mean_cluster_difference"]))
    best_effect = float(best["primary"]["mean_cluster_difference"])
    best_se = float(best["primary"].get("primary_standard_error_difference", float("nan")))
    if not np.isfinite(best_se) or best_se < 0.0:
        raise RuntimeError("The dependence-aware standard error is unavailable for the best candidate.")
    tolerance = best_se
    if best_effect <= 0.0:
        selected = min(eligible, key=lambda record: float(record["weight"]))
        selection_branch = "nonpositive_effect_smallest_eligible_fallback"
    else:
        near_best = [
            record
            for record in eligible
            if float(record["primary"]["mean_cluster_difference"]) >= best_effect - tolerance
        ]
        selected = min(near_best, key=lambda record: float(record["weight"]))
        selection_branch = "positive_effect_one_standard_error_rule"
    result = {
        "schema_version": 2,
        "status": "selected",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(str(args.protocol)),
        "selected_weight": float(selected["weight"]),
        "selection_branch": selection_branch,
        "git_commit": git_commit,
        "git_worktree_dirty": False,
        "development_efficacy_observed": best_effect > 0.0,
        "best_observed_primary_improvement": best_effect,
        "one_standard_error_tolerance": tolerance,
        "one_standard_error_source": str(best["primary"].get("primary_interval_method") or "unavailable"),
        "eligible_weight_count": len(eligible),
        "efficacy_domain_rule": (
            "intersection of complete finite exact-pair identities across all candidate weights; "
            "geometry validity is an all-attempted eligibility guard, not a row filter"
        ),
        "shared_complete_pair_structure_count": len(ordered_common_identities),
        "matched_standard_baseline_rule": (
            "domain signature, domain hashes, and all selection endpoints must match exactly across weights"
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
