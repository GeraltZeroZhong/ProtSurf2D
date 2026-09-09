#!/usr/bin/env python3
"""Freeze publication quality, performance and sensitivity benchmark configs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
from pathlib import Path

from topoppi.benchmarking.coordinate_audit import require_validated_coordinate_audit
from topoppi.benchmarking.manifest_metadata import INFERENCE_DEPENDENCY_FIELDS
from topoppi.config import benchmark_config_from_dict
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

EXCLUDED_STATUSES = {"0", "false", "no", "exclude", "excluded", "skip", "skipped"}


def require_manifest_split(path: Path, expected_split: str) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        included = [
            row
            for row in reader
            if str(row.get("include") or row.get("status") or "included").strip().lower() not in EXCLUDED_STATUSES
        ]
    if not included:
        raise ValueError(f"Manifest has no included rows: {path}")
    observed = {str(row.get("analysis_split") or "").strip().lower() for row in included}
    if observed != {expected_split}:
        raise ValueError(
            f"Manifest {path} must contain only included {expected_split} rows; observed {sorted(observed)}."
        )
    return len(included)


def included_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if str(row.get("include") or row.get("status") or "included").strip().lower() not in EXCLUDED_STATUSES
    ]


def validate_paired_protocol_manifests(
    experimental_manifest: Path,
    predicted_manifests: list[Path],
) -> dict[str, int]:
    experimental_rows = included_manifest_rows(experimental_manifest)
    experimental_ids = [str(row.get("record_id") or "").strip() for row in experimental_rows]
    if any(not value for value in experimental_ids) or len(set(experimental_ids)) != len(experimental_ids):
        raise ValueError("Experimental protocol manifest requires unique, non-empty record_id values.")
    experimental_by_id = dict(zip(experimental_ids, experimental_rows, strict=True))
    identity_fields = (
        "cluster_id",
        "family_id",
        "sequence_cluster_a",
        "sequence_cluster_b",
        "analysis_split",
        "analysis_split_component_id",
        "analysis_split_basis",
        "experimental_methods_json",
        "experimental_method_group",
        "experimental_method_contains_nmr",
        *INFERENCE_DEPENDENCY_FIELDS,
    )
    counts: dict[str, int] = {}
    for manifest in predicted_manifests:
        rows = included_manifest_rows(manifest)
        predicted_ids = [str(row.get("record_id") or "").strip() for row in rows]
        reference_ids = [str(row.get("paired_experimental_record_id") or "").strip() for row in rows]
        pair_ids = [str(row.get("paired_record_id") or "").strip() for row in rows]
        if any(not value for value in predicted_ids + reference_ids + pair_ids):
            raise ValueError(f"Predicted protocol manifest has empty pairing identifiers: {manifest}")
        if any(len(values) != len(set(values)) for values in (predicted_ids, reference_ids, pair_ids)):
            raise ValueError(f"Predicted protocol manifest pairing identifiers are not one-to-one: {manifest}")
        for row, reference_id in zip(rows, reference_ids, strict=True):
            paired_reference_id = str(row.get("paired_reference_record_id") or "").strip()
            if paired_reference_id != reference_id:
                raise ValueError(f"Predicted protocol row has inconsistent experimental reference IDs: {manifest}")
            reference = experimental_by_id.get(reference_id)
            if reference is None:
                raise ValueError(f"Predicted protocol row references an absent experiment: {reference_id}")
            for field in identity_fields:
                if not str(row.get(field) or "").strip() or str(row.get(field)) != str(reference.get(field)):
                    raise ValueError(f"Predicted/experimental protocol metadata differs for {reference_id}: {field}")
        counts[str(manifest.resolve())] = len(rows)
    return counts


def recompute_selected_weight(selection: dict[str, object]) -> tuple[float, str, float, float]:
    records = [record for record in selection.get("records", []) if isinstance(record, dict)]
    eligible = [record for record in records if bool(record.get("eligible", False))]
    if not eligible:
        raise ValueError("Weight-selection artifact has no eligible candidate.")
    for record in records:
        report = Path(str(record.get("report") or ""))
        expected_sha256 = str(record.get("report_sha256") or "").lower()
        if not report.is_file() or sha256_file(report).lower() != expected_sha256:
            raise ValueError(f"Weight-selection report is missing or has a checksum mismatch: {report}")
    try:
        best = max(eligible, key=lambda record: float(record["primary"]["mean_cluster_difference"]))
        best_effect = float(best["primary"]["mean_cluster_difference"])
        best_se = float(best["primary"].get("primary_standard_error_difference", float("nan")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Weight-selection artifact has malformed primary effects.") from exc
    if not math.isfinite(best_effect):
        raise ValueError("Weight-selection artifact has a non-finite best primary effect.")
    if not math.isfinite(best_se) or best_se < 0.0:
        raise ValueError("Weight-selection artifact lacks the required dependence-aware standard error.")
    tolerance = best_se
    if best_effect <= 0.0:
        selected = min(eligible, key=lambda record: float(record["weight"]))
        branch = "nonpositive_effect_smallest_eligible_fallback"
    else:
        near_best = [
            record
            for record in eligible
            if float(record["primary"]["mean_cluster_difference"]) >= best_effect - tolerance
        ]
        selected = min(near_best, key=lambda record: float(record["weight"]))
        branch = "positive_effect_one_standard_error_rule"
    weight = float(selected["weight"])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("Recomputed TopoPPI weight is not finite and positive.")
    return weight, branch, best_effect, tolerance


def quality_config(
    *,
    input_folder: Path,
    manifest: Path,
    output_root: Path,
    binary: Path,
    binary_sha256: str,
    weight: float,
    workers: int,
    expected_git_commit: str,
    coordinate_audit: Path,
    coordinate_audit_sha256: str,
    worker_timeout_sec: float,
    optcuts_timeout_sec: float,
    include_topology_ablation: bool = True,
) -> dict[str, object]:
    return {
        "input_folder": str(input_folder.resolve()),
        "output_root": str(output_root.resolve()),
        "chain_selection_mode": "manifest",
        "manifest_path": str(manifest.resolve()),
        "formal_mode": True,
        "expected_git_commit": expected_git_commit,
        "coordinate_audit_path": str(coordinate_audit.resolve()),
        "expected_coordinate_audit_sha256": coordinate_audit_sha256,
        "benchmark_purpose": "quality",
        "optcuts_variants": [
            "optcuts_automatic",
            "optcuts_lscm_initialized",
            "residue_aware_optcuts",
        ],
        "repetitions": 1,
        "warmup_runs": 0,
        "max_workers": workers,
        "threads_per_worker": 1,
        "contact_distance_angstrom": 6.0,
        "random_seed": 20260817,
        "bootstrap_iterations": 5000,
        "include_topology_ablation": include_topology_ablation,
        "show_tqdm": False,
        "resume": True,
        "checkpoint_interval_structures": 32,
        "worker_timeout_sec": float(worker_timeout_sec),
        "worker_memory_limit_mb": 2048.0,
        "min_chain_residues": 4,
        "per_face_sample_size_per_patch": 128,
        "surface": {
            "grid_resolution": 1.0,
            "sigma": 1.0,
            "level": 0.1,
            "padding": 10.0,
            "max_voxels": 40_000_000,
            "adaptive_resolution": False,
            "max_adaptive_resolution": 1.25,
            "smoothing_iterations": 0,
        },
        "topology": {
            "distance_cutoff": 4.0,
            "min_patch_area_angstrom2": 10.0,
            "min_patch_vertices": 3,
        },
        "parameterization": {
            "method": "auto",
            "min_vertices": 3,
            "min_face_area": 1e-12,
            "min_angle_deg": 1e-6,
            "max_aspect_ratio": 1e12,
            "uv_epsilon": 1e-6,
            "expected_euler_characteristic": 1,
            "expected_boundary_loops": 1,
            "slim_iterations": 20,
            "slim_boundary_constraint_weight": 1e11,
        },
        "optcuts": {
            "optcuts_bin": str(binary.resolve()),
            "expected_binary_sha256": binary_sha256,
            "use_input_uv": False,
            "optcuts_mode": 100,
            "optcuts_lambda_init": 0.999,
            "optcuts_distortion_bound": 4.1,
            "optcuts_use_bijectivity": True,
            "optcuts_initial_cut_option": 0,
            "residue_fragmentation_weight": weight,
            "patch_gap": 0.08,
            "timeout_sec": float(optcuts_timeout_sec),
        },
    }


def clean_git_revision(repo_root: Path) -> str:
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
    if not commit:
        raise RuntimeError("Publication protocol preparation requires a committed Git revision.")
    if status:
        raise RuntimeError("Publication protocol preparation requires a clean Git worktree.")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create publication benchmark configs from audited cohorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experimental-input",
        required=True,
        type=Path,
        help="Directory containing experimental benchmark structures.",
    )
    parser.add_argument(
        "--experimental-manifest",
        required=True,
        type=Path,
        help="Experimental test manifest CSV.",
    )
    parser.add_argument(
        "--afdb-input",
        required=True,
        type=Path,
        help="Directory containing AFDB monomer-replacement structures.",
    )
    parser.add_argument(
        "--afdb-manifest",
        required=True,
        type=Path,
        help="AFDB monomer-replacement test manifest CSV.",
    )
    parser.add_argument("--afdb-dimer-input", type=Path, help="Directory containing AFDB dimer structures.")
    parser.add_argument("--afdb-dimer-manifest", type=Path, help="AFDB dimer test manifest CSV.")
    parser.add_argument("--performance-input", type=Path, help="Input directory for the performance cohort.")
    parser.add_argument("--performance-manifest", type=Path, help="Performance-cohort manifest CSV.")
    parser.add_argument("--sensitivity-input", type=Path, help="Input directory for the sensitivity cohort.")
    parser.add_argument("--sensitivity-manifest", type=Path, help="Sensitivity-cohort manifest CSV.")
    parser.add_argument("--binary", required=True, type=Path, help="OptCuts executable used by every config.")
    parser.add_argument(
        "--weight-selection",
        required=True,
        type=Path,
        help="Frozen residue-aware weight-selection JSON.",
    )
    parser.add_argument(
        "--coordinate-audit",
        required=True,
        type=Path,
        help="Coordinate audit JSON covering the supplied manifests.",
    )
    parser.add_argument("--config-dir", required=True, type=Path, help="Directory for generated config files.")
    parser.add_argument("--result-root", required=True, type=Path, help="Root directory assigned to benchmark outputs.")
    parser.add_argument("--workers", type=int, default=8, help="Benchmark worker processes written to configs.")
    parser.add_argument(
        "--optcuts-timeout",
        type=float,
        default=300.0,
        help="OptCuts timeout in seconds for quality runs.",
    )
    parser.add_argument(
        "--operational-optcuts-timeout",
        type=float,
        default=540.0,
        help="OptCuts timeout in seconds for performance and sensitivity runs.",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=960.0,
        help="Per-structure worker timeout in seconds.",
    )
    args = parser.parse_args()

    required = (
        args.experimental_input,
        args.experimental_manifest,
        args.afdb_input,
        args.afdb_manifest,
        args.binary,
        args.weight_selection,
        args.coordinate_audit,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if not math.isfinite(args.optcuts_timeout) or args.optcuts_timeout <= 0.0:
        raise ValueError("optcuts-timeout must be finite and positive.")
    if not math.isfinite(args.operational_optcuts_timeout) or args.operational_optcuts_timeout <= 0.0:
        raise ValueError("operational-optcuts-timeout must be finite and positive.")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0.0:
        raise ValueError("worker-timeout must be finite and positive.")
    maximum_comparative_arm_count = 3
    if args.worker_timeout <= maximum_comparative_arm_count * args.optcuts_timeout:
        raise ValueError(
            "Set worker-timeout above three times optcuts-timeout to allow all three OptCuts comparison runs."
        )
    if args.worker_timeout <= args.operational_optcuts_timeout:
        raise ValueError("worker-timeout must exceed the operational OptCuts method-arm budget.")
    for label in ("performance", "sensitivity"):
        input_folder = getattr(args, f"{label}_input")
        manifest = getattr(args, f"{label}_manifest")
        if (input_folder is None) != (manifest is None):
            raise ValueError(f"{label} input and manifest must be provided together.")
        if input_folder is not None and (not input_folder.is_dir() or not manifest.is_file()):
            raise FileNotFoundError(f"Invalid {label} input or manifest.")
    if (args.afdb_dimer_input is None) != (args.afdb_dimer_manifest is None):
        raise ValueError("AFDB dimer input and manifest must be provided together.")
    if args.afdb_dimer_input is not None and (
        not args.afdb_dimer_input.is_dir() or not args.afdb_dimer_manifest.is_file()
    ):
        raise FileNotFoundError("Invalid AFDB dimer input or manifest.")

    repo_root = Path(__file__).resolve().parents[2]
    git_revision = clean_git_revision(repo_root)
    selection = json.loads(args.weight_selection.read_text(encoding="utf-8"))
    if int(selection.get("schema_version", 0)) != 2 or selection.get("status") != "selected":
        raise ValueError("Weight-selection artifact has no frozen selection.")
    if selection.get("git_worktree_dirty") is not False or str(selection.get("git_commit") or "") != git_revision:
        raise ValueError("Weight-selection artifact was not produced from the current clean Git revision.")
    weight = float(selection["selected_weight"])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("Selected TopoPPI weight must be finite and positive.")

    protocol_manifests = [args.experimental_manifest, args.afdb_manifest]
    protocol_manifests.extend(
        manifest
        for manifest in (
            args.afdb_dimer_manifest,
            args.performance_manifest,
            args.sensitivity_manifest,
        )
        if manifest is not None
    )
    for manifest in protocol_manifests:
        require_manifest_split(manifest, "test")
    predicted_quality_manifests = [args.afdb_manifest]
    if args.afdb_dimer_manifest is not None:
        predicted_quality_manifests.append(args.afdb_dimer_manifest)
    paired_protocol_counts = validate_paired_protocol_manifests(
        args.experimental_manifest,
        predicted_quality_manifests,
    )
    coordinate_audit_sha256, coordinate_audit_validations = require_validated_coordinate_audit(
        args.coordinate_audit,
        protocol_manifests,
    )
    binary_sha256 = sha256_file(args.binary)
    selection_protocol_path = Path(str(selection.get("protocol") or ""))
    expected_selection_protocol_sha256 = str(selection.get("protocol_sha256") or "").lower()
    if (
        not selection_protocol_path.is_file()
        or sha256_file(selection_protocol_path).lower() != expected_selection_protocol_sha256
    ):
        raise ValueError("Weight-selection protocol is missing or has a checksum mismatch.")
    selection_protocol = json.loads(selection_protocol_path.read_text(encoding="utf-8"))
    if str(selection_protocol.get("git_commit") or "") != git_revision:
        raise ValueError("Weight-selection protocol targets a different Git revision.")
    if str(selection_protocol.get("binary_sha256") or "").lower() != binary_sha256.lower():
        raise ValueError("Weight-selection protocol targets a different OptCuts binary.")
    selection_audit_path = Path(str(selection_protocol.get("coordinate_audit") or ""))
    selection_audit_sha256 = str(selection_protocol.get("coordinate_audit_sha256") or "").lower()
    selection_manifest_path = Path(str(selection_protocol.get("input_manifest") or ""))
    require_manifest_split(selection_manifest_path, "development")
    actual_selection_audit_sha256, _selection_audit_validations = require_validated_coordinate_audit(
        selection_audit_path,
        [selection_manifest_path],
    )
    if actual_selection_audit_sha256.lower() != selection_audit_sha256:
        raise ValueError("Weight-selection coordinate audit is missing or has a checksum mismatch.")
    recomputed_weight, recomputed_branch, recomputed_effect, recomputed_tolerance = recompute_selected_weight(selection)
    if (
        recomputed_weight != weight
        or str(selection.get("selection_branch") or "") != recomputed_branch
        or float(selection.get("best_observed_primary_improvement", float("nan"))) != recomputed_effect
        or float(selection.get("one_standard_error_tolerance", float("nan"))) != recomputed_tolerance
    ):
        raise ValueError("Weight-selection conclusion does not match the frozen selection rule.")
    selected_records = [
        record
        for record in selection.get("records", [])
        if isinstance(record, dict) and float(record.get("weight", float("nan"))) == weight
    ]
    if len(selected_records) != 1 or not bool(selected_records[0].get("eligible", False)):
        raise ValueError("Selected TopoPPI weight is not a unique eligible protocol candidate.")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[str, dict[str, object]]] = [
        (
            "experimental_quality",
            quality_config(
                input_folder=args.experimental_input,
                manifest=args.experimental_manifest,
                output_root=args.result_root / "experimental_quality",
                binary=args.binary,
                binary_sha256=binary_sha256,
                weight=weight,
                workers=args.workers,
                expected_git_commit=git_revision,
                coordinate_audit=args.coordinate_audit,
                coordinate_audit_sha256=coordinate_audit_sha256,
                worker_timeout_sec=args.worker_timeout,
                optcuts_timeout_sec=args.optcuts_timeout,
                include_topology_ablation=False,
            ),
        ),
        (
            "afdb_monomer_quality",
            quality_config(
                input_folder=args.afdb_input,
                manifest=args.afdb_manifest,
                output_root=args.result_root / "afdb_monomer_quality",
                binary=args.binary,
                binary_sha256=binary_sha256,
                weight=weight,
                workers=args.workers,
                expected_git_commit=git_revision,
                coordinate_audit=args.coordinate_audit,
                coordinate_audit_sha256=coordinate_audit_sha256,
                worker_timeout_sec=args.worker_timeout,
                optcuts_timeout_sec=args.optcuts_timeout,
                include_topology_ablation=False,
            ),
        ),
    ]
    topology_ablation = quality_config(
        input_folder=args.experimental_input,
        manifest=args.experimental_manifest,
        output_root=args.result_root / "experimental_topology_ablation",
        binary=args.binary,
        binary_sha256=binary_sha256,
        weight=weight,
        workers=args.workers,
        expected_git_commit=git_revision,
        coordinate_audit=args.coordinate_audit,
        coordinate_audit_sha256=coordinate_audit_sha256,
        worker_timeout_sec=args.worker_timeout,
        optcuts_timeout_sec=args.optcuts_timeout,
        include_topology_ablation=True,
    )
    topology_ablation["optcuts_variants"] = ["optcuts_automatic"]
    topology_ablation["optcuts"]["residue_fragmentation_weight"] = 0.0
    configs.append(("experimental_topology_ablation", topology_ablation))
    if args.afdb_dimer_input is not None:
        configs.append(
            (
                "afdb_dimer_quality",
                quality_config(
                    input_folder=args.afdb_dimer_input,
                    manifest=args.afdb_dimer_manifest,
                    output_root=args.result_root / "afdb_dimer_quality",
                    binary=args.binary,
                    binary_sha256=binary_sha256,
                    weight=weight,
                    workers=args.workers,
                    expected_git_commit=git_revision,
                    coordinate_audit=args.coordinate_audit,
                    coordinate_audit_sha256=coordinate_audit_sha256,
                    worker_timeout_sec=args.worker_timeout,
                    optcuts_timeout_sec=args.optcuts_timeout,
                    include_topology_ablation=False,
                ),
            )
        )
    if args.performance_input is not None:
        performance = quality_config(
            input_folder=args.performance_input,
            manifest=args.performance_manifest,
            output_root=args.result_root / "experimental_performance_optcuts",
            binary=args.binary,
            binary_sha256=binary_sha256,
            weight=weight,
            workers=1,
            expected_git_commit=git_revision,
            coordinate_audit=args.coordinate_audit,
            coordinate_audit_sha256=coordinate_audit_sha256,
            worker_timeout_sec=args.worker_timeout,
            optcuts_timeout_sec=args.operational_optcuts_timeout,
            include_topology_ablation=False,
        )
        performance.update(
            {
                "benchmark_purpose": "performance",
                "execution_profile": "operational_optcuts",
                "optcuts_variants": ["optcuts_automatic"],
                "include_topology_ablation": False,
                "repetitions": 3,
                "warmup_runs": 1,
                "max_workers": 1,
                "threads_per_worker": 4,
                "checkpoint_interval_structures": 1,
            }
        )
        performance["optcuts"]["residue_fragmentation_weight"] = 0.0
        configs.append(("experimental_performance_optcuts", performance))
        residue_aware_performance = copy.deepcopy(performance)
        residue_aware_performance["output_root"] = str(
            (args.result_root / "experimental_performance_topoppi").resolve()
        )
        residue_aware_performance["optcuts_variants"] = ["residue_aware_optcuts"]
        residue_aware_performance["optcuts"]["residue_fragmentation_weight"] = weight
        configs.append(("experimental_performance_topoppi", residue_aware_performance))
    if args.sensitivity_input is not None:
        configs.append(
            (
                "experimental_sensitivity_base",
                quality_config(
                    input_folder=args.sensitivity_input,
                    manifest=args.sensitivity_manifest,
                    output_root=args.result_root / "experimental_sensitivity_base",
                    binary=args.binary,
                    binary_sha256=binary_sha256,
                    weight=weight,
                    workers=args.workers,
                    expected_git_commit=git_revision,
                    coordinate_audit=args.coordinate_audit,
                    coordinate_audit_sha256=coordinate_audit_sha256,
                    worker_timeout_sec=args.worker_timeout,
                    optcuts_timeout_sec=args.optcuts_timeout,
                    include_topology_ablation=False,
                ),
            )
        )

    records = []
    for name, config in configs:
        benchmark_config_from_dict(config).validate()
        path = args.config_dir / f"{name}.json"
        dump_json_atomic(config, path)
        records.append(
            {
                "name": name,
                "config": str(path.resolve()),
                "config_sha256": sha256_file(path),
                "input_manifest": config["manifest_path"],
                "input_manifest_sha256": sha256_file(str(config["manifest_path"])),
                "output_root": config["output_root"],
            }
        )

    protocol = {
        "schema_version": 1,
        "purpose": "frozen publication benchmark protocol",
        "git_commit": git_revision,
        "git_worktree_dirty": False,
        "binary": str(args.binary.resolve()),
        "binary_sha256": binary_sha256,
        "coordinate_audit": str(args.coordinate_audit.resolve()),
        "coordinate_audit_sha256": coordinate_audit_sha256,
        "coordinate_audit_manifest_validations": coordinate_audit_validations,
        "weight_selection": str(args.weight_selection.resolve()),
        "weight_selection_sha256": sha256_file(args.weight_selection),
        "selected_residue_fragmentation_weight": weight,
        "quality_worker_count": args.workers,
        "comparative_optcuts_method_arm_timeout_sec": float(args.optcuts_timeout),
        "optcuts_method_arm_timeout_rule": (
            "one shared external-solver wall-time budget per configured OptCuts method and structure"
        ),
        "operational_optcuts_method_arm_timeout_sec": float(args.operational_optcuts_timeout),
        "worker_supervisor_timeout_sec": float(args.worker_timeout),
        "worker_supervisor_timeout_role": (
            "structure-level crash safety limit, distinct from the equal per-method OptCuts budgets"
        ),
        "prespecified_experimental_method_sensitivity": {
            "primary_analysis": "all frozen experimental structures, including NMR entries",
            "sensitivity_analysis": (
                "repeat the complete aggregate after excluding every row whose frozen official "
                "experimental_method_contains_nmr field is true"
            ),
            "role": "sensitivity only; it does not replace or select the primary analysis",
        },
        "paired_protocol_manifest_record_counts": paired_protocol_counts,
        "configs": records,
    }
    protocol_path = args.config_dir / "publication_benchmark_protocol.json"
    dump_json_atomic(protocol, protocol_path)
    print(protocol_path)


if __name__ == "__main__":
    main()
