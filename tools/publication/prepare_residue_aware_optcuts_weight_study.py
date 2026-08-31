#!/usr/bin/env python3
"""Freeze development-only TopoPPI weight-study configurations."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from topoppi.benchmarking.coordinate_audit import require_validated_coordinate_audit
from topoppi.config import benchmark_config_from_dict
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic


def validated_weights(values: list[float]) -> list[float]:
    weights = sorted(set(values))
    if not weights or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("All TopoPPI weights must be finite and positive.")
    return weights


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
        raise RuntimeError("Weight-study preparation requires a committed Git revision.")
    if status:
        raise RuntimeError("Weight-study preparation requires a clean Git worktree.")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create development-only configs for residue-aware OptCuts weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        type=Path,
        help="Directory containing development-cohort structures.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Development-cohort manifest CSV.")
    parser.add_argument("--binary", required=True, type=Path, help="OptCuts executable used by every config.")
    parser.add_argument(
        "--coordinate-audit",
        required=True,
        type=Path,
        help="Coordinate audit JSON covering the development manifest.",
    )
    parser.add_argument("--config-dir", required=True, type=Path, help="Directory for generated config files.")
    parser.add_argument("--result-root", required=True, type=Path, help="Root directory assigned to study outputs.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 20.0, 80.0],
        help="Candidate residue-footprint weights.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Benchmark worker processes written to configs.")
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Native threads assigned to each benchmark worker.",
    )
    parser.add_argument("--optcuts-timeout", type=float, default=300.0, help="OptCuts timeout in seconds.")
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=660.0,
        help="Per-structure worker timeout in seconds.",
    )
    args = parser.parse_args()

    weights = validated_weights(args.weights)
    if not math.isfinite(args.optcuts_timeout) or args.optcuts_timeout <= 0.0:
        raise ValueError("optcuts-timeout must be finite and positive.")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0.0:
        raise ValueError("worker-timeout must be finite and positive.")
    if args.worker_timeout <= 2.0 * args.optcuts_timeout:
        raise ValueError(
            "worker-timeout must exceed both matched OptCuts method-arm budgets so a later arm "
            "cannot be censored by construction."
        )
    for path in (args.input_folder, args.manifest, args.binary, args.coordinate_audit):
        if not path.exists():
            raise FileNotFoundError(path)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    git_revision = clean_git_revision(repo_root)
    binary_sha256 = sha256_file(str(args.binary))
    coordinate_audit_sha256, coordinate_audit_validations = require_validated_coordinate_audit(
        args.coordinate_audit,
        [args.manifest],
    )
    configs = []
    for weight in weights:
        label = format(weight, "g").replace(".", "p")
        config = {
            "input_folder": str(args.input_folder.resolve()),
            "output_root": str((args.result_root / f"alpha_{label}").resolve()),
            "chain_selection_mode": "manifest",
            "manifest_path": str(args.manifest.resolve()),
            "formal_mode": True,
            "expected_git_commit": git_revision,
            "coordinate_audit_path": str(args.coordinate_audit.resolve()),
            "expected_coordinate_audit_sha256": coordinate_audit_sha256,
            "benchmark_purpose": "quality",
            "optcuts_variants": [
                "optcuts_automatic",
                "residue_aware_optcuts",
            ],
            "repetitions": 1,
            "warmup_runs": 0,
            "max_workers": args.workers,
            "threads_per_worker": args.threads_per_worker,
            "contact_distance_angstrom": 6.0,
            "random_seed": 20260817,
            "bootstrap_iterations": 2000,
            "include_topology_ablation": False,
            "show_tqdm": False,
            "resume": True,
            "checkpoint_interval_structures": 32,
            "worker_timeout_sec": args.worker_timeout,
            "worker_memory_limit_mb": 2048.0,
            "min_chain_residues": 4,
            "surface": {
                "grid_resolution": 1.0,
                "sigma": 1.0,
                "level": 0.1,
                "padding": 10.0,
                "max_voxels": 40_000_000,
                "adaptive_resolution": False,
                "max_adaptive_resolution": 1.0,
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
                "optcuts_bin": str(args.binary.resolve()),
                "expected_binary_sha256": binary_sha256,
                "use_input_uv": False,
                "optcuts_mode": 100,
                "optcuts_lambda_init": 0.999,
                "optcuts_distortion_bound": 4.1,
                "optcuts_use_bijectivity": True,
                "optcuts_initial_cut_option": 0,
                "residue_fragmentation_weight": weight,
                "patch_gap": 0.08,
                "timeout_sec": args.optcuts_timeout,
            },
        }
        benchmark_config_from_dict(config).validate()
        config_path = args.config_dir / f"alpha_{label}.json"
        dump_json_atomic(config, config_path)
        configs.append(
            {
                "weight": weight,
                "config": str(config_path.resolve()),
                "config_sha256": sha256_file(str(config_path)),
                "output_root": config["output_root"],
            }
        )

    protocol = {
        "schema_version": 1,
        "purpose": "development-only selection of TopoPPI residue-fragmentation weight",
        "git_commit": git_revision,
        "git_worktree_dirty": False,
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256_file(str(args.manifest)),
        "binary": str(args.binary.resolve()),
        "binary_sha256": binary_sha256,
        "coordinate_audit": str(args.coordinate_audit.resolve()),
        "coordinate_audit_sha256": coordinate_audit_sha256,
        "coordinate_audit_manifest_validations": coordinate_audit_validations,
        "candidate_weights": weights,
        "optcuts_method_arm_timeout_sec": float(args.optcuts_timeout),
        "optcuts_method_arm_timeout_rule": (
            "one shared external-solver wall-time budget per configured OptCuts method and structure"
        ),
        "worker_supervisor_timeout_sec": float(args.worker_timeout),
        "worker_supervisor_timeout_role": (
            "structure-level crash safety limit, distinct from the equal per-method OptCuts budgets"
        ),
        "selection_rule": {
            "primary_endpoint": (
                "family-mean paired improvement in objective-weighted residue fragmentation "
                "for automatic-initialization TopoPPI versus matched standard OptCuts"
            ),
            "efficacy_domain": (
                "intersection of structures with complete exact-pair outputs for every candidate weight; "
                "the standard OptCuts baseline and source-face domain must match exactly across runs"
            ),
            "eligibility": {
                "pair_completion_rate": (
                    "both exact source-face arms are domain-complete with finite metrics and no more "
                    "than 0.02 below the best candidate rate"
                ),
                "globally_injective_usable_pair_rate": (
                    "both exact-pair arms usable and no more than 0.02 below the best candidate rate"
                ),
                "family_mean_symmetric_dirichlet_excess_over_identity_relative_improvement_minimum": -0.02,
                "family_mean_normalized_seam_relative_improvement_minimum": -0.05,
            },
            "one_standard_error_rule": (
                "among eligible candidates within one dependence-aware standard error of the "
                "largest primary improvement, select the smallest weight"
            ),
            "fallback": (
                "if no candidate is eligible, stop without selecting; if the best eligible "
                "primary improvement is non-positive, select the smallest eligible weight "
                "and flag absent development efficacy"
            ),
        },
        "configs": configs,
    }
    protocol_path = args.config_dir / "weight_selection_protocol.json"
    dump_json_atomic(protocol, protocol_path)
    print(protocol_path)


if __name__ == "__main__":
    main()
