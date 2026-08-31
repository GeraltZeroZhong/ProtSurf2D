#!/usr/bin/env python3
"""Recompute coordinate-derived manifest metadata before a formal benchmark."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa

from topoppi.benchmarking.coordinate_audit import AUDIT_PROTOCOL, AUDIT_SCHEMA_VERSION
from topoppi.benchmarking.manifest_metadata import (
    FORMAL_STRUCTURE_TYPES,
    INFERENCE_DEPENDENCY_BASIS,
    PREDICTED_STRUCTURE_TYPES,
    inference_family_id,
    observed_sequence_metadata,
    plddt_confidence_stratum,
)
from topoppi.file_utils import read_csv_rows, sha256_file
from topoppi.io.io_loader import PDBLoader
from topoppi.io.pdb_records import residue_plddt_values
from topoppi.json_utils import dump_json_atomic

ALLOWED_ANALYSIS_SPLITS = {"development", "test", "exploratory"}
DEPENDENCY_FIELDS = (
    "analysis_split",
    "analysis_split_component_id",
    "analysis_split_basis",
    "cluster_id",
    "family_id",
    "sequence_cluster_a",
    "sequence_cluster_b",
)
REFERENCE_FIELDS = (
    "record_id",
    "structure_path",
    "input_sha256",
    "chain_a",
    "chain_b",
    "sequence_a",
    "sequence_b",
    "sequence_a_sha256",
    "sequence_b_sha256",
    "chain_a_residue_count",
    "chain_b_residue_count",
)


def chain_sequence(loader: PDBLoader, chain_id: str) -> str:
    return "".join(
        protein_letters_3to1_extended[residue.get_resname()]
        for residue in loader.model[chain_id]
        if is_aa(residue, standard=False) and residue.get_resname() in protein_letters_3to1_extended
    )


def validate_plddt_manifest_metadata(
    row: dict[str, str],
    atom_confidence: np.ndarray,
    residue_confidence: np.ndarray,
) -> str:
    """Bind any declared derived pLDDT metadata to the current coordinates."""

    metric = str(row.get("confidence_metric") or "").strip().lower().replace("-", "_")
    if metric not in {"plddt", "plddt_bfactor", "b_factor_plddt"}:
        raise ValueError("predicted confidence_metric does not declare pLDDT B factors")
    expected = {
        "crop_plddt_atom_count": int(len(atom_confidence)),
        "crop_plddt_atom_minimum": float(np.min(atom_confidence)),
        "crop_plddt_atom_mean": float(np.mean(atom_confidence)),
        "crop_plddt_atom_maximum": float(np.max(atom_confidence)),
        "crop_plddt_residue_count": int(len(residue_confidence)),
        "crop_plddt_residue_minimum": float(np.min(residue_confidence)),
        "crop_plddt_residue_mean": float(np.mean(residue_confidence)),
        "crop_plddt_residue_maximum": float(np.max(residue_confidence)),
    }
    declared_fields = []
    for field, expected_value in expected.items():
        if field not in row:
            continue
        declared_fields.append(field)
        raw = row.get(field)
        raw_value = "" if raw is None else str(raw).strip()
        if not raw_value:
            raise ValueError(f"{field} is declared but empty")
        try:
            actual_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field} is not numeric") from exc
        if not math.isfinite(actual_value):
            raise ValueError(f"{field} is not finite")
        if field.endswith("_count"):
            if not actual_value.is_integer() or int(actual_value) != expected_value:
                raise ValueError(f"{field} differs from the coordinate file")
        elif not math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"{field} differs from the coordinate file")

    expected_stratum = plddt_confidence_stratum(float(np.mean(residue_confidence)))
    if "confidence_stratum" in row:
        declared_fields.append("confidence_stratum")
        if str(row.get("confidence_stratum") or "").strip() != expected_stratum:
            raise ValueError("confidence_stratum differs from residue-mean coordinate pLDDT")
    if "confidence_threshold" in row:
        declared_fields.append("confidence_threshold")
        try:
            raw_threshold = row.get("confidence_threshold")
            threshold = float("" if raw_threshold is None else str(raw_threshold).strip())
        except ValueError as exc:
            raise ValueError("confidence_threshold is not numeric") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
            raise ValueError("confidence_threshold must be finite and lie in [0, 100]")
    return "validated_declared_confidence_metadata" if declared_fields else "computed_only"


def audit_coordinate(task: tuple[str, dict[str, str]]) -> dict[str, object]:
    manifest_name, row = task
    record_id = str(row.get("record_id") or "")
    try:
        structure_type = str(row.get("structure_type") or "").strip().lower()
        if structure_type not in FORMAL_STRUCTURE_TYPES:
            raise ValueError(
                "unsupported or missing structure_type; expected one of " + ", ".join(sorted(FORMAL_STRUCTURE_TYPES))
            )
        path = Path(row["structure_path"])
        actual_sha256 = sha256_file(path)
        if actual_sha256 != row["input_sha256"]:
            raise ValueError("input_sha256 differs from the coordinate file")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            loader = PDBLoader(path)
        chain_a = row["chain_a"]
        chain_b = row["chain_b"]
        sequence = observed_sequence_metadata(
            chain_sequence(loader, chain_a),
            chain_sequence(loader, chain_b),
        )
        for field, value in sequence.items():
            if field == "sequence_semantics" and not row.get(field):
                continue
            if str(row.get(field) or "") != str(value):
                raise ValueError(f"{field} differs from the coordinate file")
        coordinates_a, atoms_a = loader.get_chain_data(chain_a)
        coordinates_b, atoms_b = loader.get_chain_data(chain_b)
        coordinates = np.concatenate((coordinates_a, coordinates_b), axis=0)
        if not len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError("selected protein heavy-atom coordinates are empty or non-finite")

        result: dict[str, object] = {
            "manifest": manifest_name,
            "record_id": record_id,
            "status": "passed",
            "input_sha256": actual_sha256,
            "chain_a": chain_a,
            "chain_b": chain_b,
            "structure_type": structure_type,
            "sequence_a_sha256": sequence["sequence_a_sha256"],
            "sequence_b_sha256": sequence["sequence_b_sha256"],
            "chain_a_residue_count": len(sequence["sequence_a"]),
            "chain_b_residue_count": len(sequence["sequence_b"]),
            "heavy_atom_count": len(coordinates),
        }
        if structure_type in PREDICTED_STRUCTURE_TYPES:
            if row.get("sequence_semantics") != "observed_residues_in_current_coordinate_input":
                raise ValueError("predicted sequence_semantics is missing or invalid")
            atom_confidence = np.asarray(
                [float(atom.get_bfactor()) for atom in (*atoms_a, *atoms_b)],
                dtype=np.float64,
            )
            if (
                not len(atom_confidence)
                or not np.isfinite(atom_confidence).all()
                or np.any((atom_confidence < 0.0) | (atom_confidence > 100.0))
            ):
                raise ValueError("predicted heavy atoms lack finite 0-100 pLDDT B factors")
            residue_confidence = np.asarray(
                residue_plddt_values([*atoms_a, *atoms_b]),
                dtype=np.float64,
            )
            manifest_validation = validate_plddt_manifest_metadata(
                row,
                atom_confidence,
                residue_confidence,
            )
            result.update(
                {
                    "plddt_summary_unit": "residue",
                    "plddt_manifest_validation": manifest_validation,
                    "plddt_confidence_stratum": plddt_confidence_stratum(float(np.mean(residue_confidence))),
                    "plddt_atom_count": int(len(atom_confidence)),
                    "plddt_atom_minimum": float(np.min(atom_confidence)),
                    "plddt_residue_count": int(len(residue_confidence)),
                    "plddt_minimum": float(np.min(residue_confidence)),
                    "plddt_mean": float(np.mean(residue_confidence)),
                    "plddt_maximum": float(np.max(residue_confidence)),
                    "plddt_atom_weighted_mean": float(np.mean(atom_confidence)),
                    "plddt_atom_maximum": float(np.max(atom_confidence)),
                }
            )
        return result
    except Exception as exc:
        return {
            "manifest": manifest_name,
            "record_id": record_id,
            "status": "failed",
            "reason": str(exc),
        }


def unique_rows(rows: list[dict[str, str]], manifest_name: str) -> None:
    for field in ("record_id", "pdb"):
        values = [str(row.get(field) or "") for row in rows]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"{manifest_name} requires unique, non-empty {field} values.")


def validate_dependency_splits(reference_rows: list[dict[str, str]]) -> dict[str, int]:
    dependency_fields = ("cluster_id", "family_id", "sequence_cluster_a", "sequence_cluster_b")
    split_by_group: dict[str, dict[str, set[str]]] = {field: defaultdict(set) for field in dependency_fields}
    component_by_group: dict[str, dict[str, set[str]]] = {field: defaultdict(set) for field in dependency_fields}
    sequence_cluster_splits: dict[str, set[str]] = defaultdict(set)
    pair_to_family: dict[tuple[str, str], set[str]] = defaultdict(set)
    family_to_pair: dict[str, set[tuple[str, str]]] = defaultdict(set)
    split_components: dict[str, set[str]] = defaultdict(set)
    split_bases: set[str] = set()
    for row in reference_rows:
        split = str(row.get("analysis_split") or "").strip()
        if split not in ALLOWED_ANALYSIS_SPLITS:
            raise ValueError(
                "Reference manifest analysis_split must be one of: " + ", ".join(sorted(ALLOWED_ANALYSIS_SPLITS))
            )
        values = {field: str(row.get(field) or "").strip() for field in dependency_fields}
        component = str(row.get("analysis_split_component_id") or "").strip()
        split_basis = str(row.get("analysis_split_basis") or "").strip()
        missing = [field for field, value in values.items() if not value]
        if not component:
            missing.append("analysis_split_component_id")
        if not split_basis:
            missing.append("analysis_split_basis")
        if missing:
            raise ValueError("Reference manifest has missing dependency metadata: " + ", ".join(missing))
        split_bases.add(split_basis)
        split_components[component].add(split)
        for field, value in values.items():
            split_by_group[field][value].add(split)
            component_by_group[field][value].add(component)
        for field in ("sequence_cluster_a", "sequence_cluster_b"):
            sequence_cluster_splits[values[field]].add(split)
        pair = tuple(sorted((values["sequence_cluster_a"], values["sequence_cluster_b"])))
        pair_to_family[pair].add(values["family_id"])
        family_to_pair[values["family_id"]].add(pair)
    leaking: dict[str, list[str]] = {
        field: [group for group, splits in groups.items() if len(splits) != 1]
        for field, groups in split_by_group.items()
    }
    leaking["sequence_cluster"] = [group for group, splits in sequence_cluster_splits.items() if len(splits) != 1]
    if any(leaking.values()):
        detail = "; ".join(f"{field}={len(groups)}" for field, groups in leaking.items() if groups)
        raise ValueError(f"Dependency groups span analysis splits: {detail}")
    fragmented_dependencies = {
        field: [group for group, components in groups.items() if len(components) != 1]
        for field, groups in component_by_group.items()
    }
    if any(fragmented_dependencies.values()):
        detail = "; ".join(f"{field}={len(groups)}" for field, groups in fragmented_dependencies.items() if groups)
        raise ValueError(f"Dependency groups map to multiple split components: {detail}")
    leaking_components = [component for component, splits in split_components.items() if len(splits) != 1]
    if leaking_components:
        raise ValueError(f"Analysis-split dependency components span splits: {len(leaking_components)}")
    if len(split_bases) != 1:
        raise ValueError("Reference manifest must use one non-empty analysis_split_basis.")
    if any(len(families) != 1 for families in pair_to_family.values()) or any(
        len(pairs) != 1 for pairs in family_to_pair.values()
    ):
        raise ValueError("Reference family_id values are inconsistent with unordered sequence-cluster pairs.")
    return {
        "homology_component_count": len(split_by_group["cluster_id"]),
        "interface_family_count": len(split_by_group["family_id"]),
        "sequence_cluster_count": len(sequence_cluster_splits),
        "analysis_split_component_count": len(split_components),
    }


def validate_paired_references(
    predicted_rows: list[dict[str, str]],
    reference_by_id: dict[str, dict[str, str]],
    manifest_name: str,
) -> None:
    pair_ids = [str(row.get("paired_record_id") or "") for row in predicted_rows]
    if any(not value for value in pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"{manifest_name} requires unique, non-empty paired_record_id values.")
    for row in predicted_rows:
        reference_id = str(row.get("paired_reference_record_id") or "")
        reference = reference_by_id.get(reference_id)
        if reference is None:
            raise ValueError(f"{manifest_name} references an unknown experimental record: {reference_id}")
        for field in REFERENCE_FIELDS:
            if str(row.get(f"paired_reference_{field}") or "") != str(reference.get(field) or ""):
                raise ValueError(f"{manifest_name} {row['record_id']} has inconsistent paired_reference_{field}.")
        if row.get("paired_experimental_record_id") != reference_id:
            raise ValueError(f"{manifest_name} {row['record_id']} has inconsistent paired_experimental_record_id.")
        if row.get("sequence_cluster_reference") != "paired_experimental_observed_sequences":
            raise ValueError(f"{manifest_name} {row['record_id']} lacks sequence-cluster provenance.")
        for field in DEPENDENCY_FIELDS:
            if row.get(field) != reference.get(field):
                raise ValueError(f"{manifest_name} {row['record_id']} changed reference {field}.")
        inference_a = str(row.get("inference_sequence_cluster_a") or "").strip()
        inference_b = str(row.get("inference_sequence_cluster_b") or "").strip()
        if not inference_a or not inference_b:
            raise ValueError(f"{manifest_name} {row['record_id']} lacks inference dependencies.")
        if row.get("inference_family_id") != inference_family_id(inference_a, inference_b):
            raise ValueError(f"{manifest_name} {row['record_id']} has an invalid inference family.")
        if row.get("inference_dependency_basis") != INFERENCE_DEPENDENCY_BASIS:
            raise ValueError(f"{manifest_name} {row['record_id']} lacks dependency provenance.")


def validate_reference_subset(
    rows: list[dict[str, str]],
    reference_by_id: dict[str, dict[str, str]],
    manifest_name: str,
) -> None:
    fields = ("pdb", *REFERENCE_FIELDS[1:], *DEPENDENCY_FIELDS)
    for row in rows:
        record_id = str(row.get("record_id") or "")
        reference = reference_by_id.get(record_id)
        if reference is None:
            raise ValueError(f"{manifest_name} contains an unknown reference record: {record_id}")
        for field in fields:
            if str(row.get(field) or "") != str(reference.get(field) or ""):
                raise ValueError(f"{manifest_name} {record_id} changed frozen field {field}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute coordinate-derived metadata across benchmark manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-manifest",
        required=True,
        type=Path,
        help="Primary experimental reference manifest CSV.",
    )
    parser.add_argument(
        "--additional-reference-manifest",
        action="append",
        default=[],
        type=Path,
        help="Additional experimental reference manifest; repeat for multiple cohorts.",
    )
    parser.add_argument(
        "--predicted-manifest",
        action="append",
        default=[],
        type=Path,
        help="Predicted-structure manifest paired to the reference; repeat as needed.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path for the coordinate audit JSON.")
    parser.add_argument("--workers", type=int, default=8, help="Coordinate-audit worker processes.")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")

    reference_manifest = (
        "experimental_reference",
        args.reference_manifest,
        read_csv_rows(args.reference_manifest),
    )
    additional_manifests = [
        (f"additional_reference_{index}_{path.stem}", path, read_csv_rows(path))
        for index, path in enumerate(args.additional_reference_manifest, start=1)
    ]
    predicted_manifests = [
        (f"predicted_{index}_{path.stem}", path, read_csv_rows(path))
        for index, path in enumerate(args.predicted_manifest, start=1)
    ]
    manifests = [reference_manifest, *additional_manifests, *predicted_manifests]
    for name, _path, rows in manifests:
        unique_rows(rows, name)
    reference_rows = manifests[0][2]
    reference_by_id = {row["record_id"]: row for row in reference_rows}
    dependency_counts = validate_dependency_splits(reference_rows)
    for name, _path, rows in additional_manifests:
        validate_reference_subset(rows, reference_by_id, name)
    for name, _path, rows in predicted_manifests:
        validate_paired_references(rows, reference_by_id, name)

    inference_cluster_splits: dict[str, set[str]] = defaultdict(set)
    inference_family_splits: dict[str, set[str]] = defaultdict(set)
    inference_cluster_components: dict[str, set[str]] = defaultdict(set)
    inference_family_components: dict[str, set[str]] = defaultdict(set)
    for _name, _path, rows in predicted_manifests:
        for row in rows:
            split = row["analysis_split"]
            component = row["analysis_split_component_id"]
            inference_cluster_splits[row["inference_sequence_cluster_a"]].add(split)
            inference_cluster_splits[row["inference_sequence_cluster_b"]].add(split)
            inference_family_splits[row["inference_family_id"]].add(split)
            inference_cluster_components[row["inference_sequence_cluster_a"]].add(component)
            inference_cluster_components[row["inference_sequence_cluster_b"]].add(component)
            inference_family_components[row["inference_family_id"]].add(component)
    leaking_inference_clusters = [cluster for cluster, splits in inference_cluster_splits.items() if len(splits) != 1]
    leaking_inference_families = [family for family, splits in inference_family_splits.items() if len(splits) != 1]
    fragmented_inference_clusters = [
        cluster for cluster, components in inference_cluster_components.items() if len(components) != 1
    ]
    fragmented_inference_families = [
        family for family, components in inference_family_components.items() if len(components) != 1
    ]
    if (
        leaking_inference_clusters
        or leaking_inference_families
        or fragmented_inference_clusters
        or fragmented_inference_families
    ):
        raise ValueError(
            "Prediction dependency groups span analysis splits or split components: "
            f"split_clusters={len(leaking_inference_clusters)}; "
            f"split_families={len(leaking_inference_families)}; "
            f"component_clusters={len(fragmented_inference_clusters)}; "
            f"component_families={len(fragmented_inference_families)}"
        )

    tasks = [(name, row) for name, _path, rows in manifests for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(audit_coordinate, tasks, chunksize=8), start=1):
            results.append(result)
            if index % 250 == 0 or index == len(tasks):
                print(f"Manifest coordinate audit: {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if result["status"] != "passed"]
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_protocol": AUDIT_PROTOCOL,
        "status": "failed" if failures else "passed",
        "coordinate_record_count": len(results),
        "coordinate_failure_count": len(failures),
        "coordinate_failure_examples": failures[:20],
        "coordinate_results": results,
        "manifest_records": {name: len(rows) for name, _path, rows in manifests},
        "manifest_sha256": {name: sha256_file(path) for name, path, _rows in manifests},
        "analysis_split_counts": dict(sorted(Counter(row["analysis_split"] for row in reference_rows).items())),
        **dependency_counts,
        "predicted_structure_type_counts": dict(
            sorted(Counter(row["structure_type"] for _name, _path, rows in predicted_manifests for row in rows).items())
        ),
        "predicted_geometry_stratum_counts": dict(
            sorted(
                Counter(
                    row["paired_geometry_stratum"] for _name, _path, rows in predicted_manifests for row in rows
                ).items()
            )
        ),
        "predicted_inference_sequence_cluster_count": len(
            {
                row[field]
                for _name, _path, rows in predicted_manifests
                for row in rows
                for field in ("inference_sequence_cluster_a", "inference_sequence_cluster_b")
            }
        ),
        "predicted_inference_family_count": len(
            {row["inference_family_id"] for _name, _path, rows in predicted_manifests for row in rows}
        ),
        "predicted_inference_sequence_cluster_cross_split_count": 0,
        "predicted_inference_family_cross_split_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(summary, args.output)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "coordinate_results"},
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
