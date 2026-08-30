"""Validation of checksum-frozen coordinate-audit artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from topoppi.benchmarking.manifest_metadata import (
    FORMAL_STRUCTURE_TYPES,
    PREDICTED_STRUCTURE_TYPES,
    plddt_confidence_stratum,
)
from topoppi.file_utils import sha256_file

AUDIT_PROTOCOL = "manifest-coordinate-audit-v4"
AUDIT_SCHEMA_VERSION = 4
EVIDENCE_BINDING_FIELDS = (
    "input_sha256",
    "chain_a",
    "chain_b",
    "structure_type",
    "sequence_a_sha256",
    "sequence_b_sha256",
    "chain_a_residue_count",
    "chain_b_residue_count",
)


def _invalid(path: Path, actual_sha256: str, reason: str) -> dict[str, object]:
    return {
        "status": "invalid",
        "path": str(path),
        "actual_sha256": actual_sha256,
        "reason": reason,
    }


def _binding_value(field: str, value: object) -> str:
    normalized = str(value or "").strip()
    if field in {"input_sha256", "structure_type"}:
        return normalized.lower()
    return normalized


def _valid_coordinate_evidence(result: dict[str, object]) -> bool:
    try:
        residue_a = int(result["chain_a_residue_count"])
        residue_b = int(result["chain_b_residue_count"])
        heavy_atoms = int(result["heavy_atom_count"])
    except (KeyError, TypeError, ValueError):
        return False
    if min(residue_a, residue_b, heavy_atoms) <= 0:
        return False
    structure_type = _binding_value("structure_type", result.get("structure_type"))
    if structure_type not in FORMAL_STRUCTURE_TYPES:
        return False
    if structure_type not in PREDICTED_STRUCTURE_TYPES:
        return True
    try:
        plddt_atom_count = int(result["plddt_atom_count"])
        plddt_count = int(result["plddt_residue_count"])
        plddt_values = [
            float(result["plddt_minimum"]),
            float(result["plddt_mean"]),
            float(result["plddt_maximum"]),
            float(result["plddt_atom_weighted_mean"]),
            float(result["plddt_atom_minimum"]),
            float(result["plddt_atom_maximum"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in plddt_values):
        return False
    try:
        expected_stratum = plddt_confidence_stratum(plddt_values[1])
    except ValueError:
        return False
    return (
        result.get("plddt_summary_unit") == "residue"
        and result.get("plddt_manifest_validation") in {"validated_declared_confidence_metadata", "computed_only"}
        and result.get("plddt_confidence_stratum") == expected_stratum
        and plddt_atom_count == heavy_atoms
        and plddt_count == residue_a + residue_b
        and plddt_values[0] <= plddt_values[1] <= plddt_values[2]
        and plddt_values[4] <= plddt_values[3] <= plddt_values[5]
    )


def validate_coordinate_audit(
    audit_path: str | Path,
    expected_audit_sha256: str,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Validate one audit and its exact manifest binding without recomputation."""

    path_value = str(audit_path).strip()
    expected_sha256 = str(expected_audit_sha256).strip().lower()
    if not path_value or not expected_sha256:
        return {"status": "missing", "reason": "coordinate-audit path/checksum is not configured"}
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return {"status": "missing", "path": str(path), "reason": "coordinate-audit file does not exist"}
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256:
        return {
            "status": "checksum_mismatch",
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "reason": "coordinate-audit checksum differs from the frozen configuration",
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "reason": f"coordinate-audit JSON cannot be read: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "reason": "coordinate-audit root is not an object",
        }
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION or payload.get("audit_protocol") != AUDIT_PROTOCOL:
        return _invalid(path, actual_sha256, f"coordinate-audit protocol is not {AUDIT_PROTOCOL}")
    if str(payload.get("status") or "").strip().lower() != "passed":
        return {
            "status": "failed",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "reason": "coordinate-audit status is not passed",
        }
    audited_manifests = payload.get("manifest_sha256")
    if not isinstance(audited_manifests, dict):
        return _invalid(path, actual_sha256, "coordinate-audit manifest checksums are missing")
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        return {
            "status": "missing",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "reason": "current manifest file does not exist",
        }
    manifest_sha256 = sha256_file(manifest)
    matching_labels = sorted(
        str(label)
        for label, digest in audited_manifests.items()
        if str(digest).strip().lower() == manifest_sha256.lower()
    )
    if not matching_labels:
        return {
            "status": "manifest_mismatch",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": "current manifest checksum is absent from the coordinate audit",
        }
    manifest_records = payload.get("manifest_records")
    if not isinstance(manifest_records, dict):
        return _invalid(path, actual_sha256, "coordinate-audit per-manifest record counts are missing")
    try:
        declared_counts = {str(label): int(count) for label, count in manifest_records.items()}
        coordinate_record_count = int(payload.get("coordinate_record_count"))
        failure_count = int(payload.get("coordinate_failure_count"))
    except (TypeError, ValueError):
        declared_counts = {}
        coordinate_record_count = -1
        failure_count = -1
    if (
        not declared_counts
        or any(count < 0 for count in declared_counts.values())
        or coordinate_record_count != sum(declared_counts.values())
        or failure_count < 0
    ):
        return _invalid(path, actual_sha256, "coordinate-audit record counts are inconsistent")
    if set(audited_manifests) != set(declared_counts):
        return _invalid(path, actual_sha256, "coordinate-audit manifest labels are inconsistent")

    coordinate_results = payload.get("coordinate_results")
    if not isinstance(coordinate_results, list) or len(coordinate_results) != coordinate_record_count:
        return _invalid(path, actual_sha256, "coordinate-audit per-record evidence is missing or incomplete")
    evidence_by_manifest: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    actual_failure_count = 0
    for result in coordinate_results:
        if not isinstance(result, dict):
            return _invalid(path, actual_sha256, "coordinate-audit evidence contains a non-object record")
        label = str(result.get("manifest") or "").strip()
        record_id = str(result.get("record_id") or "").strip()
        if label not in declared_counts or not record_id:
            return _invalid(path, actual_sha256, "coordinate-audit evidence has an unknown manifest or empty record_id")
        if record_id in evidence_by_manifest[label]:
            return _invalid(path, actual_sha256, "coordinate-audit evidence contains duplicate record_id values")
        evidence_by_manifest[label][record_id] = result
        actual_failure_count += result.get("status") != "passed"
    if any(len(evidence_by_manifest[label]) != count for label, count in declared_counts.items()):
        return _invalid(path, actual_sha256, "coordinate-audit per-manifest evidence counts are inconsistent")
    if actual_failure_count != failure_count:
        return _invalid(path, actual_sha256, "coordinate-audit declared and observed failure counts differ")
    if failure_count != 0:
        return {
            "status": "failed",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "reason": "coordinate-audit failure count is not zero",
        }
    if any(not _valid_coordinate_evidence(result) for result in coordinate_results):
        return _invalid(path, actual_sha256, "coordinate-audit contains incomplete or invalid passing evidence")

    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            current_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return _invalid(path, actual_sha256, f"current manifest records cannot be read: {exc}")
    current_manifest_record_count = len(current_rows)
    if any(declared_counts.get(label) != current_manifest_record_count for label in matching_labels):
        return {
            "status": "manifest_mismatch",
            "path": str(path),
            "actual_sha256": actual_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": "coordinate-audit record count does not match the current manifest",
        }
    current_record_ids = [str(row.get("record_id") or "").strip() for row in current_rows]
    if any(not record_id for record_id in current_record_ids) or len(set(current_record_ids)) != len(
        current_record_ids
    ):
        return _invalid(path, actual_sha256, "current manifest requires unique, non-empty record_id values")
    current_by_id = dict(zip(current_record_ids, current_rows, strict=True))
    for label in matching_labels:
        evidence_by_id = evidence_by_manifest[label]
        if set(evidence_by_id) != set(current_by_id):
            return {
                "status": "manifest_mismatch",
                "path": str(path),
                "actual_sha256": actual_sha256,
                "manifest_sha256": manifest_sha256,
                "reason": "coordinate-audit record IDs do not match the current manifest",
            }
        for record_id, row in current_by_id.items():
            evidence = evidence_by_id[record_id]
            for field in EVIDENCE_BINDING_FIELDS:
                if _binding_value(field, evidence.get(field)) != _binding_value(field, row.get(field)):
                    return {
                        "status": "manifest_mismatch",
                        "path": str(path),
                        "actual_sha256": actual_sha256,
                        "manifest_sha256": manifest_sha256,
                        "reason": f"coordinate evidence for {record_id} differs in {field}",
                    }
    return {
        "status": "validated",
        "path": str(path),
        "actual_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_labels": matching_labels,
        "coordinate_record_count": coordinate_record_count,
        "current_manifest_record_count": current_manifest_record_count,
        "coordinate_failure_count": 0,
    }


def require_validated_coordinate_audit(
    audit_path: str | Path,
    manifest_paths: list[str | Path],
) -> tuple[str, dict[str, dict[str, object]]]:
    """Validate an existing audit against every manifest before protocol freeze."""

    audit = Path(audit_path).expanduser().resolve()
    if not audit.is_file():
        raise FileNotFoundError(audit)
    checksum = sha256_file(audit)
    validations = {
        str(Path(manifest).expanduser().resolve()): validate_coordinate_audit(
            audit,
            checksum,
            manifest,
        )
        for manifest in manifest_paths
    }
    failures = {manifest: result for manifest, result in validations.items() if result.get("status") != "validated"}
    if failures:
        raise ValueError(
            "Coordinate audit is not valid for every protocol manifest: " + json.dumps(failures, sort_keys=True)
        )
    return checksum, validations


__all__ = [
    "AUDIT_PROTOCOL",
    "require_validated_coordinate_audit",
    "validate_coordinate_audit",
]
