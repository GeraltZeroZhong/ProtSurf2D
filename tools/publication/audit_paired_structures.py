#!/usr/bin/env python3
"""Audit paired structures and determine interface-benchmark eligibility."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial import cKDTree

from topoppi.benchmarking.manifest_metadata import observed_sequence_metadata
from topoppi.file_utils import read_csv_rows, sha256_file, write_csv_atomic
from topoppi.io.pdb_records import selected_protein_atom_lines
from topoppi.json_utils import dump_json_atomic
from topoppi.rigid_geometry import require_stable_rigid_fit_geometry
from topoppi.sequence_alignment import align_protein_sequences

AMINO_ACID_LETTERS = protein_letters_3to1_extended


@dataclass(frozen=True)
class Residue:
    name: str
    number: int
    insertion_code: str
    atoms: np.ndarray
    ca: np.ndarray | None


def _element(line: str) -> str:
    declared = line[76:78].strip().upper() if len(line) >= 78 else ""
    if declared:
        return declared
    atom_name = line[12:16].strip().upper().lstrip("0123456789")
    return atom_name[0] if atom_name else ""


def parse_chains(path: Path) -> dict[str, list[Residue]]:
    records: dict[str, dict[tuple[int, str], dict[str, object]]] = {}
    for line in selected_protein_atom_lines(path):
        residue_name = line[17:20].strip()
        if residue_name not in AMINO_ACID_LETTERS or _element(line) in {"H", "D"}:
            continue
        chain_id = line[21]
        key = (int(line[22:26]), line[26])
        residue = records.setdefault(chain_id, {}).setdefault(
            key,
            {"name": residue_name, "coordinates": [], "ca": None},
        )
        if residue["name"] != residue_name:
            raise ValueError(f"Microheterogeneous residue in {path}: {chain_id}{key}")
        coordinate = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        residue["coordinates"].append(coordinate)
        if line[12:16].strip() == "CA":
            residue["ca"] = coordinate
    return {
        chain_id: [
            Residue(
                name=str(record["name"]),
                number=number,
                insertion_code=insertion,
                atoms=np.asarray(record["coordinates"], dtype=np.float64),
                ca=record["ca"],
            )
            for (number, insertion), record in residue_map.items()
        ]
        for chain_id, residue_map in records.items()
    }


def align_residues(
    reference: list[Residue],
    mobile: list[Residue],
) -> tuple[dict[int, int], dict[str, float | int]]:
    reference_sequence = "".join(AMINO_ACID_LETTERS[residue.name] for residue in reference)
    mobile_sequence = "".join(AMINO_ACID_LETTERS[residue.name] for residue in mobile)
    pairs, report = align_protein_sequences(reference_sequence, mobile_sequence)
    mobile_to_reference = {mobile_index: reference_index for reference_index, mobile_index in pairs}
    return mobile_to_reference, {
        "aligned_residue_count": report["aligned_residue_count"],
        "alignment_identity": report["alignment_identity"],
        "reference_coverage": report["reference_coverage"],
        "mobile_coverage": report["mobile_coverage"],
        "alignment_score": report["alignment_score"],
        "optimal_alignment_count": report["optimal_alignment_count"],
        "optimal_correspondence_count": report["optimal_correspondence_count"],
        "consensus_pair_count": report["consensus_pair_count"],
        "selected_pair_consensus_fraction": report["selected_pair_consensus_fraction"],
        "selected_alignment_rule": report["selected_alignment_rule"],
    }


def residue_sequence(residues: list[Residue]) -> str:
    return "".join(AMINO_ACID_LETTERS[residue.name] for residue in residues)


def validate_observed_sequence_metadata(
    row: dict[str, str],
    chain_a: list[Residue],
    chain_b: list[Residue],
    label: str,
) -> None:
    actual = observed_sequence_metadata(
        residue_sequence(chain_a),
        residue_sequence(chain_b),
    )
    for field, value in actual.items():
        if field == "sequence_semantics" and not row.get(field):
            continue
        if str(row.get(field) or "") != str(value):
            raise ValueError(f"{label} manifest {field} differs from its coordinate input.")


def validate_paired_reference_metadata(
    experimental_row: dict[str, str],
    predicted_row: dict[str, str],
) -> None:
    fields = (
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
    for field in fields:
        predicted_field = f"paired_reference_{field}"
        if str(predicted_row.get(predicted_field) or "") != str(experimental_row.get(field) or ""):
            raise ValueError(f"Predicted manifest {predicted_field} differs from the paired experimental record.")
    if predicted_row.get("paired_experimental_record_id") != experimental_row["record_id"]:
        raise ValueError("Predicted paired_experimental_record_id differs from the experimental record.")
    if predicted_row.get("sequence_cluster_reference") != "paired_experimental_observed_sequences":
        raise ValueError("Predicted sequence-cluster provenance is not declared.")
    for field in (
        "analysis_split",
        "analysis_split_component_id",
        "cluster_id",
        "family_id",
        "sequence_cluster_a",
        "sequence_cluster_b",
    ):
        if str(predicted_row.get(field) or "") != str(experimental_row.get(field) or ""):
            raise ValueError(f"Predicted {field} differs from the paired experimental record.")


def residue_contacts(
    chain_a: list[Residue],
    chain_b: list[Residue],
    cutoff: float,
) -> set[tuple[int, int]]:
    coordinates_b = np.concatenate([residue.atoms for residue in chain_b], axis=0)
    residue_indices_b = np.concatenate(
        [np.full(len(residue.atoms), index, dtype=np.int64) for index, residue in enumerate(chain_b)]
    )
    tree = cKDTree(coordinates_b)
    contacts = set()
    for index_a, residue in enumerate(chain_a):
        neighborhoods = tree.query_ball_point(residue.atoms, r=float(cutoff))
        for neighbors in neighborhoods:
            contacts.update((index_a, int(residue_indices_b[index_b])) for index_b in neighbors)
    return contacts


def cross_chain_distance_counts(
    chain_a: list[Residue],
    chain_b: list[Residue],
    clash_cutoff: float,
) -> dict[str, float | int]:
    coordinates_a = np.concatenate([residue.atoms for residue in chain_a], axis=0)
    coordinates_b = np.concatenate([residue.atoms for residue in chain_b], axis=0)
    tree_b = cKDTree(coordinates_b)
    neighborhoods = tree_b.query_ball_point(coordinates_a, r=float(clash_cutoff))
    clash_pairs = int(sum(len(neighbors) for neighbors in neighborhoods))
    involved_a = int(sum(bool(neighbors) for neighbors in neighborhoods))
    involved_b = len({index for neighbors in neighborhoods for index in neighbors})
    nearest, _ = tree_b.query(coordinates_a, k=1)
    return {
        "cross_chain_minimum_heavy_atom_distance_angstrom": float(np.min(nearest)),
        "cross_chain_clash_atom_pair_count": clash_pairs,
        "cross_chain_clash_atom_fraction": (involved_a + involved_b) / (len(coordinates_a) + len(coordinates_b)),
    }


def _mapped_contact_set(
    contacts: set[tuple[int, int]],
    map_a: dict[int, int],
    map_b: dict[int, int],
) -> set[tuple[int, int]]:
    return {(map_a[index_a], map_b[index_b]) for index_a, index_b in contacts if index_a in map_a and index_b in map_b}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def contact_comparison(
    experimental_a: list[Residue],
    experimental_b: list[Residue],
    predicted_a: list[Residue],
    predicted_b: list[Residue],
    map_a: dict[int, int],
    map_b: dict[int, int],
    cutoff: float,
) -> dict[str, float | int]:
    experimental = residue_contacts(experimental_a, experimental_b, cutoff)
    predicted_raw = residue_contacts(predicted_a, predicted_b, cutoff)
    mapped_reference_a = set(map_a.values())
    mapped_reference_b = set(map_b.values())
    comparable_experimental = {
        pair for pair in experimental if pair[0] in mapped_reference_a and pair[1] in mapped_reference_b
    }
    predicted = _mapped_contact_set(predicted_raw, map_a, map_b)
    intersection = comparable_experimental & predicted
    mapped_union = comparable_experimental | predicted
    predicted_unmapped_count = len(predicted_raw) - len(predicted)
    full_union_count = len(experimental | predicted) + predicted_unmapped_count
    interface_a_reference = {left for left, _right in experimental}
    interface_a_predicted = {left for left, _right in predicted}
    interface_b_reference = {right for _left, right in experimental}
    interface_b_predicted = {right for _left, right in predicted}
    return {
        "experimental_contact_count_total": len(experimental),
        "experimental_contact_count_comparable": len(comparable_experimental),
        "experimental_contact_mapping_coverage": _ratio(len(comparable_experimental), len(experimental)),
        "predicted_contact_count_total": len(predicted_raw),
        "predicted_contact_count_mapped": len(predicted),
        "predicted_contact_count_unmapped": predicted_unmapped_count,
        "contact_true_positive_count": len(intersection),
        "contact_recall_fnat": _ratio(len(intersection), len(experimental)),
        "contact_precision": _ratio(len(intersection), len(predicted_raw)),
        "contact_jaccard": _ratio(len(intersection), full_union_count),
        "contact_recall_mapped_domain": _ratio(len(intersection), len(comparable_experimental)),
        "contact_precision_mapped_domain": _ratio(len(intersection), len(predicted)),
        "contact_jaccard_mapped_domain": _ratio(len(intersection), len(mapped_union)),
        "interface_residue_a_count_reference": len(interface_a_reference),
        "interface_residue_b_count_reference": len(interface_b_reference),
        "interface_residue_a_mapping_coverage": _ratio(
            len(interface_a_reference & mapped_reference_a), len(interface_a_reference)
        ),
        "interface_residue_b_mapping_coverage": _ratio(
            len(interface_b_reference & mapped_reference_b), len(interface_b_reference)
        ),
        "interface_residue_a_recall": _ratio(
            len(interface_a_reference & interface_a_predicted), len(interface_a_reference)
        ),
        "interface_residue_b_recall": _ratio(
            len(interface_b_reference & interface_b_predicted), len(interface_b_reference)
        ),
    }


def interface_benchmark_eligibility(
    contact_metrics: dict[str, float | int],
    predicted_row: dict[str, str],
) -> tuple[bool, str]:
    accession_a = str(predicted_row.get("afdb_accession_a") or "").strip()
    accession_b = str(predicted_row.get("afdb_accession_b") or "").strip()
    if (
        str(predicted_row.get("structure_type") or "").strip().lower() == "afdb"
        and accession_a
        and accession_a == accession_b
    ):
        return False, "homodimer_partner_role_is_not_identifiable"
    return True, ""


def _paired_ca(
    reference: list[Residue],
    mobile: list[Residue],
    mobile_to_reference: dict[int, int],
    reference_subset: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (reference[reference_index].ca, mobile[mobile_index].ca)
        for mobile_index, reference_index in mobile_to_reference.items()
        if (reference_subset is None or reference_index in reference_subset)
        and reference[reference_index].ca is not None
        and mobile[mobile_index].ca is not None
    ]
    if not pairs:
        return np.empty((0, 3)), np.empty((0, 3))
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.float64),
        np.asarray([pair[1] for pair in pairs], dtype=np.float64),
    )


def pose_comparison(
    experimental_a: list[Residue],
    experimental_b: list[Residue],
    predicted_a: list[Residue],
    predicted_b: list[Residue],
    map_a: dict[int, int],
    map_b: dict[int, int],
    contact_cutoff: float,
) -> dict[str, float | int]:
    reference_a, mobile_a = _paired_ca(experimental_a, predicted_a, map_a)
    reference_b, mobile_b = _paired_ca(experimental_b, predicted_b, map_b)
    if len(reference_a) < 3 or not len(reference_b):
        raise ValueError("Pose audit needs at least three receptor and one ligand matched C-alpha atoms.")
    geometry_report = require_stable_rigid_fit_geometry(reference_a, mobile_a)
    superimposer = SVDSuperimposer()
    superimposer.set(reference_a, mobile_a)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    fitted_a = mobile_a @ rotation + translation
    fitted_b = mobile_b @ rotation + translation
    receptor_rmsd = float(np.sqrt(np.mean(np.sum((reference_a - fitted_a) ** 2, axis=1))))
    ligand_rmsd = float(np.sqrt(np.mean(np.sum((reference_b - fitted_b) ** 2, axis=1))))

    interface_b = {right for _left, right in residue_contacts(experimental_a, experimental_b, contact_cutoff)}
    interface_reference_b, interface_mobile_b = _paired_ca(
        experimental_b,
        predicted_b,
        map_b,
        reference_subset=interface_b,
    )
    interface_fitted_b = interface_mobile_b @ rotation + translation
    interface_reference_ca_count = sum(experimental_b[index].ca is not None for index in interface_b)
    interface_ligand_rmsd = (
        float(np.sqrt(np.mean(np.sum((interface_reference_b - interface_fitted_b) ** 2, axis=1))))
        if len(interface_reference_b)
        else float("nan")
    )
    return {
        "receptor_fit_ca_count": len(reference_a),
        "receptor_fit_ca_rmsd_angstrom": receptor_rmsd,
        "ligand_ca_count": len(reference_b),
        "ligand_ca_rmsd_after_receptor_fit_angstrom": ligand_rmsd,
        "interface_ligand_reference_ca_count": int(interface_reference_ca_count),
        "interface_ligand_ca_count": len(interface_reference_b),
        "interface_ligand_ca_mapping_coverage": _ratio(len(interface_reference_b), interface_reference_ca_count),
        "interface_ligand_ca_rmsd_after_receptor_fit_angstrom": interface_ligand_rmsd,
        **{f"receptor_fit_{key}": value for key, value in geometry_report.items()},
    }


def audit_pair(
    experimental_row: dict[str, str],
    predicted_row: dict[str, str],
    contact_cutoff: float,
    clash_cutoff: float,
) -> dict[str, object]:
    if experimental_row["paired_record_id"] != predicted_row["paired_record_id"]:
        raise ValueError("Experimental and predicted paired_record_id values differ.")
    if experimental_row["pdb_id"] != predicted_row["pdb_id"]:
        raise ValueError("Experimental and predicted PDB identifiers differ.")
    experimental_path = Path(experimental_row["structure_path"])
    predicted_path = Path(predicted_row["structure_path"])
    experimental_sha256 = sha256_file(str(experimental_path))
    predicted_sha256 = sha256_file(str(predicted_path))
    if experimental_sha256 != experimental_row["input_sha256"]:
        raise ValueError("Experimental structure checksum differs from its paired manifest.")
    if predicted_sha256 != predicted_row["input_sha256"]:
        raise ValueError("Predicted structure checksum differs from its paired manifest.")
    experimental = parse_chains(experimental_path)
    predicted = parse_chains(predicted_path)
    chain_a_experimental = experimental[experimental_row["chain_a"]]
    chain_b_experimental = experimental[experimental_row["chain_b"]]
    chain_a_predicted = predicted[predicted_row["chain_a"]]
    chain_b_predicted = predicted[predicted_row["chain_b"]]
    validate_observed_sequence_metadata(
        experimental_row,
        chain_a_experimental,
        chain_b_experimental,
        "Experimental",
    )
    validate_observed_sequence_metadata(
        predicted_row,
        chain_a_predicted,
        chain_b_predicted,
        "Predicted",
    )
    validate_paired_reference_metadata(experimental_row, predicted_row)
    map_a, alignment_a = align_residues(chain_a_experimental, chain_a_predicted)
    map_b, alignment_b = align_residues(chain_b_experimental, chain_b_predicted)
    contact_metrics = contact_comparison(
        chain_a_experimental,
        chain_b_experimental,
        chain_a_predicted,
        chain_b_predicted,
        map_a,
        map_b,
        contact_cutoff,
    )
    benchmark_eligible, benchmark_exclusion_reason = interface_benchmark_eligibility(
        contact_metrics,
        predicted_row,
    )
    return {
        "paired_record_id": predicted_row["paired_record_id"],
        "experimental_record_id": experimental_row["record_id"],
        "predicted_record_id": predicted_row["record_id"],
        "pdb_id": experimental_row["pdb_id"],
        "analysis_split": experimental_row.get("analysis_split") or "",
        "cluster_id": experimental_row.get("cluster_id") or "",
        "structure_type": predicted_row.get("structure_type") or "",
        "experimental_input_sha256": experimental_sha256,
        "predicted_input_sha256": predicted_sha256,
        **{f"alignment_a_{key}": value for key, value in alignment_a.items()},
        **{f"alignment_b_{key}": value for key, value in alignment_b.items()},
        **contact_metrics,
        **{
            f"experimental_{key}": value
            for key, value in cross_chain_distance_counts(
                chain_a_experimental, chain_b_experimental, clash_cutoff
            ).items()
        },
        **{
            f"predicted_{key}": value
            for key, value in cross_chain_distance_counts(chain_a_predicted, chain_b_predicted, clash_cutoff).items()
        },
        **pose_comparison(
            chain_a_experimental,
            chain_b_experimental,
            chain_a_predicted,
            chain_b_predicted,
            map_a,
            map_b,
            contact_cutoff,
        ),
        "contact_cutoff_angstrom": contact_cutoff,
        "clash_cutoff_angstrom": clash_cutoff,
        "benchmark_eligible": benchmark_eligible,
        "benchmark_exclusion_reason": benchmark_exclusion_reason,
        "status": "accepted",
        "reason": "",
    }


def audit_pair_task(task) -> dict[str, object]:
    experimental_row, predicted_row, contact_cutoff, clash_cutoff = task
    try:
        if experimental_row is None:
            raise ValueError("No experimental record has the predicted paired_record_id.")
        return audit_pair(experimental_row, predicted_row, contact_cutoff, clash_cutoff)
    except (KeyError, OSError, PDBException, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            "paired_record_id": predicted_row.get("paired_record_id") or "",
            "predicted_record_id": predicted_row.get("record_id") or "",
            "pdb_id": predicted_row.get("pdb_id") or "",
            "structure_type": predicted_row.get("structure_type") or "",
            "status": "excluded",
            "reason": str(exc),
        }


def finite_summary(rows: list[dict[str, object]], key: str) -> dict[str, float | int]:
    values = np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def paired_rows_by_id(
    experimental_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    experimental_ids = [str(row.get("paired_record_id") or "").strip() for row in experimental_rows]
    predicted_ids = [str(row.get("paired_record_id") or "").strip() for row in predicted_rows]
    if any(not value for value in experimental_ids + predicted_ids):
        raise ValueError("Paired manifests require non-empty paired_record_id values.")
    if len(set(experimental_ids)) != len(experimental_ids):
        raise ValueError("Experimental manifest contains duplicate paired_record_id values.")
    if len(set(predicted_ids)) != len(predicted_ids):
        raise ValueError("Predicted manifest contains duplicate paired_record_id values.")
    if set(experimental_ids) != set(predicted_ids):
        raise ValueError("Experimental and predicted manifests do not contain the same paired_record_id set.")
    return dict(zip(experimental_ids, experimental_rows, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure geometry and interface eligibility for paired structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experimental-manifest",
        required=True,
        type=Path,
        help="Experimental reference manifest CSV.",
    )
    parser.add_argument(
        "--predicted-manifest",
        required=True,
        type=Path,
        help="Predicted-structure manifest CSV paired by record ID.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for paired audit artifacts.")
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=6.0,
        help="Maximum heavy-atom distance in angstroms for an interface contact.",
    )
    parser.add_argument(
        "--clash-cutoff",
        type=float,
        default=2.0,
        help="Heavy-atom distance in angstroms counted as a clash.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Paired-audit worker processes.")
    args = parser.parse_args()
    if not args.experimental_manifest.is_file() or not args.predicted_manifest.is_file():
        raise FileNotFoundError("experimental-manifest and predicted-manifest must exist.")
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if (
        not math.isfinite(args.contact_cutoff)
        or not math.isfinite(args.clash_cutoff)
        or args.contact_cutoff <= 0.0
        or args.clash_cutoff <= 0.0
        or args.clash_cutoff >= args.contact_cutoff
    ):
        raise ValueError("Cutoffs must be finite and satisfy 0 < clash-cutoff < contact-cutoff.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experimental_rows = read_csv_rows(args.experimental_manifest)
    predicted_rows = read_csv_rows(args.predicted_manifest)
    if not experimental_rows or not predicted_rows:
        raise ValueError("Paired structure audit requires non-empty manifests.")
    experimental_by_pair = paired_rows_by_id(experimental_rows, predicted_rows)
    tasks = [
        (
            experimental_by_pair.get(predicted_row["paired_record_id"]),
            predicted_row,
            args.contact_cutoff,
            args.clash_cutoff,
        )
        for predicted_row in predicted_rows
    ]
    audit_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(audit_pair_task, tasks, chunksize=8), start=1):
            audit_rows.append(result)
            if index % 100 == 0 or index == len(predicted_rows):
                print(f"Paired structure QC: {index}/{len(predicted_rows)}", flush=True)

    audit_path = args.output_dir / "paired_structure_qc.csv"
    write_csv_atomic(audit_path, audit_rows)
    accepted = [row for row in audit_rows if row["status"] == "accepted"]
    eligible = [row for row in accepted if bool(row.get("benchmark_eligible"))]
    benchmark_exclusion_counts = Counter(
        str(row.get("benchmark_exclusion_reason") or "unspecified")
        for row in accepted
        if not bool(row.get("benchmark_eligible"))
    )
    summary = {
        "schema_version": 6,
        "manifest_sequence_validation": (
            "observed chain sequences, residue counts, and sequence hashes are recomputed from "
            "both coordinate inputs; predicted paired_reference_* metadata must match the "
            "experimental record"
        ),
        "sequence_alignment": "semiglobal with free terminal and penalized internal gaps",
        "contact_comparison": {
            "primary_domain": (
                "all experimental and predicted residue contacts; predicted contacts outside "
                "the sequence correspondence count as false positives"
            ),
            "mapped_domain_diagnostics": (
                "conditional metrics restricted to contacts whose residues have an explicit sequence correspondence"
            ),
        },
        "experimental_manifest_sha256": sha256_file(str(args.experimental_manifest)),
        "predicted_manifest_sha256": sha256_file(str(args.predicted_manifest)),
        "contact_cutoff_angstrom": args.contact_cutoff,
        "clash_cutoff_angstrom": args.clash_cutoff,
        "predicted_record_count": len(predicted_rows),
        "accepted_pair_count": len(accepted),
        "excluded_pair_count": len(audit_rows) - len(accepted),
        "benchmark_eligible_pair_count": len(eligible),
        "benchmark_ineligible_pair_count": len(accepted) - len(eligible),
        "benchmark_exclusion_reason_counts": dict(sorted(benchmark_exclusion_counts.items())),
        "benchmark_eligibility_rule": (
            "all sequence-mapped predicted chain pairs are retained even when the predicted "
            "coordinates contain no residue-residue heavy-atom contact at the frozen cutoff; "
            "exact AFDB homodimers are retained in attrition data but excluded because "
            "partner-specific chain roles are not identifiable"
        ),
        "status_counts": dict(sorted(Counter(row["status"] for row in audit_rows).items())),
        "metrics": {
            key: finite_summary(accepted, key)
            for key in (
                "contact_recall_fnat",
                "contact_precision",
                "contact_jaccard",
                "experimental_contact_mapping_coverage",
                "interface_residue_a_mapping_coverage",
                "interface_residue_b_mapping_coverage",
                "interface_ligand_ca_mapping_coverage",
                "predicted_cross_chain_clash_atom_pair_count",
                "predicted_cross_chain_clash_atom_fraction",
                "ligand_ca_rmsd_after_receptor_fit_angstrom",
                "interface_ligand_ca_rmsd_after_receptor_fit_angstrom",
            )
        },
        "audit_sha256": sha256_file(str(audit_path)),
    }
    dump_json_atomic(summary, args.output_dir / "paired_structure_qc_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
