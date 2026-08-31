#!/usr/bin/env python3
"""Build AFDB-monomer replacement complexes in the experimental docking pose."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.SVDSuperimposer import SVDSuperimposer

from topoppi.benchmarking.manifest_metadata import (
    observed_sequence_metadata,
    paired_reference_metadata,
    plddt_confidence_stratum,
)
from topoppi.file_utils import read_csv_rows, sha256_file, write_csv_atomic
from topoppi.io.afdb_download import (
    download_pdb_cached,
    download_sidecar_path,
    project_uniprot_intervals_to_coordinates,
)
from topoppi.io.pdb_records import selected_protein_atom_lines
from topoppi.json_utils import dump_json_atomic
from topoppi.rigid_geometry import require_stable_rigid_fit_geometry
from topoppi.sequence_alignment import align_protein_sequences

AFDB_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
AMINO_ACID_LETTERS = protein_letters_3to1_extended
UPSTREAM_COMPLEX_FIELDS = frozenset(
    {
        "afdb_match_status",
        "afdb_match_reason_code",
        "afdb_match_reason",
        "afdb_model_id",
        "afdb_exact_candidate_count",
        "afdb_iptm",
        "afdb_ipsae",
        "afdb_ipsae_stratum",
        "afdb_pdockq",
        "afdb_pdockq2",
        "afdb_lis",
        "afdb_provider",
        "afdb_model_metadata_sha256",
        "afdb_complex_query_sha256_a",
        "afdb_complex_query_sha256_b",
        "afdb_complex_query_retrieved_at_utc_a",
        "afdb_complex_query_retrieved_at_utc_b",
    }
)


def file_timestamp_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def monomer_source_metadata(row: dict[str, str]) -> dict[str, str]:
    """Retain mapping evidence without relabelling an upstream complex as this model."""

    return {key: value for key, value in row.items() if key not in UPSTREAM_COMPLEX_FIELDS}


def verify_experimental_coordinate(path: Path, expected_sha256: str) -> None:
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("Experimental manifest has a malformed input_sha256.")
    actual = sha256_file(path)
    if actual.lower() != expected:
        raise ValueError(f"Experimental coordinate checksum mismatch: expected {expected}, got {actual}.")


def fetch_json_cached(url: str, path: Path, timeout: float) -> object:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(5):
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            payload: object = []
            break
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(min(float(response.headers.get("Retry-After") or 2**attempt), 30.0))
            continue
        response.raise_for_status()
        payload = response.json()
        break
    else:
        raise RuntimeError(f"Transient HTTP failure persisted for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary, path)
    return payload


def prediction_record(payload: object, accession: str) -> dict[str, object]:
    records = [record for record in payload if isinstance(record, dict)] if isinstance(payload, list) else []
    canonical_id = f"AF-{accession}-F1"
    matches = [
        record for record in records if str(record.get("uniprotAccession") or "") == accession and record.get("pdbUrl")
    ]
    canonical = [
        record
        for record in matches
        if str(record.get("entryId") or record.get("modelEntityId") or "") == canonical_id
        and str(record.get("providerId") or "") == "GDM"
    ]
    if len(canonical) != 1:
        raise ValueError(f"AFDB metadata does not expose one canonical GDM monomer PDB for {accession}")
    return canonical[0]


def parse_intervals(value: str) -> list[tuple[int, int]]:
    intervals = [(int(start), int(end)) for start, end in json.loads(value)]
    if not intervals or any(start < 1 or end < start for start, end in intervals):
        raise ValueError("UniProt crop intervals must be non-empty positive inclusive ranges.")
    return intervals


@dataclass
class ResidueAtoms:
    name: str
    number: int
    insertion_code: str
    lines: list[str]
    ca: np.ndarray | None = None


def parse_protein_chains(path: Path) -> dict[str, list[ResidueAtoms]]:
    residue_maps: dict[str, dict[tuple[int, str], ResidueAtoms]] = {}
    for line in selected_protein_atom_lines(path):
        if len(line) < 66:
            continue
        name = line[17:20].strip()
        if name not in AMINO_ACID_LETTERS:
            continue
        chain_id = line[21]
        number = int(line[22:26])
        insertion_code = line[26]
        key = (number, insertion_code)
        chain = residue_maps.setdefault(chain_id, {})
        residue = chain.get(key)
        if residue is None:
            residue = ResidueAtoms(name, number, insertion_code, [])
            chain[key] = residue
        elif residue.name != name:
            raise ValueError(f"Microheterogeneous residue is unsupported in {path}: {chain_id}{key}")
        residue.lines.append(line)
        if line[12:16].strip() == "CA":
            residue.ca = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
    return {chain_id: list(residues.values()) for chain_id, residues in residue_maps.items()}


def crop_residues(residues: list[ResidueAtoms], intervals: list[tuple[int, int]]) -> list[ResidueAtoms]:
    result = [residue for residue in residues if any(start <= residue.number <= end for start, end in intervals)]
    if not result:
        raise ValueError("UniProt mapping intervals removed every AFDB monomer residue.")
    expected_count = len({number for start, end in intervals for number in range(start, end + 1)})
    if len(result) != expected_count:
        raise ValueError("AFDB monomer PDB numbering does not cover every projected coordinate.")
    return result


def projection_manifest_metadata(report: dict[str, object], side: str) -> dict[str, object]:
    return {
        f"afdb_coordinate_projection_{key}_{side}": (
            json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
        )
        for key, value in report.items()
    }


def residue_sequence(residues: list[ResidueAtoms]) -> str:
    return "".join(AMINO_ACID_LETTERS[residue.name] for residue in residues)


def aligned_ca_pairs(
    experimental: list[ResidueAtoms],
    predicted: list[ResidueAtoms],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sequence_experimental = residue_sequence(experimental)
    sequence_predicted = residue_sequence(predicted)
    residue_pairs, sequence_report = align_protein_sequences(
        sequence_experimental,
        sequence_predicted,
    )
    pairs = []
    for left, right in residue_pairs:
        if experimental[left].ca is None or predicted[right].ca is None:
            continue
        pairs.append((experimental[left].ca, predicted[right].ca))
    if not pairs:
        raise ValueError("Sequence alignment produced no paired C-alpha atoms.")
    reference = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    mobile = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    return (
        reference,
        mobile,
        {
            "aligned_ca_count": len(pairs),
            "aligned_residue_count": sequence_report["aligned_residue_count"],
            "alignment_identity": sequence_report["alignment_identity"],
            "experimental_sequence_coverage": sequence_report["reference_coverage"],
            "predicted_sequence_coverage": sequence_report["mobile_coverage"],
            "alignment_score": sequence_report["alignment_score"],
            "experimental_ca_coverage": len(pairs) / max(1, sum(residue.ca is not None for residue in experimental)),
            "predicted_ca_coverage": len(pairs) / max(1, sum(residue.ca is not None for residue in predicted)),
            "experimental_residue_count": len(experimental),
            "predicted_crop_residue_count": len(predicted),
            "optimal_alignment_count": sequence_report["optimal_alignment_count"],
            "optimal_correspondence_count": sequence_report["optimal_correspondence_count"],
            "consensus_pair_count": sequence_report["consensus_pair_count"],
            "selected_pair_consensus_fraction": sequence_report["selected_pair_consensus_fraction"],
            "selected_alignment_rule": sequence_report["selected_alignment_rule"],
        },
    )


def superposition_transform(
    experimental: list[ResidueAtoms],
    predicted: list[ResidueAtoms],
    *,
    minimum_aligned_ca: int,
    minimum_identity: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    reference, mobile, report = aligned_ca_pairs(experimental, predicted)
    if len(reference) < minimum_aligned_ca:
        raise ValueError(f"Only {len(reference)} aligned C-alpha atoms; need {minimum_aligned_ca}.")
    if float(report["alignment_identity"]) < minimum_identity:
        raise ValueError("Experimental/AFDB crop sequence identity is below the configured threshold.")
    geometry_report = require_stable_rigid_fit_geometry(reference, mobile)
    superimposer = SVDSuperimposer()
    superimposer.set(reference, mobile)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    return (
        rotation,
        translation,
        {
            **report,
            **geometry_report,
            "alignment_ca_rmsd_angstrom": float(superimposer.get_rms()),
        },
    )


def transform_chain_lines(
    residues: list[ResidueAtoms],
    rotation: np.ndarray,
    translation: np.ndarray,
    target_chain: str,
    first_serial: int,
) -> tuple[list[str], list[float]]:
    output = []
    plddt = []
    serial = first_serial
    for residue in residues:
        for line in residue.lines:
            coordinate = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            transformed = coordinate @ rotation + translation
            output.append(
                f"{line[:6]}{serial:5d}{line[11:21]}{target_chain}{line[22:30]}"
                f"{transformed[0]:8.3f}{transformed[1]:8.3f}{transformed[2]:8.3f}{line[54:]}"
            )
            plddt.append(float(line[60:66]))
            serial += 1
    return output, plddt


def plddt_report(values: list[float]) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("AFDB monomer atoms do not all contain finite 0-100 pLDDT B factors.")
    return {
        "crop_plddt_atom_count": int(len(values)),
        "crop_plddt_atom_mean": float(np.mean(values)),
        "crop_plddt_atom_minimum": float(np.min(values)),
        "crop_plddt_atom_maximum": float(np.max(values)),
    }


def residue_plddt_report(residues: list[ResidueAtoms]) -> dict[str, float | int]:
    values = []
    for residue in residues:
        ca_lines = [line for line in residue.lines if line[12:16].strip() == "CA"]
        if len(ca_lines) != 1:
            raise ValueError("Each AFDB crop residue must contain exactly one C-alpha pLDDT value.")
        atom_values = np.asarray(
            [float(line[60:66]) for line in residue.lines],
            dtype=np.float64,
        )
        if (
            not len(atom_values)
            or not np.isfinite(atom_values).all()
            or np.any((atom_values < 0.0) | (atom_values > 100.0))
        ):
            raise ValueError("AFDB monomer residues contain invalid pLDDT B factors.")
        if float(np.ptp(atom_values)) > 0.011:
            raise ValueError("Atoms within one AFDB monomer residue contain inconsistent pLDDT values.")
        values.append(float(ca_lines[0][60:66]))
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any((array < 0.0) | (array > 100.0)):
        raise ValueError("AFDB monomer residues do not all contain finite 0-100 C-alpha pLDDT values.")
    return {
        "crop_plddt_residue_count": int(len(array)),
        "crop_plddt_residue_mean": float(np.mean(array)),
        "crop_plddt_residue_minimum": float(np.min(array)),
        "crop_plddt_residue_maximum": float(np.max(array)),
    }


def write_complex(path: Path, chain_a: list[str], chain_b: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wt", encoding="ascii", newline="\n") as handle:
        handle.writelines(chain_a)
        handle.write("TER\n")
        handle.writelines(chain_b)
        handle.write("TER\nEND\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align AlphaFold DB monomers into experimental docking poses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--afdb-match-audit",
        required=True,
        type=Path,
        help="AFDB match-audit CSV containing paired monomer accessions.",
    )
    parser.add_argument(
        "--experimental-folder",
        required=True,
        type=Path,
        help="Directory containing the experimental complex structures.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for replacement complexes.")
    parser.add_argument("--cache-dir", type=Path, help="Directory for cached AFDB prediction metadata.")
    parser.add_argument("--raw-monomer-dir", type=Path, help="Directory for downloaded AFDB monomer files.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent download and alignment workers.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--minimum-aligned-ca",
        type=int,
        default=10,
        help="Minimum aligned C-alpha atoms for pose fitting.",
    )
    parser.add_argument(
        "--minimum-identity",
        type=float,
        default=0.70,
        help="Minimum sequence identity for accepting a monomer alignment.",
    )
    args = parser.parse_args()
    if not args.afdb_match_audit.is_file() or not args.experimental_folder.is_dir():
        raise FileNotFoundError("afdb-match-audit and experimental-folder must exist.")
    if args.workers <= 0 or args.minimum_aligned_ca < 3:
        raise ValueError("workers must be positive and minimum-aligned-ca must be at least three.")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError("timeout must be finite and positive.")
    for name, value in (("minimum-identity", args.minimum_identity),):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.cache_dir or args.output_dir / "cache" / "prediction"
    raw_dir = args.raw_monomer_dir or args.output_dir / "raw_monomers"
    complex_dir = args.output_dir / "replacement_complexes"
    rows = [
        row
        for row in read_csv_rows(args.afdb_match_audit)
        if row.get("afdb_accession_a") and row.get("afdb_accession_b")
    ]
    if not rows:
        raise ValueError("AFDB match audit contains no two-accession mapped rows.")
    record_ids = [str(row.get("record_id") or "").strip() for row in rows]
    pdb_ids = [str(row.get("pdb_id") or "").strip() for row in rows]
    if (
        any(not value for value in record_ids)
        or len(set(record_ids)) != len(record_ids)
        or any(not value for value in pdb_ids)
        or len(set(pdb_ids)) != len(pdb_ids)
    ):
        raise ValueError("Mapped rows require unique, non-empty record_id and pdb_id values.")
    accessions = sorted({row[key] for row in rows for key in ("afdb_accession_a", "afdb_accession_b")})

    def metadata_worker(accession):
        try:
            payload = fetch_json_cached(
                AFDB_PREDICTION_URL.format(accession=accession),
                metadata_dir / f"{accession}.json",
                args.timeout,
            )
            return accession, prediction_record(payload, accession), ""
        except ValueError as exc:
            return accession, None, str(exc)

    records = {}
    metadata_errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (accession, record, error) in enumerate(executor.map(metadata_worker, accessions), start=1):
            if record is None:
                metadata_errors[accession] = error
            else:
                records[accession] = record
            if index % 100 == 0 or index == len(accessions):
                print(f"AFDB monomer metadata: {index}/{len(accessions)}", flush=True)

    def download_worker(accession):
        path = raw_dir / f"AF-{accession}-F1.pdb"
        provenance = download_pdb_cached(str(records[accession]["pdbUrl"]), path, args.timeout)
        return accession, {**provenance, "path": str(path.resolve())}

    downloads = {}
    downloadable_accessions = sorted(records)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (accession, provenance) in enumerate(
            executor.map(download_worker, downloadable_accessions), start=1
        ):
            downloads[accession] = provenance
            if index % 100 == 0 or index == len(downloadable_accessions):
                print(f"AFDB monomer PDB: {index}/{len(downloadable_accessions)}", flush=True)

    def build_worker(row):
        try:
            source_metadata = monomer_source_metadata(row)
            experimental_path = args.experimental_folder / row["pdb"]
            verify_experimental_coordinate(experimental_path, row.get("input_sha256", ""))
            experimental_chains = parse_protein_chains(experimental_path)
            experimental_a = experimental_chains[row["chain_a"]]
            experimental_b = experimental_chains[row["chain_b"]]
            accession_a = row["afdb_accession_a"]
            accession_b = row["afdb_accession_b"]
            unavailable = [accession for accession in (accession_a, accession_b) if accession not in records]
            if unavailable:
                details = "; ".join(f"{accession}: {metadata_errors[accession]}" for accession in unavailable)
                raise ValueError(f"AFDB monomer unavailable ({details})")
            predicted_chains_a = parse_protein_chains(Path(downloads[accession_a]["path"]))
            predicted_chains_b = parse_protein_chains(Path(downloads[accession_b]["path"]))
            if len(predicted_chains_a) != 1 or len(predicted_chains_b) != 1:
                raise ValueError("An AFDB monomer PDB does not contain exactly one protein chain.")
            intervals_a, projection_a = project_uniprot_intervals_to_coordinates(
                parse_intervals(row["afdb_intervals_a"]),
                records[accession_a],
            )
            intervals_b, projection_b = project_uniprot_intervals_to_coordinates(
                parse_intervals(row["afdb_intervals_b"]),
                records[accession_b],
            )
            predicted_a = crop_residues(
                next(iter(predicted_chains_a.values())),
                intervals_a,
            )
            predicted_b = crop_residues(
                next(iter(predicted_chains_b.values())),
                intervals_b,
            )
            rotation_a, translation_a, alignment_a = superposition_transform(
                experimental_a,
                predicted_a,
                minimum_aligned_ca=args.minimum_aligned_ca,
                minimum_identity=args.minimum_identity,
            )
            rotation_b, translation_b, alignment_b = superposition_transform(
                experimental_b,
                predicted_b,
                minimum_aligned_ca=args.minimum_aligned_ca,
                minimum_identity=args.minimum_identity,
            )
            chain_a_lines, plddt_a = transform_chain_lines(predicted_a, rotation_a, translation_a, "A", 1)
            chain_b_lines, plddt_b = transform_chain_lines(
                predicted_b,
                rotation_b,
                translation_b,
                "B",
                len(chain_a_lines) + 1,
            )
            confidence = {
                **plddt_report(plddt_a + plddt_b),
                **residue_plddt_report(predicted_a + predicted_b),
            }
            filename = f"{row['pdb_id']}__afdb_monomer_replacement.pdb"
            output_path = complex_dir / filename
            write_complex(output_path, chain_a_lines, chain_b_lines)
            paired_record_id = f"pdbbind-afdb-monomer:{row['pdb_id']}"
            result = {
                **source_metadata,
                **paired_reference_metadata(row),
                **observed_sequence_metadata(
                    residue_sequence(predicted_a),
                    residue_sequence(predicted_b),
                ),
                **{f"alignment_a_{key}": value for key, value in alignment_a.items()},
                **{f"alignment_b_{key}": value for key, value in alignment_b.items()},
                **confidence,
                **projection_manifest_metadata(projection_a, "a"),
                **projection_manifest_metadata(projection_b, "b"),
                "status": "accepted",
                "reason": "",
                "record_id": f"afdb-monomer-replacement:{row['pdb_id']}",
                "paired_record_id": paired_record_id,
                "paired_experimental_record_id": row["record_id"],
                "pdb": filename,
                "input_sha256": sha256_file(str(output_path)),
                "chain_a": "A",
                "chain_b": "B",
                "afdb_model_id": f"{records[accession_a]['modelEntityId']}+{records[accession_b]['modelEntityId']}",
                "afdb_model_id_a": records[accession_a]["modelEntityId"],
                "afdb_model_id_b": records[accession_b]["modelEntityId"],
                "afdb_raw_pdb_url_a": downloads[accession_a]["url"],
                "afdb_raw_pdb_url_b": downloads[accession_b]["url"],
                "afdb_raw_pdb_sha256_a": downloads[accession_a]["sha256"],
                "afdb_raw_pdb_sha256_b": downloads[accession_b]["sha256"],
                "afdb_raw_pdb_retrieved_at_utc_a": downloads[accession_a]["retrieved_at_utc"],
                "afdb_raw_pdb_retrieved_at_utc_b": downloads[accession_b]["retrieved_at_utc"],
                "afdb_raw_pdb_download_sidecar_sha256_a": sha256_file(
                    download_sidecar_path(Path(downloads[accession_a]["path"]))
                ),
                "afdb_raw_pdb_download_sidecar_sha256_b": sha256_file(
                    download_sidecar_path(Path(downloads[accession_b]["path"]))
                ),
                "afdb_prediction_metadata_sha256_a": sha256_file(str(metadata_dir / f"{accession_a}.json")),
                "afdb_prediction_metadata_sha256_b": sha256_file(str(metadata_dir / f"{accession_b}.json")),
                "afdb_prediction_metadata_retrieved_at_utc_a": file_timestamp_utc(metadata_dir / f"{accession_a}.json"),
                "afdb_prediction_metadata_retrieved_at_utc_b": file_timestamp_utc(metadata_dir / f"{accession_b}.json"),
                "afdb_latest_version_a": records[accession_a].get("latestVersion") or "",
                "afdb_latest_version_b": records[accession_b].get("latestVersion") or "",
                "afdb_model_created_date_a": records[accession_a].get("modelCreatedDate") or "",
                "afdb_model_created_date_b": records[accession_b].get("modelCreatedDate") or "",
                "afdb_tool_used_a": records[accession_a].get("toolUsed") or "",
                "afdb_tool_used_b": records[accession_b].get("toolUsed") or "",
                "afdb_provider_a": records[accession_a].get("providerId") or "",
                "afdb_provider_b": records[accession_b].get("providerId") or "",
                "dataset_source": "AlphaFold DB monomer predictions aligned to PDBbind partner chains",
                "source_accession": f"{accession_a}+{accession_b}",
                "license_or_terms": "AlphaFold DB terms: https://alphafold.ebi.ac.uk/faq",
                "structure_type": "afdb_monomer_replacement",
                "structure_method": (
                    "AFDB monomers independently rigidly superposed to experimental chains; "
                    "relative docking pose is experimental"
                ),
                "resolution_angstrom": "",
                "confidence_metric": "plddt_bfactor",
                "confidence_source": "AlphaFold DB monomer PDB B-factor fields",
                "confidence_threshold": 70.0,
                "confidence_stratum": plddt_confidence_stratum(float(confidence["crop_plddt_residue_mean"])),
                "selection_mode": (
                    "chain-sequence-validated UniProt-mapped AFDB monomers cropped to mapped intervals and "
                    "independently superposed by sequence-matched C-alpha atoms"
                ),
                "structure_path": str(output_path.resolve()),
                "hotspot_residues_a": "",
                "prolif_file": "",
                "prolif_sha256": "",
            }
            experimental = {
                **source_metadata,
                "paired_record_id": paired_record_id,
                "paired_afdb_monomer_record_id": result["record_id"],
            }
            return result, experimental
        except (KeyError, OSError, PDBException, TypeError, ValueError) as exc:
            return {**row, "status": "excluded", "reason": str(exc)}, None

    audit_rows = []
    predicted_rows = []
    experimental_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (predicted, experimental) in enumerate(executor.map(build_worker, rows), start=1):
            audit_rows.append(predicted)
            if experimental is not None:
                predicted_rows.append(predicted)
                experimental_rows.append(experimental)
            if index % 50 == 0 or index == len(rows):
                print(f"AFDB monomer replacement: {index}/{len(rows)}", flush=True)

    write_csv_atomic(args.output_dir / "afdb_monomer_replacement_audit.csv", audit_rows)
    predicted_manifest = args.output_dir / "afdb_monomer_replacement_manifest.csv"
    experimental_manifest = args.output_dir / "pdbbind_monomer_matched_manifest.csv"
    write_csv_atomic(predicted_manifest, predicted_rows)
    write_csv_atomic(experimental_manifest, experimental_rows)
    summary = {
        "schema_version": 4,
        "design": "AFDB monomer geometry in the experimental relative docking pose",
        "scientific_scope": (
            "tests sensitivity to AFDB coordinate/domain availability and predicted partner geometry "
            "after independent placement in the experimental pose; it does not test predicted complex "
            "pose accuracy"
        ),
        "mapped_candidate_count": len(rows),
        "unique_accession_count": len(accessions),
        "afdb_monomer_accession_count": len(records),
        "afdb_monomer_unavailable_accession_count": len(metadata_errors),
        "successful_replacement_count": len(predicted_rows),
        "failed_replacement_count": len(rows) - len(predicted_rows),
        "minimum_aligned_ca": args.minimum_aligned_ca,
        "minimum_identity": args.minimum_identity,
        "rigid_fit_inclusion_rule": (
            "minimum matched C-alpha count, minimum sequence identity, and stable non-collinear fit geometry; "
            "whole-chain coverage is recorded but not used for selection"
        ),
        "current_input_sequence_rule": (
            "sequence_a/b and chain_a/b_residue_count describe observed residues in each "
            "generated AFDB replacement coordinate input"
        ),
        "paired_reference_rule": ("paired_reference_* fields retain the corresponding experimental input metadata"),
        "dependency_group_rule": (
            "sequence_cluster_a/b, cluster_id, family_id, and analysis_split are inherited from "
            "the paired experimental record"
        ),
        "predicted_manifest_sha256": sha256_file(str(predicted_manifest)),
        "experimental_manifest_sha256": sha256_file(str(experimental_manifest)),
        "afdb_match_audit_sha256": sha256_file(str(args.afdb_match_audit)),
    }
    dump_json_atomic(summary, args.output_dir / "afdb_monomer_replacement_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
