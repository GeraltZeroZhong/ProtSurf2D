#!/usr/bin/env python3
"""Download, crop, and manifest AFDB dimers matched to PDBbind interfaces."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException

from topoppi.benchmarking.manifest_metadata import (
    ipsae_confidence_stratum,
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

AFDB_PREDICTION_URL = "https://alphafold.ebi.ac.uk/api/prediction/{model_id}"
AMINO_ACID_LETTERS = protein_letters_3to1_extended


def file_timestamp_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_json_cached(url: str, path: Path, timeout: float) -> object:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(5):
        response = requests.get(url, timeout=timeout)
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


def chain_assignments(payload: object, accession_a: str, accession_b: str):
    records = [record for record in payload if isinstance(record, dict)] if isinstance(payload, list) else []
    by_accession = {}
    for record in records:
        by_accession.setdefault(str(record.get("uniprotAccession") or ""), []).append(record)
    if accession_a == accession_b:
        matches = sorted(by_accession.get(accession_a, []), key=lambda record: str(record.get("chainId")))
        if len(matches) != 2:
            raise ValueError("AFDB homodimer metadata does not contain exactly two matching chains.")
        return matches[0], matches[1]
    matches_a = by_accession.get(accession_a, [])
    matches_b = by_accession.get(accession_b, [])
    if len(matches_a) != 1 or len(matches_b) != 1:
        raise ValueError("AFDB heterodimer metadata does not uniquely map both UniProt accessions.")
    return matches_a[0], matches_b[0]


def parse_intervals(value: str) -> list[tuple[int, int]]:
    intervals = [(int(start), int(end)) for start, end in json.loads(value)]
    if not intervals or any(start < 1 or end < start for start, end in intervals):
        raise ValueError("UniProt crop intervals must be non-empty positive inclusive ranges.")
    return intervals


def in_intervals(residue_number: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= residue_number <= end for start, end in intervals)


def projection_manifest_metadata(report: dict[str, object], side: str) -> dict[str, object]:
    return {
        f"afdb_coordinate_projection_{key}_{side}": (
            json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
        )
        for key, value in report.items()
    }


def confidence_manifest_metadata(mean_plddt: float, ipsae: float) -> dict[str, object]:
    """Keep residue-level pLDDT and complex-level ipSAE strata distinct."""

    return {
        "confidence_metric": "plddt_bfactor",
        "confidence_source": "AlphaFold DB model PDB B-factor field",
        "confidence_threshold": 70.0,
        "confidence_stratum": plddt_confidence_stratum(mean_plddt),
        "afdb_ipsae_stratum": ipsae_confidence_stratum(ipsae),
    }


def crop_model(
    raw_path: Path,
    output_path: Path,
    source_chain_a: str,
    source_chain_b: str,
    intervals_a: list[tuple[int, int]],
    intervals_b: list[tuple[int, int]],
) -> dict[str, object]:
    selections = {
        source_chain_id: (target_chain_id, intervals)
        for source_chain_id, target_chain_id, intervals in (
            (source_chain_a, "A", intervals_a),
            (source_chain_b, "B", intervals_b),
        )
    }
    if len(selections) != 2:
        raise ValueError("AFDB metadata assigned both partners to the same source chain.")
    output_lines = {"A": [], "B": []}
    residues: dict[str, dict[tuple[int, str], str]] = {"A": {}, "B": {}}
    residue_plddt: dict[str, dict[tuple[int, str], float]] = {"A": {}, "B": {}}
    plddt = []
    for line in selected_protein_atom_lines(raw_path):
        if len(line) < 66:
            continue
        source_chain_id = line[21]
        selection = selections.get(source_chain_id)
        residue_name = line[17:20].strip()
        if selection is None or residue_name not in AMINO_ACID_LETTERS:
            continue
        target_chain_id, intervals = selection
        residue_number = int(line[22:26])
        if not in_intervals(residue_number, intervals):
            continue
        residue_key = (residue_number, line[26])
        previous_name = residues[target_chain_id].setdefault(residue_key, residue_name)
        if previous_name != residue_name:
            raise ValueError(f"Microheterogeneous residue is unsupported in {raw_path}: {source_chain_id}{residue_key}")
        confidence = float(line[60:66])
        previous_confidence = residue_plddt[target_chain_id].setdefault(
            residue_key,
            confidence,
        )
        if abs(previous_confidence - confidence) > 0.011:
            raise ValueError(
                f"Atoms within one AFDB residue contain inconsistent pLDDT values: {source_chain_id}{residue_key}"
            )
        output_lines[target_chain_id].append(line[:21] + target_chain_id + line[22:])
        plddt.append(confidence)
    if not residues["A"] or not residues["B"]:
        raise ValueError("AFDB crop removed every residue from at least one metadata-declared chain.")
    expected_residue_counts = {
        "A": len({number for start, end in intervals_a for number in range(start, end + 1)}),
        "B": len({number for start, end in intervals_b for number in range(start, end + 1)}),
    }
    for chain_id in ("A", "B"):
        if len(residues[chain_id]) != expected_residue_counts[chain_id]:
            raise ValueError(
                f"AFDB PDB residue numbering does not cover every projected coordinate in chain {chain_id}."
            )
    plddt_values = [value for value in plddt if math.isfinite(value)]
    if len(plddt_values) != len(plddt) or any(not 0.0 <= value <= 100.0 for value in plddt_values):
        raise ValueError("AFDB crop contains invalid pLDDT B factors.")
    residue_plddt_values = [value for chain_id in ("A", "B") for value in residue_plddt[chain_id].values()]
    if len(residue_plddt_values) != len(residues["A"]) + len(residues["B"]) or any(
        not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in residue_plddt_values
    ):
        raise ValueError("AFDB crop lacks one finite 0-100 pLDDT value per residue.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("wt", encoding="ascii", newline="\n") as handle:
        for chain_id in ("A", "B"):
            handle.writelines(output_lines[chain_id])
            handle.write("TER\n")
        handle.write("END\n")
    os.replace(temporary, output_path)
    sequence_a = "".join(AMINO_ACID_LETTERS[name] for name in residues["A"].values())
    sequence_b = "".join(AMINO_ACID_LETTERS[name] for name in residues["B"].values())
    return {
        **observed_sequence_metadata(sequence_a, sequence_b),
        "crop_residue_count_a": len(residues["A"]),
        "crop_residue_count_b": len(residues["B"]),
        "crop_atom_count_a": len(output_lines["A"]),
        "crop_atom_count_b": len(output_lines["B"]),
        "crop_plddt_atom_mean": sum(plddt_values) / len(plddt_values),
        "crop_plddt_atom_minimum": min(plddt_values),
        "crop_plddt_atom_maximum": max(plddt_values),
        "crop_plddt_residue_count": len(residue_plddt_values),
        "crop_plddt_residue_mean": sum(residue_plddt_values) / len(residue_plddt_values),
        "crop_plddt_residue_minimum": min(residue_plddt_values),
        "crop_plddt_residue_maximum": max(residue_plddt_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and crop matched AlphaFold DB dimer structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matched-candidates",
        required=True,
        type=Path,
        help="Candidate-match CSV from match_afdb_complexes.py.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for models and the output manifest.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent metadata and model downloads.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Network timeout in seconds.")
    args = parser.parse_args()
    if not args.matched_candidates.is_file():
        raise FileNotFoundError(args.matched_candidates)
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError("timeout must be finite and positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_cache = args.output_dir / "cache" / "prediction"
    raw_dir = args.output_dir / "raw_models"
    cropped_dir = args.output_dir / "cropped_models"
    rows = read_csv_rows(args.matched_candidates)
    if not rows:
        raise ValueError("Matched-candidates table contains no rows.")
    record_ids = [str(row.get("record_id") or "").strip() for row in rows]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Matched candidates require unique, non-empty record_id values.")

    def metadata_worker(model_id):
        payload = fetch_json_cached(
            AFDB_PREDICTION_URL.format(model_id=model_id),
            metadata_cache / f"{model_id}.json",
            args.timeout,
        )
        return model_id, payload

    model_ids = sorted({row["afdb_model_id"] for row in rows})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        metadata = dict(executor.map(metadata_worker, model_ids))

    model_download_spec = {}
    for model_id in model_ids:
        payload = metadata[model_id]
        records = payload if isinstance(payload, list) else []
        urls = {str(record.get("pdbUrl") or "") for record in records if record.get("pdbUrl")}
        if len(urls) != 1:
            raise ValueError(f"AFDB metadata does not expose one PDB URL for {model_id}")
        model_download_spec[model_id] = urls.pop()

    model_metadata = {}
    for model_id in model_ids:
        records = [record for record in metadata[model_id] if isinstance(record, dict)]
        model_metadata[model_id] = {
            key: next(iter(values)) if len(values) == 1 else "|".join(sorted(map(str, values)))
            for key in ("latestVersion", "modelCreatedDate", "toolUsed", "providerId")
            if (values := {record.get(key) for record in records if record.get(key) is not None})
        }

    def download_worker(model_id):
        path = raw_dir / f"{model_id}.pdb"
        provenance = download_pdb_cached(model_download_spec[model_id], path, args.timeout)
        return model_id, {**provenance, "path": str(path.resolve())}

    downloads = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (model_id, provenance) in enumerate(executor.map(download_worker, model_ids), start=1):
            downloads[model_id] = provenance
            if index % 50 == 0 or index == len(model_ids):
                print(f"AFDB PDB: {index}/{len(model_ids)}", flush=True)

    def crop_worker(row):
        model_id = row["afdb_model_id"]
        try:
            record_a, record_b = chain_assignments(metadata[model_id], row["afdb_accession_a"], row["afdb_accession_b"])
            intervals_a, projection_a = project_uniprot_intervals_to_coordinates(
                parse_intervals(row["afdb_intervals_a"]),
                record_a,
            )
            intervals_b, projection_b = project_uniprot_intervals_to_coordinates(
                parse_intervals(row["afdb_intervals_b"]),
                record_b,
            )
            filename = f"{row['pdb_id']}__{model_id}.pdb"
            output_path = cropped_dir / filename
            crop_stats = crop_model(
                Path(downloads[model_id]["path"]),
                output_path,
                str(record_a["chainId"]),
                str(record_b["chainId"]),
                intervals_a,
                intervals_b,
            )
            paired_record_id = f"pdbbind-afdb:{row['pdb_id']}"
            result = {
                **row,
                **paired_reference_metadata(row),
                **crop_stats,
                **projection_manifest_metadata(projection_a, "a"),
                **projection_manifest_metadata(projection_b, "b"),
                "status": "accepted",
                "reason": "",
                "record_id": f"afdb:{row['pdb_id']}:{model_id}",
                "paired_record_id": paired_record_id,
                "paired_experimental_record_id": row["record_id"],
                "pdb": filename,
                "input_sha256": sha256_file(str(output_path)),
                "chain_a": "A",
                "chain_b": "B",
                "afdb_source_chain_a": record_a["chainId"],
                "afdb_source_chain_b": record_b["chainId"],
                "afdb_raw_pdb_url": downloads[model_id]["url"],
                "afdb_raw_pdb_sha256": downloads[model_id]["sha256"],
                "afdb_raw_pdb_size_bytes": downloads[model_id]["size_bytes"],
                "afdb_raw_pdb_retrieved_at_utc": downloads[model_id]["retrieved_at_utc"],
                "afdb_raw_pdb_download_sidecar_sha256": sha256_file(
                    download_sidecar_path(Path(downloads[model_id]["path"]))
                ),
                "afdb_prediction_metadata_sha256": sha256_file(str(metadata_cache / f"{model_id}.json")),
                "afdb_prediction_metadata_retrieved_at_utc": file_timestamp_utc(metadata_cache / f"{model_id}.json"),
                "afdb_latest_version": model_metadata[model_id].get("latestVersion") or "",
                "afdb_model_created_date": model_metadata[model_id].get("modelCreatedDate") or "",
                "afdb_tool_used": model_metadata[model_id].get("toolUsed") or "",
                "afdb_provider": model_metadata[model_id].get("providerId") or row.get("afdb_provider") or "",
                "dataset_source": "AlphaFold Protein Structure Database complex predictions",
                "source_accession": model_id,
                "license_or_terms": "AlphaFold DB terms: https://alphafold.ebi.ac.uk/faq",
                "structure_type": "afdb",
                "structure_method": "AlphaFold DB complex prediction",
                "resolution_angstrom": "",
                **confidence_manifest_metadata(
                    float(crop_stats["crop_plddt_residue_mean"]),
                    float(row["afdb_ipsae"]),
                ),
                "selection_mode": (
                    "sequence-validated UniProt-mapped AFDB dimer chains; PDB-observed mapped UniProt intervals"
                ),
                "structure_path": str(output_path.resolve()),
                "hotspot_residues_a": "",
                "prolif_file": "",
                "prolif_sha256": "",
            }
            experimental = {
                **row,
                "paired_record_id": paired_record_id,
                "paired_afdb_record_id": result["record_id"],
            }
            return result, experimental
        except (KeyError, OSError, PDBException, TypeError, ValueError) as exc:
            return {**row, "status": "excluded", "reason": str(exc)}, None

    audit_rows = []
    predicted_rows = []
    experimental_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (predicted, experimental) in enumerate(executor.map(crop_worker, rows), start=1):
            audit_rows.append(predicted)
            if experimental is not None:
                predicted_rows.append(predicted)
                experimental_rows.append(experimental)
            if index % 50 == 0 or index == len(rows):
                print(f"AFDB crop: {index}/{len(rows)}", flush=True)

    write_csv_atomic(args.output_dir / "afdb_crop_audit.csv", audit_rows)
    afdb_manifest = args.output_dir / "afdb_matched_manifest.csv"
    experimental_manifest = args.output_dir / "pdbbind_matched_experimental_manifest.csv"
    write_csv_atomic(afdb_manifest, predicted_rows)
    write_csv_atomic(experimental_manifest, experimental_rows)
    summary = {
        "schema_version": 3,
        "matched_candidate_count": len(rows),
        "unique_afdb_model_count": len(model_ids),
        "successful_crop_count": len(predicted_rows),
        "failed_crop_count": len(rows) - len(predicted_rows),
        "afdb_manifest_sha256": sha256_file(str(afdb_manifest)),
        "matched_experimental_manifest_sha256": sha256_file(str(experimental_manifest)),
        "matched_candidates_sha256": sha256_file(str(args.matched_candidates)),
        "crop_rule": (
            "retain the union of mapped UniProt intervals observed in each selected PDB chain; "
            "mapping may come from SIFTS or the declared sequence-validated UniProt fallback; "
            "project UniProt positions through AFDB uniprotStart/sequenceStart metadata and record "
            "any model-range truncation"
        ),
        "chain_rule": "AFDB chains are accession-matched and renamed A/B in receptor/ligand order",
        "current_input_sequence_rule": (
            "sequence_a/b and chain_a/b_residue_count describe observed residues in each cropped AFDB coordinate input"
        ),
        "paired_reference_rule": ("paired_reference_* fields retain the corresponding experimental input metadata"),
        "dependency_group_rule": (
            "sequence_cluster_a/b, cluster_id, family_id, and analysis_split are inherited from "
            "the paired experimental record"
        ),
    }
    dump_json_atomic(summary, args.output_dir / "afdb_download_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
