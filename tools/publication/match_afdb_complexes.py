#!/usr/bin/env python3
"""Match PDBbind chain pairs to current AlphaFold DB dimer predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa

from topoppi.file_utils import read_csv_rows, sha256_file, write_csv_atomic
from topoppi.io.io_loader import PDBLoader
from topoppi.json_utils import dump_json_atomic
from topoppi.sequence_alignment import align_protein_sequences

PDBE_MAPPING_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
AFDB_COMPLEX_URL = "https://alphafold.ebi.ac.uk/api/complex/{accession}"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def file_timestamp_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_json_cached(url: str, path: Path, timeout: float) -> object:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(5):
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            payload: object = []
            break
        if response.status_code == 429 or response.status_code >= 500:
            retry_after = float(response.headers.get("Retry-After") or 2**attempt)
            time.sleep(min(retry_after, 30.0))
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


def fetch_many(items, worker, workers: int, label: str):
    results = {}
    ordered = sorted(set(items))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (key, value) in enumerate(executor.map(worker, ordered), start=1):
            results[key] = value
            if index % 100 == 0 or index == len(ordered):
                print(f"{label}: {index}/{len(ordered)}", flush=True)
    return results


def merged_intervals(intervals) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if start and end and end >= start)
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_length(intervals) -> int:
    return sum(end - start + 1 for start, end in intervals)


def optional_int(value: object) -> int | None:
    return int(value) if value is not None and str(value).strip() else None


def _author_residue_key(number: int, insertion: str) -> tuple[int, str]:
    return int(number), str(insertion or "").strip().upper()


def _segment_contains_residue(segment: dict[str, object], residue_id: tuple[int, str]) -> bool:
    if segment["author_start"] is None or segment["author_end"] is None:
        return False
    start = _author_residue_key(int(segment["author_start"]), str(segment["author_start_insertion"]))
    end = _author_residue_key(int(segment["author_end"]), str(segment["author_end_insertion"]))
    if end < start:
        start, end = end, start
    return start <= residue_id <= end


def _observed_chain_residue_ids(loader: PDBLoader, chain_id: str) -> list[tuple[int, str]]:
    return [
        _author_residue_key(residue.id[1], residue.id[2])
        for residue in loader.model[chain_id]
        if is_aa(residue, standard=False) and residue.get_resname() in protein_letters_3to1_extended
    ]


def observed_pair_residue_ids(task) -> tuple[str, dict[str, list[tuple[int, str]]]]:
    record_id, structure_path, expected_sha256, chain_a, chain_b = task
    actual_sha256 = sha256_file(structure_path)
    if actual_sha256.lower() != str(expected_sha256 or "").strip().lower():
        raise ValueError(
            f"Experimental coordinate checksum mismatch for {record_id}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    loader = PDBLoader(structure_path)
    return record_id, {
        "a": _observed_chain_residue_ids(loader, chain_a),
        "b": _observed_chain_residue_ids(loader, chain_b),
    }


def _uniprot_sequences(search_payload: object) -> dict[str, str]:
    results = search_payload.get("results", []) if isinstance(search_payload, dict) else []
    return {
        str(record.get("primaryAccession") or "").strip(): str((record.get("sequence") or {}).get("value") or "")
        .strip()
        .upper()
        for record in results
        if record.get("primaryAccession") and (record.get("sequence") or {}).get("value")
    }


def _crop_sequence(sequence: str, intervals: list[tuple[int, int]]) -> str:
    if any(start < 1 or end > len(sequence) for start, end in intervals):
        return ""
    return "".join(sequence[start - 1 : end] for start, end in intervals)


def _intervals_from_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    intervals = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            intervals.append((start + 1, previous + 1))
            start = index
        previous = index
    intervals.append((start + 1, previous + 1))
    return intervals


def select_chain_accession(
    mapping_payload: object,
    pdb_id: str,
    chain_id: str,
    observed_residue_ids: list[tuple[int, str]],
    chain_sequence: str,
    search_payload: object,
) -> dict[str, object]:
    root = mapping_payload.get(pdb_id, {}) if isinstance(mapping_payload, dict) else {}
    uniprot = root.get("UniProt", {}) if isinstance(root, dict) else {}
    sequences = _uniprot_sequences(search_payload)
    candidates = []
    for accession, metadata in uniprot.items():
        mappings = [item for item in metadata.get("mappings", []) if str(item.get("chain_id") or "") == chain_id]
        valid_mappings = [
            item for item in mappings if item.get("unp_start") is not None and item.get("unp_end") is not None
        ]
        intervals = merged_intervals((item.get("unp_start"), item.get("unp_end")) for item in valid_mappings)
        uniprot_interval_count = interval_length(intervals)
        if not uniprot_interval_count:
            continue
        segments = sorted(
            (
                {
                    "unp_start": int(item["unp_start"]),
                    "unp_end": int(item["unp_end"]),
                    "author_start": optional_int(item["start"].get("author_residue_number")),
                    "author_start_insertion": str(item["start"].get("author_insertion_code") or ""),
                    "author_end": optional_int(item["end"].get("author_residue_number")),
                    "author_end_insertion": str(item["end"].get("author_insertion_code") or ""),
                    "identity": float(item.get("identity") or 0.0),
                    "coverage": float(item.get("coverage") or 0.0),
                }
                for item in valid_mappings
            ),
            key=lambda item: (
                item["unp_start"],
                item["unp_end"],
                item["author_start"] if item["author_start"] is not None else -1,
                item["author_start_insertion"],
            ),
        )
        raw_weights = [int(item["unp_end"]) - int(item["unp_start"]) + 1 for item in valid_mappings]
        sifts_weighted_identity = sum(
            float(item.get("identity") or 0.0) * weight
            for item, weight in zip(valid_mappings, raw_weights, strict=True)
        ) / sum(raw_weights)

        uniprot_sequence = sequences.get(str(accession), "")
        candidate_sequence = _crop_sequence(uniprot_sequence, intervals) if uniprot_sequence else ""
        if candidate_sequence:
            _, alignment_report = align_protein_sequences(chain_sequence, candidate_sequence)
            mapped_count = int(alignment_report["aligned_residue_count"])
            weighted_identity = float(alignment_report["alignment_identity"])
            experimental_coverage = float(alignment_report["reference_coverage"])
            mapping_evidence = "sifts_interval_semiglobal_sequence_alignment"
            alignment_score = float(alignment_report["alignment_score"])
            alignment_ambiguity = {
                "optimal_alignment_count": int(alignment_report["optimal_alignment_count"]),
                "optimal_correspondence_count": int(alignment_report["optimal_correspondence_count"]),
                "selected_pair_consensus_fraction": float(alignment_report["selected_pair_consensus_fraction"]),
            }
        else:
            residue_identities = {
                residue_id: max(
                    float(segment["identity"]) for segment in segments if _segment_contains_residue(segment, residue_id)
                )
                for residue_id in observed_residue_ids
                if any(_segment_contains_residue(segment, residue_id) for segment in segments)
            }
            mapped_count = len(residue_identities)
            weighted_identity = sum(residue_identities.values()) / mapped_count if mapped_count else 0.0
            experimental_coverage = mapped_count / len(observed_residue_ids)
            mapping_evidence = "sifts_author_residue_intersection"
            alignment_score = ""
            alignment_ambiguity = {
                "optimal_alignment_count": "",
                "optimal_correspondence_count": "",
                "selected_pair_consensus_fraction": "",
            }
        if not mapped_count:
            continue
        candidates.append(
            {
                "accession": accession,
                "intervals": intervals,
                "mapped_residue_count": mapped_count,
                "uniprot_interval_residue_count": uniprot_interval_count,
                "experimental_sequence_coverage": experimental_coverage,
                "weighted_identity": weighted_identity,
                "sifts_weighted_identity": sifts_weighted_identity,
                "mapping_evidence": mapping_evidence,
                "alignment_score": alignment_score,
                **alignment_ambiguity,
                "mapping_count": len(valid_mappings),
                "segments": segments,
            }
        )
    if not candidates:
        raise ValueError(f"No SIFTS UniProt mapping for PDB {pdb_id} chain {chain_id}")
    selected = dict(
        min(
            candidates,
            key=lambda item: (
                -float(item["experimental_sequence_coverage"]),
                -float(item["weighted_identity"]),
                -float(item["sifts_weighted_identity"]),
                -int(item["mapped_residue_count"]),
                -int(item["uniprot_interval_residue_count"]),
                str(item["accession"]),
            ),
        )
    )
    selected["candidate_accession_count"] = len(candidates)
    selected["candidates"] = candidates
    selected["mapping_method"] = "pdbe_sifts_chain_mapping"
    return selected


def uniprot_pdb_search_url(pdb_id: str, candidate_limit: int) -> str:
    query = urlencode(
        {
            "query": f"xref:pdb-{pdb_id}",
            "fields": "accession,sequence,length",
            "format": "json",
            "size": int(candidate_limit),
        }
    )
    return f"{UNIPROT_SEARCH_URL}?{query}"


def uniprot_cache_path(cache_dir: Path, pdb_id: str, candidate_limit: int) -> Path:
    return cache_dir / f"{pdb_id}.size{candidate_limit}.json"


def require_complete_uniprot_search(search_payload: object, candidate_limit: int) -> None:
    results = search_payload.get("results", []) if isinstance(search_payload, dict) else []
    if len(results) >= candidate_limit:
        raise ValueError(
            f"UniProt PDB-linked search reached its {candidate_limit}-record limit; "
            "the accession candidate set may be truncated."
        )


def select_sequence_matched_accession(
    search_payload: object,
    chain_sequence: str,
    *,
    minimum_aligned_residues: int,
    minimum_identity: float,
    minimum_chain_coverage: float,
    minimum_pair_consensus: float = 0.90,
) -> dict[str, object]:
    """Select a PDB-linked UniProt entry by an auditable chain-sequence alignment."""

    sequence = "".join(str(chain_sequence).split()).upper()
    if not sequence:
        raise ValueError("Experimental chain sequence is empty")
    results = search_payload.get("results", []) if isinstance(search_payload, dict) else []
    candidates = []
    for record in results:
        accession = str(record.get("primaryAccession") or "").strip()
        candidate_sequence = str((record.get("sequence") or {}).get("value") or "").strip().upper()
        if not accession or not candidate_sequence:
            continue
        accession_alignments = []
        for alignment_mode in ("local", "semiglobal"):
            pairs, report = align_protein_sequences(
                sequence,
                candidate_sequence,
                mode=alignment_mode,
            )
            aligned_count = int(report["aligned_residue_count"])
            identity = float(report["alignment_identity"])
            coverage = float(report["reference_coverage"])
            pair_consensus = float(report["selected_pair_consensus_fraction"])
            if (
                aligned_count < minimum_aligned_residues
                or identity < minimum_identity
                or coverage < minimum_chain_coverage
                or pair_consensus < minimum_pair_consensus
            ):
                continue
            accession_alignments.append(
                {
                    "accession": accession,
                    "intervals": _intervals_from_indices([right for _, right in pairs]),
                    "mapped_residue_count": aligned_count,
                    "weighted_identity": identity,
                    "experimental_sequence_coverage": coverage,
                    "uniprot_sequence_length": len(candidate_sequence),
                    "alignment_score": float(report["alignment_score"]),
                    "alignment_mode": alignment_mode,
                    "optimal_alignment_count": int(report["optimal_alignment_count"]),
                    "optimal_correspondence_count": int(report["optimal_correspondence_count"]),
                    "selected_pair_consensus_fraction": float(report["selected_pair_consensus_fraction"]),
                }
            )
        if accession_alignments:
            candidates.append(
                min(
                    accession_alignments,
                    key=lambda item: (
                        -float(item["experimental_sequence_coverage"]),
                        -float(item["weighted_identity"]),
                        -int(item["mapped_residue_count"]),
                        -float(item["alignment_score"]),
                        str(item["alignment_mode"]),
                    ),
                )
            )
    if not candidates:
        raise ValueError(
            "No PDB-linked UniProt sequence passes the chain mapping thresholds "
            f"(aligned>={minimum_aligned_residues}, identity>={minimum_identity}, "
            f"coverage>={minimum_chain_coverage}, pair_consensus>={minimum_pair_consensus})"
        )
    selected = dict(
        min(
            candidates,
            key=lambda item: (
                -float(item["experimental_sequence_coverage"]),
                -float(item["weighted_identity"]),
                -int(item["mapped_residue_count"]),
                abs(int(item["uniprot_sequence_length"]) - len(sequence)),
                str(item["accession"]),
            ),
        )
    )
    selected["candidate_accession_count"] = len(candidates)
    selected["candidates"] = candidates
    selected["segments"] = []
    selected["mapping_evidence"] = f"pdb_xref_{selected['alignment_mode']}_sequence_alignment"
    selected["mapping_method"] = "uniprot_pdb_xref_sequence_alignment"
    return selected


def mapping_output(mapping: dict[str, object], suffix: str) -> dict[str, object]:
    method = str(mapping["mapping_method"])
    coverage = float(mapping["experimental_sequence_coverage"])
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"Experimental sequence coverage outside [0, 1]: {coverage}")
    output = {
        f"afdb_accession_{suffix}": mapping["accession"],
        f"afdb_intervals_{suffix}": json.dumps(mapping["intervals"], separators=(",", ":")),
        f"afdb_mapping_method_{suffix}": method,
        f"afdb_mapping_evidence_{suffix}": mapping["mapping_evidence"],
        f"afdb_mapping_aligned_residue_count_{suffix}": mapping["mapped_residue_count"],
        f"afdb_mapping_identity_{suffix}": mapping["weighted_identity"],
        f"afdb_mapping_experimental_coverage_{suffix}": coverage,
        f"afdb_mapping_candidate_accession_count_{suffix}": mapping["candidate_accession_count"],
        f"afdb_mapping_optimal_alignment_count_{suffix}": mapping.get("optimal_alignment_count", ""),
        f"afdb_mapping_optimal_correspondence_count_{suffix}": mapping.get("optimal_correspondence_count", ""),
        f"afdb_mapping_selected_pair_consensus_fraction_{suffix}": mapping.get("selected_pair_consensus_fraction", ""),
        f"afdb_mapping_candidates_{suffix}": json.dumps(mapping["candidates"], separators=(",", ":")),
    }
    if method == "pdbe_sifts_chain_mapping":
        output.update(
            {
                f"sifts_segments_{suffix}": json.dumps(mapping["segments"], separators=(",", ":")),
                f"sifts_mapped_residue_count_{suffix}": mapping["mapped_residue_count"],
                f"sifts_uniprot_interval_residue_count_{suffix}": mapping["uniprot_interval_residue_count"],
                f"sifts_identity_{suffix}": mapping["sifts_weighted_identity"],
                f"sifts_candidate_accession_count_{suffix}": mapping["candidate_accession_count"],
            }
        )
    return output


def composition_accessions(candidate: dict[str, object]) -> list[str]:
    composition = candidate.get("complexComposition") or []
    if composition:
        accessions = []
        for item in composition:
            if not isinstance(item, dict) or item.get("identifierType") != "uniprotAccession":
                return []
            identifier = str(item.get("identifier") or "").strip()
            raw_stoichiometry = item.get("stoichiometry", 1)
            if not identifier or isinstance(raw_stoichiometry, bool):
                return []
            try:
                stoichiometry = int(raw_stoichiometry)
                numeric_stoichiometry = float(raw_stoichiometry)
            except (TypeError, ValueError, OverflowError):
                return []
            if stoichiometry < 1 or not math.isfinite(numeric_stoichiometry) or numeric_stoichiometry != stoichiometry:
                return []
            accessions.extend([identifier] * stoichiometry)
        return accessions
    legacy = candidate.get("uniprotAccession") or []
    if not isinstance(legacy, list):
        return []
    accessions = [str(value or "").strip() for value in legacy]
    return accessions if all(accessions) else []


def finite_metric(candidate: dict[str, object], name: str) -> float:
    try:
        value = float(candidate.get(name, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def optional_metric(candidate: dict[str, object], name: str) -> float | str:
    value = finite_metric(candidate, name)
    return value if math.isfinite(value) else ""


def select_exact_model(
    accession_a: str,
    accession_b: str,
    candidate_payloads: list[object],
) -> tuple[dict[str, object], int]:
    expected = Counter((accession_a, accession_b))
    exact = {}
    for payload in candidate_payloads:
        for candidate in payload if isinstance(payload, list) else []:
            if not isinstance(candidate, dict):
                continue
            if Counter(composition_accessions(candidate)) != expected:
                continue
            iptm = finite_metric(candidate, "complexPredictionAccuracy_ipTM")
            ipsae = finite_metric(candidate, "complexPredictionAccuracy_ipSAE")
            if not math.isfinite(iptm) or not math.isfinite(ipsae):
                continue
            exact[str(candidate["modelEntityId"])] = candidate
    if not exact:
        raise ValueError("No exact AFDB dimer with finite ipTM and ipSAE")

    def rank(candidate):
        metrics = [
            finite_metric(candidate, name)
            for name in (
                "complexPredictionAccuracy_ipSAE",
                "complexPredictionAccuracy_ipTM",
                "complexPredictionAccuracy_pDockQ2",
                "complexPredictionAccuracy_pDockQ",
                "complexPredictionAccuracy_LIS",
            )
        ]
        normalized = [value if math.isfinite(value) else -1.0 for value in metrics]
        return (*(-value for value in normalized), str(candidate["modelEntityId"]))

    return min(exact.values(), key=rank), len(exact)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match experimental chain pairs to AlphaFold DB dimer candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experimental-manifest",
        required=True,
        type=Path,
        help="Experimental benchmark manifest CSV.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for match tables and query caches.")
    parser.add_argument("--cache-dir", type=Path, help="Directory for PDBe, UniProt, and AFDB API caches.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent mapping and matching workers.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--uniprot-candidates",
        type=int,
        default=500,
        help="Maximum UniProt search candidates per query.",
    )
    parser.add_argument(
        "--minimum-fallback-aligned-residues",
        type=int,
        default=10,
        help="Minimum aligned residues for a sequence-fallback match.",
    )
    parser.add_argument(
        "--minimum-fallback-identity",
        type=float,
        default=0.70,
        help="Minimum sequence identity for a fallback match.",
    )
    parser.add_argument(
        "--minimum-fallback-chain-coverage",
        type=float,
        default=0.70,
        help="Minimum per-chain sequence coverage for a fallback match.",
    )
    parser.add_argument(
        "--minimum-fallback-pair-consensus",
        type=float,
        default=0.90,
        help="Minimum pair-level consensus for a fallback match.",
    )
    args = parser.parse_args()
    if not args.experimental_manifest.is_file():
        raise FileNotFoundError(args.experimental_manifest)
    if args.workers <= 0 or args.minimum_fallback_aligned_residues <= 0:
        raise ValueError("workers and minimum aligned residues must be positive.")
    if not 1 <= args.uniprot_candidates <= 500:
        raise ValueError("uniprot-candidates must be between 1 and the UniProt API maximum of 500.")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        raise ValueError("timeout must be finite and positive.")
    for name, value in (
        ("minimum-fallback-identity", args.minimum_fallback_identity),
        ("minimum-fallback-chain-coverage", args.minimum_fallback_chain_coverage),
        ("minimum-fallback-pair-consensus", args.minimum_fallback_pair_consensus),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir or args.output_dir / "cache"
    mapping_cache = cache_root / "pdbe_sifts"
    uniprot_cache = cache_root / "uniprot_pdb_xref"
    complex_cache = cache_root / "afdb_complex"
    rows = read_csv_rows(args.experimental_manifest)
    if not rows:
        raise ValueError("Experimental manifest contains no rows.")
    record_ids = [str(row.get("record_id") or "").strip() for row in rows]
    if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Experimental manifest requires unique, non-empty record_id values.")

    def mapping_worker(pdb_id):
        payload = fetch_json_cached(
            PDBE_MAPPING_URL.format(pdb_id=pdb_id),
            mapping_cache / f"{pdb_id}.json",
            args.timeout,
        )
        return pdb_id, payload

    mappings = fetch_many((row["pdb_id"] for row in rows), mapping_worker, args.workers, "SIFTS")

    def uniprot_worker(pdb_id):
        cache_path = uniprot_cache_path(uniprot_cache, pdb_id, args.uniprot_candidates)
        payload = fetch_json_cached(
            uniprot_pdb_search_url(pdb_id, args.uniprot_candidates),
            cache_path,
            args.timeout,
        )
        require_complete_uniprot_search(payload, args.uniprot_candidates)
        return pdb_id, payload

    uniprot_payloads = fetch_many(
        (row["pdb_id"] for row in rows),
        uniprot_worker,
        args.workers,
        "UniProt PDB xref",
    )
    residue_tasks = [
        (
            row["record_id"],
            row["structure_path"],
            row["input_sha256"],
            row["chain_a"],
            row["chain_b"],
        )
        for row in rows
    ]
    observed_residues = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (record_id, residue_ids) in enumerate(
            executor.map(observed_pair_residue_ids, residue_tasks, chunksize=16),
            start=1,
        ):
            observed_residues[record_id] = residue_ids
            if index % 100 == 0 or index == len(residue_tasks):
                print(f"Observed chain residues: {index}/{len(residue_tasks)}", flush=True)
    provisional_rows = []
    for row in rows:
        resolved = {}
        errors = {}
        for suffix in ("a", "b"):
            try:
                residue_ids = observed_residues[row["record_id"]][suffix]
                if len(residue_ids) != len(row[f"sequence_{suffix}"]):
                    raise ValueError(
                        f"Observed residue count changed for chain {row[f'chain_{suffix}']}: "
                        f"{len(residue_ids)} != {len(row[f'sequence_{suffix}'])}"
                    )
                resolved[suffix] = select_chain_accession(
                    mappings[row["pdb_id"]],
                    row["pdb_id"],
                    row[f"chain_{suffix}"],
                    residue_ids,
                    row[f"sequence_{suffix}"],
                    uniprot_payloads[row["pdb_id"]],
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors[suffix] = str(exc)
        provisional_rows.append((row, resolved, errors))
    mapped_rows = []
    for row, resolved, errors in provisional_rows:
        for suffix in ("a", "b"):
            if suffix in resolved:
                continue
            try:
                resolved[suffix] = select_sequence_matched_accession(
                    uniprot_payloads[row["pdb_id"]],
                    row[f"sequence_{suffix}"],
                    minimum_aligned_residues=args.minimum_fallback_aligned_residues,
                    minimum_identity=args.minimum_fallback_identity,
                    minimum_chain_coverage=args.minimum_fallback_chain_coverage,
                    minimum_pair_consensus=args.minimum_fallback_pair_consensus,
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors[suffix] = f"{errors[suffix]}; UniProt sequence fallback: {exc}"
        if len(resolved) != 2:
            mapped_rows.append(
                {
                    **row,
                    "afdb_match_status": "unmatched",
                    "afdb_match_reason_code": "uniprot_chain_mapping_missing",
                    "afdb_match_reason": " | ".join(
                        f"chain {suffix.upper()}: {errors[suffix]}" for suffix in sorted(errors)
                    ),
                }
            )
            continue
        mapped_rows.append(
            {
                **row,
                **mapping_output(resolved["a"], "a"),
                **mapping_output(resolved["b"], "b"),
                "afdb_match_status": "mapped",
                "afdb_match_reason": "",
                "pdbe_mapping_metadata_sha256": sha256_file(str(mapping_cache / f"{row['pdb_id']}.json")),
                "pdbe_mapping_metadata_retrieved_at_utc": file_timestamp_utc(mapping_cache / f"{row['pdb_id']}.json"),
                "uniprot_search_metadata_sha256": sha256_file(
                    str(uniprot_cache_path(uniprot_cache, row["pdb_id"], args.uniprot_candidates))
                ),
                "uniprot_search_metadata_retrieved_at_utc": file_timestamp_utc(
                    uniprot_cache_path(uniprot_cache, row["pdb_id"], args.uniprot_candidates)
                ),
            }
        )
    accessions = {
        str(row[key])
        for row in mapped_rows
        if row["afdb_match_status"] == "mapped"
        for key in ("afdb_accession_a", "afdb_accession_b")
    }

    def complex_worker(accession):
        payload = fetch_json_cached(
            AFDB_COMPLEX_URL.format(accession=accession),
            complex_cache / f"{accession}.json",
            args.timeout,
        )
        return accession, payload

    complexes = fetch_many(accessions, complex_worker, args.workers, "AFDB complex")
    audit_rows = []
    matched_rows = []
    for row in mapped_rows:
        if row["afdb_match_status"] != "mapped":
            audit_rows.append(row)
            continue
        try:
            model, exact_count = select_exact_model(
                str(row["afdb_accession_a"]),
                str(row["afdb_accession_b"]),
                [complexes[str(row["afdb_accession_a"])], complexes[str(row["afdb_accession_b"])]],
            )
            matched = {
                **row,
                "afdb_match_status": "matched",
                "afdb_model_id": model["modelEntityId"],
                "afdb_exact_candidate_count": exact_count,
                "afdb_iptm": finite_metric(model, "complexPredictionAccuracy_ipTM"),
                "afdb_ipsae": finite_metric(model, "complexPredictionAccuracy_ipSAE"),
                "afdb_pdockq": optional_metric(model, "complexPredictionAccuracy_pDockQ"),
                "afdb_pdockq2": optional_metric(model, "complexPredictionAccuracy_pDockQ2"),
                "afdb_lis": optional_metric(model, "complexPredictionAccuracy_LIS"),
                "afdb_provider": model.get("providerId") or "",
                "afdb_model_metadata_sha256": hashlib.sha256(
                    json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "afdb_complex_query_sha256_a": sha256_file(str(complex_cache / f"{row['afdb_accession_a']}.json")),
                "afdb_complex_query_sha256_b": sha256_file(str(complex_cache / f"{row['afdb_accession_b']}.json")),
                "afdb_complex_query_retrieved_at_utc_a": file_timestamp_utc(
                    complex_cache / f"{row['afdb_accession_a']}.json"
                ),
                "afdb_complex_query_retrieved_at_utc_b": file_timestamp_utc(
                    complex_cache / f"{row['afdb_accession_b']}.json"
                ),
            }
            audit_rows.append(matched)
            matched_rows.append(matched)
        except (KeyError, TypeError, ValueError) as exc:
            unmatched = {
                **row,
                "afdb_match_status": "unmatched",
                "afdb_match_reason_code": "afdb_exact_dimer_unavailable",
                "afdb_match_reason": str(exc),
            }
            audit_rows.append(unmatched)

    write_csv_atomic(args.output_dir / "afdb_match_audit.csv", audit_rows)
    write_csv_atomic(args.output_dir / "afdb_matched_candidates.csv", matched_rows)
    reason_counts = Counter(row.get("afdb_match_reason_code") or "matched" for row in audit_rows)
    summary = {
        "schema_version": 5,
        "experimental_record_count": len(rows),
        "uniprot_mapped_record_count": sum(row["afdb_match_status"] != "unmatched" for row in mapped_rows),
        "sifts_mapped_both_chains_record_count": sum(
            row.get("afdb_mapping_method_a") == "pdbe_sifts_chain_mapping"
            and row.get("afdb_mapping_method_b") == "pdbe_sifts_chain_mapping"
            for row in mapped_rows
        ),
        "sequence_fallback_used_record_count": sum(
            "uniprot_pdb_xref_sequence_alignment"
            in {row.get("afdb_mapping_method_a"), row.get("afdb_mapping_method_b")}
            for row in mapped_rows
        ),
        "afdb_matched_record_count": len(matched_rows),
        "afdb_match_fraction": len(matched_rows) / len(rows),
        "status_reason_counts": dict(sorted(reason_counts.items())),
        "experimental_manifest_sha256": sha256_file(str(args.experimental_manifest)),
        "pdbe_mapping_endpoint": PDBE_MAPPING_URL,
        "sifts_candidate_ranking": (
            "semiglobal experimental-sequence coverage and identity within the SIFTS UniProt intervals, "
            "SIFTS-reported identity, aligned residues, and interval length descending; accession ascending"
        ),
        "mapping_coverage_definition": (
            "fraction of the coordinate-derived experimental chain sequence aligned to the accession-specific "
            "SIFTS UniProt intervals; author-residue intersection is used only if a UniProt sequence is unavailable"
        ),
        "uniprot_search_endpoint": UNIPROT_SEARCH_URL,
        "uniprot_sequence_fallback": {
            "candidate_limit": args.uniprot_candidates,
            "minimum_aligned_residues": args.minimum_fallback_aligned_residues,
            "minimum_identity": args.minimum_fallback_identity,
            "minimum_chain_coverage": args.minimum_fallback_chain_coverage,
            "minimum_pair_consensus": args.minimum_fallback_pair_consensus,
            "alignment": (
                "best threshold-passing local or semiglobal correspondence; semiglobal uses free terminal "
                "and penalized internal gaps"
            ),
            "ranking": "coverage, identity, aligned residues descending; length difference and accession ascending",
            "selection_independent_of_afdb_model_availability": True,
        },
        "afdb_complex_endpoint": AFDB_COMPLEX_URL,
        "model_ranking": "ipSAE, ipTM, pDockQ2, pDockQ, LIS descending; model ID ascending",
    }
    dump_json_atomic(summary, args.output_dir / "afdb_match_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
