#!/usr/bin/env python3
"""Build a traceable dominant-chain-pair table from the PDBbind 2020R1 PP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import is_aa

from topoppi.file_utils import sha256_file, write_csv_atomic
from topoppi.io.io_loader import PDBLoader
from topoppi.json_utils import dump_json_atomic

INDEX_PATTERN = re.compile(
    r"^(?P<pdb>[0-9A-Za-z]{4})\s+(?P<resolution>\S+)\s+(?P<year>\d{4})\s+"
    r"(?P<binding>\S+)\s+//.*\((?P<receptor>[0-9A-Za-z]+)\|(?P<ligand>[0-9A-Za-z]+)\)"
)
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_EXPERIMENT_QUERY = """
query TopoPPIExperimentMetadata($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl { method }
    rcsb_entry_info { resolution_combined }
  }
}
""".strip()


def parse_index(path: Path) -> list[dict[str, str]]:
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = INDEX_PATTERN.match(line)
        if not match:
            raise ValueError(f"Cannot parse index line {line_number}: {raw_line}")
        record = match.groupdict()
        record["pdb_id"] = record.pop("pdb").lower()
        record["index_line_number"] = str(line_number)
        record["index_line_sha256"] = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
        records.append(record)
    identifiers = [record["pdb_id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("PDBbind PP index contains duplicate PDB identifiers.")
    return records


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    raw = response.headers.get("Retry-After") if response is not None else None
    try:
        delay = float(raw) if raw is not None else float(2**attempt)
    except ValueError:
        delay = float(2**attempt)
    return min(max(delay, 0.0), 30.0)


def experimental_method_group(methods: list[str]) -> str:
    normalized = set(methods)
    if normalized == {"X-RAY DIFFRACTION"}:
        return "x_ray_diffraction"
    if normalized == {"SOLUTION NMR"}:
        return "solution_nmr"
    if normalized == {"ELECTRON MICROSCOPY"}:
        return "electron_microscopy"
    return "multiple_or_other"


def normalized_experiment_record(
    *,
    pdb_id: str,
    methods: list[object],
    resolutions: list[object],
    source: str,
    source_details: dict[str, object] | None = None,
) -> dict[str, object]:
    if not pdb_id:
        raise ValueError("RCSB metadata record has no PDB identifier.")
    normalized_methods = sorted({str(method).strip().upper() for method in methods if str(method).strip()})
    if not normalized_methods:
        raise ValueError(f"RCSB metadata has no experimental method for {pdb_id}.")
    normalized_resolutions = []
    for value in resolutions:
        if value in {None, "", ".", "?"}:
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"RCSB metadata has an invalid resolution for {pdb_id}.")
        normalized_resolutions.append(numeric)
    normalized_resolutions = sorted(set(normalized_resolutions))
    return {
        "pdb_id": pdb_id.lower(),
        "experimental_methods": normalized_methods,
        "resolution_combined_angstrom": normalized_resolutions,
        "resolution_angstrom": normalized_resolutions[0] if len(normalized_resolutions) == 1 else "",
        "experimental_method_group": experimental_method_group(normalized_methods),
        "experimental_method_contains_nmr": any("NMR" in method for method in normalized_methods),
        "source": source,
        **(source_details or {}),
    }


def experiment_record_from_graphql(entry: dict[str, object]) -> dict[str, object]:
    pdb_id = str(entry.get("rcsb_id") or "").strip().lower()
    methods = [block.get("method") for block in entry.get("exptl") or [] if isinstance(block, dict)]
    entry_info = entry.get("rcsb_entry_info") or {}
    resolutions = entry_info.get("resolution_combined") or [] if isinstance(entry_info, dict) else []
    return normalized_experiment_record(
        pdb_id=pdb_id,
        methods=methods,
        resolutions=list(resolutions),
        source="rcsb_data_api_graphql",
    )


def experiment_record_from_mmcif(path: Path, pdb_id: str, url: str) -> dict[str, object]:
    values = MMCIF2Dict(str(path))
    methods = values.get("_exptl.method", [])
    if not isinstance(methods, list):
        methods = [methods]
    resolutions = []
    for key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        field = values.get(key, [])
        resolutions.extend(field if isinstance(field, list) else [field])
    return normalized_experiment_record(
        pdb_id=pdb_id,
        methods=methods,
        resolutions=resolutions,
        source="rcsb_official_mmcif_fallback",
        source_details={
            "source_url": url,
            "source_sha256": sha256_file(path),
        },
    )


def _graphql_batch(entry_ids: list[str], timeout: float) -> list[dict[str, object]]:
    response: requests.Response | None = None
    for attempt in range(5):
        try:
            response = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": RCSB_EXPERIMENT_QUERY, "variables": {"ids": entry_ids}},
                timeout=timeout,
            )
        except requests.RequestException:
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(_retry_delay(response, attempt))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise ValueError(f"RCSB GraphQL returned errors: {payload['errors']}")
        data = payload.get("data") or {}
        entries = data.get("entries") or [] if isinstance(data, dict) else []
        return [entry for entry in entries if isinstance(entry, dict)]
    raise RuntimeError("Transient RCSB GraphQL failure persisted after five attempts.")


def _download_mmcif_cached(pdb_id: str, directory: Path, timeout: float) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pdb_id.lower()}.cif"
    url = RCSB_MMCIF_URL.format(pdb_id=pdb_id.upper())
    if path.is_file() and path.read_text(encoding="utf-8", errors="ignore").lstrip().startswith("data_"):
        return path, url
    temporary = path.with_suffix(".cif.part")
    response: requests.Response | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException:
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(_retry_delay(response, attempt))
            continue
        response.raise_for_status()
        temporary.write_bytes(response.content)
        if not temporary.read_text(encoding="utf-8", errors="ignore").lstrip().startswith("data_"):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"RCSB fallback response is not an mmCIF file for {pdb_id}.")
        os.replace(temporary, path)
        return path, url
    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Transient RCSB mmCIF failure persisted for {pdb_id}.")


def validate_experiment_metadata_cache(payload: object, pdb_ids: list[str]) -> dict[str, object]:
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("RCSB experiment metadata cache has an unsupported schema.")
    expected = sorted({pdb_id.lower() for pdb_id in pdb_ids})
    if payload.get("requested_pdb_ids") != expected:
        raise ValueError("RCSB experiment metadata cache targets a different PDB identifier set.")
    expected_query_sha256 = hashlib.sha256(RCSB_EXPERIMENT_QUERY.encode("utf-8")).hexdigest()
    if payload.get("graphql_query_sha256") != expected_query_sha256:
        raise ValueError("RCSB experiment metadata cache was built with a different GraphQL query.")
    records = payload.get("records")
    if not isinstance(records, dict) or sorted(records) != expected:
        raise ValueError("RCSB experiment metadata cache is incomplete.")
    for pdb_id, record in records.items():
        if not isinstance(record, dict) or not record.get("experimental_methods"):
            raise ValueError(f"RCSB experiment metadata cache has an invalid record for {pdb_id}.")
        normalized = normalized_experiment_record(
            pdb_id=pdb_id,
            methods=list(record["experimental_methods"]),
            resolutions=list(record.get("resolution_combined_angstrom") or []),
            source=str(record.get("source") or ""),
        )
        for field in (
            "experimental_methods",
            "resolution_combined_angstrom",
            "experimental_method_group",
            "experimental_method_contains_nmr",
        ):
            if record.get(field) != normalized[field]:
                raise ValueError(f"RCSB experiment metadata cache has inconsistent {field} for {pdb_id}.")
    return payload


def rcsb_experiment_metadata(
    pdb_ids: list[str],
    cache_path: Path,
    *,
    timeout: float,
    batch_size: int,
) -> dict[str, object]:
    requested = sorted({pdb_id.lower() for pdb_id in pdb_ids})
    if cache_path.is_file():
        return validate_experiment_metadata_cache(
            json.loads(cache_path.read_text(encoding="utf-8")),
            requested,
        )

    records: dict[str, dict[str, object]] = {}
    for start in range(0, len(requested), batch_size):
        batch = [pdb_id.upper() for pdb_id in requested[start : start + batch_size]]
        for entry in _graphql_batch(batch, timeout):
            record = experiment_record_from_graphql(entry)
            pdb_id = str(record["pdb_id"])
            if pdb_id in records:
                raise ValueError(f"RCSB GraphQL returned duplicate metadata for {pdb_id}.")
            records[pdb_id] = record

    missing_graphql = sorted(set(requested) - set(records))
    mmcif_dir = cache_path.parent / "rcsb_mmcif_fallback"
    for pdb_id in missing_graphql:
        path, url = _download_mmcif_cached(pdb_id, mmcif_dir, timeout)
        records[pdb_id] = experiment_record_from_mmcif(path, pdb_id, url)
    payload: dict[str, object] = {
        "schema_version": 1,
        "retrieved_at_utc": _utc_now(),
        "requested_pdb_ids": requested,
        "graphql_url": RCSB_GRAPHQL_URL,
        "graphql_query": RCSB_EXPERIMENT_QUERY,
        "graphql_query_sha256": hashlib.sha256(RCSB_EXPERIMENT_QUERY.encode("utf-8")).hexdigest(),
        "graphql_missing_pdb_ids": missing_graphql,
        "records": dict(sorted(records.items())),
    }
    validate_experiment_metadata_cache(payload, requested)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(payload, cache_path)
    return payload


def attach_experiment_metadata(
    records: list[dict[str, str]],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    metadata_records = metadata["records"]
    enriched = []
    for record in records:
        experiment = metadata_records[record["pdb_id"]]
        methods = list(experiment["experimental_methods"])
        resolutions = list(experiment["resolution_combined_angstrom"])
        enriched.append(
            {
                **record,
                "pdbbind_index_resolution_angstrom": record["resolution"],
                "structure_method": "; ".join(methods),
                "experimental_methods_json": json.dumps(methods, separators=(",", ":")),
                "experimental_method_group": experiment["experimental_method_group"],
                "experimental_method_contains_nmr": experiment["experimental_method_contains_nmr"],
                "rcsb_resolution_combined_angstrom_json": json.dumps(resolutions, separators=(",", ":")),
                "rcsb_experiment_metadata_source": experiment["source"],
                "resolution_angstrom": resolutions[0] if len(resolutions) == 1 else "",
                "resolution_angstrom_semantics": "single_official_rcsb_resolution_combined_value_or_empty",
            }
        )
    return enriched


def chain_sequence(loader: PDBLoader, chain_id: str) -> str:
    return "".join(
        protein_letters_3to1_extended[residue.get_resname()]
        for residue in loader.model[chain_id]
        if is_aa(residue, standard=False) and residue.get_resname() in protein_letters_3to1_extended
    )


def process_record(task: tuple[dict[str, str], str, float, int]) -> dict[str, object]:
    record, structure_dir, cutoff, min_residues = task
    pdb_id = record["pdb_id"]
    path = Path(structure_dir) / f"{pdb_id}_complex.pdb"
    base: dict[str, object] = {
        **record,
        "pdb": path.name,
        "record_id": f"pdbbind2020r1:{pdb_id}",
        "structure_path": str(path.resolve()),
        "receptor_chain_group": record["receptor"],
        "ligand_chain_group": record["ligand"],
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        loader = PDBLoader(str(path))
        chain_a, chain_b, details = loader.select_contact_chain_pair_between_groups(
            tuple(record["receptor"]),
            tuple(record["ligand"]),
            min_chain_residues=min_residues,
            distance_cutoff=cutoff,
        )
        sequence_a = chain_sequence(loader, chain_a)
        sequence_b = chain_sequence(loader, chain_b)
        if not sequence_a or not sequence_b:
            raise ValueError("Selected chain pair has an empty protein sequence.")
        if int(details["contact_residue_pair_count"]) == 0:
            raise ValueError(f"Selected chain pair has no heavy-atom contact within {cutoff:g} Å.")
        return {
            **base,
            "status": "accepted",
            "reason": "",
            "input_sha256": sha256_file(str(path)),
            "chain_a": chain_a,
            "chain_b": chain_b,
            "sequence_a": sequence_a,
            "sequence_b": sequence_b,
            "sequence_a_sha256": hashlib.sha256(sequence_a.encode("ascii")).hexdigest(),
            "sequence_b_sha256": hashlib.sha256(sequence_b.encode("ascii")).hexdigest(),
            "chain_a_residue_count": len(sequence_a),
            "chain_b_residue_count": len(sequence_b),
            "coordinate_header_structure_method": str(
                loader.structure.header.get("structure_method") or "not_declared"
            ),
            **details,
        }
    except (FileNotFoundError, OSError, PDBException, ValueError) as exc:
        return {**base, "status": "excluded", "reason": str(exc)}


def write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            for side in ("a", "b"):
                handle.write(f">{row['record_id']}|{side}|chain={row[f'chain_{side}']}\n{row[f'sequence_{side}']}\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select dominant chain pairs from the PDBbind 2020R1 PP archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", required=True, type=Path, help="PDBbind protein-protein index file.")
    parser.add_argument(
        "--structure-dir",
        required=True,
        type=Path,
        help="Directory containing the PDBbind complex structures.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for pair tables and metadata.")
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=6.0,
        help="Maximum heavy-atom contact distance in angstroms.",
    )
    parser.add_argument(
        "--min-chain-residues",
        type=int,
        default=1,
        help="Minimum protein residues required in a candidate chain.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Structure-analysis worker processes.")
    parser.add_argument(
        "--rcsb-metadata-cache",
        type=Path,
        help="Path for reusable RCSB experiment metadata JSON.",
    )
    parser.add_argument("--metadata-timeout", type=float, default=60.0, help="RCSB request timeout in seconds.")
    parser.add_argument(
        "--metadata-batch-size",
        type=int,
        default=100,
        help="PDB entries requested per RCSB metadata batch.",
    )
    args = parser.parse_args()
    if not args.index.is_file():
        raise FileNotFoundError(args.index)
    if not args.structure_dir.is_dir():
        raise FileNotFoundError(args.structure_dir)
    if not math.isfinite(args.distance_cutoff) or args.distance_cutoff <= 0.0:
        raise ValueError("distance-cutoff must be finite and positive.")
    if args.min_chain_residues <= 0 or args.workers <= 0:
        raise ValueError("min-chain-residues and workers must be positive integers.")
    if not math.isfinite(args.metadata_timeout) or args.metadata_timeout <= 0.0:
        raise ValueError("metadata-timeout must be finite and positive.")
    if args.metadata_batch_size <= 0:
        raise ValueError("metadata-batch-size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_records = parse_index(args.index)
    if not index_records:
        raise ValueError("PDBbind PP index contains no records.")
    metadata_cache = args.rcsb_metadata_cache or args.output_dir / "cache" / "rcsb_experiment_metadata.json"
    experiment_metadata = rcsb_experiment_metadata(
        [record["pdb_id"] for record in index_records],
        metadata_cache,
        timeout=float(args.metadata_timeout),
        batch_size=int(args.metadata_batch_size),
    )
    index_records = attach_experiment_metadata(index_records, experiment_metadata)
    tasks = [
        (record, str(args.structure_dir), float(args.distance_cutoff), int(args.min_chain_residues))
        for record in index_records
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(process_record, tasks, chunksize=16))
    accepted = [row for row in rows if row["status"] == "accepted"]
    write_csv_atomic(args.output_dir / "pdbbind_r1_chain_pair_audit.csv", rows)
    write_csv_atomic(args.output_dir / "pdbbind_r1_selected_pairs.csv", accepted)
    write_fasta(args.output_dir / "pdbbind_r1_selected_chains.fasta", accepted)
    summary = {
        "schema_version": 2,
        "index_path": str(args.index.resolve()),
        "index_sha256": sha256_file(str(args.index)),
        "structure_dir": str(args.structure_dir.resolve()),
        "index_record_count": len(index_records),
        "accepted_record_count": len(accepted),
        "excluded_record_count": len(rows) - len(accepted),
        "distance_cutoff_angstrom": float(args.distance_cutoff),
        "minimum_chain_residues": int(args.min_chain_residues),
        "rcsb_experiment_metadata_cache": str(metadata_cache.resolve()),
        "rcsb_experiment_metadata_cache_sha256": sha256_file(metadata_cache),
        "rcsb_graphql_query_sha256": experiment_metadata["graphql_query_sha256"],
        "rcsb_graphql_missing_record_count": len(experiment_metadata["graphql_missing_pdb_ids"]),
        "experimental_method_counts": dict(sorted(Counter(row["structure_method"] for row in accepted).items())),
        "experimental_method_group_counts": dict(
            sorted(Counter(row["experimental_method_group"] for row in accepted).items())
        ),
        "experimental_method_contains_nmr_count": sum(
            bool(row["experimental_method_contains_nmr"]) for row in accepted
        ),
    }
    dump_json_atomic(summary, args.output_dir / "pdbbind_r1_selection_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
