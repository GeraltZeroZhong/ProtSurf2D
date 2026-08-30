#!/usr/bin/env python3
"""Select a deterministic size-stratified, cluster-diverse manifest subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

EXCLUDED_STATUSES = {"0", "false", "no", "exclude", "excluded", "skip", "skipped"}
ALLOWED_ANALYSIS_SPLITS = {"development", "test", "exploratory"}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest has no header.")
        return list(reader), list(reader.fieldnames)


def stable_key(seed: int, row: dict[str, str]) -> str:
    identity = row.get("record_id") or row.get("pdb") or ""
    return hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()


def structure_size(row: dict[str, str]) -> int:
    try:
        counts = [float(row["chain_a_residue_count"]), float(row["chain_b_residue_count"])]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Size-stratified selection requires chain_a_residue_count and chain_b_residue_count."
        ) from None
    if any(not math.isfinite(value) or value <= 0.0 or not value.is_integer() for value in counts):
        raise ValueError("Chain residue counts must be finite positive integers.")
    return sum(int(value) for value in counts)


def analysis_split(row: dict[str, str]) -> str:
    value = str(row.get("analysis_split") or "").strip().lower()
    if value not in ALLOWED_ANALYSIS_SPLITS:
        raise ValueError(
            "Included manifest rows require analysis_split in: " + ", ".join(sorted(ALLOWED_ANALYSIS_SPLITS))
        )
    return value


def allocate(total: int, stratum_sizes: list[int]) -> list[int]:
    if total > sum(stratum_sizes):
        raise ValueError("Requested subset is larger than the eligible manifest.")
    exact = [total * size / sum(stratum_sizes) for size in stratum_sizes]
    result = [min(size, int(value)) for size, value in zip(stratum_sizes, exact, strict=True)]
    order = sorted(
        range(len(stratum_sizes)),
        key=lambda index: (exact[index] - result[index], stratum_sizes[index], -index),
        reverse=True,
    )
    while sum(result) < total:
        progressed = False
        for index in order:
            if result[index] < stratum_sizes[index]:
                result[index] += 1
                progressed = True
                if sum(result) == total:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the requested subset across strata.")
    return result


def choose_group_diverse(
    rows: list[dict[str, str]],
    count: int,
    seed: int,
    diversity_key: str,
) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: stable_key(seed, row))
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        group = str(
            row.get(diversity_key)
            or row.get("family_id")
            or row.get("cluster_id")
            or row.get("record_id")
            or row.get("pdb")
        ).strip()
        by_group[group].append(row)
    selected = []
    groups = sorted(by_group, key=lambda value: stable_key(seed, {"record_id": value}))
    while len(selected) < count:
        progressed = False
        for group in groups:
            if by_group[group]:
                selected.append(by_group[group].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a deterministic, size-stratified benchmark subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Source benchmark manifest CSV.")
    parser.add_argument(
        "--output-manifest",
        required=True,
        type=Path,
        help="Path for the selected manifest CSV.",
    )
    parser.add_argument(
        "--output-record-ids",
        required=True,
        type=Path,
        help="Path for the selected record-ID text file.",
    )
    parser.add_argument("--output-summary", type=Path, help="Optional path for the selection summary JSON.")
    parser.add_argument("--count", required=True, type=int, help="Number of records to select.")
    parser.add_argument(
        "--analysis-split",
        default="test",
        help="Manifest split from which records are selected.",
    )
    parser.add_argument("--size-strata", type=int, default=8, help="Number of structure-size strata.")
    parser.add_argument("--seed", type=int, default=20260817, help="Seed for deterministic within-stratum ranking.")
    parser.add_argument(
        "--diversity-key",
        choices=("family_id", "cluster_id"),
        default="family_id",
        help="Manifest field used to spread selections across groups.",
    )
    args = parser.parse_args()
    if args.count <= 0 or args.size_strata <= 0:
        raise ValueError("count and size-strata must be positive.")

    rows, fields = read_rows(args.manifest)
    split = args.analysis_split.strip().lower()
    if split not in ALLOWED_ANALYSIS_SPLITS:
        raise ValueError("analysis-split must be development, test, or exploratory.")
    included_rows = [
        row
        for row in rows
        if str(row.get("include") or row.get("status") or "included").strip().lower() not in EXCLUDED_STATUSES
    ]
    eligible = [row for row in included_rows if analysis_split(row) == split]
    if args.count > len(eligible):
        raise ValueError(f"Requested {args.count} rows from only {len(eligible)} eligible rows.")
    eligible_record_ids = [str(row.get("record_id") or "").strip() for row in eligible]
    if any(not value for value in eligible_record_ids) or len(set(eligible_record_ids)) != len(eligible_record_ids):
        raise ValueError("Eligible rows require unique, non-empty record_id values.")
    ranked = sorted(eligible, key=lambda row: (structure_size(row), row.get("record_id") or row.get("pdb")))
    stratum_count = min(args.size_strata, len(ranked))
    boundaries = [index * len(ranked) // stratum_count for index in range(stratum_count + 1)]
    strata = [ranked[boundaries[index] : boundaries[index + 1]] for index in range(stratum_count)]
    counts = allocate(args.count, [len(rows_in_stratum) for rows_in_stratum in strata])
    selected = []
    for index, (rows_in_stratum, count) in enumerate(zip(strata, counts, strict=True)):
        selected.extend(
            choose_group_diverse(
                rows_in_stratum,
                count,
                args.seed + index,
                args.diversity_key,
            )
        )
    selected.sort(key=lambda row: (structure_size(row), row.get("record_id") or row.get("pdb")))
    record_ids = [str(row.get("record_id") or "").strip() for row in selected]

    summary_path = args.output_summary or args.output_manifest.with_suffix(".selection.json")
    output_paths = {
        args.output_manifest.resolve(),
        args.output_record_ids.resolve(),
        summary_path.resolve(),
    }
    if len(output_paths) != 3 or args.manifest.resolve() in output_paths:
        raise ValueError("Source manifest, selected manifest, record IDs, and summary require distinct paths.")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=args.output_manifest.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    os.replace(temporary, args.output_manifest)
    args.output_record_ids.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output_record_ids.parent, delete=False) as handle:
        temporary_ids = Path(handle.name)
        handle.write("\n".join(record_ids) + "\n")
    os.replace(temporary_ids, args.output_record_ids)
    summary = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(str(args.manifest)),
        "analysis_split": split,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "size_strata": stratum_count,
        "stratum_allocations": counts,
        "diversity_key": args.diversity_key,
        "unique_selected_family_count": len(
            {str(row.get("family_id") or "").strip() for row in selected if str(row.get("family_id") or "").strip()}
        ),
        "unique_selected_cluster_count": len(
            {str(row.get("cluster_id") or "").strip() for row in selected if str(row.get("cluster_id") or "").strip()}
        ),
        "random_seed": args.seed,
        "output_manifest_sha256": sha256_file(str(args.output_manifest)),
        "output_record_ids_sha256": sha256_file(str(args.output_record_ids)),
    }
    summary["output_summary"] = str(summary_path.resolve())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
