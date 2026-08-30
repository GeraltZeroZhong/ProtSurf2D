#!/usr/bin/env python3
"""Generate ProLIF records and bind them to a benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from topoppi.file_utils import sha256_file, write_csv_atomic
from topoppi.interactions.interaction_engine import generate_prolif_interactions

EXCLUDED_STATUSES = {"0", "false", "no", "exclude", "excluded", "skip", "skipped"}


def included(row: dict[str, str]) -> bool:
    value = str(row.get("include") or row.get("status") or "included").strip().lower()
    return value not in EXCLUDED_STATUSES


def structure_path(row: dict[str, str], manifest: Path, structure_dir: Path | None) -> Path:
    filename = str(row.get("pdb") or "").strip()
    if structure_dir is not None:
        if not filename:
            raise ValueError("Every included row needs pdb when --structure-dir is used.")
        return structure_dir / filename
    declared = str(row.get("staged_structure_path") or row.get("structure_path") or "").strip()
    if declared:
        path = Path(declared).expanduser()
        return path if path.is_absolute() else manifest.parent / path
    if not filename:
        raise ValueError("Every included row needs pdb, staged_structure_path, or structure_path.")
    return manifest.parent / filename


def prepare_manifest(
    manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    *,
    structure_dir: Path | None = None,
) -> int:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest has no header.")
        rows = list(reader)
        fields = [
            name for name in reader.fieldnames if name not in {"prolif_path", "interaction_file", "interaction_sha256"}
        ]

    for name in ("input_sha256", "prolif_file", "prolif_sha256"):
        if name not in fields:
            fields.append(name)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = 0
    output_rows = []
    for row in rows:
        row = {
            key: value
            for key, value in row.items()
            if key not in {"prolif_path", "interaction_file", "interaction_sha256"}
        }
        if not included(row):
            row["prolif_file"] = ""
            row["prolif_sha256"] = ""
            output_rows.append(row)
            continue

        source = structure_path(row, manifest, structure_dir).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_sha256 = sha256_file(source)
        declared_sha256 = str(row.get("input_sha256") or "").strip().lower()
        if declared_sha256 and declared_sha256 != actual_sha256:
            raise ValueError(
                f"Input checksum mismatch for {source.name}: expected {declared_sha256}, got {actual_sha256}"
            )

        chain_a = str(row.get("chain_a") or "").strip()
        chain_b = str(row.get("chain_b") or "").strip()
        if not chain_a or not chain_b:
            raise ValueError(f"Included row {source.name!r} needs chain_a and chain_b.")

        prolif_path = Path(
            generate_prolif_interactions(
                source,
                chain_a,
                chain_b,
                source_sha256=actual_sha256,
                output_dir=output_dir,
            )
        ).resolve()
        row["input_sha256"] = actual_sha256
        row["prolif_file"] = os.path.relpath(prolif_path, output_manifest.parent)
        row["prolif_sha256"] = sha256_file(prolif_path)
        output_rows.append(row)
        prepared += 1

    write_csv_atomic(output_manifest, output_rows, fields)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chain-pair ProLIF JSON files and write their paths and SHA-256 values to a manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Input benchmark manifest CSV.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Prepared benchmark manifest CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for ProLIF JSON files. The default is a prolif directory beside the output manifest.",
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        help="Directory containing every pdb filename; overrides structure paths stored in the manifest.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.output_manifest.parent / "prolif"
    prepared = prepare_manifest(
        args.manifest.resolve(),
        args.output_manifest.resolve(),
        output_dir.resolve(),
        structure_dir=args.structure_dir.resolve() if args.structure_dir else None,
    )
    print(f"Prepared ProLIF evidence for {prepared} included structures: {args.output_manifest.resolve()}")


if __name__ == "__main__":
    main()
