#!/usr/bin/env python3
"""Materialize an exact, checksum-verified benchmark input directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from topoppi.file_utils import sha256_file, write_csv_atomic
from topoppi.json_utils import dump_json_atomic

EXCLUDED_STATUSES = {"0", "false", "no", "exclude", "excluded", "skip", "skipped"}
ALLOWED_ANALYSIS_SPLITS = {"development", "test", "exploratory"}


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest has no header.")
        return list(reader), list(reader.fieldnames)


def included(row: dict[str, str]) -> bool:
    value = str(row.get("include") or row.get("status") or "included").strip().lower()
    return value not in EXCLUDED_STATUSES


def analysis_split(row: dict[str, str]) -> str:
    value = str(row.get("analysis_split") or "").strip().lower()
    if value not in ALLOWED_ANALYSIS_SPLITS:
        raise ValueError(
            "Included manifest rows require analysis_split in: " + ", ".join(sorted(ALLOWED_ANALYSIS_SPLITS))
        )
    return value


def source_path(row: dict[str, str], source_dir: Path | None) -> Path:
    declared = str(row.get("structure_path") or "").strip()
    if declared:
        path = Path(declared).expanduser()
        if not path.is_absolute():
            raise ValueError(f"structure_path must be absolute: {declared}")
        return path
    filename = str(row.get("pdb") or "").strip()
    if not filename or source_dir is None:
        raise ValueError("Every selected row needs structure_path or --source-dir plus pdb.")
    return source_dir / filename


def materialize(source: Path, target: Path, *, copy: bool) -> str:
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Staged target is not a file: {target}")
        return "reused"
    if copy:
        shutil.copy2(source, target)
        return "copied"
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied_cross_filesystem"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize structure files selected by a benchmark manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Source benchmark manifest CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for staged structures and manifest.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Structure directory used when manifest rows provide only a pdb filename.",
    )
    parser.add_argument(
        "--analysis-split",
        action="append",
        default=[],
        help="Include one analysis split; repeat to include multiple splits.",
    )
    parser.add_argument(
        "--record-id-file",
        type=Path,
        help="Text file containing the record IDs to include, one per line.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Create independent file copies; the default uses hard links when possible.",
    )
    args = parser.parse_args()

    rows, fields = read_manifest(args.manifest)
    split_filter = {value.strip().lower() for value in args.analysis_split if value.strip()}
    if not split_filter.issubset(ALLOWED_ANALYSIS_SPLITS):
        raise ValueError("analysis-split must be development, test, or exploratory.")
    record_filter = None
    if args.record_id_file:
        record_filter = {
            line.strip()
            for line in args.record_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not record_filter:
            raise ValueError("Record-ID filter is empty.")

    selected = []
    seen_names: dict[str, str] = {}
    seen_record_ids = set()
    for row in rows:
        if not included(row):
            continue
        split = analysis_split(row)
        if split_filter and split not in split_filter:
            continue
        record_id = str(row.get("record_id") or "").strip()
        if record_filter is not None and record_id not in record_filter:
            continue
        filename = str(row.get("pdb") or "").strip()
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Unsafe or missing pdb filename: {filename!r}")
        source = source_path(row, args.source_dir.resolve() if args.source_dir else None).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = str(row.get("input_sha256") or "").strip().lower()
        if len(expected) != 64:
            raise ValueError(f"Missing or malformed input_sha256 for {filename}")
        actual = sha256_file(str(source))
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {source}: expected {expected}, got {actual}")
        if filename in seen_names:
            raise ValueError(f"Duplicate staged filename: {filename}")
        seen_names[filename] = actual
        if not record_id:
            raise ValueError(f"Missing record_id for {filename}")
        if record_id in seen_record_ids:
            raise ValueError(f"Duplicate staged record_id: {record_id}")
        seen_record_ids.add(record_id)
        selected.append((row, source, filename, actual))

    if record_filter is not None:
        missing = sorted(record_filter - seen_record_ids)
        if missing:
            raise ValueError(f"Record IDs absent from selected manifest rows: {missing[:8]}")
    if not selected:
        raise ValueError("No included manifest row passed the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_existing = {filename for _row, _source, filename, _sha in selected}
    allowed_existing.update({"benchmark_manifest.csv", "staging_summary.json"})
    unexpected = sorted(path.name for path in args.output_dir.iterdir() if path.name not in allowed_existing)
    if unexpected:
        raise ValueError(f"Output directory contains unplanned entries: {unexpected[:8]}")

    actions: dict[str, int] = {}
    staged_rows = []
    for row, source, filename, expected in selected:
        target = args.output_dir / filename
        action = materialize(source, target, copy=args.copy)
        if sha256_file(str(target)) != expected:
            raise ValueError(f"Staged checksum mismatch: {target}")
        actions[action] = actions.get(action, 0) + 1
        staged_rows.append(
            {
                **row,
                "pdb": filename,
                "input_sha256": expected,
                "staged_structure_path": str(target.resolve()),
            }
        )

    output_fields = [*fields]
    if "staged_structure_path" not in output_fields:
        output_fields.append("staged_structure_path")
    manifest_path = args.output_dir / "benchmark_manifest.csv"
    write_csv_atomic(manifest_path, staged_rows, output_fields)
    summary = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(str(args.manifest)),
        "staged_manifest": str(manifest_path.resolve()),
        "staged_manifest_sha256": sha256_file(str(manifest_path)),
        "selected_structure_count": len(staged_rows),
        "analysis_split_filter": sorted(split_filter),
        "record_id_filter_file": str(args.record_id_file.resolve()) if args.record_id_file else None,
        "materialization_actions": actions,
    }
    dump_json_atomic(summary, args.output_dir / "staging_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
