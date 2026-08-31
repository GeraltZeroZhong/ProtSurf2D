"""Integrity checks for completed benchmark evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from topoppi.file_utils import sha256_file

BENCHMARK_ARTIFACT_FILENAMES = {
    "report_filename": "benchmark_report.json",
    "summary_filename": "benchmark_summary.csv",
    "checkpoint_filename": "benchmark_checkpoint.json",
    "manifest_filename": "benchmark_manifest.csv",
    "failures_filename": "benchmark_failures.csv",
    "per_patch_filename": "benchmark_per_patch.csv",
    "per_face_sample_filename": "benchmark_per_face_sample.csv",
    "per_residue_filename": "benchmark_per_residue.csv.gz",
    "provenance_filename": "benchmark_provenance.csv.gz",
    "optcuts_execution_filename": "benchmark_optcuts_executions.jsonl.gz",
}


def read_json_object(path: Path, label: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def validate_benchmark_evidence_bundle(
    report_path: Path,
    report: Mapping[str, object],
) -> str:
    """Verify the exact artifact set, byte counts, hashes, and config fingerprint."""

    config = report.get("config")
    runtime = report.get("runtime")
    if not isinstance(config, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError(f"Benchmark report lacks config/runtime metadata: {report_path}")
    checksum_name = str(config.get("artifact_checksums_filename") or "benchmark_artifact_checksums.json").strip()
    checksum_path = report_path.parent / checksum_name
    if not checksum_path.is_file():
        raise ValueError(f"Benchmark evidence checksum manifest is missing: {checksum_path}")
    checksums = read_json_object(checksum_path, "Benchmark evidence checksum manifest")
    if checksums.get("algorithm") != "sha256" or not isinstance(checksums.get("artifacts"), list):
        raise ValueError(f"Benchmark evidence checksum manifest is invalid: {checksum_path}")
    if str(checksums.get("config_fingerprint") or "").strip() != str(runtime.get("config_fingerprint") or "").strip():
        raise ValueError(f"Benchmark evidence checksum manifest has a stale config fingerprint: {checksum_path}")

    declared: dict[str, tuple[int, str]] = {}
    for raw in checksums["artifacts"]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Benchmark evidence checksum entry is invalid: {checksum_path}")
        filename = str(raw.get("filename") or "").strip()
        digest = str(raw.get("sha256") or "").strip().lower()
        try:
            byte_count = int(raw.get("bytes"))
        except (TypeError, ValueError):
            byte_count = -1
        if (
            not filename
            or Path(filename).name != filename
            or filename in declared
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or byte_count < 0
        ):
            raise ValueError(f"Benchmark evidence checksum entry is invalid: {checksum_path}")
        declared[filename] = (byte_count, digest)

    expected_name_list = [
        str(config.get(field) or default).strip() for field, default in BENCHMARK_ARTIFACT_FILENAMES.items()
    ]
    if len(set(expected_name_list)) != len(expected_name_list):
        raise ValueError(f"Benchmark configuration reuses an evidence artifact filename: {report_path}")
    expected_names = set(expected_name_list)
    if expected_names != set(declared):
        missing = sorted(expected_names - set(declared))
        unexpected = sorted(set(declared) - expected_names)
        raise ValueError(
            "Benchmark evidence checksum manifest has the wrong artifact set: "
            f"missing={missing}; unexpected={unexpected}"
        )
    configured_report_name = str(
        config.get("report_filename") or BENCHMARK_ARTIFACT_FILENAMES["report_filename"]
    ).strip()
    if report_path.name != configured_report_name:
        raise ValueError(f"Report filename differs from its frozen benchmark configuration: {report_path}")
    for filename in sorted(expected_names):
        artifact = report_path.parent / filename
        if not artifact.is_file():
            raise ValueError(f"Benchmark evidence artifact is missing: {artifact}")
        expected_bytes, expected_sha256 = declared[filename]
        if artifact.stat().st_size != expected_bytes or sha256_file(artifact).lower() != expected_sha256:
            raise ValueError(f"Benchmark evidence artifact checksum differs: {artifact}")
    return sha256_file(report_path)


__all__ = [
    "BENCHMARK_ARTIFACT_FILENAMES",
    "read_json_object",
    "validate_benchmark_evidence_bundle",
]
