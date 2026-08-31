"""Checksum- and URL-bound downloads for AlphaFold DB coordinate files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import requests

from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic


def file_timestamp_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def looks_like_pdb(path: Path) -> bool:
    with path.open("rt", encoding="utf-8", errors="ignore") as handle:
        return any(line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA" for line in handle)


def download_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".download.json")


def _merged_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered or any(start < 1 or end < start for start, end in ordered):
        raise ValueError("UniProt intervals must be non-empty positive inclusive ranges.")
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _metadata_integer(record: Mapping[str, object], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool):
        raise ValueError(f"AFDB metadata field {name} is not an integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AFDB metadata field {name} is not an integer.") from exc
    if integer < 1 or str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"AFDB metadata field {name} is not a positive integer.")
    return integer


def project_uniprot_intervals_to_coordinates(
    intervals: Sequence[tuple[int, int]],
    record: Mapping[str, object],
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    """Project requested UniProt ranges onto AFDB PDB residue numbering."""

    requested = _merged_intervals(intervals)
    uniprot_start = _metadata_integer(record, "uniprotStart")
    uniprot_end = _metadata_integer(record, "uniprotEnd")
    sequence_start = _metadata_integer(record, "sequenceStart")
    sequence_end = _metadata_integer(record, "sequenceEnd")
    if uniprot_end < uniprot_start or sequence_end < sequence_start:
        raise ValueError("AFDB metadata contains a reversed sequence interval.")
    if uniprot_end - uniprot_start != sequence_end - sequence_start:
        raise ValueError("AFDB metadata cannot be represented by an exact residue-number offset.")

    available = []
    for start, end in requested:
        clipped_start = max(start, uniprot_start)
        clipped_end = min(end, uniprot_end)
        if clipped_start <= clipped_end:
            available.append((clipped_start, clipped_end))
    available = _merged_intervals(available) if available else []
    if not available:
        raise ValueError("Requested UniProt intervals do not overlap the AFDB model sequence range.")

    offset = sequence_start - uniprot_start
    coordinate_intervals = [(start + offset, end + offset) for start, end in available]
    requested_count = sum(end - start + 1 for start, end in requested)
    available_count = sum(end - start + 1 for start, end in available)
    return coordinate_intervals, {
        "metadata_uniprot_start": uniprot_start,
        "metadata_uniprot_end": uniprot_end,
        "metadata_sequence_start": sequence_start,
        "metadata_sequence_end": sequence_end,
        "coordinate_residue_number_offset": offset,
        "requested_uniprot_intervals": [list(interval) for interval in requested],
        "available_uniprot_intervals": [list(interval) for interval in available],
        "coordinate_intervals": [list(interval) for interval in coordinate_intervals],
        "requested_uniprot_residue_count": requested_count,
        "available_uniprot_residue_count": available_count,
        "available_uniprot_fraction": available_count / requested_count,
        "requested_interval_truncated_to_model": available_count < requested_count,
    }


def validated_cached_download(url: str, path: Path) -> dict[str, object] | None:
    sidecar = download_sidecar_path(path)
    if not path.is_file() or not sidecar.is_file() or not looks_like_pdb(path):
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or str(payload.get("url") or "") != url:
        return None
    actual_sha256 = sha256_file(path)
    if (
        str(payload.get("sha256") or "").lower() != actual_sha256.lower()
        or int(payload.get("size_bytes", -1)) != path.stat().st_size
    ):
        return None
    retrieved_at = str(payload.get("retrieved_at_utc") or "").strip()
    if not retrieved_at:
        return None
    return {
        "url": url,
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
        "retrieved_at_utc": retrieved_at,
    }


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    raw = response.headers.get("Retry-After") if response is not None else None
    try:
        delay = float(raw) if raw is not None else float(2**attempt)
    except ValueError:
        delay = float(2**attempt)
    return min(max(delay, 0.0), 30.0)


def download_pdb_cached(url: str, path: Path, timeout: float) -> dict[str, object]:
    """Download a PDB and reuse it only when a sidecar binds URL and bytes."""

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("AFDB download timeout must be finite and positive.")
    cached = validated_cached_download(url, path)
    if cached is not None:
        return cached

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_network_error: requests.RequestException | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
        except requests.RequestException as exc:
            last_network_error = exc
            time.sleep(_retry_delay(None, attempt))
            continue
        with response:
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(_retry_delay(response, attempt))
                continue
            response.raise_for_status()
            digest = hashlib.sha256()
            size = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        if not looks_like_pdb(temporary):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"AFDB response is not a PDB coordinate file: {url}")
        os.replace(temporary, path)
        provenance = {
            "url": url,
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "retrieved_at_utc": file_timestamp_utc(path),
        }
        dump_json_atomic(provenance, download_sidecar_path(path))
        return provenance
    temporary.unlink(missing_ok=True)
    detail = f": {last_network_error}" if last_network_error is not None else ""
    raise RuntimeError(f"Transient HTTP failure persisted for {url}{detail}")


__all__ = [
    "download_pdb_cached",
    "download_sidecar_path",
    "file_timestamp_utc",
    "looks_like_pdb",
    "project_uniprot_intervals_to_coordinates",
    "validated_cached_download",
]
