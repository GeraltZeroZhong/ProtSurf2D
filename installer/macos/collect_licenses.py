"""Collect third-party package metadata and bundled license texts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import site
from importlib import metadata
from pathlib import Path

LICENSE_NAMES = ("license", "copying", "notice", "copyright")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", value).strip("-") or "package"


def _license_entry_path(value: str) -> Path:
    parts = []
    for part in re.split(r"[/\\]+", value):
        if not part or part == ".":
            continue
        parts.append("_parent" if part == ".." else _safe_name(part))
    return Path(*parts) if parts else Path("LICENSE")


def _copy_license_files(source: Path, destination: Path, output_dir: Path) -> list[dict[str, str]]:
    if not source.is_dir():
        return []
    copied = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "path": str(target.relative_to(output_dir)),
            }
        )
    return copied


def _conda_records(prefix: Path, license_root: Path) -> list[dict[str, object]]:
    records = []
    for metadata_file in sorted((prefix / "conda-meta").glob("*.json")):
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        name = str(payload.get("name") or metadata_file.stem)
        version = str(payload.get("version") or "unknown")
        source_url = str(payload.get("url") or "")
        link = payload.get("link") if isinstance(payload.get("link"), dict) else {}
        extracted_dir = Path(str(payload.get("extracted_package_dir") or link.get("source") or ""))
        destination = license_root / "conda" / f"{_safe_name(name)}-{_safe_name(version)}"
        copied = _copy_license_files(
            extracted_dir / "info" / "licenses",
            destination,
            license_root.parent,
        )
        records.append(
            {
                "manager": "conda",
                "name": name,
                "version": version,
                "license": str(payload.get("license") or "unspecified"),
                "source": source_url if source_url.startswith(("https://", "http://")) else "",
                "license_files": copied,
            }
        )
    return records


def _python_records(prefix: Path, license_root: Path) -> list[dict[str, object]]:
    search_paths = [Path(path) for path in site.getsitepackages() if Path(path).is_relative_to(prefix)]
    records = []
    for distribution in sorted(
        metadata.distributions(path=[str(path) for path in search_paths]),
        key=lambda item: (item.metadata.get("Name", "").lower(), item.version),
    ):
        name = distribution.metadata.get("Name") or "unknown"
        version = distribution.version or "unknown"
        destination = license_root / "python" / f"{_safe_name(name)}-{_safe_name(version)}"
        copied = []
        declared_license_files = set(distribution.metadata.get_all("License-File") or ())
        for entry in distribution.files or ():
            entry_name = str(entry)
            filename = Path(entry_name).name.lower()
            if (
                not filename.startswith(LICENSE_NAMES)
                and entry_name not in declared_license_files
                and "/licenses/" not in f"/{entry_name.lower()}"
            ):
                continue
            source = Path(distribution.locate_file(entry))
            if not source.is_file():
                continue
            target = destination / _license_entry_path(entry_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(
                {
                    "path": str(target.relative_to(license_root.parent)),
                }
            )
        license_expression = distribution.metadata.get("License-Expression")
        license_name = license_expression or distribution.metadata.get("License") or "unspecified"
        source_url = distribution.metadata.get("Home-page") or ""
        records.append(
            {
                "manager": "python",
                "name": name,
                "version": version,
                "license": license_name.strip() or "unspecified",
                "source": source_url if source_url.startswith(("https://", "http://")) else "",
                "license_files": sorted(
                    {item["path"]: item for item in copied}.values(),
                    key=lambda item: item["path"],
                ),
            }
        )
    return records


def _merge_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        key = (
            re.sub(r"[-_.]+", "-", str(record["name"])).lower(),
            str(record["version"]),
        )
        if key not in merged:
            merged[key] = {
                "name": record["name"],
                "version": record["version"],
                "managers": [record["manager"]],
                "license": record["license"],
                "source": record["source"],
                "license_files": list(record["license_files"]),
            }
            continue

        current = merged[key]
        current["managers"] = sorted({str(item) for item in current["managers"]} | {str(record["manager"])})
        current_license = str(current["license"]).strip()
        candidate_license = str(record["license"]).strip()
        if current_license.lower() in {"", "unknown", "unspecified"} and candidate_license:
            current["license"] = candidate_license
        if not current["source"] and record["source"]:
            current["source"] = record["source"]
        files = [*current["license_files"], *record["license_files"]]
        current["license_files"] = sorted(
            {str(item["path"]): item for item in files}.values(),
            key=lambda item: str(item["path"]),
        )
    return sorted(
        merged.values(),
        key=lambda item: (str(item["name"]).lower(), str(item["version"])),
    )


def collect(prefix: Path, output_dir: Path) -> None:
    license_root = output_dir / "licenses"
    license_root.mkdir(parents=True, exist_ok=True)
    records = _merge_records(_conda_records(prefix, license_root) + _python_records(prefix, license_root))
    unresolved = [
        {
            "name": record["name"],
            "version": record["version"],
            "managers": record["managers"],
        }
        for record in records
        if str(record["license"]).strip().lower() in {"", "unknown", "unspecified"} and not record["license_files"]
    ]

    lines = [
        "TopoPPI third-party notices",
        "",
        "This bundle records package metadata from the environment used to build the application.",
        "Bundled license files remain authoritative for their respective components.",
        "",
    ]
    if unresolved:
        names = ", ".join(f"{item['name']} {item['version']}" for item in unresolved)
        lines.extend(
            [
                "Package metadata did not identify a license or license file for: " + names,
                "These entries remain in the machine-readable inventory for review.",
                "",
            ]
        )
    for record in records:
        managers = ", ".join(str(item) for item in record["managers"])
        lines.append(f"{record['name']} {record['version']} ({managers})")
        lines.append(f"License: {record['license']}")
        if record["source"]:
            lines.append(f"Source: {record['source']}")
        if record["license_files"]:
            files = ", ".join(str(item["path"]) for item in record["license_files"])
            lines.append(f"Bundled license files: {files}")
        lines.append("")
    (output_dir / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "THIRD_PARTY_LICENSES.json").write_text(
        json.dumps(
            {"schema_version": "1.0", "packages": records, "unresolved": unresolved},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="Conda environment prefix to inspect")
    parser.add_argument("output_dir", type=Path, help="Directory for notices and license texts")
    args = parser.parse_args()
    collect(args.prefix.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
