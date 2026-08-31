from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from topoppi.file_utils import sha256_file


def _load_script():
    path = Path(__file__).parents[1] / "tools" / "publication" / "prepare_manifest_prolif.py"
    spec = importlib.util.spec_from_file_location("test_prepare_manifest_prolif_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load_script()


def test_prepare_manifest_generates_and_binds_included_rows(tmp_path, monkeypatch):
    structure = tmp_path / "complex.cif"
    structure.write_text("example structure", encoding="utf-8")
    source_sha256 = sha256_file(structure)
    manifest = tmp_path / "input.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "pdb",
                "staged_structure_path",
                "input_sha256",
                "chain_a",
                "chain_b",
                "include",
                "prolif_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "included",
                "pdb": structure.name,
                "staged_structure_path": "/stale/staged/complex.cif",
                "input_sha256": source_sha256,
                "chain_a": "surface",
                "chain_b": "partner",
                "include": "true",
                "prolif_path": "legacy.json",
            }
        )
        writer.writerow(
            {
                "record_id": "excluded",
                "pdb": "excluded.pdb",
                "include": "false",
                "prolif_path": "old.json",
            }
        )

    calls = []

    def fake_generate(path, chain_a, chain_b, *, source_sha256, output_dir):
        calls.append((Path(path), chain_a, chain_b, source_sha256))
        target = Path(output_dir) / "complex.surface-partner.prolif.json"
        target.write_text('{"interactions": []}', encoding="utf-8")
        return str(target)

    monkeypatch.setattr(PREPARE, "generate_prolif_interactions", fake_generate)
    output_manifest = tmp_path / "prepared" / "manifest.csv"
    output_dir = tmp_path / "prepared" / "prolif"

    count = PREPARE.prepare_manifest(manifest, output_manifest, output_dir, structure_dir=tmp_path)

    assert count == 1
    assert calls == [(structure, "surface", "partner", source_sha256)]
    with output_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert "prolif_path" not in reader.fieldnames
        assert reader.fieldnames[-2:] == ["prolif_file", "prolif_sha256"]
    assert rows[0]["prolif_file"] == "prolif/complex.surface-partner.prolif.json"
    assert rows[0]["prolif_sha256"] == sha256_file(output_dir / "complex.surface-partner.prolif.json")
    assert rows[1]["prolif_file"] == ""
    assert rows[1]["prolif_sha256"] == ""


def test_prepare_manifest_rejects_a_declared_input_mismatch(tmp_path):
    structure = tmp_path / "complex.pdb"
    structure.write_text("example structure", encoding="utf-8")
    manifest = tmp_path / "input.csv"
    manifest.write_text(
        f"pdb,input_sha256,chain_a,chain_b\n{structure.name},{'0' * 64},A,B\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input checksum mismatch"):
        PREPARE.prepare_manifest(manifest, tmp_path / "output.csv", tmp_path / "prolif")
