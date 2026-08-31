"""PDB record helpers that follow Bio.PDB's atom-conformer selection."""

from __future__ import annotations

import math
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


def residue_plddt_values(atoms) -> list[float]:
    """Return one validated AlphaFold pLDDT value per observed residue."""

    grouped: dict[tuple[str, tuple[object, ...]], dict[str, object]] = {}
    for atom in atoms:
        residue = atom.get_parent()
        chain = residue.get_parent()
        key = (str(chain.id), tuple(residue.id))
        block = grouped.setdefault(key, {"all": [], "ca": []})
        value = float(atom.get_bfactor())
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("Predicted atoms must contain finite 0-100 pLDDT B factors.")
        block["all"].append(value)
        if str(atom.get_name()).strip() == "CA":
            block["ca"].append(value)

    values = []
    for block in grouped.values():
        all_values = block["all"]
        ca_values = block["ca"]
        if len(ca_values) != 1:
            raise ValueError("Each predicted protein residue must contain exactly one C-alpha pLDDT value.")
        if max(all_values) - min(all_values) > 0.011:
            raise ValueError("Atoms within one predicted residue contain inconsistent pLDDT values.")
        values.append(float(ca_values[0]))
    if not values:
        raise ValueError("Predicted structure contains no residue-level pLDDT values.")
    return values


def _line_atom_key(line: str) -> tuple[int, str, str, str, str, int, str]:
    return (
        int(line[6:11]),
        line[12:16].strip(),
        line[16],
        line[17:20].strip(),
        line[21],
        int(line[22:26]),
        line[26],
    )


def _selected_atom_key(atom) -> tuple[int, str, str, str, str, int, str]:
    residue = atom.get_parent()
    chain = residue.get_parent()
    return (
        int(atom.get_serial_number()),
        str(atom.get_name()).strip(),
        str(atom.get_altloc()),
        str(residue.get_resname()).strip(),
        str(chain.id),
        int(residue.id[1]),
        str(residue.id[2]),
    )


def _is_hydrogen(atom) -> bool:
    """Mirror the heavy-atom rule used by :class:`topoppi.io.PDBLoader`."""

    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element in {"H", "D"}
    name = str(atom.get_name()).strip().upper().lstrip("0123456789")
    return name.startswith(("H", "D"))


def selected_protein_atom_lines(path: str | Path) -> list[str]:
    """Return first-model protein-residue atom lines selected by Bio.PDB.

    Bio.PDB selects one child of each disordered atom, normally the conformer
    with highest occupancy.  Returning the corresponding source records keeps
    publication preprocessing consistent with :class:`topoppi.io.PDBLoader`
    while retaining fixed-width PDB fields needed for coordinate rewriting.
    """

    path = Path(path)
    model = PDBParser(QUIET=True).get_structure("P", str(path))[0]
    selected = {
        _selected_atom_key(atom)
        for chain in model
        for residue in chain
        if is_aa(residue, standard=False)
        for atom in residue
        if not _is_hydrogen(atom)
    }

    lines = []
    matched = set()
    with path.open("rt", encoding="ascii", errors="strict") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            key = _line_atom_key(line)
            if key in selected:
                lines.append(line)
                matched.add(key)

    missing = selected - matched
    if missing:
        raise ValueError(f"Could not recover {len(missing)} Bio.PDB-selected protein atom records from {path}.")
    return lines
