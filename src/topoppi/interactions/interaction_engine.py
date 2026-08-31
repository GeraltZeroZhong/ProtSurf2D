import json
import logging
import os
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from topoppi import __version__
from topoppi.file_utils import sha256_file
from topoppi.json_utils import dump_json_atomic

logger = logging.getLogger("InteractionEngine")

INTERACTION_SCHEMA_VERSION = 3

PROLIF_TO_STANDARD = {
    "HydrogenBond": "HydrogenBond",
    "HBAcceptor": "HydrogenBond",
    "HBDonor": "HydrogenBond",
    "Hydrophobic": "Hydrophobic",
    "PiStacking": "PiStacking",
    "FaceToFace": "PiStacking",
    "EdgeToFace": "PiStacking",
    "PiCation": "PiCation",
    "CationPi": "PiCation",
    "Cationic": "Ionic",
    "Anionic": "Ionic",
    "Ionic": "Ionic",
    "HalogenBond": "HalogenBond",
    "XBAcceptor": "HalogenBond",
    "XBDonor": "HalogenBond",
    "MetalAcceptor": "MetalCoordination",
    "MetalDonor": "MetalCoordination",
    "MetalCoordination": "MetalCoordination",
    "VdWContact": "VdWContact",
}

_RESIDUE_TOKEN_RE = re.compile(r"(?<!\d)(-?\d+)([A-Za-z]?)")
_PROLIF_RESIDUE_RE = re.compile(r"^\s*([A-Za-z0-9 ]*?)(-?\d+)(?:\.[^.\s]+)?\s*$")


def normalize_interaction_name(name):
    if not name:
        return None
    compact = str(name).replace(" ", "").replace("-", "")
    return PROLIF_TO_STANDARD.get(compact, "Other")


def residue_sequence_token(value):
    """Return a canonical PDB residue sequence token such as ``42`` or ``42A``."""
    match = _RESIDUE_TOKEN_RE.search(str(value).strip())
    if match is None:
        return None
    sequence = int(match.group(1))
    insertion = match.group(2).upper()
    return f"{sequence}{insertion}"


def _prolif_residue_identity(value):
    match = _PROLIF_RESIDUE_RE.match(str(value))
    if match is None:
        return None
    return match.group(1).strip().upper(), int(match.group(2))


def _chain_residue_token_map(atom_group):
    mapping = defaultdict(set)
    for residue in atom_group.residues:
        sequence = int(residue.resid)
        insertion = str(getattr(residue, "icode", "") or "").strip().upper()
        token = f"{sequence}{insertion}"
        name = str(residue.resname).strip().upper()
        mapping[(name, sequence)].add(token)
        mapping[(None, sequence)].add(token)
    return mapping


def _resolve_generated_residue_token(value, token_map):
    fallback = residue_sequence_token(value)
    if fallback is None or token_map is None:
        return fallback
    identity = _prolif_residue_identity(value)
    if identity is not None:
        candidates = token_map.get(identity, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            return None
        sequence = identity[1]
    else:
        sequence = int(_RESIDUE_TOKEN_RE.search(fallback).group(1))
    candidates = token_map.get((None, sequence), set())
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        return None
    return fallback


def _to_records(dataframe, residue_tokens_a=None, residue_tokens_b=None, log=None):
    records = []
    if dataframe is None or dataframe.empty:
        return records
    skipped_ambiguous = 0
    observed = set()
    for column, values in dataframe.items():
        if not isinstance(column, tuple) or len(column) < 3:
            continue
        if not values.fillna(False).astype(bool).any():
            continue
        ligand_res, protein_res, interaction_name = column[-3:]
        res_a = _resolve_generated_residue_token(protein_res, residue_tokens_a)
        res_b = _resolve_generated_residue_token(ligand_res, residue_tokens_b)
        if res_a is None or res_b is None:
            skipped_ambiguous += 1
            continue
        interaction = normalize_interaction_name(interaction_name)
        key = (res_a, res_b, interaction)
        if key in observed:
            continue
        observed.add(key)
        records.append({"res_a_seq": res_a, "res_b_seq": res_b, "interaction": interaction})
    if skipped_ambiguous and log is not None:
        log.warning(
            "Skipped %d ProLIF interaction columns whose insertion-coded residue could not be resolved uniquely.",
            skipped_ambiguous,
        )
    return records


def _select_chain_atoms(universe, chain):
    for query in (f"chainID {chain}", f"segid {chain}"):
        atoms = universe.select_atoms(query)
        if len(atoms):
            return atoms
    return None


@contextmanager
def _prolif_structure_input(structure_path, chain_a, chain_b):
    """Yield a structure path and chain IDs that MDAnalysis can read."""

    source = Path(structure_path)
    if source.suffix.lower() not in {".cif", ".mmcif"}:
        yield str(source), str(chain_a), str(chain_b)
        return

    from Bio.PDB import PDBIO
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    from topoppi.io.io_loader import PDBLoader

    parsed_model = PDBLoader(source).model
    converted = Structure("TopoPPI")
    model = Model(0)
    converted.add(model)
    surrogate_ids = ("A", "B")
    for source_id, surrogate_id in zip((str(chain_a), str(chain_b)), surrogate_ids, strict=True):
        if source_id not in parsed_model:
            raise ValueError(f"Chain {source_id!r} was not found in the mmCIF structure.")
        target_chain = Chain(surrogate_id)
        for residue in parsed_model[source_id]:
            target_chain.add(residue.copy())
        model.add(target_chain)

    with TemporaryDirectory(prefix="topoppi-prolif-") as tmpdir:
        converted_path = Path(tmpdir) / "selected-chains.pdb"
        writer = PDBIO()
        writer.set_structure(converted)
        writer.save(str(converted_path))
        yield str(converted_path), *surrogate_ids


def _atom_group_to_pdb_block(atom_group):
    """Serialize an AtomGroup in memory for RDKit's PDB chemical perception."""

    from MDAnalysis.coordinates.PDB import PDBWriter
    from MDAnalysis.lib.util import NamedStream

    buffer = StringIO()
    stream = NamedStream(buffer, "topoppi-prolif-chain.pdb", close=False)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Found no information for attr: 'formalcharges'.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Unit cell dimensions not found.*",
            )
            with PDBWriter(
                stream,
                n_atoms=atom_group.n_atoms,
                bonds=None,
                multiframe=False,
            ) as writer:
                writer.write(atom_group)
        return buffer.getvalue()
    finally:
        buffer.close()


def _mda_to_prolif_with_explicit_hydrogen(atom_group, chain_label, plf, log):
    """Build an isolated, hydrogenated ProLIF molecule for fingerprinting.

    The in-memory PDB round-trip lets RDKit perceive standard protein bond
    orders before hydrogens are added.  Both operations create fingerprint-only
    molecule objects; the source AtomGroup and its coordinates are read but
    never modified.
    """

    from rdkit import Chem, rdBase

    pdb_block = _atom_group_to_pdb_block(atom_group)
    quiet_rdkit = logging.getLogger().getEffectiveLevel() > logging.DEBUG
    with rdBase.BlockLogs() if quiet_rdkit else nullcontext():
        mol = Chem.MolFromPDBBlock(
            pdb_block,
            sanitize=True,
            removeHs=False,
            proximityBonding=True,
        )
    if mol is None:
        raise ValueError(
            f"Could not prepare Chain {str(chain_label).strip()!r} for interaction detection. "
            "Check that the chain contains complete protein residues, or provide a ProLIF JSON file."
        )

    input_hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    with rdBase.BlockLogs() if quiet_rdkit else nullcontext():
        prepared = Chem.AddHs(
            mol,
            addCoords=True,
            addResidueInfo=True,
        )
    explicit_hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in prepared.GetAtoms())
    if not explicit_hydrogen_count:
        raise ValueError(
            f"Could not add hydrogens to Chain {str(chain_label).strip()!r}. "
            "Check the selected structure, or provide a ProLIF JSON file."
        )

    log.debug(
        "Prepared an isolated ProLIF fingerprint molecule for Chain %s with %d explicit hydrogens (%d added).",
        chain_label,
        explicit_hydrogen_count,
        explicit_hydrogen_count - input_hydrogen_count,
    )
    return plf.Molecule(prepared)


def generate_prolif_interactions(
    pdb_path,
    chain_a,
    chain_b,
    log=None,
    *,
    source_sha256=None,
    output_dir=None,
):
    """Generate or reuse a chain-pair ProLIF sidecar."""

    log = log or logger
    target_dir = os.path.abspath(output_dir or os.path.dirname(os.path.abspath(pdb_path)))
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    chain_tag = f"{str(chain_a).strip()}-{str(chain_b).strip()}".replace(os.sep, "_")
    out_path = os.path.join(target_dir, f"{pdb_name}.{chain_tag}.prolif.json")
    current_sha = source_sha256
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            source_sha = existing.get("source_sha256") if isinstance(existing, dict) else None
            schema_version = existing.get("interaction_schema_version") if isinstance(existing, dict) else None
        except (OSError, json.JSONDecodeError):
            source_sha = None
            schema_version = None
        current_sha = current_sha or sha256_file(pdb_path)
        if source_sha == current_sha and schema_version == INTERACTION_SCHEMA_VERSION:
            log.info("Found existing ProLIF output: %s", os.path.basename(out_path))
            return out_path
        log.warning("Existing ProLIF output has a stale source hash or interaction schema; regenerating.")

    quiet_third_party = logging.getLogger().getEffectiveLevel() > logging.DEBUG
    try:
        with warnings.catch_warnings():
            if quiet_third_party:
                warnings.simplefilter("ignore", DeprecationWarning)
            import MDAnalysis as mda

            # MDAnalysis enables its own deprecation notices during import.
            if quiet_third_party:
                warnings.simplefilter("ignore", DeprecationWarning)
            import prolif as plf
    except ImportError as exc:
        raise RuntimeError(
            "ProLIF auto-generation requires MDAnalysis, ProLIF, and RDKit. "
            "Install or repair the TopoPPI interaction dependencies."
        ) from exc

    current_sha = current_sha or sha256_file(pdb_path)

    with _prolif_structure_input(pdb_path, chain_a, chain_b) as (mda_path, mda_chain_a, mda_chain_b):
        universe = mda.Universe(mda_path)
        chain_a_atoms = _select_chain_atoms(universe, mda_chain_a)
        chain_b_atoms = _select_chain_atoms(universe, mda_chain_b)

        if chain_a_atoms is None or chain_b_atoms is None:
            raise ValueError(
                f"ProLIF atom selection failed for chain {str(chain_a).strip()!r} or {str(chain_b).strip()!r}."
            )

        mol_a = _mda_to_prolif_with_explicit_hydrogen(chain_a_atoms, chain_a, plf, log)
        mol_b = _mda_to_prolif_with_explicit_hydrogen(chain_b_atoms, chain_b, plf, log)
        fp = plf.Fingerprint()
        fp.run_from_iterable([mol_b], mol_a, progress=False)
        records = _to_records(
            fp.to_dataframe(),
            residue_tokens_a=_chain_residue_token_map(chain_a_atoms),
            residue_tokens_b=_chain_residue_token_map(chain_b_atoms),
            log=log,
        )
    payload = {
        "engine": "prolif",
        "interaction_schema_version": INTERACTION_SCHEMA_VERSION,
        "topoppi_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_pdb": os.path.abspath(pdb_path),
        "source_sha256": current_sha,
        "chain_a": str(chain_a).strip(),
        "chain_b": str(chain_b).strip(),
        "interactions": records,
    }
    dump_json_atomic(payload, out_path)
    log.info("ProLIF JSON generated successfully: %s", os.path.basename(out_path))
    return out_path


def load_prolif_document(json_path):
    """Load one object-schema ProLIF document and its normalized records."""

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ProLIF JSON root must be an object.")
    interactions = payload.get("interactions", [])
    if not isinstance(interactions, list):
        raise ValueError("ProLIF JSON interactions must be a list.")
    residue_types = defaultdict(set)
    partners = defaultdict(dict)
    for item in interactions:
        if not isinstance(item, dict):
            continue
        res_a = residue_sequence_token(item.get("res_a_seq"))
        res_b = residue_sequence_token(item.get("res_b_seq"))
        i_type = normalize_interaction_name(item.get("interaction"))
        if res_a is None or not i_type:
            continue
        residue_types[res_a].add(i_type)
        if res_b is not None:
            partners[res_a][res_b] = partners[res_a].get(res_b, 0) + 1
    return payload, dict(residue_types), dict(partners)


def _residue_label_lookup(atoms):
    """Map canonical PDB residue tokens to TopoPPI residue labels."""

    lookup = {}
    for atom in atoms:
        residue = atom.get_parent()
        chain = residue.get_parent()
        token = residue_sequence_token(f"{int(residue.id[1])}{str(residue.id[2]).strip().upper()}")
        label = f"{chain.id}:{residue.get_resname()}:{token}"
        previous = lookup.setdefault(token, label)
        if previous != label:
            raise ValueError(f"Residue token {token!r} is ambiguous within Chain {chain.id!r}.")
    return lookup


def load_prolif_partner_map(
    json_path,
    atoms_a,
    atoms_b,
    *,
    expected_chain_a=None,
    expected_chain_b=None,
    expected_source_sha256=None,
    require_bindings=False,
):
    """Return the authoritative ProLIF Chain-A-to-B residue partner map.

    Only records whose Chain A and Chain B residue tokens resolve against the
    supplied structure are retained. Values count distinct interaction records
    for a residue pair. Residue weights use the number of partner keys.
    """

    payload, _residue_types, token_partners = load_prolif_document(json_path)
    file_engine = str(payload.get("engine") or "").strip().lower()
    file_chain_a = str(payload.get("chain_a") or "").strip()
    file_chain_b = str(payload.get("chain_b") or "").strip()
    file_source_sha256 = str(payload.get("source_sha256") or payload.get("input_sha256") or "").strip().lower()
    requested_a = str(expected_chain_a or "").strip()
    requested_b = str(expected_chain_b or "").strip()
    requested_source_sha256 = str(expected_source_sha256 or "").strip().lower()
    if require_bindings and (not file_chain_a or not file_chain_b or not file_source_sha256):
        raise ValueError("Formal interaction JSON must declare chain_a, chain_b, and source_sha256/input_sha256.")
    if file_engine and file_engine != "prolif":
        raise ValueError(f"Interaction JSON engine must be 'prolif', got {file_engine!r}.")
    if file_chain_a and requested_a and file_chain_a != requested_a:
        raise ValueError(f"ProLIF chain_a mismatch: {file_chain_a} != {requested_a}")
    if file_chain_b and requested_b and file_chain_b != requested_b:
        raise ValueError(f"ProLIF chain_b mismatch: {file_chain_b} != {requested_b}")
    if file_source_sha256 and requested_source_sha256 and file_source_sha256 != requested_source_sha256:
        raise ValueError("ProLIF source checksum does not match the selected structure.")

    labels_a = _residue_label_lookup(atoms_a)
    labels_b = _residue_label_lookup(atoms_b)
    partner_map = {}
    for token_a, partners in token_partners.items():
        label_a = labels_a.get(token_a)
        if label_a is None:
            continue
        resolved = {
            labels_b[token_b]: int(count)
            for token_b, count in partners.items()
            if token_b in labels_b and int(count) > 0
        }
        if resolved:
            partner_map[label_a] = dict(sorted(resolved.items()))
    return dict(sorted(partner_map.items()))
