import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger("InteractionEngine")

PROLIF_TO_STANDARD = {
    "HydrogenBond": "HydrogenBond",
    "HBAcceptor": "HBAcceptor",
    "HBDonor": "HBDonor",
    "Hydrophobic": "Hydrophobic",
    "PiStacking": "PiStacking",
    "PiCation": "PiCation",
    "CationPi": "CationPi",
    "Cationic": "Cationic",
    "Anionic": "Anionic",
    "Ionic": "Cationic",
    "HalogenBond": "HalogenBond",
    "MetalAcceptor": "MetalAcceptor",
    "MetalDonor": "MetalDonor",
    "VdWContact": "VdWContact",
}


def _normalize_interaction_name(name):
    if not name:
        return None
    compact = str(name).replace(" ", "").replace("-", "")
    return PROLIF_TO_STANDARD.get(compact) or PROLIF_TO_STANDARD.get(str(name)) or "VdWContact"


def _extract_seq_id(value):
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _to_records(dataframe):
    records = []
    if dataframe is None or dataframe.empty:
        return records
    stacked = dataframe.stack(list(range(dataframe.columns.nlevels))).reset_index()
    stacked = stacked[stacked[0].astype(bool)]
    for _, row in stacked.iterrows():
        col_values = row.tolist()[2:-1]
        if len(col_values) < 3:
            continue
        ligand_res = col_values[0]
        protein_res = col_values[1]
        interaction_name = col_values[2]
        records.append(
            {
                "res_a_seq": _extract_seq_id(protein_res),
                "res_b_seq": _extract_seq_id(ligand_res),
                "interaction": _normalize_interaction_name(interaction_name),
            }
        )
    return [r for r in records if r["res_a_seq"] is not None]


def generate_prolif_interactions(pdb_path, chain_a, chain_b, log=None):
    log = log or logger
    try:
        import MDAnalysis as mda
        import prolif as plf
    except ImportError:
        log.warning("ProLIF/MDAnalysis not installed; skip ProLIF auto-generation.")
        return None

    pdb_dir = os.path.dirname(os.path.abspath(pdb_path))
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    out_path = os.path.join(pdb_dir, f"{pdb_name}.prolif.json")
    if os.path.exists(out_path):
        log.info(f"Found existing ProLIF output: {os.path.basename(out_path)}")
        return out_path

    universe = mda.Universe(pdb_path)

    chain_queries = [f"chainID {chain_a}", f"segid {chain_a}"]
    chain_a_atoms = None
    for query in chain_queries:
        atoms = universe.select_atoms(query)
        if len(atoms) > 0:
            chain_a_atoms = atoms
            break
    chain_queries = [f"chainID {chain_b}", f"segid {chain_b}"]
    chain_b_atoms = None
    for query in chain_queries:
        atoms = universe.select_atoms(query)
        if len(atoms) > 0:
            chain_b_atoms = atoms
            break

    if chain_a_atoms is None or chain_b_atoms is None:
        log.warning("ProLIF atom selection failed for one of the target chains.")
        return None

    mol_a = plf.Molecule.from_mda(chain_a_atoms)
    mol_b = plf.Molecule.from_mda(chain_b_atoms)
    fp = plf.Fingerprint()
    fp.run_from_iterable([mol_b], mol_a)
    records = _to_records(fp.to_dataframe())
    payload = {
        "engine": "prolif",
        "chain_a": str(chain_a).strip(),
        "chain_b": str(chain_b).strip(),
        "interactions": records,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info(f"ProLIF JSON generated successfully: {os.path.basename(out_path)}")
    return out_path


def load_prolif_data(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    interactions = payload.get("interactions", payload if isinstance(payload, list) else [])
    residue_types = defaultdict(set)
    partners = defaultdict(dict)
    for item in interactions:
        res_a = _extract_seq_id(item.get("res_a_seq"))
        res_b = _extract_seq_id(item.get("res_b_seq"))
        i_type = _normalize_interaction_name(item.get("interaction"))
        if res_a is None or not i_type:
            continue
        residue_types[res_a].add(i_type)
        if res_b is not None:
            partners[res_a][res_b] = partners[res_a].get(res_b, 0) + 1
    return dict(residue_types), dict(partners)
