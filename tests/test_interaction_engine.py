import json
import logging
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFIO, PDBParser

from topoppi.interactions.interaction_engine import (
    INTERACTION_SCHEMA_VERSION,
    _mda_to_prolif_with_explicit_hydrogen,
    _to_records,
    generate_prolif_interactions,
    load_prolif_document,
    load_prolif_partner_map,
)
from topoppi.io.io_loader import PDBLoader

FIXTURES = Path(__file__).parent / "fixtures"


class InteractionEngineTests(unittest.TestCase):
    def test_mmcif_suffixes_support_automatic_prolif_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix, chain_a, chain_b in ((".cif", "A", "B"), (".mmcif", "surface", "partner")):
                with self.subTest(suffix=suffix, chain_a=chain_a, chain_b=chain_b):
                    structure = PDBParser(QUIET=True).get_structure("complex", FIXTURES / "1bvk.pdb")
                    if len(chain_a) > 1:
                        structure[0]["A"].id = chain_a
                        structure[0]["B"].id = chain_b
                    input_path = Path(tmp) / f"complex{suffix}"
                    output_dir = Path(tmp) / suffix.removeprefix(".")
                    output_dir.mkdir()
                    writer = MMCIFIO()
                    writer.set_structure(structure)
                    writer.save(str(input_path))

                    result = generate_prolif_interactions(
                        input_path,
                        chain_a,
                        chain_b,
                        output_dir=output_dir,
                    )

                    payload = json.loads(Path(result).read_text(encoding="utf-8"))
                    self.assertEqual(payload["chain_a"], chain_a)
                    self.assertEqual(payload["chain_b"], chain_b)
                    self.assertEqual(payload["engine"], "prolif")
                    self.assertTrue(payload["interactions"])

    def test_prolif_preparation_adds_hydrogens_without_mutating_source(self):
        import MDAnalysis as mda
        from rdkit import Chem

        universe = mda.Universe(FIXTURES / "1bvk.pdb")
        source = universe.select_atoms("chainID A")
        source_coordinates = source.positions.copy()
        source_atom_count = source.n_atoms

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import prolif as plf

            prepared = _mda_to_prolif_with_explicit_hydrogen(
                source,
                "A",
                plf,
                logging.getLogger(),
            )

        np.testing.assert_array_equal(source.positions, source_coordinates)
        self.assertEqual(source.n_atoms, source_atom_count)
        self.assertEqual(
            sum(atom.GetAtomicNum() != 1 for atom in prepared.GetAtoms()),
            source_atom_count,
        )
        hydrogens = [atom for atom in prepared.GetAtoms() if atom.GetAtomicNum() == 1]
        self.assertTrue(hydrogens)
        self.assertTrue(all(atom.GetPDBResidueInfo() is not None for atom in hydrogens))
        self.assertEqual(prepared.n_residues, len(source.residues))
        self.assertIn(
            Chem.BondType.DOUBLE,
            {bond.GetBondType() for bond in prepared.GetBonds()},
        )
        self.assertIn(
            Chem.BondType.AROMATIC,
            {bond.GetBondType() for bond in prepared.GetBonds()},
        )
        self.assertFalse(any("hydrogen atom" in str(item.message).lower() for item in caught))

    def test_explicit_hydrogen_preparation_invalidates_old_sidecars(self):
        self.assertEqual(INTERACTION_SCHEMA_VERSION, 3)

    def test_load_prolif_document_normalizes_records(self):
        _payload, residue_types, partners = load_prolif_document(str(FIXTURES / "prolif_interactions.json"))

        self.assertEqual(residue_types["1"], {"HydrogenBond"})
        self.assertEqual(residue_types["2A"], {"Hydrophobic"})
        self.assertEqual(partners["1"]["2"], 1)
        self.assertEqual(partners["2A"]["5"], 1)
        self.assertNotIn("bad", residue_types)

    def test_prolif_partner_map_resolves_authoritative_structure_labels(self):
        loader = PDBLoader(FIXTURES / "tiny_complex.pdb")
        _coords_a, atoms_a = loader.get_chain_data("A")
        _coords_b, atoms_b = loader.get_chain_data("B")

        partners = load_prolif_partner_map(
            FIXTURES / "prolif_interactions.json",
            atoms_a,
            atoms_b,
            expected_chain_a="A",
            expected_chain_b="B",
        )

        self.assertEqual(partners, {"A:GLY:1": {"B:ALA:2": 1}})

    def test_prolif_partner_map_rejects_a_declared_source_mismatch(self):
        loader = PDBLoader(FIXTURES / "tiny_complex.pdb")
        _coords_a, atoms_a = loader.get_chain_data("A")
        _coords_b, atoms_b = loader.get_chain_data("B")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interactions.json"
            path.write_text(
                json.dumps(
                    {
                        "chain_a": "A",
                        "chain_b": "B",
                        "source_sha256": "a" * 64,
                        "interactions": [
                            {
                                "res_a_seq": "1",
                                "res_b_seq": "2",
                                "interaction": "Hydrophobic",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source checksum"):
                load_prolif_partner_map(
                    path,
                    atoms_a,
                    atoms_b,
                    expected_chain_a="A",
                    expected_chain_b="B",
                    expected_source_sha256="b" * 64,
                )

    def test_dataframe_records_use_all_three_column_levels(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("GLU5.B", "LYS10.A", "XBAcceptor"),
                ("PHE8.B", "TYR11.A", "FaceToFace"),
                ("ASP9.B", "ARG12.A", "Anionic"),
                ("GLY10.B", "ALA13.A", "FutureInteraction"),
            ],
            names=["ligand", "protein", "interaction"],
        )
        dataframe = pd.DataFrame([[True, True, True, True]], columns=columns)

        records = _to_records(dataframe)

        self.assertEqual(
            records,
            [
                {"res_a_seq": "10", "res_b_seq": "5", "interaction": "HalogenBond"},
                {"res_a_seq": "11", "res_b_seq": "8", "interaction": "PiStacking"},
                {"res_a_seq": "12", "res_b_seq": "9", "interaction": "Ionic"},
                {"res_a_seq": "13", "res_b_seq": "10", "interaction": "Other"},
            ],
        )

    def test_dataframe_records_recover_unique_insertion_codes(self):
        columns = pd.MultiIndex.from_tuples([("GLY5.B", "SER10.A", "HBDonor"), ("GLY6.B", "SER11.A", "HBAcceptor")])
        dataframe = pd.DataFrame([[True, True]], columns=columns)
        tokens_a = {
            ("SER", 10): {"10A"},
            (None, 10): {"10A"},
            ("SER", 11): {"11", "11A"},
            (None, 11): {"11", "11A"},
        }

        records = _to_records(dataframe, residue_tokens_a=tokens_a)

        self.assertEqual(records, [{"res_a_seq": "10A", "res_b_seq": "5", "interaction": "HydrogenBond"}])
