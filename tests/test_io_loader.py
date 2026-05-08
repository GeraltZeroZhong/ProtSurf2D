import unittest
from pathlib import Path

from topoppi.io.io_loader import PDBLoader


FIXTURES = Path(__file__).parent / "fixtures"


class PDBLoaderTests(unittest.TestCase):
    def test_loads_chains_and_filters_heteroatoms(self):
        loader = PDBLoader(str(FIXTURES / "tiny_complex.pdb"))

        self.assertEqual(loader.get_protein_chain_ids(), ["A", "B"])
        coords_a, atoms_a = loader.get_chain_data("A")
        coords_b, atoms_b = loader.get_chain_data("B")

        self.assertEqual(coords_a.shape, (6, 3))
        self.assertEqual(len(atoms_a), 6)
        self.assertEqual(coords_b.shape, (6, 3))
        self.assertEqual(len(atoms_b), 6)
        self.assertEqual(loader.get_chain_residue_count("A"), 2)

    def test_missing_chain_raises(self):
        loader = PDBLoader(str(FIXTURES / "tiny_complex.pdb"))
        with self.assertRaises(ValueError):
            loader.get_chain_data("Z")

    def test_1bvk_fixture_is_available(self):
        loader = PDBLoader(str(FIXTURES / "1bvk.pdb"))

        self.assertEqual(loader.get_protein_chain_ids(), ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(loader.get_chain_residue_count("A"), 108)
        coords_a, atoms_a = loader.get_chain_data("A")
        self.assertEqual(coords_a.shape, (843, 3))
        self.assertEqual(len(atoms_a), 843)
