import unittest
from pathlib import Path

from topoppi.interactions.interaction_engine import load_prolif_data


FIXTURES = Path(__file__).parent / "fixtures"


class InteractionEngineTests(unittest.TestCase):
    def test_load_prolif_data_normalizes_records(self):
        residue_types, partners = load_prolif_data(str(FIXTURES / "prolif_interactions.json"))

        self.assertEqual(residue_types[1], {"HydrogenBond"})
        self.assertEqual(residue_types[2], {"Hydrophobic"})
        self.assertEqual(partners[1][2], 1)
        self.assertEqual(partners[2][5], 1)
        self.assertNotIn("bad", residue_types)
