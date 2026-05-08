import tempfile
import unittest
from pathlib import Path

from topoppi.config import TopoPPIRunConfig
from topoppi.errors import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_config_validates_input_path_and_numeric_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "x.pdb"
            pdb.write_text("END\n", encoding="utf-8")
            config = TopoPPIRunConfig(pdb_file=str(pdb), chain_a="A", chain_b="B")
            config.validate()

            bad = TopoPPIRunConfig(
                pdb_file=str(pdb),
                chain_a="A",
                chain_b="B",
                topology=config.topology.__class__(distance_cutoff=0),
            )
            with self.assertRaises(ConfigurationError):
                bad.validate()
