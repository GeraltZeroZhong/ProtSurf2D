import unittest
from unittest import mock

from topoppi import cli


class CliTests(unittest.TestCase):
    def test_cli_uses_headless_optcuts_mode(self):
        with mock.patch("topoppi.cli.run_interface_mapping") as run_interface_mapping:
            exit_code = cli.main(["input.pdb", "-A", "A", "-B", "B", "--optcuts-bin", "OptCuts_bin"])

        self.assertEqual(exit_code, 0)
        config = run_interface_mapping.call_args.args[0]
        self.assertEqual(config.optcuts.optcuts_mode, config.optcuts.optcuts_headless_mode)


if __name__ == "__main__":
    unittest.main()
