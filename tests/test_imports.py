import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_package_import_is_lightweight(self):
        import topoppi
        from topoppi._version import __version__

        self.assertEqual(topoppi.__version__, __version__)

    def test_gui_module_import_does_not_start_tk(self):
        import topoppi.gui

        self.assertTrue(callable(topoppi.gui.main))
