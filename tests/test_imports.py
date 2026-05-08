import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_package_import_is_lightweight(self):
        import topoppi

        self.assertEqual(topoppi.__version__, "1.0.0")

    def test_gui_module_import_does_not_start_tk(self):
        import topoppi.gui

        self.assertTrue(callable(topoppi.gui.main))
