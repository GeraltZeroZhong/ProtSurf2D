import importlib.util
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topoppi_macos_licenses", ROOT / "installer" / "macos" / "collect_licenses.py"
)
LICENSES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LICENSES)


class MacOSLicenseCollectionTests(unittest.TestCase):
    def test_same_named_python_license_files_keep_their_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir, "environment")
            site_packages = prefix / "lib" / "python3.10" / "site-packages"
            first = site_packages / "vendor-a" / "LICENSE"
            second = site_packages / "vendor-b" / "LICENSE"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("first license", encoding="utf-8")
            second.write_text("second license", encoding="utf-8")
            package_metadata = Message()
            package_metadata["Name"] = "sample"
            package_metadata["License"] = "MIT"

            class Distribution:
                metadata = package_metadata
                version = "1.0"
                files = [Path("vendor-a/LICENSE"), Path("vendor-b/LICENSE")]

                @staticmethod
                def locate_file(entry):
                    return site_packages / entry

            license_root = Path(tmpdir, "output", "licenses")
            with (
                mock.patch.object(LICENSES.site, "getsitepackages", return_value=[str(site_packages)]),
                mock.patch.object(
                    LICENSES.metadata,
                    "distributions",
                    return_value=[Distribution()],
                ),
            ):
                records = LICENSES._python_records(prefix, license_root)

            copied = [Path(item["path"]) for item in records[0]["license_files"]]
            self.assertEqual(
                copied,
                [
                    Path("licenses/python/sample-1.0/vendor-a/LICENSE"),
                    Path("licenses/python/sample-1.0/vendor-b/LICENSE"),
                ],
            )
            self.assertEqual((license_root.parent / copied[0]).read_text(), "first license")
            self.assertEqual((license_root.parent / copied[1]).read_text(), "second license")

    def test_unknown_transitive_license_is_reported_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir, "environment")
            output = Path(tmpdir, "output")
            metadata = prefix / "conda-meta" / "virtual-helper-1.0-0.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps({"name": "virtual-helper", "version": "1.0"}),
                encoding="utf-8",
            )

            LICENSES.collect(prefix, output)

            inventory = json.loads((output / "THIRD_PARTY_LICENSES.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["unresolved"][0]["name"], "virtual-helper")
            self.assertIn(
                "virtual-helper 1.0",
                (output / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8"),
            )

    def test_duplicate_package_records_are_merged(self):
        records = [
            {
                "manager": manager,
                "name": name,
                "version": "1.0",
                "license": license_name,
                "source": "",
                "license_files": [],
            }
            for manager, name, license_name in (
                ("conda", "sample_pkg", "unspecified"),
                ("python", "sample-pkg", "MIT"),
            )
        ]

        merged = LICENSES._merge_records(records)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["managers"], ["conda", "python"])
        self.assertEqual(merged[0]["license"], "MIT")


if __name__ == "__main__":
    unittest.main()
