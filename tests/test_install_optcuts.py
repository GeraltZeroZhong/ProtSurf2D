import hashlib
import stat
import tempfile
import unittest
from pathlib import Path

from topoppi.install_optcuts import ARTIFACT_NAME, default_url, install_optcuts


class InstallOptCutsTests(unittest.TestCase):
    def test_default_url_uses_release_tag_and_artifact_name(self):
        self.assertEqual(
            default_url("1.1"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v1.1/{ARTIFACT_NAME}",
        )
        self.assertEqual(
            default_url("v1.1"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v1.1/{ARTIFACT_NAME}",
        )

    def test_installs_file_url_with_checksum_and_executable_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / ARTIFACT_NAME
            payload = b"fake optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()

            install_dir = tmp_path / "bin"
            target = install_optcuts(
                url=source.as_uri(),
                checksum=checksum,
                install_dir=install_dir,
                skip_platform_check=True,
            )

            self.assertEqual(target, install_dir / "OptCuts_bin")
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)

    def test_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / ARTIFACT_NAME
            payload = b"fake optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()

            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            target = install_dir / "OptCuts_bin"
            target.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "already exists"):
                install_optcuts(
                    url=source.as_uri(),
                    checksum=checksum,
                    install_dir=install_dir,
                    skip_platform_check=True,
                )

            self.assertEqual(target.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
