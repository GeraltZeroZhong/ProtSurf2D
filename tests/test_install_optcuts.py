import hashlib
import io
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from topoppi import __version__
from topoppi.install_optcuts import (
    ARTIFACT_NAME,
    PLATFORM_ARTIFACTS,
    build_parser,
    default_install_dir,
    default_url,
    install_optcuts,
    main,
    platform_artifact,
    resolve_expected_checksum,
)


class InstallOptCutsTests(unittest.TestCase):
    def test_help_does_not_show_none_as_a_user_default(self):
        self.assertNotIn("default: None", build_parser().format_help())

    def test_version_flag_reports_the_installed_package(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(SystemExit, "0"):
            main(["--version"])
        self.assertEqual(output.getvalue().strip(), f"topoppi-install-optcuts {__version__}")

    def test_default_url_uses_release_tag_and_artifact_name(self):
        self.assertEqual(
            default_url(__version__),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v{__version__}/{ARTIFACT_NAME}",
        )
        self.assertEqual(
            default_url(f"v{__version__}"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v{__version__}/{ARTIFACT_NAME}",
        )

    def test_default_url_can_target_windows_artifact(self):
        self.assertEqual(
            default_url(__version__, "windows-x86_64"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v{__version__}/"
            "OptCuts_bin-windows-x86_64.exe",
        )

    def test_auto_platform_accepts_windows_amd64(self):
        with (
            mock.patch("topoppi.install_optcuts.platform.system", return_value="Windows"),
            mock.patch("topoppi.install_optcuts.platform.machine", return_value="AMD64"),
        ):
            self.assertEqual(platform_artifact("auto").platform_key, "windows-x86_64")

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
            )

            self.assertEqual(target, install_dir / "OptCuts_bin")
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)

    def test_installs_windows_file_url_with_exe_target_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact = PLATFORM_ARTIFACTS["windows-x86_64"]
            source = tmp_path / artifact.artifact_name
            payload = b"fake windows optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()

            install_dir = tmp_path / "Scripts"
            target = install_optcuts(
                url=source.as_uri(),
                checksum=checksum,
                install_dir=install_dir,
                platform_key=artifact.platform_key,
            )

            self.assertEqual(target, install_dir / "OptCuts_bin.exe")
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)

    def test_reads_checksum_from_release_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact = PLATFORM_ARTIFACTS["windows-x86_64"]
            source = tmp_path / artifact.artifact_name
            payload = b"fake windows optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()
            Path(str(source) + ".sha256").write_text(f"{checksum}  {artifact.artifact_name}\n", encoding="utf-8")

            self.assertEqual(resolve_expected_checksum(artifact, url=source.as_uri()), checksum)

    def test_default_install_dir_prefers_active_virtualenv(self):
        with (
            mock.patch.dict("topoppi.install_optcuts.os.environ", {}, clear=True),
            mock.patch("topoppi.install_optcuts.sys.prefix", "/tmp/topoppi-venv"),
            mock.patch("topoppi.install_optcuts.sys.base_prefix", "/usr"),
        ):
            self.assertEqual(default_install_dir("linux-x86_64"), Path("/tmp/topoppi-venv/bin"))
            self.assertEqual(default_install_dir("windows-x86_64"), Path("/tmp/topoppi-venv/Scripts"))

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
                )

            self.assertEqual(target.read_text(encoding="utf-8"), "existing")

    def test_force_replaces_an_existing_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / ARTIFACT_NAME
            payload = b"updated optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()

            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            target = install_dir / "OptCuts_bin"
            target.write_bytes(b"old binary")

            installed = install_optcuts(
                url=source.as_uri(),
                checksum=checksum,
                install_dir=install_dir,
                force=True,
            )

            self.assertEqual(installed, target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
