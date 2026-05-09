import hashlib
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from topoppi.install_optcuts import (
    ARTIFACT_NAME,
    PLATFORM_ARTIFACTS,
    SidecarArtifact,
    default_install_dir,
    default_url,
    install_optcuts,
    platform_artifact,
    resolve_expected_checksum,
)


class InstallOptCutsTests(unittest.TestCase):
    def test_default_url_uses_release_tag_and_artifact_name(self):
        self.assertEqual(
            default_url("1.2"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v1.2/{ARTIFACT_NAME}",
        )
        self.assertEqual(
            default_url("v1.2"),
            f"https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v1.2/{ARTIFACT_NAME}",
        )

    def test_default_url_can_target_windows_artifact(self):
        self.assertEqual(
            default_url("1.2", "windows-x86_64"),
            "https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v1.2/OptCuts_bin-windows-x86_64.exe",
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
                skip_platform_check=True,
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
                skip_platform_check=True,
            )

            self.assertEqual(target, install_dir / "OptCuts_bin.exe")
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)

    def test_installs_sidecar_artifact_next_to_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / ARTIFACT_NAME
            payload = b"fake optcuts binary"
            source.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()

            sidecar = SidecarArtifact("libigl_stb_image-linux-x86_64.so", "libigl_stb_image.so")
            sidecar_source = tmp_path / sidecar.artifact_name
            sidecar_payload = b"fake sidecar library"
            sidecar_source.write_bytes(sidecar_payload)
            sidecar_checksum = hashlib.sha256(sidecar_payload).hexdigest()

            install_dir = tmp_path / "bin"
            target = install_optcuts(
                url=source.as_uri(),
                checksum=checksum,
                install_dir=install_dir,
                sidecars=((sidecar, sidecar_source.as_uri(), sidecar_checksum),),
                skip_platform_check=True,
            )

            self.assertEqual(target, install_dir / "OptCuts_bin")
            self.assertEqual((install_dir / "libigl_stb_image.so").read_bytes(), sidecar_payload)
            self.assertTrue((install_dir / "libigl_stb_image.so").stat().st_mode & stat.S_IXUSR)

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
                    skip_platform_check=True,
                )

            self.assertEqual(target.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
