"""Install the external OptCuts binary used by TopoPPI."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from topoppi._version import __version__

LINUX_X86_64_SHA256 = "8f973b20dbf0db83409317dd267f6b674cfa9e9173fb77c260af70104e01426d"
DEFAULT_PLATFORM_KEY = "linux-x86_64"
RELEASE_URL_TEMPLATE = "https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v{version}/{artifact}"


class InstallError(RuntimeError):
    """Raised when OptCuts installation cannot be completed."""


@dataclass(frozen=True)
class PlatformArtifact:
    platform_key: str
    artifact_name: str
    target_name: str
    systems: tuple[str, ...]
    machines: tuple[str, ...]
    sha256: str | None = None


PLATFORM_ARTIFACTS = {
    "linux-x86_64": PlatformArtifact(
        platform_key="linux-x86_64",
        artifact_name="OptCuts_bin-linux-x86_64",
        target_name="OptCuts_bin",
        systems=("linux",),
        machines=("x86_64",),
        sha256=LINUX_X86_64_SHA256,
    ),
    "windows-x86_64": PlatformArtifact(
        platform_key="windows-x86_64",
        artifact_name="OptCuts_bin-windows-x86_64.exe",
        target_name="OptCuts_bin.exe",
        systems=("windows",),
        machines=("x86_64",),
    ),
}

ARTIFACT_NAME = PLATFORM_ARTIFACTS[DEFAULT_PLATFORM_KEY].artifact_name
DEFAULT_SHA256 = LINUX_X86_64_SHA256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topoppi-install-optcuts",
        description="Download and install the OptCuts binary for TopoPPI.",
    )
    parser.add_argument(
        "--version",
        default=__version__,
        help="TopoPPI release version to download from. Defaults to the installed package version.",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", *PLATFORM_ARTIFACTS),
        default="auto",
        help="Release artifact platform to install. Defaults to the current platform.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override the release artifact URL. Useful for testing or private mirrors.",
    )
    parser.add_argument(
        "--checksum",
        default=None,
        help="Expected SHA256 checksum. Defaults to the built-in Linux checksum or the release .sha256 sidecar.",
    )
    parser.add_argument(
        "--checksum-url",
        default=None,
        help="Override URL for the SHA256 sidecar. Defaults to '<artifact URL>.sha256' when needed.",
    )
    parser.add_argument(
        "--install-dir",
        default=None,
        help="Directory to install OptCuts into. Defaults to the active Conda env, then a user-local bin directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing OptCuts_bin.",
    )
    parser.add_argument(
        "--skip-platform-check",
        action="store_true",
        help="Skip the Linux x86-64 platform check.",
    )
    return parser


def normalize_machine(machine: str) -> str:
    normalized = machine.lower()
    if normalized in {"amd64", "x64"}:
        return "x86_64"
    return normalized


def current_platform_key() -> str:
    system = platform.system().lower()
    machine = normalize_machine(platform.machine())
    for platform_key, artifact in PLATFORM_ARTIFACTS.items():
        if system in artifact.systems and machine in artifact.machines:
            return platform_key
    supported = ", ".join(sorted(PLATFORM_ARTIFACTS))
    raise InstallError(f"No packaged OptCuts artifact for {system or 'unknown'} {machine or 'unknown'}. Supported: {supported}.")


def platform_artifact(platform_key: str | None = DEFAULT_PLATFORM_KEY) -> PlatformArtifact:
    resolved_key = current_platform_key() if platform_key in {None, "auto"} else platform_key
    try:
        return PLATFORM_ARTIFACTS[resolved_key]
    except KeyError as exc:
        supported = ", ".join(sorted(PLATFORM_ARTIFACTS))
        raise InstallError(f"Unsupported OptCuts artifact platform '{resolved_key}'. Supported: {supported}.") from exc


def default_url(version: str, platform_key: str | None = DEFAULT_PLATFORM_KEY) -> str:
    normalized = version[1:] if version.startswith("v") else version
    artifact = platform_artifact(platform_key)
    return RELEASE_URL_TEMPLATE.format(version=normalized, artifact=artifact.artifact_name)


def default_install_dir(platform_key: str | None = None) -> Path:
    artifact = platform_artifact(platform_key)
    scripts_dir = "Scripts" if artifact.platform_key.startswith("windows-") else "bin"
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix) / scripts_dir
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return Path(sys.prefix) / scripts_dir
    if artifact.platform_key.startswith("windows-"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "TopoPPI" / "bin"
        return Path.home() / "AppData" / "Local" / "TopoPPI" / "bin"
    return Path.home() / ".local" / "bin"


def ensure_supported_platform(artifact: PlatformArtifact, skip: bool = False) -> None:
    if skip:
        return
    system = platform.system().lower()
    machine = normalize_machine(platform.machine())
    if system not in artifact.systems or machine not in artifact.machines:
        raise InstallError(
            f"The {artifact.platform_key} OptCuts artifact cannot be installed on "
            f"{system or 'unknown'} {machine or 'unknown'}."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError) as exc:
        raise InstallError(f"Failed to download OptCuts from {url}: {exc}") from exc


def read_text_url(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except (UnicodeDecodeError, urllib.error.URLError, OSError) as exc:
        raise InstallError(f"Failed to read checksum from {url}: {exc}") from exc


def parse_sha256(text: str) -> str:
    for token in text.replace("*", " ").split():
        candidate = token.strip().lower()
        if len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate):
            return candidate
    raise InstallError("No SHA256 digest found in checksum sidecar.")


def resolve_expected_checksum(
    artifact: PlatformArtifact,
    *,
    url: str,
    checksum: str | None = None,
    checksum_url: str | None = None,
) -> str:
    if checksum:
        return checksum.strip().lower()
    if artifact.sha256:
        return artifact.sha256
    return parse_sha256(read_text_url(checksum_url or f"{url}.sha256"))


def install_optcuts(
    *,
    url: str,
    checksum: str,
    install_dir: Path,
    platform_key: str | None = DEFAULT_PLATFORM_KEY,
    target_name: str | None = None,
    force: bool = False,
    skip_platform_check: bool = False,
) -> Path:
    artifact = platform_artifact(platform_key)
    ensure_supported_platform(artifact, skip=skip_platform_check)
    install_dir = install_dir.expanduser().resolve()
    target = install_dir / (target_name or artifact.target_name)
    if target.exists() and not force:
        raise InstallError(f"{target} already exists. Use --force to overwrite it.")

    install_dir.mkdir(parents=True, exist_ok=True)
    expected = checksum.strip().lower()
    if not expected:
        raise InstallError("Expected SHA256 checksum cannot be empty.")

    with tempfile.TemporaryDirectory(prefix="topoppi-optcuts-") as tmp:
        tmp_path = Path(tmp) / artifact.artifact_name
        download_file(url, tmp_path)
        actual = sha256_file(tmp_path)
        if actual.lower() != expected:
            raise InstallError(f"Checksum mismatch for {url}: expected {expected}, got {actual}.")

        shutil.move(str(tmp_path), target)
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifact = platform_artifact(args.platform)
        url = args.url or default_url(args.version, artifact.platform_key)
        checksum = resolve_expected_checksum(artifact, url=url, checksum=args.checksum, checksum_url=args.checksum_url)
        install_dir = Path(args.install_dir) if args.install_dir else default_install_dir(artifact.platform_key)
        target = install_optcuts(
            url=url,
            checksum=checksum,
            install_dir=install_dir,
            platform_key=artifact.platform_key,
            force=args.force,
            skip_platform_check=args.skip_platform_check,
        )
    except InstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Installed {target.name} to {target}")
    print(f"Verify with: {target.name}")
    if artifact.platform_key.startswith("windows-"):
        print(f"Or set: set TOPOPPI_OPTCUTS_BIN={target}")
    else:
        print(f"Or set: export TOPOPPI_OPTCUTS_BIN={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
