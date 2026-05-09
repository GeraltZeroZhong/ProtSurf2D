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
from pathlib import Path
from typing import Sequence

from topoppi._version import __version__

ARTIFACT_NAME = "OptCuts_bin-linux-x86_64"
DEFAULT_SHA256 = "8f973b20dbf0db83409317dd267f6b674cfa9e9173fb77c260af70104e01426d"
RELEASE_URL_TEMPLATE = "https://github.com/GeraltZeroZhong/TopoPPI/releases/download/v{version}/{artifact}"


class InstallError(RuntimeError):
    """Raised when OptCuts installation cannot be completed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topoppi-install-optcuts",
        description="Download and install the Linux x86-64 OptCuts binary for TopoPPI.",
    )
    parser.add_argument(
        "--version",
        default=__version__,
        help="TopoPPI release version to download from. Defaults to the installed package version.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override the release artifact URL. Useful for testing or private mirrors.",
    )
    parser.add_argument(
        "--checksum",
        default=DEFAULT_SHA256,
        help="Expected SHA256 checksum for the OptCuts binary.",
    )
    parser.add_argument(
        "--install-dir",
        default=None,
        help="Directory to install OptCuts_bin into. Defaults to $CONDA_PREFIX/bin, then ~/.local/bin.",
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


def default_url(version: str) -> str:
    normalized = version[1:] if version.startswith("v") else version
    return RELEASE_URL_TEMPLATE.format(version=normalized, artifact=ARTIFACT_NAME)


def default_install_dir() -> Path:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix) / "bin"
    return Path.home() / ".local" / "bin"


def ensure_supported_platform(skip: bool = False) -> None:
    if skip:
        return
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux" or machine not in {"x86_64", "amd64"}:
        raise InstallError(
            "The packaged OptCuts artifact is only available for Linux x86-64. "
            "Build OptCuts manually and set TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin."
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


def install_optcuts(
    *,
    url: str,
    checksum: str,
    install_dir: Path,
    force: bool = False,
    skip_platform_check: bool = False,
) -> Path:
    ensure_supported_platform(skip=skip_platform_check)
    install_dir = install_dir.expanduser().resolve()
    target = install_dir / "OptCuts_bin"
    if target.exists() and not force:
        raise InstallError(f"{target} already exists. Use --force to overwrite it.")

    install_dir.mkdir(parents=True, exist_ok=True)
    expected = checksum.strip().lower()
    if not expected:
        raise InstallError("Expected SHA256 checksum cannot be empty.")

    with tempfile.TemporaryDirectory(prefix="topoppi-optcuts-") as tmp:
        tmp_path = Path(tmp) / ARTIFACT_NAME
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
    url = args.url or default_url(args.version)
    install_dir = Path(args.install_dir) if args.install_dir else default_install_dir()

    try:
        target = install_optcuts(
            url=url,
            checksum=args.checksum,
            install_dir=install_dir,
            force=args.force,
            skip_platform_check=args.skip_platform_check,
        )
    except InstallError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Installed OptCuts_bin to {target}")
    print("Verify with: OptCuts_bin")
    print(f"Or set: export TOPOPPI_OPTCUTS_BIN={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
