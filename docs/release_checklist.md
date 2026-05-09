# Release Checklist

Use this checklist for GitHub releases and PyPI/TestPyPI publication.

## Before Tagging

- Confirm the working tree contains only intentional changes.
- Update `src/topoppi/_version.py`, `CHANGELOG.md`, and `CITATION.cff` to the same version.
- For final minor releases, use a PEP 440 version such as `1.1` and tag `v1.1`.
- For pre-releases, use a PEP 440 version such as `1.1rc1`, tag `v1.1rc1`, and mark the GitHub release as a pre-release.
- Confirm runtime defaults in `topoppi.config` match README and reproducibility docs.
- Confirm README examples use installed commands and `topoppi.*` imports.
- Confirm the release target is Python 3.10, matching `requires-python`, classifiers, `environment.yml`, and CI.
- Confirm the PyPI project has GitHub Trusted Publishing configured for the `pypi` environment.
- Confirm `tools/OptCuts/NOTICE.md` describes the exact binary source, license, platform, and release artifact policy.
- For Windows one-click releases, confirm the `Windows Installer` workflow builds `OptCuts_bin-windows-x86_64.exe`, embeds it in the setup executable, and attaches the standalone `.exe` plus `.sha256` sidecar to the same GitHub release.
- Check for accidental secrets and large files:

```bash
rg -n --glob '!docs/release_checklist.md' "(API[_-]?KEY|SECRET|TOKEN|PASSWORD|PRIVATE KEY|BEGIN RSA|ghp_|pypi-|AWS_ACCESS)" .
find . -type f -size +5M -not -path './.git/*' -print
```

- Recreate or verify the Conda environment from `environment.yml`.
- Install the package in editable mode:

```bash
pip install -e ".[dev,benchmark,interactions,meshio]"
```

## Validation

```bash
python -m pytest
python -m ruff check .
python -m topoppi.cli --help
python -m topoppi.install_optcuts --help
MPLBACKEND=Agg python -m topoppi.cli tests/fixtures/1bvk.pdb -A A -B C \
  --prolif tests/fixtures/prolif_interactions.json \
  --optcuts-bin tools/OptCuts/OptCuts_bin \
  -o /tmp/topoppi-release-smoke.png
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
mkdir -p release-assets
install -m 755 tools/OptCuts/OptCuts_bin release-assets/OptCuts_bin-linux-x86_64
sha256sum release-assets/OptCuts_bin-linux-x86_64
tar -tzf dist/*.tar.gz | grep -E 'tools/OptCuts/OptCuts_bin|tests/fixtures/.*(_cutoff|\.topoppi\.json)' && exit 1 || true
```

For Windows installer validation, run the `Windows Installer` workflow. It
builds the Windows OptCuts executable, embeds it in the setup executable, and
uploads both the setup and standalone OptCuts artifacts. To build locally on
Windows, first run `tools\OptCuts\build_windows_optcuts.ps1` and then run Inno
Setup:

```powershell
.\tools\OptCuts\build_windows_optcuts.ps1 -OutputDir installer\windows
Push-Location installer\windows
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  /DMyAppVersion=X.Y `
  /DMyPackageSpec=https://github.com/GeraltZeroZhong/TopoPPI/archive/refs/tags/vX.Y.zip `
  TopoPPI.iss
Pop-Location
```

In a fresh environment, install the built wheel and run:

```bash
topoppi --help
topoppi-install-optcuts --help
topoppi-gui
```

For GUI-facing releases, open the window with `tests/fixtures/1bvk.pdb`,
confirm the single-run and benchmark modes are legible, and refresh
`docs/assets/3ff22687bd403a67cd66caeacc95baee.png` if the layout changed.

For full pipeline validation, install or point to OptCuts:

```bash
export TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin
topoppi <input.pdb> -A A -B B -o interface_map.png
```

## Publication

Create the annotated tag before publishing. The tag version must match
`src/topoppi/_version.py`, `CHANGELOG.md`, and `CITATION.cff`.

```bash
git tag -a vX.Y -m "TopoPPI vX.Y"
git push origin vX.Y
```

The `Publish` workflow runs only from version tags, verifies the tag matches
the package version, and publishes through PyPI Trusted Publishing. It also
creates or updates the GitHub release and
attaches `OptCuts_bin-linux-x86_64`, checksum sidecars, and any optional
`OptCuts_bin-windows-x86_64.exe` binary present under `tools/OptCuts`.
The separate `Windows Installer` workflow builds and attaches
`TopoPPI-X.Y-windows-x86_64-setup.exe`, its `.sha256` sidecar, and standalone
Windows OptCuts artifacts.
If manual upload is ever required, upload exact filenames from a freshly
cleaned `dist/` build and `release-assets/` directory instead of globbing
both locations together.

```bash
python -m twine upload --repository testpypi dist/topoppi-X.Y*
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps topoppi==X.Y
python -m twine upload dist/topoppi-X.Y*
```

After PyPI upload and GitHub release creation:

- Attach reproducibility artifacts if applicable.
- Record benchmark dataset version, parameters, hardware, and output checksums.

## Binary Policy

The bundled `tools/OptCuts/OptCuts_bin` is not included in the Python package distribution.
Release artifact names expected by `topoppi-install-optcuts` are:

- `OptCuts_bin-linux-x86_64`
- `OptCuts_bin-windows-x86_64.exe`

The expected Linux SHA256 is:

```text
8f973b20dbf0db83409317dd267f6b674cfa9e9173fb77c260af70104e01426d
```

Document platform support, binary provenance, and licensing separately for any GitHub release artifact.
Do not attach OptCuts binary artifacts unless `tools/OptCuts/NOTICE.md` and `src/topoppi/install_optcuts.py` are current.
For Windows artifacts, attach a matching `OptCuts_bin-windows-x86_64.exe.sha256` sidecar; the setup executable embeds the Windows binary, and `topoppi-install-optcuts` can also use the standalone sidecar when installing from a release.
