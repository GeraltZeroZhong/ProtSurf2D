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
tar -tzf dist/*.tar.gz | grep -E 'tests/fixtures/.*(_cutoff|\.topoppi\.json)' && exit 1 || true
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
attaches `OptCuts_bin-linux-x86_64` plus `OptCuts_bin-linux-x86_64.sha256`.
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
The release artifact name expected by `topoppi-install-optcuts` is `OptCuts_bin-linux-x86_64`.
The expected SHA256 is:

```text
8f973b20dbf0db83409317dd267f6b674cfa9e9173fb77c260af70104e01426d
```

Document platform support, binary provenance, and licensing separately for any GitHub release artifact.
Do not attach OptCuts binary artifacts unless `tools/OptCuts/NOTICE.md` and `src/topoppi/install_optcuts.py` are current.
