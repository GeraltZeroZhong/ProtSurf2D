# Release Checklist

Use this checklist for GitHub releases and PyPI/TestPyPI publication.

## Before Tagging

- Confirm the working tree contains only intentional changes.
- Update `src/topoppi/__init__.py`, `pyproject.toml`, `CHANGELOG.md`, and `CITATION.cff` to the same version.
- Confirm runtime defaults in `topoppi.config` match README and reproducibility docs.
- Confirm README examples use installed commands and `topoppi.*` imports.
- Recreate or verify the Conda environment from `environment.yml`.
- Install the package in editable mode:

```bash
pip install -e ".[dev,benchmark,interactions,meshio]"
```

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests
python -m topoppi.cli --help
python -m build
twine check dist/*
```

In a fresh environment, install the built wheel and run:

```bash
topoppi --help
topoppi-gui
```

For full pipeline validation, install or point to OptCuts:

```bash
export TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin
topoppi <input.pdb> -A A -B B -o interface_map.png
```

## Publication

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ topoppi
twine upload dist/*
```

After PyPI upload:

- Create the GitHub tag and release notes.
- Attach reproducibility artifacts if applicable.
- Record benchmark dataset version, parameters, hardware, and output checksums.

## Binary Policy

The bundled `tools/OptCuts/OptCuts_bin` is not included in the Python package distribution.
Document platform support and licensing separately for any binary release artifact.
