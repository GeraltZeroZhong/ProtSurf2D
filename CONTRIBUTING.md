# Contributing to TopoPPI

This guide covers the path from a source checkout to a tested change.

## Set up the development environment

TopoPPI 2.0 uses Python 3.10. Create the Conda environment and install the project in editable mode:

```bash
conda env create -f environment.yml
conda activate topoppi-dev
python -m pip install -e ".[dev,benchmark,meshio]"
```

On Linux x86-64, install the bundled OptCuts executable before running the complete mapping pipeline:

```bash
bash tools/OptCuts/install_optcuts.sh
command -v OptCuts_bin
```

Windows and macOS development requires a native OptCuts build. The instructions live beside their installers:

- [Windows installer guide](./installer/windows/README.md)
- [macOS application guide](./installer/macos/README.md)

## Make a focused change

- Keep code, comments, commit messages, and repository documentation in English.
- Preserve scientific behavior unless the change intentionally updates it and includes matching tests and documentation.
- Prefer direct code paths and clear errors. Add validation where it protects a real user-facing boundary.
- Keep fixtures small, synthetic, and redistributable. Leave downloaded structures, generated benchmark results, and large binaries outside the repository.
- Update the changelog and user documentation when a command, default, output, or installation step changes.

## Run the checks

Run the same core checks used in continuous integration:

```bash
python -m pytest
python -m ruff check .
```

On a headless Linux machine, run the tests with `xvfb-run -a python -m pytest`
after installing Xvfb and xauth. CI uses this command so the Tk editing tests
execute alongside the pipeline tests. Windows and macOS run Tk tests through
their native display systems.

Tests that launch the external OptCuts executable use the `requires_optcuts` marker. Dataset-scale or long-running tests use `slow`. A new test should carry one of these markers only when its runtime or dependency warrants it.

For packaging changes, also build and inspect both distributions:

```bash
python -m build
python -m twine check dist/*
```

For user-facing command changes, exercise the relevant help page and one representative workflow. Examples include:

```bash
topoppi --help
topoppi render --help
topoppi-benchmark --help
topoppi-install-optcuts --help
```

The platform workflows also run `installer/smoke_footprints.py` against a saved
atlas from the installed CLI. This checks the packaged GUI's atlas opening,
highlighting, CSV annotation and save operations.

## Open a pull request

Describe the user problem, the chosen behavior, and the checks you ran. Include screenshots for visible desktop changes and a small output example for schema or command-line changes. Keep unrelated cleanup in a separate change so reviewers can trace each result to its implementation.
