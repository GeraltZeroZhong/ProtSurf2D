# Contributing

## Development Setup

```bash
conda env create -f environment.yml
conda activate bio3d
pip install -e ".[dev,benchmark,interactions,meshio]"
```

Install OptCuts into the active environment when running the full pipeline:

```bash
bash tools/OptCuts/install_optcuts.sh
which OptCuts_bin
```

## Tests

Run the lightweight test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

With pytest installed:

```bash
pytest
```

Tests that require the external OptCuts binary should be marked `requires_optcuts`.
Slow or dataset-scale tests should be marked `slow` and must not be required for basic pull-request validation.

## Code Style

Use English for code, comments, commit messages, and project documentation inside the repository.
Prefer small, reviewable changes and preserve core algorithm behavior unless a clear bug is identified.

```bash
ruff check src tests
```

## Fixture Policy

Keep default fixtures small, synthetic, and redistributable.
Do not add downloaded PDB datasets, generated benchmark outputs, or large binary artifacts to the repository.
