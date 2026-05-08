# Reproducibility

This document defines the minimum information needed to reproduce TopoPPI runs and benchmark tables.

## Environment

Use the Conda environment as the primary research environment:

```bash
conda env create -f environment.yml
conda activate bio3d
pip install -e ".[dev,benchmark,interactions,meshio]"
```

Record:

- operating system and CPU/GPU details,
- Python version,
- Conda package export,
- TopoPPI version and git commit,
- OptCuts binary path and checksum,
- whether `TOPOPPI_OPTCUTS_BIN` or `--optcuts-bin` was used.

## Single-Structure Runs

Record the exact command, input structure accession/source, chain IDs, and parameters:

```bash
topoppi input.pdb -A A -B B \
  --cutoff 9.0 \
  --res 2.0 \
  --sigma 1.0 \
  --patch-gap 0.08 \
  -o interface_map.png
```

The authoritative default values are defined in `topoppi.config.DEFAULT_RUN_CONFIG`.

Expected outputs:

- rendered interface image,
- optional `<input_basename>.<chain_a>-<chain_b>.prolif.json`,
- optional GUI figure sidecar manifest (`<figure>.topoppi.json`),
- process logs.

The GUI manifest records the TopoPPI version, input and ProLIF file checksums,
selected chains, configuration blocks, resolved OptCuts artifact, git commit,
stage timings, optimizer diagnostics, style settings, and the run log.

## Benchmark Runs

Benchmark jobs are selected from `.pdb` files in an input folder.
The current benchmark preprocessing rules are:

- file must contain at least two protein chains,
- selected chains must each have more than 10 amino acids,
- configured chain IDs are used for every accepted file,
- files missing either configured chain are skipped.

Expected benchmark outputs:

- `benchmark_report.json`,
- `benchmark_summary.csv`,
- `benchmark_checkpoint.json`.

Benchmark reports include the runtime worker count and a configuration
fingerprint. Resume mode only accepts checkpoints with a matching fingerprint;
`new` writes to a timestamped output directory, and `overwrite` removes the
known report/checkpoint/CSV files before running.

For published tables, archive the input file list, parameter configuration, output files, and checksums.

The repository includes `tests/fixtures/1bvk.pdb` as a redistributable smoke-test structure fixture.
