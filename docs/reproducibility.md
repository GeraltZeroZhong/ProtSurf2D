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
- optional `<input_basename>.prolif.json`,
- process logs.

## Benchmark Runs

Benchmark jobs are selected from `.pdb` files in an input folder.
The current benchmark preprocessing rules are:

- file must contain at least two protein chains,
- selected chains must each have more than 10 amino acids,
- if both `A` and `B` chains exist, use `A/B`,
- otherwise use the first two protein chains in structure order.

Expected benchmark outputs:

- `benchmark_report.json`,
- `benchmark_summary.csv`,
- `benchmark_checkpoint.json`.

For published tables, archive the input file list, parameter configuration, output files, and checksums.

The repository includes `tests/fixtures/1bvk.pdb` as a redistributable smoke-test structure fixture.
