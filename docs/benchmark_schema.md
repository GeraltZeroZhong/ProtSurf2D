# Benchmark Output Schema

TopoPPI benchmark runs produce JSON and CSV files under the configured output directory.

## `benchmark_report.json`

Top-level fields:

- `created_at`: UTC timestamp.
- `topoppi_version`: package version used for the run.
- `config`: serialized `BenchmarkConfig`.
- `runtime`: execution metadata, including `worker_count` and `config_fingerprint`.
- `preprocessing`: accepted/skipped file metadata and chain-selection rules.
- `files`: per-structure result records.
- `summary`: aggregate metrics from all valid structures.

Per-structure records include:

- `pdb`: input filename.
- `chain_selection`: selected receptor and ligand chains.
- `patch_count`: extracted interface patch count.
- `mesh_stats`: surface vertex and face counts.
- `lscm_raw`, `lscm_optcuts`, `harmonic_raw`, `spherical_raw`, `cylindrical_raw`: quality blocks.
- `topology_repair`: parameterization and topology-gate diagnostics.
- `timing`: stage timing and scalability proxies.
- `memory`: peak RSS if `psutil` is available.
- `topology_optimization`: energy and seam-length comparison.
- `optcuts_ablation`: before/after OptCuts comparison.
- `atlas_trainability`: raster-density metrics for downstream learning workflows.
- `error`: present only for failed structures.

## `benchmark_summary.csv`

The CSV flattens the most important per-structure fields for spreadsheet/statistical analysis.
Column names are intentionally stable across patch releases.

## Empty or Fully Skipped Inputs

If preprocessing finds no valid structures, TopoPPI still writes:

- `benchmark_report.json` with the preprocessing skip reasons,
- `benchmark_summary.csv` with headers,
- `benchmark_checkpoint.json` with the matching config fingerprint.

The run then raises an error so CLI/GUI callers can surface the invalid input set clearly.

If schema changes are needed for a minor release, update this document and `CHANGELOG.md`.
