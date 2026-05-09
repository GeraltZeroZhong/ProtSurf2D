# TopoPPI

TopoPPI maps **protein-protein interaction (PPI) interfaces** from 3D structures into annotated 2D UV atlases. It is built for interactive inspection, reproducible figure export, and benchmark-style evaluation across structure sets.

The toolkit includes:

- a **Basic/Advanced Tkinter GUI** for single-structure analysis, interaction styling, benchmark launching, and reproducible export,
- a **command-line pipeline** for one-shot interface map generation,
- a **benchmark framework** for multi-structure evaluation with JSON/CSV reports,
- optional typed interaction annotation through ProLIF, with geometric fallback when no ProLIF data is available.

The pipeline loads protein chains from PDB/mmCIF files, builds a receptor surface, extracts interface patches against a partner chain, flattens patches to UV space, optimizes UVs with [**OptCuts**](https://github.com/liminchen/OptCuts), and renders annotated interface maps.

<img width="1920" height="1032" alt="TopoPPI GUI showing the Basic workflow and interface map" src="./docs/assets/3ff22687bd403a67cd66caeacc95baee.png" />

---

## Quick Start (GUI, ~5 minutes)

If you only want to launch the GUI and run your first structure quickly:

1. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate bio3d
```

2. Install bundled OptCuts into the active Conda environment:

```bash
bash tools/OptCuts/install_optcuts.sh
which OptCuts_bin
```

3. Install TopoPPI in editable mode:

```bash
pip install -e ".[benchmark,interactions,meshio]"
```

4. Launch GUI:

```bash
topoppi-gui
```

5. In the GUI:
   - Use **Basic** to load a `.pdb`/`.cif` structure file,
   - Review detected chains and use **Swap A/B** if the surface/partner assignment is reversed,
   - Choose which interaction types to display and adjust colors if needed,
   - Click **Run Single Analysis** to generate and auto-save the interface map.

> Note: [OptCuts](https://github.com/liminchen/OptCuts) is required by the current pipeline. Running without OptCuts is intentionally unsupported.

---

## Project Overview

### Core workflow

1. **Load structure data** for Chain A (receptor/surface chain) and Chain B (ligand chain).
2. **Generate molecular surface** for Chain A.
3. **Extract interface patches** using a distance cutoff to Chain B atoms.
4. **Parameterize patches** with LSCM.
5. **Optimize UV patches** with OptCuts (required in current pipeline).
6. **Visualize and export** annotated 2D interface maps.

### Main entry points

- `topoppi`: installed command-line pipeline.
- `topoppi-gui`: installed Tkinter GUI.
- `topoppi.pipeline.run_interface_mapping`: importable single-structure API.
- `topoppi.benchmarking`: benchmark engine, metrics, aggregation, and CSV/JSON reporting.

All user-facing defaults live in `topoppi.config`.

---

## Installation & Requirements

### System requirements

- Python **3.10** (recommended via Conda)
- OS with Tk support (for GUI mode)
- **libigl Python bindings 2.6.x** (package: `igl`)
- **ProLIF + MDAnalysis are recommended** for automatic typed interaction generation; the GUI can leave ProLIF blank and fall back to geometric heuristics if generation is unavailable
- **OptCuts binary (`OptCuts_bin`) must be installed** and available in your PATH (or passed via `--optcuts-bin`)

### Create environment

```bash
conda env create -f environment.yml
conda activate bio3d
pip install -e ".[benchmark,interactions,meshio]"
```

The repository includes `pyproject.toml` for standard Python packaging and future PyPI publication.

### Install bundled [OptCuts](https://github.com/liminchen/OptCuts) binary

The repository includes a helper script that installs `tools/OptCuts/OptCuts_bin` into your active Conda environment:

```bash
bash tools/OptCuts/install_optcuts.sh
```

After installation, verify:

```bash
which OptCuts_bin
```

> Required: the current pipeline does **not** support running without [OptCuts](https://github.com/liminchen/OptCuts). You can also set `TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin`.

### Python dependencies

See `environment.yml` and `pyproject.toml` for the authoritative lists.
The full Conda environment includes ProLIF/MDAnalysis for automatic interaction generation.

Main dependencies include:

- `numpy`, `scipy`, `matplotlib`
- `biopython`, `scikit-image`, `trimesh`, `igl` (**2.6.x**)
- `networkx`, `rtree`, `shapely`, `pillow`
- `openbabel`, `MDAnalysis`, `prolif`

---

## Usage

### 1) GUI mode

Launch the desktop app:

```bash
topoppi-gui
```

GUI supports:

- Basic single-file analysis with structure loading, chain preview, A/B swapping, interaction filters, and color controls,
- Advanced controls for chain IDs, cutoffs, surface parameters, OptCuts export, labels, layout, and output directories,
- folder-level benchmark runs with explicit `resume`, `new`, and `overwrite` modes,
- interaction-type filtering, custom colors, and style presets (`Exploration`, `Publication`, `High contrast`),
- recent input/output path pickers,
- inline validation before launching a run,
- staged progress (`Load`, `Surface`, `Patch`, `OptCuts`, `Render`) and cooperative cancellation,
- optional OptCuts frame export,
- auto-saved figures and sidecar manifests for reproducibility.

### 2) Command-line mode

Run the full pipeline on a single PDB/mmCIF structure:

```bash
topoppi <input.pdb|input.cif> [options]
```

Example:

```bash
topoppi ./data/1abc.pdb -A A -B B -o interface_map.png --cutoff 9.0 --res 1.0 --sigma 1.5
```

### 3) Benchmark mode (via GUI workflow)

Select **Benchmark**, choose a folder containing `.pdb` files, select a run mode, and run **Run Benchmark** in GUI.
Benchmark preprocessing uses the configured surface/partner chain IDs for every file.
Outputs are written under:

- `benchmark_report.json`
- `benchmark_summary.csv`
- `benchmark_checkpoint.json` (resume support)

### 4) Output artifacts (what gets written)

#### Single-run CLI / GUI outputs

- Main rendered interface image (default: `interface_map.png`, or your `-o/--output` value)
- GUI single runs auto-save a figure to the selected save directory and write a `<figure>.topoppi.json` sidecar manifest
- Auto-generated ProLIF JSON when `--prolif` is not provided (saved as `<input_basename>.<chain_a>-<chain_b>.prolif.json` beside the input file)

#### Benchmark outputs

- `benchmark_report.json`: full structured report with runtime worker count and config fingerprint
- `benchmark_summary.csv`: tabular summary for all processed structures
- `benchmark_checkpoint.json`: resume/checkpoint state

---

## Configuration

### Command-line options (`topoppi`)

- `pdb_file`: input structure file (`.pdb` or `.cif`)
- `-A, --chain-a`: receptor/surface chain ID
- `-B, --chain-b`: ligand chain ID
- `--prolif`: optional ProLIF JSON path
- `--cutoff`: interface distance cutoff (Å)
- `--res`: surface grid resolution (Å)
- `--sigma`: Gaussian smoothing sigma
- `-o, --output` (default `interface_map.png`): output image path
- `--optcuts-bin` (default `OptCuts_bin`): OptCuts executable path/name
- `--patch-gap`: minimum spacing between charts in global UV atlas
- `-v, --verbose`: verbose logging

CLI defaults are read from `topoppi.config.DEFAULT_RUN_CONFIG`.

### Benchmark configuration (`BenchmarkConfig`)

`topoppi.config.BenchmarkConfig` defines reusable benchmark settings, including:

- input/output roots,
- chain IDs,
- nested surface/topology/parameterization/OptCuts configuration,
- parallelism (`max_workers`),
- resume behavior (`resume`),
- minimum patch validity thresholds.

### ProLIF behavior

If `--prolif` is not provided (or the GUI ProLIF field is left blank), TopoPPI tries to auto-generate `<input_basename>.<chain_a>-<chain_b>.prolif.json` using MDAnalysis + ProLIF. If generation is unavailable or fails, visualization falls back to geometric interaction heuristics.

---

## Examples

### Test fixture: 1BVK

The repository includes `tests/fixtures/1bvk.pdb` for smoke tests and reproducible examples.

```bash
topoppi tests/fixtures/1bvk.pdb -A A -B C -o 1bvk_interface.png
```

### CLI: basic run

```bash
topoppi ./data/complex.pdb -A A -B B -o complex_interface.png
```

### CLI: custom OptCuts binary and tighter patch gap

```bash
topoppi ./data/complex.pdb -A A -B C \
  --optcuts-bin /usr/local/bin/OptCuts_bin \
  --patch-gap 0.05 \
  --output complex_interface_optcuts.png
```

### CLI: use existing ProLIF interactions

```bash
topoppi ./data/complex.pdb -A A -B B \
  --prolif ./data/complex.prolif.json \
  --output complex_with_prolif.png
```

### Python: run one interface map programmatically

```python
from topoppi.config import TopoPPIRunConfig
from topoppi.pipeline import run_interface_mapping

result = run_interface_mapping(
    TopoPPIRunConfig(
        pdb_file="./data/complex.pdb",
        chain_a="A",
        chain_b="B",
        output_file="complex_interface.png",
    )
)
print(result.to_dict())
```

### Python: run benchmark programmatically

```python
from dataclasses import replace

from topoppi.config import BenchmarkConfig, DEFAULT_RUN_CONFIG
from topoppi.benchmarking import BenchmarkRunner

config = BenchmarkConfig(
    input_folder="./dataset",
    output_root="./benchmark_results",
    chain_a="A",
    chain_b="B",
    surface=replace(DEFAULT_RUN_CONFIG.surface, grid_resolution=1.0, sigma=1.0),
    topology=replace(DEFAULT_RUN_CONFIG.topology, distance_cutoff=9.0),
    optcuts=replace(DEFAULT_RUN_CONFIG.optcuts, optcuts_bin="OptCuts_bin").for_headless(),
    resume=True,
)

runner = BenchmarkRunner(config=config, log_fn=print)
report = runner.run()
print(report["summary"])
```

---

## Project Structure

```text
TopoPPI/
├─ pyproject.toml                 # Python package metadata and console scripts
├─ environment.yml                # Conda environment definition
├─ docs/                          # Release and reproducibility notes
├─ tests/                         # Lightweight smoke/unit tests
├─ tools/
│  └─ OptCuts/
│     ├─ OptCuts_bin              # Bundled OptCuts executable
│     ├─ install_optcuts.sh       # Installer script for Conda env
│     └─ LICENSE.txt              # OptCuts license
└─ src/
   └─ topoppi/
      ├─ cli.py                   # CLI entry point
      ├─ config.py                # Central runtime, benchmark, GUI, and OptCuts configuration
      ├─ pipeline.py              # Importable single-run API
      ├─ io/                      # PDB/mmCIF loading and chain extraction
      ├─ mesh/                    # Surface generation, topology, parameterization
      ├─ optimization/            # OptCuts-based UV optimization
      ├─ interactions/            # ProLIF integration and interaction normalization
      ├─ visualization/           # 2D interface rendering
      ├─ atlas/                   # Atlas metrics
      ├─ gui_app/                 # GUI mixins and application orchestration
      └─ benchmarking/            # Benchmark runner, metrics, reporting
```

---

## Troubleshooting

### `OptCuts_bin` not found

Symptoms:
- CLI/GUI fails when entering optimization stage
- `which OptCuts_bin` returns empty

Fix:

```bash
bash tools/OptCuts/install_optcuts.sh
which OptCuts_bin
```

If still missing, pass an explicit binary path:

```bash
topoppi <input.pdb|input.cif> -A <chainA> -B <chainB> --optcuts-bin /absolute/path/to/OptCuts_bin
```

### ProLIF/MDAnalysis import errors

Symptoms:
- Import errors for `prolif` or `MDAnalysis`

Fix:
- Recreate environment from `environment.yml`
- Confirm packages are installed in the active env

```bash
conda env create -f environment.yml
conda activate bio3d
python -c "import MDAnalysis, prolif; print('ok')"
```

### `igl` / libigl compatibility issues

Symptoms:
- Import failures or runtime issues in geometry/parameterization stages

Fix:
- Use the project environment and keep `igl` in the documented `2.6.x` range.

---

## Changelog

- **v1.0** is the initial public release of TopoPPI.
- **v1.1** content is being prepared for the next release.

See [CHANGELOG.md](./CHANGELOG.md) for release history.

---

## License

This project is distributed under the terms of the **MIT License**. See [LICENSE](./LICENSE).

The bundled OptCuts binary has its own license. See [`tools/OptCuts/LICENSE.txt`](./tools/OptCuts/LICENSE.txt).
