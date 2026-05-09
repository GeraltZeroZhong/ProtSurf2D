# TopoPPI

<p align="center">
  <img width="96" height="96" alt="TopoPPI icon" src="./src/topoppi/assets/topoppi.png" />
</p>

> **Release:** This README targets `topoppi 1.2`. The matching GitHub release tag is `v1.2`.

TopoPPI maps **protein-protein interaction (PPI) interfaces** from 3D structures into annotated 2D UV atlases. It is built for interactive inspection, reproducible figure export, and benchmark-style evaluation across structure sets.

The toolkit includes:

- a **Basic/Advanced Tkinter GUI** for single-structure analysis, interaction styling, benchmark launching, and reproducible export,
- a **command-line pipeline** for one-shot interface map generation,
- a **benchmark framework** for multi-structure evaluation with JSON/CSV reports,
- optional typed interaction annotation through ProLIF, with geometric fallback when no ProLIF data is available.

The pipeline loads protein chains from PDB/mmCIF files, builds a receptor surface, extracts interface patches against a partner chain, flattens patches to UV space, optimizes UVs with [**OptCuts**](https://github.com/liminchen/OptCuts), and renders annotated interface maps.

<img width="1920" height="1032" alt="TopoPPI GUI showing the Basic workflow and interface map" src="./docs/assets/3ff22687bd403a67cd66caeacc95baee.png" />

---

## Quick Start (GUI, shortest path)

### Windows x86-64

For Windows users, use the one-click installer from the [TopoPPI v1.2 release](https://github.com/GeraltZeroZhong/TopoPPI/releases/tag/v1.2):

```text
TopoPPI-1.2-windows-x86_64-setup.exe
```

Download it, keep your internet connection on, double-click it, and finish the installer. It installs TopoPPI, Python, the required scientific packages, and the bundled Windows OptCuts executable into `%LOCALAPPDATA%\TopoPPI`; no separate Conda, Python, or OptCuts setup is needed. If Windows SmartScreen warns that the unsigned installer is from an unknown publisher, continue only if the file came from the official GitHub release.

After installation, open **TopoPPI GUI** from the Windows Start Menu.

### Linux x86-64

Use Conda for the scientific stack, then install TopoPPI and OptCuts:

```bash
conda create -n topoppi -c conda-forge python=3.10 tk igl=2.6.* numpy scipy biopython scikit-image matplotlib trimesh networkx pillow rtree shapely openbabel mdanalysis rdkit psutil tqdm meshio pip
conda activate topoppi
pip install "topoppi[benchmark,interactions,meshio]"
topoppi-install-optcuts
topoppi-gui
```

### Run Your First Structure

In the GUI:

1. Use **Basic** to load a `.pdb` or `.cif` structure file.
2. Review detected chains and use **Swap A/B** if the surface/partner assignment is reversed.
3. Choose which interaction types to display and adjust colors if needed.
4. Click **Run Single Analysis** to generate and auto-save the interface map.

> Note: [OptCuts](https://github.com/liminchen/OptCuts) is required by the current pipeline. The Windows installer includes the Windows OptCuts executable. PyPI wheels do not include OptCuts, so Linux x86-64 users should run `topoppi-install-optcuts` after installing TopoPPI.

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
- `topoppi-install-optcuts`: download and install the OptCuts binary for a supported platform from the matching GitHub release.
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

### Create environment from a source checkout

```bash
conda env create -f environment.yml
conda activate bio3d
pip install -e ".[benchmark,interactions,meshio]"
```

The repository includes `pyproject.toml` for standard Python packaging and PyPI publication.

### Install from PyPI

```bash
conda create -n topoppi python=3.10
conda activate topoppi
pip install "topoppi[benchmark,interactions,meshio]"
topoppi-install-optcuts
which OptCuts_bin
```

For this release, `topoppi-install-optcuts` downloads from the matching GitHub release tag, for example `v1.2`. It auto-selects Linux x86-64 or Windows x86-64 when the release provides the matching OptCuts artifact. On Linux x86-64 it also installs the required `libigl_stb_image.so` runtime library next to `OptCuts_bin`. On other platforms, build OptCuts manually and set `TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin`.

### Windows one-click installer

Windows x86-64 users can install TopoPPI without managing Conda manually by downloading the setup executable from a GitHub release:

```text
TopoPPI-<version>-windows-x86_64-setup.exe
```

The installer creates an isolated TopoPPI environment under `%LOCALAPPDATA%\TopoPPI`, installs the matching TopoPPI release, installs the bundled Windows OptCuts executable, and creates Start Menu launchers for **TopoPPI GUI** and **TopoPPI CLI**.

The setup executable built by GitHub Actions embeds:

```text
OptCuts_bin-windows-x86_64.exe
OptCuts_bin-windows-x86_64.exe.sha256
```

The same files are also attached to the GitHub release as standalone artifacts for users who prefer `topoppi-install-optcuts`.

### Install [OptCuts](https://github.com/liminchen/OptCuts) binary

For PyPI installs, use the installed downloader:

```bash
topoppi-install-optcuts
```

From a source checkout, the repository also includes a helper script that installs `tools/OptCuts/OptCuts_bin` and its Linux runtime sidecar into your active Conda environment:

```bash
bash tools/OptCuts/install_optcuts.sh
```

After installation, verify:

```bash
which OptCuts_bin
```

> Required: the current pipeline does **not** support running without [OptCuts](https://github.com/liminchen/OptCuts). You can also set `TOPOPPI_OPTCUTS_BIN=/absolute/path/to/OptCuts_bin`.
> PyPI source and wheel distributions intentionally do not include `tools/OptCuts`; `topoppi-install-optcuts` downloads the release-provided binary/runtime artifacts instead.

### Python dependencies

See `environment.yml` and `pyproject.toml` for the authoritative lists.
The full Conda environment includes ProLIF/MDAnalysis for automatic interaction generation.

Main dependencies include:

- `numpy`, `scipy`, `matplotlib`
- `biopython`, `scikit-image`, `trimesh`, `igl` (**2.6.x**)
- `networkx`, `rtree`, `shapely`, `pillow`
- optional interaction extras: `MDAnalysis`, `prolif`, `rdkit`

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
- `--prolif` (`--arpeggio` alias): optional ProLIF JSON path
- `--cutoff`: interface distance cutoff (Å)
- `--res`: surface grid resolution (Å)
- `--sigma`: Gaussian smoothing sigma
- `-o, --output` (default `interface_map.png`): output image path
- `--optcuts-bin` (default `OptCuts_bin`): OptCuts executable path/name
- `--patch-gap`: minimum spacing between charts in global UV atlas
- `--show`: display the Matplotlib figure after saving
- `-v, --verbose`: verbose logging

CLI defaults are read from `topoppi.config.DEFAULT_RUN_CONFIG`; CLI runs force OptCuts headless mode so they work on display-less runners and servers.

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
│     ├─ OptCuts_bin              # Source-checkout convenience executable
│     ├─ install_optcuts.sh       # Installer script for Conda env
│     ├─ NOTICE.md                # Binary provenance and packaging policy
│     └─ LICENSE.txt              # OptCuts license
├─ installer/
│  └─ windows/                    # Inno Setup bootstrap installer
└─ src/
	   └─ topoppi/
	      ├─ cli.py                   # CLI entry point
	      ├─ config.py                # Central runtime, benchmark, GUI, and OptCuts configuration
	      ├─ install_optcuts.py       # Release artifact downloader for OptCuts
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
topoppi-install-optcuts
which OptCuts_bin
```

From a source checkout, this also works:

```bash
bash tools/OptCuts/install_optcuts.sh
which OptCuts_bin
```

If still missing, pass an explicit binary path:

```bash
topoppi <input.pdb|input.cif> -A <chainA> -B <chainB> --optcuts-bin /absolute/path/to/OptCuts_bin
```

### Windows setup fails while installing OptCuts

Symptoms:
- The setup executable creates the Python environment but stops during `topoppi-install-optcuts`
- The error mentions `OptCuts_bin-windows-x86_64.exe`

Fix:
- Download the setup executable produced by the `Windows Installer` workflow; it embeds the Windows OptCuts executable.
- Confirm the GitHub release also contains both `OptCuts_bin-windows-x86_64.exe` and `OptCuts_bin-windows-x86_64.exe.sha256` for fallback installs.
- If you built OptCuts yourself, set `TOPOPPI_OPTCUTS_BIN` to the absolute path of your `OptCuts_bin.exe`

### Linux OptCuts shared library error

Symptoms:
- The error mentions `libigl_stb_image.so: cannot open shared object file`

Fix:
- Re-run `topoppi-install-optcuts --force`; it installs both `OptCuts_bin` and `libigl_stb_image.so`.
- From a source checkout, run `bash tools/OptCuts/install_optcuts.sh` again so the sidecar is copied next to `OptCuts_bin`.

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

- **v1.2** adds Windows x86-64 installer scaffolding and Windows-aware OptCuts artifact installation, plus Linux OptCuts runtime sidecar handling.
- **v1.1** adds improved GUI workflows, benchmark reporting, reproducibility manifests, PyPI packaging, and one-command Linux x86-64 OptCuts installation.
- **v1.0** is the initial public release of TopoPPI.

See [CHANGELOG.md](./CHANGELOG.md) for release history.

---

## License

This project is distributed under the terms of the **MIT License**. See [LICENSE](./LICENSE).

The source checkout includes a Linux x86-64 OptCuts binary and runtime sidecar for convenience. They are not included in the Python package distribution; see [`tools/OptCuts/NOTICE.md`](./tools/OptCuts/NOTICE.md) and [`tools/OptCuts/LICENSE.txt`](./tools/OptCuts/LICENSE.txt) before redistributing binary artifacts.
