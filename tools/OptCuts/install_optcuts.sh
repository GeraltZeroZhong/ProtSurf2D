#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_bin="${script_dir}/OptCuts_bin"
source_lib="${script_dir}/libigl_stb_image.so"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX is not set. Activate the target Conda environment first." >&2
  exit 1
fi

if [[ ! -f "${source_bin}" ]]; then
  echo "ERROR: OptCuts_bin was not found at ${source_bin}." >&2
  exit 1
fi

install_dir="${CONDA_PREFIX}/bin"
mkdir -p "${install_dir}"

echo "Installing OptCuts into ${install_dir}..."
install -m 755 "${source_bin}" "${install_dir}/OptCuts_bin"
if [[ -f "${source_lib}" ]]; then
  install -m 755 "${source_lib}" "${install_dir}/libigl_stb_image.so"
fi

echo "Installation complete!"
echo "You can now run OptCuts_bin directly in your Conda environment."
echo "Example: OptCuts_bin 10 input.obj 0.999 1 0 4.1 1 0 mytest"
