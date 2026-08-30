#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_path="${1:-${script_dir}/OptCuts_bin}"
source_repo="${2:-https://github.com/liminchen/OptCuts.git}"
upstream_commit="cd2302671af7954f263b0ea93d8419aa943d54be"
patch_path="${script_dir}/residue_aware/optcuts-cd230267.patch"
source_provenance_patch_path="${script_dir}/residue_aware/source-vertex-provenance-cd230267.patch"
reproducibility_patch_path="${script_dir}/reproducible/candidate-validity-cd230267.patch"
obj_output_precision_patch_path="${script_dir}/reproducible/obj-output-precision-cd230267.patch"
static_stb_patch_path="${script_dir}/reproducible/static-stb-cd230267.patch"
sparse_local_solves_patch_path="${script_dir}/reproducible/sparse-local-solves-cd230267.patch"
oscillation_tolerance_patch_path="${script_dir}/reproducible/oscillation-tolerance-cd230267.patch"
topology_cycle_acceleration_patch_path="${script_dir}/reproducible/topology-cycle-acceleration-cd230267.patch"
mpl2_sparse_solver_patch_path="${script_dir}/reproducible/mpl2-sparse-solver-cd230267.patch"
core_dir="${script_dir}/residue_aware"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/topoppi-optcuts-build.XXXXXX")"
source_dir="${build_root}/OptCuts"
build_dir="${build_root}/build"

cleanup() {
  rm -rf -- "${build_root}"
}
trap cleanup EXIT

git clone "${source_repo}" "${source_dir}"
git -C "${source_dir}" fetch --depth 1 origin "${upstream_commit}"
git -C "${source_dir}" checkout --detach "${upstream_commit}"
git -C "${source_dir}" apply "${reproducibility_patch_path}"
git -C "${source_dir}" apply "${obj_output_precision_patch_path}"
git -C "${source_dir}" apply "${static_stb_patch_path}"
git -C "${source_dir}" apply "${sparse_local_solves_patch_path}"
git -C "${source_dir}" apply "${oscillation_tolerance_patch_path}"
git -C "${source_dir}" apply "${patch_path}"
git -C "${source_dir}" apply "${source_provenance_patch_path}"
git -C "${source_dir}" apply "${topology_cycle_acceleration_patch_path}"
git -C "${source_dir}" apply "${mpl2_sparse_solver_patch_path}"
cp "${core_dir}/ResidueFootprintEnergy.hpp" "${source_dir}/src/ResidueFootprintEnergy.hpp"
cp "${core_dir}/ResidueFootprintEnergy.cpp" "${source_dir}/src/ResidueFootprintEnergy.cpp"

SOURCE_DATE_EPOCH="$(git -C "${source_dir}" show -s --format=%ct "${upstream_commit}")"
export SOURCE_DATE_EPOCH
prefix_map="-ffile-prefix-map=${build_root}=. -fdebug-prefix-map=${build_root}=."
cxx_release_flags="-O3 -DNDEBUG -DEIGEN_MPL2_ONLY ${prefix_map}"
platform_args=()
linker_flags=()
for flag in ${LDFLAGS:-}; do
  case "${flag}" in
    -Wl,-rpath,*|-Wl,-rpath=*) ;;
    *) linker_flags+=("${flag}") ;;
  esac
done
export LDFLAGS="${linker_flags[*]-}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  platform_args+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0")
else
  compiler="${CXX:-c++}"
  installed_specs="$("${compiler}" -print-file-name=specs 2>/dev/null || true)"
  raw_specs="${build_root}/gcc.specs"
  clean_specs="${build_root}/gcc-release.specs"
  if [[ -f "${installed_specs}" ]] &&
    grep -Eq '%\{!static:-rpath /[^}]+' "${installed_specs}" &&
    "${compiler}" -dumpspecs >"${raw_specs}" 2>/dev/null; then
    sed -E 's/[[:space:]]+%\{!static:-rpath [^}]+\}//' \
      "${raw_specs}" >"${clean_specs}"
    cxx_release_flags+=" -specs=${clean_specs}"
  fi
  platform_args+=("-DCMAKE_EXE_LINKER_FLAGS_RELEASE=-static-libstdc++ -static-libgcc")
fi
cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SKIP_RPATH=TRUE \
  -DCMAKE_SKIP_BUILD_RPATH=TRUE \
  -DTBB_NO_DATE=ON \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG ${prefix_map}" \
  -DCMAKE_CXX_FLAGS_RELEASE="${cxx_release_flags}" \
  "${platform_args[@]}"

case "$(uname -s)" in
  Darwin) audit_os="macOS" ;;
  Linux) audit_os="Linux" ;;
  *) audit_os="Unix" ;;
esac
cat >"${build_dir}/ext/tbb/version_string.ver" <<EOF
#define __TBB_VERSION_STRINGS(N) \
#N": BUILD_HOST         release-builder" ENDL \
#N": BUILD_OS           ${audit_os}" ENDL \
#N": BUILD_KERNEL       generic" ENDL \
#N": BUILD_COMPILER     C++" ENDL \
#N": BUILD_LIBC         system" ENDL \
#N": BUILD_LD           system" ENDL \
#N": BUILD_TARGET       native" ENDL \
#N": BUILD_COMMAND      TopoPPI release build" ENDL

#define __TBB_DATETIME "Unknown"
EOF
cmake --build "${build_dir}" --parallel "${OPTCUTS_BUILD_JOBS:-4}"

mkdir -p "$(dirname "${output_path}")"
install -m 755 "${build_dir}/OptCuts_bin" "${output_path}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  strip -x "${output_path}"
  codesign --force --sign - "${output_path}"
else
  strip --strip-unneeded "${output_path}"
  if readelf -d "${output_path}" | grep -Eq '(RPATH|RUNPATH)'; then
    echo "OptCuts artifact contains an embedded runtime search path." >&2
    exit 1
  fi
fi

artifact_metadata="$(strings -a "${output_path}")"
build_paths=("${build_root}")
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  build_paths+=("${CONDA_PREFIX}")
fi
for build_path in "${build_paths[@]}"; do
  if grep -Fq "${build_path}" <<<"${artifact_metadata}"; then
    echo "OptCuts artifact contains a build-environment path: ${build_path}" >&2
    exit 1
  fi
done
if grep -Eiq 'microsoft-standard|[/\\]Users[/\\]runner[/\\]|[/\\]home[/\\]runner[/\\]|runner[/\\]work[/\\]' <<<"${artifact_metadata}"; then
  echo "OptCuts artifact contains runner-specific build metadata." >&2
  exit 1
fi
if grep -Eq 'TBB: BUILD_OS[[:space:]]+.*[0-9]' <<<"${artifact_metadata}"; then
  echo "OptCuts artifact contains a versioned TBB BUILD_OS value." >&2
  exit 1
fi
for expected in \
  "TBB: BUILD_HOST         release-builder" \
  "TBB: BUILD_OS           ${audit_os}" \
  "TBB: BUILD_KERNEL       generic" \
  "TBB: BUILD_COMPILER     C++"; do
  if ! grep -Fq "${expected}" <<<"${artifact_metadata}"; then
    echo "OptCuts artifact is missing neutral TBB metadata: ${expected}" >&2
    exit 1
  fi
done

echo "Built the TopoPPI OptCuts executable from ${upstream_commit}"
echo "Output: ${output_path}"
