# OptCuts Binary Notice

This directory contains a convenience Linux x86-64 build of `OptCuts_bin` for
source checkout users who run the current TopoPPI pipeline.

- Upstream project: https://github.com/liminchen/OptCuts
- Upstream commit checked during release audit: `cd2302671af7954f263b0ea93d8419aa943d54be`
- Binary format: ELF 64-bit LSB pie executable, x86-64, dynamically linked
- Linux runtime sidecar: `libigl_stb_image.so` with `$ORIGIN` runtime lookup
- GitHub release artifact name: `OptCuts_bin-linux-x86_64`
- Linux runtime release artifact name: `libigl_stb_image-linux-x86_64.so`
- Optional Windows release artifact name: `OptCuts_bin-windows-x86_64.exe`
- Linux OptCuts SHA256: `0395b2b34f359b59a230e4833e320a55f81d12d90404f1c72b30c3eb8aef3e9f`
- Linux runtime SHA256: `996a27b49b5b42b5c97554898ab3e943baa4c08969df89f7c4f6e54dabbbf65f`
- Python package policy: `tools/OptCuts/OptCuts_bin` and
  `tools/OptCuts/libigl_stb_image.so` are excluded from TopoPPI sdist and wheel
  distributions.

Windows artifacts are not stored in this source checkout. The `Windows Installer`
workflow builds `OptCuts_bin-windows-x86_64.exe` from the upstream OptCuts
commit, embeds it in `TopoPPI-<version>-windows-x86_64-setup.exe`, and attaches
both the standalone executable and `.sha256` sidecar to the same GitHub release.

The upstream repository ships `LICENSE.txt` with MIT License text but no
copyright notice line. The local `LICENSE.txt` is copied from that upstream
license file so the binary provenance is explicit for GitHub source archives
and any separately published binary artifacts.
