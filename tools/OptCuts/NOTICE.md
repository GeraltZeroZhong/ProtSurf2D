# OptCuts Binary Notice

This directory contains a convenience Linux x86-64 build of `OptCuts_bin` for
source checkout users who run the current TopoPPI pipeline.

- Upstream project: https://github.com/liminchen/OptCuts
- Upstream commit checked during release audit: `cd2302671af7954f263b0ea93d8419aa943d54be`
- Binary format: ELF 64-bit LSB pie executable, x86-64, dynamically linked
- GitHub release artifact name: `OptCuts_bin-linux-x86_64`
- SHA256: `8f973b20dbf0db83409317dd267f6b674cfa9e9173fb77c260af70104e01426d`
- Python package policy: `tools/OptCuts/OptCuts_bin` is excluded from TopoPPI
  sdist and wheel distributions.

The upstream repository ships `LICENSE.txt` with MIT License text but no
copyright notice line. The local `LICENSE.txt` is copied from that upstream
license file so the binary provenance is explicit for GitHub source archives
and any separately published binary artifacts.
