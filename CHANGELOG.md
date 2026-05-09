# Changelog

All notable changes to TopoPPI are documented here.

## [Unreleased]

## [1.2] - 2026-05-09

- Added Windows x86-64 bootstrap installer scaffolding for one-click TopoPPI installation.
- Added Windows-aware OptCuts artifact resolution in `topoppi-install-optcuts`.
- Added Linux OptCuts runtime sidecar installation for `libigl_stb_image.so`.
- Added release workflow support for optional Windows OptCuts artifacts and Windows setup executables.
- Documented Windows installer requirements, release artifacts, and troubleshooting paths.

## [1.1] - 2026-05-09

- Improved GUI mode switching, sticky run controls, styling, run logs, validation, figure auto-save, manifests, and Matplotlib navigation.
- Added benchmark `resume`/`new`/`overwrite` modes, worker selection, config fingerprints, and invalid-input reports.
- Made benchmark preprocessing honor configured chain IDs consistently.
- Added ProLIF metadata validation and richer single-run reproducibility manifests.
- Hardened release packaging, PyPI metadata, and OptCuts installation checks.
- Added the `topoppi-install-optcuts` downloader for Linux x86-64 GitHub release artifacts.

## [1.0.0] - 2026-05-08

- Initial public release of TopoPPI.
