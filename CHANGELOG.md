# Changelog

All notable changes to TopoPPI are documented here.

## [Unreleased]

## [1.3] - 2026-08-30

### Desktop and installation

- Added self-contained macOS disk images for Apple Silicon and Intel, including the matching native OptCuts executable and an offline runtime.
- Added visible first-launch preparation and recovery guidance to the macOS app.
- Improved the Windows setup flow with a console-free GUI launcher, a persistent configured command prompt, an automation wrapper, direct installation of the bundled OptCuts executable, complete ProLIF dependency checks, and reliable bootstrap failure reporting.
- Added native platform smoke runs to the Windows and macOS release workflows.
- Expanded platform documentation for installation, upgrades, repair, removal, and local release builds.
- Added the installed version, project help, issue reporting, and an output-folder control to the Basic desktop workflow.
- Bound saving and redraw operations to the last successful single-run snapshot, so cancelled, failed, and benchmark tasks cannot mix figures with another run's settings.

### Interface mapping

- Added the residue-footprint-aware OptCuts objective as the complete TopoPPI method, identified in benchmark output as `residue_aware_optcuts`, together with a matched geometry-only ablation.
- Added seam-preserving per-face-corner UV coordinates across OBJ input and output, metrics, atlas packing, and visualization.
- Added LSCM initialization for OptCuts, automatic-initialization comparison, timeout and cancellation support, binary and artifact hashes, and audited upstream provenance.
- Added deterministic area-matched chart packing with recorded transforms, spacing, overlap, utilization, and waste measurements.
- Updated distortion metrics to use a common similarity scale, original 3-D area weighting, Jacobian singular values, and global-reflection-corrected flip counts.
- Added internal cut-edge seam counts and normalized 3-D seam length.
- Made supplied and generated ProLIF records the shared source for interaction membership, residue weights, GUI patch filtering, labels, and colors. Geometric interaction inference remains available through an explicit option.
- Bound formal benchmark residue weights to the manifest's checked ProLIF record across comparative and operational profiles.
- Added chemically perceived, explicitly hydrogenated RDKit copies for ProLIF fingerprinting while preserving the source structure coordinates.
- Added automatic ProLIF generation for PDB, CIF, and mmCIF inputs, including multi-character mmCIF chain identifiers.
- Aligned CLI, GUI, Python, and Windows application output around the same interaction and visualization rules.
- Unified the GUI, CLI, and Python interface cutoff at `4.0 Å` and changed the default display threshold to one interaction residue for small interfaces.
- Moved structure, chain, and output validation ahead of ProLIF and OptCuts work; missing-chain errors now list available protein chains.
- Placed generated ProLIF records beside the selected output image and added `--version` to each public command.

### Benchmarking and reproducibility

- Rebuilt comparisons around one provenance-tracked 3-D face domain shared by LSCM, harmonic, SLIM, spherical, cylindrical, and the selected OptCuts methods.
- Added isolated repetitions, process-tree memory sampling, fixed thread settings, warm-ups, repeatability checks, and explicit resource metadata.
- Added formal manifests, coordinate audits, input and interaction checksums, deterministic chain selection, inclusion logs, per-patch retention, pLDDT propagation, per-face samples, and compressed provenance mappings.
- Added cluster-aware paired inference, cluster bootstrap confidence intervals, effect sizes, Benjamini-Hochberg correction, and JSON `null` handling for unavailable values.
- Added surface-grid and memory preflight, controlled adaptive grid spacing, read-only benchmark preflight, fingerprint-bound resume, and one-factor or factorial sensitivity plans.
- Added experimental and AlphaFold DB paired-structure preparation, confidence and geometry strata, and paired transport analysis tools.
- Added a manifest preparation command that generates ProLIF records in bulk and fills the formal `prolif_file` and `prolif_sha256` fields.
- Added concise terminal summaries, field-specific configuration errors, `topoppi-benchmark verify`, safe output-root checks for every profile, and a runnable quick-start configuration.

### Maintenance

- Removed obsolete compatibility paths, repeated validation, unreachable guards, unused wrappers, and duplicate file helpers across the core pipeline, GUI, and publication tools.
- Expanded regression and real-binary integration coverage for mapping, seams, packing, provenance, benchmark evidence, installers, and end-to-end execution.
- Reworked the README, benchmark schema, platform guides, OptCuts notices, contribution guide, and publication-tool help around task-oriented user workflows.

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

## [1.0] - 2026-04-06

- Initial public release of TopoPPI.
