"""Command-line interface for TopoPPI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Optional, Sequence

from topoppi import __version__
from topoppi.config import DEFAULT_RUN_CONFIG, TopoPPIRunConfig
from topoppi.errors import TopoPPIError
from topoppi.logging_utils import setup_logging
from topoppi.pipeline import run_interface_mapping


def build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_RUN_CONFIG
    parser = argparse.ArgumentParser(
        prog="topoppi",
        description="Create an annotated 2D interface map from a protein complex.",
        epilog="Example: topoppi complex.pdb -A A -B B -o interface_map.png",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    structure = parser.add_argument_group("structure and interactions")
    structure.add_argument("pdb_file", help="Input PDB or mmCIF structure")
    structure.add_argument(
        "-A",
        "--chain-a",
        dest="chain_a",
        default=defaults.chain_a,
        help="Chain used to build the molecular surface",
    )
    structure.add_argument(
        "-B",
        "--chain-b",
        dest="chain_b",
        default=defaults.chain_b,
        help="Partner chain used to locate the interface",
    )
    structure.add_argument(
        "--prolif",
        default=argparse.SUPPRESS,
        help="Existing ProLIF JSON; leave empty to generate one beside the output image",
    )
    structure.add_argument(
        "--geometric-fallback-distance",
        dest="contact_distance",
        type=float,
        default=defaults.contact_distance_angstrom,
        help="Heavy-atom cutoff for distance-based interaction fallback in Angstroms",
    )
    structure.add_argument(
        "--residue-scope",
        choices=("interaction", "patch"),
        default=defaults.visualization.residue_scope,
        help="Residues to annotate: ProLIF interactions or the full surface patch",
    )
    structure.add_argument(
        "--min-points",
        dest="min_points",
        type=int,
        default=defaults.visualization.min_points,
        help="Minimum Chain A interaction residues required to display a patch",
    )
    structure.add_argument(
        "--uniform-residue-color",
        action="store_false",
        dest="color_by_interaction_type",
        default=argparse.SUPPRESS,
        help="Use one color for annotated residues; the default colors show interaction types",
    )
    structure.add_argument(
        "--geometric-interaction-fallback",
        action="store_true",
        default=defaults.visualization.use_geometric_interaction_fallback,
        help="Use distance-based interaction assignments when ProLIF generation fails",
    )

    surface = parser.add_argument_group("surface generation")
    surface.add_argument(
        "--cutoff",
        type=float,
        default=defaults.topology.distance_cutoff,
        help="Chain B distance used to extract the interface patch in Angstroms",
    )
    surface.add_argument(
        "--res",
        type=float,
        default=defaults.surface.grid_resolution,
        help="Surface grid spacing in Angstroms",
    )
    surface.add_argument(
        "--sigma",
        type=float,
        default=defaults.surface.sigma,
        help="Gaussian smoothing scale in Angstroms",
    )
    surface.add_argument(
        "--surface-level", type=float, default=defaults.surface.level, help="Gaussian-density isosurface level"
    )
    surface.add_argument(
        "--surface-padding", type=float, default=defaults.surface.padding, help="Grid padding in Angstroms"
    )
    surface.add_argument(
        "--max-voxels", type=int, default=defaults.surface.max_voxels, help="Maximum dense-grid voxel count"
    )
    surface.add_argument(
        "--max-adaptive-resolution",
        type=float,
        default=defaults.surface.max_adaptive_resolution,
        help="Largest grid spacing allowed during automatic coarsening in Angstroms",
    )
    surface.add_argument(
        "--no-adaptive-resolution",
        action="store_false",
        dest="adaptive_resolution",
        default=argparse.SUPPRESS,
        help="Stop when the requested grid spacing exceeds the voxel budget",
    )

    mapping = parser.add_argument_group("UV mapping and OptCuts")
    mapping.add_argument(
        "--parameterization",
        choices=("auto", "lscm", "harmonic", "slim", "spherical", "cylindrical"),
        default=defaults.parameterization.method,
        help="Initial UV parameterization",
    )
    mapping.add_argument(
        "--slim-iterations",
        type=int,
        default=defaults.parameterization.slim_iterations,
        help="SLIM iterations when --parameterization=slim",
    )
    mapping.add_argument(
        "--slim-boundary-constraint-weight",
        type=float,
        default=defaults.parameterization.slim_boundary_constraint_weight,
        help="Soft convex-boundary constraint weight for SLIM",
    )
    mapping.add_argument(
        "--optcuts-bin",
        default=defaults.optcuts.optcuts_bin,
        help="OptCuts executable path or command name",
    )
    mapping.add_argument(
        "--patch-gap",
        type=float,
        default=defaults.optcuts.patch_gap,
        help="Chart gap as a fraction of sqrt(total 3-D chart area)",
    )
    mapping.add_argument(
        "--optcuts-lambda", type=float, default=defaults.optcuts.optcuts_lambda_init, help="OptCuts initial lambda"
    )
    mapping.add_argument(
        "--optcuts-distortion-bound",
        type=float,
        default=defaults.optcuts.optcuts_distortion_bound,
        help="OptCuts distortion bound",
    )
    mapping.add_argument(
        "--optcuts-initial-cut-option",
        type=int,
        choices=(0, 1),
        default=defaults.optcuts.optcuts_initial_cut_option,
        help="OptCuts initial cut option (0=random two-edge, 1=farthest two-point)",
    )
    mapping.add_argument(
        "--no-optcuts-bijectivity",
        action="store_false",
        dest="optcuts_use_bijectivity",
        default=argparse.SUPPRESS,
        help="Disable OptCuts bijectivity enforcement",
    )
    mapping.add_argument(
        "--optcuts-initialization",
        choices=("provided", "automatic"),
        default="provided" if defaults.optcuts.use_input_uv else "automatic",
        help="Use the selected UV parameterization or OptCuts automatic initialization",
    )
    mapping.add_argument(
        "--optcuts-timeout",
        type=float,
        default=defaults.optcuts.timeout_sec,
        help="OptCuts timeout per patch in seconds",
    )
    mapping.add_argument(
        "--residue-fragmentation-weight",
        type=float,
        default=defaults.optcuts.residue_fragmentation_weight,
        help="Weight of TopoPPI's residue-footprint objective; 0 selects the geometry-only ablation",
    )
    mapping.add_argument(
        "--expected-optcuts-sha256",
        default=argparse.SUPPRESS,
        help="Expected OptCuts SHA-256 for a frozen run",
    )

    output = parser.add_argument_group("output and logging")
    output.add_argument("--output", "-o", default=defaults.output_file, help="Output image")
    output.add_argument("--show", action="store_true", help="Open the Matplotlib figure after saving")
    output.add_argument("--verbose", "-v", action="store_true", help="Show debug logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("topoppi.cli")

    config = TopoPPIRunConfig(
        pdb_file=args.pdb_file,
        chain_a=args.chain_a,
        chain_b=args.chain_b,
        output_file=args.output,
        prolif_file=getattr(args, "prolif", None),
        contact_distance_angstrom=args.contact_distance,
        surface=replace(
            DEFAULT_RUN_CONFIG.surface,
            grid_resolution=args.res,
            sigma=args.sigma,
            level=args.surface_level,
            padding=args.surface_padding,
            max_voxels=args.max_voxels,
            adaptive_resolution=getattr(
                args,
                "adaptive_resolution",
                DEFAULT_RUN_CONFIG.surface.adaptive_resolution,
            ),
            max_adaptive_resolution=args.max_adaptive_resolution,
        ),
        topology=replace(DEFAULT_RUN_CONFIG.topology, distance_cutoff=args.cutoff),
        parameterization=replace(
            DEFAULT_RUN_CONFIG.parameterization,
            method=args.parameterization,
            slim_iterations=args.slim_iterations,
            slim_boundary_constraint_weight=args.slim_boundary_constraint_weight,
        ),
        optcuts=replace(
            DEFAULT_RUN_CONFIG.optcuts,
            optcuts_bin=args.optcuts_bin,
            patch_gap=args.patch_gap,
            optcuts_lambda_init=args.optcuts_lambda,
            optcuts_distortion_bound=args.optcuts_distortion_bound,
            optcuts_initial_cut_option=args.optcuts_initial_cut_option,
            optcuts_use_bijectivity=getattr(
                args,
                "optcuts_use_bijectivity",
                DEFAULT_RUN_CONFIG.optcuts.optcuts_use_bijectivity,
            ),
            use_input_uv=args.optcuts_initialization == "provided",
            residue_fragmentation_weight=args.residue_fragmentation_weight,
            timeout_sec=args.optcuts_timeout,
            expected_binary_sha256=getattr(
                args,
                "expected_optcuts_sha256",
                DEFAULT_RUN_CONFIG.optcuts.expected_binary_sha256,
            ),
        ).for_headless(),
        visualization=replace(
            DEFAULT_RUN_CONFIG.visualization,
            show_plot=args.show,
            min_points=args.min_points,
            residue_scope=args.residue_scope,
            color_by_interaction_type=getattr(
                args,
                "color_by_interaction_type",
                DEFAULT_RUN_CONFIG.visualization.color_by_interaction_type,
            ),
            use_geometric_interaction_fallback=args.geometric_interaction_fallback,
        ),
    )
    try:
        run_interface_mapping(config, log=log)
    except TopoPPIError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
