"""Command-line interface for TopoPPI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from typing import Optional, Sequence

from topoppi.config import DEFAULT_RUN_CONFIG, TopoPPIRunConfig
from topoppi.errors import TopoPPIError
from topoppi.logging_utils import setup_logging
from topoppi.pipeline import run_interface_mapping


def build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_RUN_CONFIG
    parser = argparse.ArgumentParser(prog="topoppi", description="TopoPPI: protein interface 2D map generator")
    parser.add_argument("pdb_file", help="Path to input PDB/mmCIF file")
    parser.add_argument("-A", "--chain-a", "--chain_a", dest="chain_a", default=defaults.chain_a, help="Chain ID for the receptor/surface chain")
    parser.add_argument("-B", "--chain-b", "--chain_b", dest="chain_b", default=defaults.chain_b, help="Chain ID for the ligand chain")
    parser.add_argument("--prolif", "--arpeggio", dest="prolif", default=None, help="Path to ProLIF interaction JSON")
    parser.add_argument("--cutoff", type=float, default=defaults.topology.distance_cutoff, help="Interface distance cutoff in Angstroms")
    parser.add_argument("--res", type=float, default=defaults.surface.grid_resolution, help="Grid resolution for surface generation in Angstroms")
    parser.add_argument("--sigma", type=float, default=defaults.surface.sigma, help="Gaussian smoothing sigma")
    parser.add_argument("--output", "-o", default=defaults.output_file, help="Output image filename")
    parser.add_argument("--optcuts-bin", default=defaults.optcuts.optcuts_bin, help="Path/name for OptCuts executable")
    parser.add_argument("--patch-gap", type=float, default=defaults.optcuts.patch_gap, help="Padding/min-gap between charts in global UV")
    parser.add_argument("--show", action="store_true", help="Display the Matplotlib figure after saving")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
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
        prolif_file=args.prolif,
        surface=replace(DEFAULT_RUN_CONFIG.surface, grid_resolution=args.res, sigma=args.sigma),
        topology=replace(DEFAULT_RUN_CONFIG.topology, distance_cutoff=args.cutoff),
        parameterization=DEFAULT_RUN_CONFIG.parameterization,
        optcuts=replace(DEFAULT_RUN_CONFIG.optcuts, optcuts_bin=args.optcuts_bin, patch_gap=args.patch_gap),
        visualization=replace(DEFAULT_RUN_CONFIG.visualization, show_plot=args.show),
    )
    try:
        run_interface_mapping(config, log=log)
    except TopoPPIError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
