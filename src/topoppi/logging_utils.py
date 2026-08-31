"""Logging setup helpers."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for CLI entry points."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s" if verbose else "%(levelname)s: %(message)s"),
        datefmt="%H:%M:%S",
    )
    dependency_level = logging.NOTSET if verbose else logging.ERROR
    for name in ("MDAnalysis", "prolif"):
        logging.getLogger(name).setLevel(dependency_level)
