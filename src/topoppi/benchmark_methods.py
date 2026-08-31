"""Canonical benchmark method identifiers and OptCuts pairing rules."""

from __future__ import annotations

PARAMETERIZATION_METHODS = ("lscm", "harmonic", "slim", "spherical", "cylindrical")
STANDARD_OPTCUTS_METHODS = ("optcuts_automatic", "optcuts_lscm_initialized")
RESIDUE_AWARE_OPTCUTS_METHODS = ("residue_aware_optcuts",)
OPTCUTS_VARIANTS = (*STANDARD_OPTCUTS_METHODS, *RESIDUE_AWARE_OPTCUTS_METHODS)
DEFAULT_STANDARD_METHODS = (*PARAMETERIZATION_METHODS, *STANDARD_OPTCUTS_METHODS)

RESIDUE_AWARE_BASELINE = {
    "residue_aware_optcuts": "optcuts_automatic",
}


def resolved_optcuts_variants(
    configured: tuple[str, ...] | None,
    *,
    residue_fragmentation_weight: float,
) -> tuple[str, ...]:
    """Resolve the weight-dependent default or an explicit frozen list."""

    if configured is None:
        return (
            *STANDARD_OPTCUTS_METHODS,
            *(RESIDUE_AWARE_OPTCUTS_METHODS if residue_fragmentation_weight > 0.0 else ()),
        )
    return tuple(configured)


__all__ = [
    "RESIDUE_AWARE_BASELINE",
    "RESIDUE_AWARE_OPTCUTS_METHODS",
    "DEFAULT_STANDARD_METHODS",
    "OPTCUTS_VARIANTS",
    "PARAMETERIZATION_METHODS",
    "STANDARD_OPTCUTS_METHODS",
    "resolved_optcuts_variants",
]
