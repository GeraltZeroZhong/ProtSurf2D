"""Shared interaction metadata used by visualization and GUI code."""

from __future__ import annotations

INTERACTION_TYPES = (
    "HydrogenBond",
    "Ionic",
    "Hydrophobic",
    "PiStacking",
    "PiCation",
    "HalogenBond",
    "MetalCoordination",
    "PolarContact",
    "VdWContact",
    "Other",
)

INTERACTION_COLORS = {
    "HydrogenBond": "#1f5eff",
    "Ionic": "#d97706",
    "Hydrophobic": "#6b7280",
    "PiStacking": "#7c3aed",
    "PiCation": "#f97316",
    "HalogenBond": "#06b6d4",
    "MetalCoordination": "#7c4a24",
    "PolarContact": "#4f8fba",
    "VdWContact": "#008080",
    "Other": "#9ca3af",
}

DEFAULT_ACTIVE_INTERACTION_TYPES = set(INTERACTION_TYPES)
