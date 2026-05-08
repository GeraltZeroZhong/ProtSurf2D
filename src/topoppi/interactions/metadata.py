"""Shared interaction metadata used by visualization and GUI code."""

from __future__ import annotations

INTERACTION_TYPES = (
    "VdWContact",
    "HydrogenBond",
    "Hydrophobic",
    "PiStacking",
    "PiCation",
    "CationPi",
    "Cationic",
    "Anionic",
    "HalogenBond",
    "MetalCoordination",
)

INTERACTION_COLORS = {
    "VdWContact": "#008080",
    "HydrogenBond": "#1f5eff",
    "Hydrophobic": "#6b7280",
    "PiStacking": "#7c3aed",
    "PiCation": "#f97316",
    "CationPi": "#ef4444",
    "Cationic": "#f59e0b",
    "Anionic": "#d97706",
    "HalogenBond": "#06b6d4",
    "MetalCoordination": "#7c4a24",
}

DEFAULT_ACTIVE_INTERACTION_TYPES = set(INTERACTION_TYPES)
