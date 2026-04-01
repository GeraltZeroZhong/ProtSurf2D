from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import trimesh

from src.atlas_state import AtlasState


@dataclass
class SeamMoveResult:
    accepted: bool
    delta_energy: float
    details: str


class SeamOptimizer:
    """
    Placeholder seam updater for alternating optimization.

    Current behavior:
    - Maintains seam metadata and computes a seam regularization term.
    - Provides an extension point for future edge cut/uncut operations.
    """

    def __init__(self, seam_weight: float = 0.1, enable_updates: bool = False):
        self.seam_weight = float(seam_weight)
        self.enable_updates = bool(enable_updates)

    def seam_length_proxy(self, patch: trimesh.Trimesh) -> float:
        # Use boundary edges as seam proxy for open patch charts.
        try:
            boundary_edges = patch.edges_boundary
            if boundary_edges is None or len(boundary_edges) == 0:
                return 0.0
            verts = patch.vertices
            e0 = verts[boundary_edges[:, 0]]
            e1 = verts[boundary_edges[:, 1]]
            return float(np.linalg.norm(e1 - e0, axis=1).sum())
        except Exception:
            return 0.0

    def evaluate_energy(self, patches: List[trimesh.Trimesh]) -> float:
        return self.seam_weight * sum(self.seam_length_proxy(p) for p in patches)

    def optimize_step(self, atlas: AtlasState, patches: List[trimesh.Trimesh]) -> SeamMoveResult:
        baseline = self.evaluate_energy(patches)
        if not self.enable_updates:
            for p in patches:
                p.metadata["seam_energy"] = baseline
            return SeamMoveResult(False, 0.0, "seam updates disabled")

        # This step intentionally keeps topology unchanged for now.
        # Seam mutation operators (edge cut/uncut/reroute) can be inserted here.
        for p in patches:
            p.metadata["seam_energy"] = baseline
        return SeamMoveResult(False, 0.0, "no seam move accepted (operator TODO)")
