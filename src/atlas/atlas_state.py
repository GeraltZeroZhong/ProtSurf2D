from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import trimesh


@dataclass
class ChartState:
    """State of one chart in the joint atlas optimization."""

    patch_index: int
    uv_local: np.ndarray
    uv_current: np.ndarray
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    rotation: float = 0.0

    def transformed_uv(self, use_translation: bool = True, use_rotation: bool = True) -> np.ndarray:
        uv = self.uv_current
        if uv is None or len(uv) == 0:
            return uv

        centered = uv - uv.mean(axis=0)
        if use_rotation:
            c = float(np.cos(self.rotation))
            s = float(np.sin(self.rotation))
            rot = np.array([[c, -s], [s, c]], dtype=np.float64)
            centered = centered @ rot.T

        if use_translation:
            centered = centered + self.translation

        return centered


@dataclass
class AtlasState:
    """Container shared by U/S/G alternating steps."""

    charts: Dict[int, ChartState]

    @classmethod
    def from_patches(cls, patches: List[trimesh.Trimesh]) -> "AtlasState":
        charts: Dict[int, ChartState] = {}
        for idx, patch in enumerate(patches):
            uv = patch.metadata.get("uv")
            if uv is None or len(uv) == 0:
                continue
            uv = np.asarray(uv, dtype=np.float64)
            charts[idx] = ChartState(
                patch_index=idx,
                uv_local=uv.copy(),
                uv_current=uv.copy(),
            )
        return cls(charts=charts)

    def chart_ids(self) -> Iterable[int]:
        return sorted(self.charts.keys())

    def set_grid_initial_layout(self, spacing: float = 1.35) -> None:
        ids = list(self.chart_ids())
        if not ids:
            return
        cols = int(np.ceil(np.sqrt(len(ids))))
        for layout_idx, cid in enumerate(ids):
            row = layout_idx // cols
            col = layout_idx % cols
            self.charts[cid].translation = np.array([col * spacing, -row * spacing], dtype=np.float64)

    def write_back(self, patches: List[trimesh.Trimesh]) -> None:
        for cid in self.chart_ids():
            state = self.charts[cid]
            patch = patches[cid]
            patch.metadata["uv"] = state.uv_current.copy()
            patch.metadata["uv_global"] = state.transformed_uv()

    def update_local_uv(self, patch_index: int, uv_new: np.ndarray) -> None:
        if patch_index not in self.charts:
            return
        uv_new = np.asarray(uv_new, dtype=np.float64)
        self.charts[patch_index].uv_current = uv_new

    def set_pose(self, patch_index: int, translation: Optional[np.ndarray] = None, rotation: Optional[float] = None) -> None:
        if patch_index not in self.charts:
            return
        if translation is not None:
            self.charts[patch_index].translation = np.asarray(translation, dtype=np.float64)
        if rotation is not None:
            self.charts[patch_index].rotation = float(rotation)
