from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from src.atlas_state import AtlasState


@dataclass
class GroupConstraint:
    anchor_chart: int
    member_chart: int
    target_offset: np.ndarray


class AtlasConstraintEvaluator:
    """Atlas-level penalties: overlap, grouping, padding."""

    def __init__(self, padding: float = 0.08, overlap_weight: float = 1.0, group_weight: float = 0.0):
        self.padding = float(padding)
        self.overlap_weight = float(overlap_weight)
        self.group_weight = float(group_weight)

    @staticmethod
    def bbox(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return uv.min(axis=0), uv.max(axis=0)

    @staticmethod
    def bbox_overlap_vector(box_a, box_b, padding=0.0) -> Optional[np.ndarray]:
        a_min, a_max = box_a
        b_min, b_max = box_b
        overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
        overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
        if overlap_x <= -padding or overlap_y <= -padding:
            return None

        overlap_x += padding
        overlap_y += padding

        center_a = (a_min + a_max) * 0.5
        center_b = (b_min + b_max) * 0.5
        direction = center_b - center_a
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        else:
            direction = direction / norm

        return direction * min(overlap_x, overlap_y)

    def overlap_energy_and_pushes(self, atlas: AtlasState) -> Tuple[float, Dict[int, np.ndarray]]:
        pushes: Dict[int, np.ndarray] = {cid: np.zeros(2, dtype=np.float64) for cid in atlas.chart_ids()}
        energy = 0.0
        ids = list(atlas.chart_ids())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ia, ib = ids[i], ids[j]
                uv_a = atlas.charts[ia].transformed_uv()
                uv_b = atlas.charts[ib].transformed_uv()
                vec = self.bbox_overlap_vector(self.bbox(uv_a), self.bbox(uv_b), self.padding)
                if vec is None:
                    continue
                mag = float(np.linalg.norm(vec))
                energy += self.overlap_weight * mag * mag
                pushes[ia] -= 0.5 * self.overlap_weight * vec
                pushes[ib] += 0.5 * self.overlap_weight * vec
        return energy, pushes

    def group_energy_and_pushes(self, atlas: AtlasState, groups: Iterable[GroupConstraint]) -> Tuple[float, Dict[int, np.ndarray]]:
        pushes: Dict[int, np.ndarray] = {cid: np.zeros(2, dtype=np.float64) for cid in atlas.chart_ids()}
        if self.group_weight <= 0:
            return 0.0, pushes

        energy = 0.0
        for g in groups:
            if g.anchor_chart not in atlas.charts or g.member_chart not in atlas.charts:
                continue
            t_a = atlas.charts[g.anchor_chart].translation
            t_b = atlas.charts[g.member_chart].translation
            err = (t_b - t_a) - np.asarray(g.target_offset, dtype=np.float64)
            energy += self.group_weight * float(np.dot(err, err))
            pushes[g.anchor_chart] += self.group_weight * err
            pushes[g.member_chart] -= self.group_weight * err
        return energy, pushes
