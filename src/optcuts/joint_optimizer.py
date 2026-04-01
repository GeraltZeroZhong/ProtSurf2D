from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh

try:
    import meshio
except Exception:  # optional dependency
    meshio = None

from src.atlas_constraints import AtlasConstraintEvaluator, GroupConstraint
from src.atlas_state import AtlasState
from src.metrics import UVAtlasMetrics
from src.parameterization import Parameterizer
from src.seam_optimizer import SeamOptimizer

logger = logging.getLogger("UVOOptimizer")


@dataclass
class UVOptimizerConfig:
    enabled: bool = False
    use_optcuts: bool = True
    optcuts_bin: str = "OptCuts_bin"

    # Outer alternating loop
    max_iterations: int = 40

    # G-step controls
    overlap_weight: float = 1.0
    internal_weight: float = 1.0
    patch_gap: float = 0.08
    rotation_enabled: bool = True
    global_scale_enabled: bool = False

    # S-step controls
    seam_weight: float = 0.1
    enable_seam_update: bool = False

    # Atlas constraints
    group_weight: float = 0.0


class OptCutsUVOptimizer:
    """
    Joint optimizer for local parameterization + seam proxy + atlas placement.

    Uses alternating optimization across:
      - U-step: per-chart UV refinement
      - S-step: seam objective/update hook
      - G-step: global chart translation/rotation non-overlap optimization
    """

    def __init__(self, config: Optional[UVOptimizerConfig] = None):
        self.config = config or UVOptimizerConfig()
        self.parameterizer = Parameterizer()
        self.seam_optimizer = SeamOptimizer(
            seam_weight=self.config.seam_weight,
            enable_updates=self.config.enable_seam_update,
        )
        self.last_report: Dict[str, object] = {}

    def optimize_patches(self, patches: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        start_ts = time.perf_counter()
        if not patches:
            self.last_report = {"status": "empty_input"}
            return patches

        # Optional OptCuts refinement
        for patch in patches:
            uv = patch.metadata.get("uv")
            if uv is None:
                continue
            if self.config.enabled and self.config.use_optcuts:
                opt_uv = self._run_optcuts_for_patch(patch, uv)
                if opt_uv is not None:
                    patch.metadata["uv_optcuts"] = opt_uv
                    patch.metadata["uv"] = opt_uv

        if not self.config.enabled:
            for patch in patches:
                if patch.metadata.get("uv") is not None:
                    patch.metadata["uv_global"] = patch.metadata["uv"].copy()
            self.last_report = self._build_report(
                patches=patches,
                objective_history=[],
                iteration_time=0.0,
                u_failures=0,
                u_attempts=0,
            )
            return patches

        atlas = AtlasState.from_patches(patches)
        if not list(atlas.chart_ids()):
            self.last_report = {"status": "no_valid_uv_charts"}
            return patches
        atlas.set_grid_initial_layout(spacing=1.35)

        constraints = AtlasConstraintEvaluator(
            padding=self.config.patch_gap,
            overlap_weight=self.config.overlap_weight,
            group_weight=self.config.group_weight,
        )
        groups: List[GroupConstraint] = []

        objective_history: List[float] = []
        u_failures_total = 0
        u_attempts_total = 0

        for outer_iter in range(max(1, self.config.max_iterations)):
            failures, attempts = self._u_step(atlas, patches)
            u_failures_total += failures
            u_attempts_total += attempts
            seam_result = self.seam_optimizer.optimize_step(atlas, patches)
            g_energy = self._g_step(atlas, constraints, groups)

            distortion_mean = self._global_distortion_mean(atlas, patches)
            seam_energy = self.seam_optimizer.evaluate_energy(patches)
            objective = distortion_mean + seam_energy + g_energy
            objective_history.append(float(objective))

            overlap_area = UVAtlasMetrics.atlas_bbox_overlap_area(
                [atlas.charts[cid].transformed_uv() for cid in atlas.chart_ids()]
            )
            logger.info(
                "[JointOpt] iter=%d, seam=%s, obj=%.6f, g_energy=%.6f, overlap_area=%.6f",
                outer_iter + 1,
                seam_result.details,
                objective,
                g_energy,
                overlap_area,
            )

        atlas.write_back(patches)
        elapsed = time.perf_counter() - start_ts
        self.last_report = self._build_report(
            patches=patches,
            objective_history=objective_history,
            iteration_time=elapsed,
            u_failures=u_failures_total,
            u_attempts=u_attempts_total,
        )
        for p in patches:
            p.metadata["joint_opt_report"] = self.last_report
        return patches

    def _u_step(self, atlas: AtlasState, patches: List[trimesh.Trimesh]) -> tuple[int, int]:
        failures = 0
        attempts = 0
        for cid in atlas.chart_ids():
            patch = patches[cid]
            uv_old = atlas.charts[cid].uv_current
            attempts += 1
            uv_new = self.parameterizer.refine_patch_uv(
                patch,
                uv_init=uv_old,
                blend_strength=0.8,
            )
            if uv_new is not None:
                atlas.update_local_uv(cid, uv_new)
            else:
                atlas.update_local_uv(cid, uv_old)
                failures += 1
        return failures, attempts

    def _g_step(self, atlas: AtlasState, constraints: AtlasConstraintEvaluator, groups: List[GroupConstraint]) -> float:
        energy_overlap, pushes_overlap = constraints.overlap_energy_and_pushes(atlas)
        energy_group, pushes_group = constraints.group_energy_and_pushes(atlas, groups)

        ids = list(atlas.chart_ids())
        for cid in ids:
            push = pushes_overlap.get(cid, np.zeros(2)) + pushes_group.get(cid, np.zeros(2))
            atlas.charts[cid].translation += push

        if self.config.rotation_enabled and len(ids) > 1:
            # Light-weight rotation jitter to improve packing under overlap pressure.
            for cid in ids:
                mag = np.linalg.norm(pushes_overlap.get(cid, np.zeros(2)))
                atlas.charts[cid].rotation += float(np.clip(0.01 * mag, -0.05, 0.05))

        if self.config.internal_weight > 0:
            shrink = 1.0 / (1.0 + 0.02 * self.config.internal_weight)
            for cid in ids:
                atlas.charts[cid].translation *= shrink

        return energy_overlap + energy_group

    def _global_distortion_mean(self, atlas: AtlasState, patches: List[trimesh.Trimesh]) -> float:
        vals = []
        for cid in atlas.chart_ids():
            ds = UVAtlasMetrics.distortion_stats(patches[cid], atlas.charts[cid].uv_current)
            vals.append(ds["mean"])
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _build_report(
        self,
        patches: List[trimesh.Trimesh],
        objective_history: List[float],
        iteration_time: float,
        u_failures: int,
        u_attempts: int,
    ) -> Dict[str, object]:
        uv_list = [p.metadata.get("uv_global", p.metadata.get("uv")) for p in patches if p.metadata.get("uv") is not None]

        flip_vals = []
        dist_vals = []
        angle_vals = []
        area_vals = []
        seam_total = 0.0
        for p in patches:
            uv = p.metadata.get("uv_global", p.metadata.get("uv"))
            if uv is None:
                continue
            flip_vals.append(UVAtlasMetrics.flip_rate(p, uv))
            dist_vals.append(UVAtlasMetrics.distortion_stats(p, uv))
            angle_vals.append(UVAtlasMetrics.angle_distortion_stats(p, uv))
            area_vals.append(UVAtlasMetrics.area_distortion_stats(p, uv))
            seam_total += self.seam_optimizer.seam_length_proxy(p)

        def _agg(stats_list, key):
            vals = [s[key] for s in stats_list] if stats_list else []
            return float(np.mean(vals)) if vals else float("inf")

        overlap_area = UVAtlasMetrics.atlas_bbox_overlap_area(uv_list) if uv_list else 0.0
        padding_viol = UVAtlasMetrics.padding_violations(uv_list, self.config.patch_gap) if uv_list else 0
        utilization = UVAtlasMetrics.atlas_utilization(uv_list) if uv_list else 0.0

        obj_drop = 0.0
        if len(objective_history) >= 2:
            obj_drop = float(objective_history[0] - objective_history[-1])

        failure_rate = float(u_failures / max(1, u_attempts))
        return {
            "parameterization_quality": {
                "flip_rate_mean": float(np.mean(flip_vals)) if flip_vals else 1.0,
                "distortion": {"mean": _agg(dist_vals, "mean"), "max": _agg(dist_vals, "max"), "p95": _agg(dist_vals, "p95")},
                "angle_distortion": {"mean": _agg(angle_vals, "mean"), "max": _agg(angle_vals, "max"), "p95": _agg(angle_vals, "p95")},
                "area_distortion": {"mean": _agg(area_vals, "mean"), "max": _agg(area_vals, "max"), "p95": _agg(area_vals, "p95")},
            },
            "topology_complexity": {
                "seam_total_length": float(seam_total),
                "chart_count": int(len(uv_list)),
            },
            "atlas_usability": {
                "overlap_area": float(overlap_area),
                "padding_violations": int(padding_viol),
                "utilization": float(utilization),
            },
            "stability_efficiency": {
                "objective_history": [float(v) for v in objective_history],
                "objective_drop": float(obj_drop),
                "total_time_sec": float(iteration_time),
                "failure_rate": float(failure_rate),
            },
        }

    def get_last_report(self) -> Dict[str, object]:
        return dict(self.last_report)

    def _run_optcuts_for_patch(self, patch: trimesh.Trimesh, fallback_uv: np.ndarray) -> Optional[np.ndarray]:
        bin_path = self.config.optcuts_bin
        resolved_bin = shutil.which(bin_path) if not os.path.isabs(bin_path) else bin_path
        if not resolved_bin or not os.path.exists(resolved_bin):
            logger.warning("OptCuts binary not found (%s). Skip external optimization.", bin_path)
            return None

        try:
            with tempfile.TemporaryDirectory(prefix="optcuts_") as tmpdir:
                in_obj = os.path.join(tmpdir, "patch_in.obj")
                out_obj = os.path.join(tmpdir, "patch_out.obj")
                patch.export(in_obj)

                cmd = [resolved_bin, "--input", in_obj, "--output", out_obj]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning("OptCuts failed (code=%s): %s", proc.returncode, proc.stderr.strip())
                    return None

                if not os.path.exists(out_obj):
                    logger.warning("OptCuts output OBJ not found: %s", out_obj)
                    return None

                uv = self._read_uv_from_obj(out_obj)
                if uv is None:
                    return None

                if len(uv) != len(fallback_uv):
                    logger.warning("OptCuts UV vertex count mismatch (%d vs %d).", len(uv), len(fallback_uv))
                    return None

                return uv
        except Exception as exc:
            logger.warning("OptCuts execution error: %s", exc)
            return None

    @staticmethod
    def _read_uv_from_obj(obj_path: str) -> Optional[np.ndarray]:
        try:
            loaded = trimesh.load(obj_path, process=False)
            if isinstance(loaded, trimesh.Trimesh):
                vis = getattr(loaded, "visual", None)
                uv = getattr(vis, "uv", None)
                if uv is not None and len(uv) > 0:
                    return np.asarray(uv, dtype=np.float64)
        except Exception:
            pass

        if meshio is not None:
            try:
                mesh = meshio.read(obj_path)
                if "obj:vt" in mesh.point_data:
                    uv = np.asarray(mesh.point_data["obj:vt"], dtype=np.float64)
                    return uv[:, :2]
            except Exception:
                pass

        logger.warning("Failed to parse UV from OptCuts OBJ: %s", obj_path)
        return None
