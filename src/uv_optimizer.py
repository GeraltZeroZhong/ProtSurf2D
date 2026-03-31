import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import trimesh

try:
    import meshio
except Exception:  # optional dependency
    meshio = None

logger = logging.getLogger("UVOOptimizer")


@dataclass
class UVOptimizerConfig:
    enabled: bool = False
    use_optcuts: bool = True
    optcuts_bin: str = "OptCuts_bin"
    overlap_weight: float = 1.0
    internal_weight: float = 1.0
    max_iterations: int = 60
    patch_gap: float = 0.08


class OptCutsUVOptimizer:
    """
    Stage-2 UV optimizer:
    1) (Optional) Per-patch OptCuts call to improve local UV quality.
    2) Build a shared global UV space for all patches.
    3) Run overlap-aware layout optimization with an internal distortion prior.
    """

    def __init__(self, config: Optional[UVOptimizerConfig] = None):
        self.config = config or UVOptimizerConfig()

    def optimize_patches(self, patches: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        if not patches:
            return patches

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
            return patches

        base_uvs = [p.metadata.get("uv") for p in patches]
        valid_indices = [i for i, uv in enumerate(base_uvs) if uv is not None and len(uv) > 0]
        if not valid_indices:
            return patches

        origins = {i: base_uvs[i].copy() for i in valid_indices}
        global_uvs = self._initialize_global_layout(origins)
        optimized = self._optimize_joint_layout(origins, global_uvs)

        for i, patch in enumerate(patches):
            uv = optimized.get(i)
            if uv is None:
                continue
            patch.metadata["uv_global"] = uv

        return patches

    def _initialize_global_layout(self, uv_dict):
        n = len(uv_dict)
        cols = int(np.ceil(np.sqrt(n)))
        spacing = 1.35
        global_uvs = {}
        for idx, patch_idx in enumerate(sorted(uv_dict.keys())):
            row = idx // cols
            col = idx % cols
            offset = np.array([col * spacing, -row * spacing], dtype=np.float64)
            uv = uv_dict[patch_idx]
            centered = uv - uv.mean(axis=0)
            global_uvs[patch_idx] = centered + offset
        return global_uvs

    def _optimize_joint_layout(self, uv_origins, uv_global):
        if len(uv_global) < 2:
            return uv_global

        translations = {
            i: uv_global[i].mean(axis=0) - uv_origins[i].mean(axis=0)
            for i in uv_global
        }

        for _ in range(max(1, self.config.max_iterations)):
            moved = False
            keys = sorted(uv_global.keys())
            for a_pos in range(len(keys)):
                for b_pos in range(a_pos + 1, len(keys)):
                    ia, ib = keys[a_pos], keys[b_pos]
                    box_a = self._bbox(uv_origins[ia] + translations[ia])
                    box_b = self._bbox(uv_origins[ib] + translations[ib])
                    overlap = self._bbox_overlap_vector(box_a, box_b, self.config.patch_gap)
                    if overlap is None:
                        continue
                    moved = True
                    push = overlap * self.config.overlap_weight
                    translations[ia] -= 0.5 * push
                    translations[ib] += 0.5 * push

            if not moved:
                break

            if self.config.internal_weight > 0:
                shrink = 1.0 / (1.0 + 0.05 * self.config.internal_weight)
                for i in translations:
                    translations[i] *= shrink

        result = {}
        for i in uv_global:
            result[i] = uv_origins[i] + translations[i]
        return result

    @staticmethod
    def _bbox(uv):
        return uv.min(axis=0), uv.max(axis=0)

    @staticmethod
    def _bbox_overlap_vector(box_a, box_b, gap=0.0):
        a_min, a_max = box_a
        b_min, b_max = box_b
        overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
        overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
        if overlap_x <= -gap or overlap_y <= -gap:
            return None

        overlap_x = overlap_x + gap
        overlap_y = overlap_y + gap

        center_a = (a_min + a_max) * 0.5
        center_b = (b_min + b_max) * 0.5
        direction = center_b - center_a
        if np.linalg.norm(direction) < 1e-8:
            direction = np.array([1.0, 0.0])
        direction = direction / np.linalg.norm(direction)

        magnitude = min(overlap_x, overlap_y)
        return direction * magnitude

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
