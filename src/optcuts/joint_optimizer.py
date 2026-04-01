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

from src.metrics import UVAtlasMetrics

logger = logging.getLogger("UVOOptimizer")


@dataclass
class UVOptimizerConfig:
    optcuts_bin: str = "OptCuts_bin"
    patch_gap: float = 0.08


class OptCutsUVOptimizer:
    """OptCuts-only UV optimizer (no alternating U/S/G loop, no fallback path)."""

    def __init__(self, config: Optional[UVOptimizerConfig] = None):
        self.config = config or UVOptimizerConfig()
        self.last_report: Dict[str, object] = {}

    def optimize_patches(self, patches: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
        start_ts = time.perf_counter()
        if not patches:
            self.last_report = {"status": "empty_input"}
            return patches

        for idx, patch in enumerate(patches):
            uv = patch.metadata.get("uv")
            if uv is None:
                raise RuntimeError(f"Patch {idx} is missing initial UV before OptCuts.")
            opt_uv = self._run_optcuts_for_patch(patch, uv)
            patch.metadata["uv_optcuts"] = opt_uv
            patch.metadata["uv"] = opt_uv
            patch.metadata["uv_global"] = opt_uv.copy()

        elapsed = time.perf_counter() - start_ts
        self.last_report = self._build_report(patches=patches, iteration_time=elapsed)
        for p in patches:
            p.metadata["joint_opt_report"] = self.last_report
        return patches

    def _build_report(
        self,
        patches: List[trimesh.Trimesh],
        iteration_time: float,
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
            seam_total += 0.0

        def _agg(stats_list, key):
            vals = [s[key] for s in stats_list] if stats_list else []
            return float(np.mean(vals)) if vals else float("inf")

        overlap_area = UVAtlasMetrics.atlas_bbox_overlap_area(uv_list) if uv_list else 0.0
        padding_viol = UVAtlasMetrics.padding_violations(uv_list, self.config.patch_gap) if uv_list else 0
        utilization = UVAtlasMetrics.atlas_utilization(uv_list) if uv_list else 0.0

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
                "objective_history": [],
                "objective_drop": 0.0,
                "total_time_sec": float(iteration_time),
                "failure_rate": 0.0,
            },
        }

    def get_last_report(self) -> Dict[str, object]:
        return dict(self.last_report)

    def _run_optcuts_for_patch(self, patch: trimesh.Trimesh, reference_uv: np.ndarray) -> np.ndarray:
        bin_path = self.config.optcuts_bin
        resolved_bin = shutil.which(bin_path) if not os.path.isabs(bin_path) else bin_path
        if not resolved_bin or not os.path.exists(resolved_bin):
            raise RuntimeError(f"OptCuts binary not found: {bin_path}")

        try:
            with tempfile.TemporaryDirectory(prefix="optcuts_") as tmpdir:
                in_obj = os.path.join(tmpdir, "patch_in.obj")
                patch.export(in_obj)

                # The bundled binary is invoked via positional parameters (see tools/OptCuts/install_optcuts.sh).
                # Keep the output inside the temporary directory by setting cwd.
                run_tag = "patch"
                cmd = [
                    resolved_bin,
                    "10",       # target face count / simplification setting
                    in_obj,      # input obj
                    "0.999",    # distortion bound
                    "1",        # mode
                    "0",        # initial cut option
                    "4.1",      # b_d (>=4.1 according to binary warnings)
                    "1",        # normalize UV
                    "0",        # output option
                    run_tag,     # output tag
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
                if proc.returncode != 0:
                    raise RuntimeError(f"OptCuts failed (code={proc.returncode}): {proc.stderr.strip()}")

                out_obj = self._locate_optcuts_output_obj(tmpdir)
                if not os.path.exists(out_obj):
                    raise RuntimeError(f"OptCuts output OBJ not found: {out_obj}")

                uv = self._read_uv_from_obj(out_obj, expected_vertex_count=len(reference_uv))
                if uv is None:
                    raise RuntimeError(f"Failed to parse UV from OptCuts output: {out_obj}")

                if len(uv) != len(reference_uv):
                    raise RuntimeError(f"OptCuts UV vertex count mismatch ({len(uv)} vs {len(reference_uv)})")

                return uv
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"OptCuts execution error: {exc}") from exc

    @staticmethod
    def _locate_optcuts_output_obj(tmpdir: str) -> str:
        candidate_paths = [
            os.path.join(tmpdir, "output", "finalResult_mesh.obj"),
            os.path.join(tmpdir, "finalResult_mesh.obj"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path

        for root, _, files in os.walk(tmpdir):
            if "finalResult_mesh.obj" in files:
                return os.path.join(root, "finalResult_mesh.obj")
        return os.path.join(tmpdir, "output", "finalResult_mesh.obj")

    @staticmethod
    def _read_uv_from_obj(obj_path: str, expected_vertex_count: Optional[int] = None) -> Optional[np.ndarray]:
        try:
            loaded = trimesh.load(obj_path, process=False)
            if isinstance(loaded, trimesh.Trimesh):
                vis = getattr(loaded, "visual", None)
                uv = getattr(vis, "uv", None)
                if uv is not None and len(uv) > 0:
                    uv = np.asarray(uv, dtype=np.float64)
                    if expected_vertex_count is None or len(uv) == expected_vertex_count:
                        return uv
        except Exception:
            pass

        if meshio is not None:
            try:
                mesh = meshio.read(obj_path)
                if "obj:vt" in mesh.point_data:
                    uv = np.asarray(mesh.point_data["obj:vt"], dtype=np.float64)
                    if expected_vertex_count is None or len(uv) == expected_vertex_count:
                        return uv[:, :2]
            except Exception:
                pass

        if expected_vertex_count is not None:
            try:
                uv = OptCutsUVOptimizer._read_uv_from_obj_manual(obj_path, expected_vertex_count)
                if uv is not None:
                    return uv
            except Exception:
                pass

        logger.warning("Failed to parse UV from OptCuts OBJ: %s", obj_path)
        return None

    @staticmethod
    def _read_uv_from_obj_manual(obj_path: str, expected_vertex_count: int) -> Optional[np.ndarray]:
        texcoords = []
        vertex_uv_accum = [[] for _ in range(expected_vertex_count)]
        pending_pairs = []

        with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("vt "):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            texcoords.append((float(parts[1]), float(parts[2])))
                        except ValueError:
                            continue
                    continue

                if not line.startswith("f "):
                    continue

                face_tokens = line.strip().split()[1:]
                for token in face_tokens:
                    if "/" not in token:
                        continue
                    chunks = token.split("/")
                    if len(chunks) < 2 or not chunks[0] or not chunks[1]:
                        continue

                    try:
                        v_raw = int(chunks[0])
                        vt_raw = int(chunks[1])
                    except ValueError:
                        continue

                    pending_pairs.append((v_raw, vt_raw))

        if not texcoords:
            return None

        for v_raw, vt_raw in pending_pairs:
            v_idx = (expected_vertex_count + v_raw) if v_raw < 0 else (v_raw - 1)
            vt_idx = (len(texcoords) + vt_raw) if vt_raw < 0 else (vt_raw - 1)
            if 0 <= v_idx < expected_vertex_count and 0 <= vt_idx < len(texcoords):
                vertex_uv_accum[v_idx].append(texcoords[vt_idx])

        uv = np.zeros((expected_vertex_count, 2), dtype=np.float64)
        assigned = 0
        for i, candidates in enumerate(vertex_uv_accum):
            if not candidates:
                continue
            assigned += 1
            if len(candidates) == 1:
                uv[i] = candidates[0]
            else:
                uv[i] = np.mean(np.asarray(candidates, dtype=np.float64), axis=0)

        if assigned != expected_vertex_count:
            logger.warning(
                "OBJ UV manual parse assigned %d/%d vertices for %s",
                assigned,
                expected_vertex_count,
                obj_path,
            )
            return None
        return uv
