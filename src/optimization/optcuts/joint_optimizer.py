from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh
from PIL import Image

try:
    import meshio
except Exception:  # optional dependency
    meshio = None

from src.atlas.metrics import UVAtlasMetrics

logger = logging.getLogger("UVOOptimizer")


@dataclass
class UVOptimizerConfig:
    optcuts_bin: str = "OptCuts_bin"
    patch_gap: float = 0.08
    optcuts_prog_mode: int = 1
    # OptCuts CLI positional argument (README: initialCutOption).
    # NOTE: this is not a frame-export switch.
    optcuts_initial_cut_option: int = 0
    save_optcuts_frames: bool = False
    # Export every frame by default. Larger values can be used to downsample.
    optcuts_frame_stride: int = 1
    # Upscale exported frames when the source image is too small.
    # The value is treated as the minimum target for BOTH width and height,
    # so exported frames are guaranteed to not stay at tiny dimensions (e.g. 320px wide).
    # 0 disables upscaling.
    optcuts_min_frame_long_edge: int = 3200
    optcuts_frames_dir: str = ""
    # Keep original GIF artifacts (if any) for direct playback/debugging.
    optcuts_copy_raw_gif: bool = True


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
            opt_uv = self._run_optcuts_for_patch(patch, uv, patch_index=idx)
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

    def _run_optcuts_for_patch(self, patch: trimesh.Trimesh, reference_uv: np.ndarray, patch_index: int) -> np.ndarray:
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
                    "10",       # mode: offline optimization with visualization outputs
                    in_obj,      # input mesh path
                    "0.999",    # lambda_init
                    str(int(self.config.optcuts_prog_mode)),  # testID
                    "0",        # methodType
                    "4.1",      # distortionBound
                    "1",        # useBijectivity
                    str(int(self.config.optcuts_initial_cut_option)),
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

                self._maybe_export_optcuts_frames(tmpdir=tmpdir, patch_index=patch_index)
                return uv
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"OptCuts execution error: {exc}") from exc

    def _maybe_export_optcuts_frames(self, tmpdir: str, patch_index: int) -> None:
        if not self.config.save_optcuts_frames:
            return

        output_dir = self.config.optcuts_frames_dir.strip()
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "optcuts_frames")
        os.makedirs(output_dir, exist_ok=True)

        png_paths = []
        raster_paths = []
        gif_paths = []
        for root, _, files in os.walk(tmpdir):
            for name in files:
                lower = name.lower()
                full_path = os.path.join(root, name)
                if lower.endswith(".png"):
                    png_paths.append(full_path)
                    raster_paths.append(full_path)
                elif lower.endswith((".bmp", ".jpg", ".jpeg", ".tif", ".tiff")):
                    raster_paths.append(full_path)
                elif lower.endswith(".gif"):
                    gif_paths.append(os.path.join(root, name))

        patch_dir = os.path.join(output_dir, f"patch_{patch_index:03d}")
        os.makedirs(patch_dir, exist_ok=True)

        stride = max(1, int(self.config.optcuts_frame_stride))

        def _frame_sort_key(path: str):
            # OptCuts frame names vary by build/output folder and are not always pure digits.
            # Build a natural sort key from all digit chunks in basename so names such as
            # 1.png / 10.png / frame_2_0.png are ordered consistently.
            base = os.path.splitext(os.path.basename(path))[0]
            chunks = re.findall(r"\d+", base)
            nums = tuple(int(c) for c in chunks) if chunks else ()
            return (0 if chunks else 1, nums, base, path)

        ordered_png_paths = sorted(png_paths, key=_frame_sort_key)
        ordered_raster_paths = sorted(raster_paths, key=_frame_sort_key)

        def _is_diagnostic_png(path: str) -> bool:
            base = os.path.basename(path).lower()
            stem = os.path.splitext(base)[0]
            return base == "finalresult.png" or stem.endswith("_distortion") or stem.endswith("_seam")

        def _is_viewer_like_png(path: str) -> bool:
            # Viewer snapshot names vary across OptCuts builds; keep this permissive.
            # Examples seen in practice:
            # - 0.png / 1.png / 2.png
            # - frame_0001.png / viewer_001.png / iter_10.png
            # We only exclude known static diagnostic outputs.
            base = os.path.basename(path).lower()
            stem = os.path.splitext(base)[0]
            if _is_diagnostic_png(path):
                return False
            if stem.isdigit():
                return True
            return any(tag in stem for tag in ("frame", "viewer", "iter", "step", "anim"))

        # Prefer likely viewer timeline PNGs; if unavailable, still prefer any non-diagnostic
        # PNG before falling back to GIF extraction (to avoid irreversible GIF quantization).
        viewer_like = [path for path in ordered_png_paths if _is_viewer_like_png(path)]
        non_diagnostic_pngs = [path for path in ordered_png_paths if not _is_diagnostic_png(path)]
        non_diagnostic_rasters = [path for path in ordered_raster_paths if not _is_diagnostic_png(path)]

        # Prefer PNG/raster viewer snapshots whenever they provide an actual frame sequence.
        # Only if sequence rasters are unavailable do we fallback to GIF extraction.
        frame_candidates = viewer_like if viewer_like else non_diagnostic_pngs
        prefer_png_sequence = bool(frame_candidates) and len(frame_candidates) > 1
        if not prefer_png_sequence and len(non_diagnostic_rasters) > 1:
            # Some OptCuts builds dump snapshots as JPG/BMP/TIFF instead of PNG.
            # Use those rasters directly (converted to PNG on save) to keep sharpness.
            frame_candidates = non_diagnostic_rasters
            prefer_png_sequence = True

        if prefer_png_sequence:
            # Sample by rank (every Nth frame after sorting), not by raw numeric ID.
            # This avoids dropping almost all frames when only a subset of files have
            # digit-only basenames.
            selected = frame_candidates[::stride]
            if not selected:
                selected = [frame_candidates[0]]
            for idx, src_path in enumerate(selected):
                base_name = os.path.basename(src_path)
                dst_name = f"{idx:04d}_{base_name}"
                dst_path = os.path.join(patch_dir, dst_name)
                self._copy_png_with_min_resolution(
                    src_path=src_path,
                    dst_path=dst_path,
                    min_long_edge=max(0, int(self.config.optcuts_min_frame_long_edge)),
                )
            logger.info(
                "OptCuts PNG frames exported for patch %d: %d image(s) -> %s",
                patch_index,
                len(selected),
                patch_dir,
            )
            if not viewer_like:
                logger.warning(
                    "No explicit viewer-like PNG naming pattern found for patch %d. "
                    "Exported non-diagnostic raster sequence to preserve quality.",
                    patch_index,
                )
            return

        # No sharp raster sequence found; fallback to GIF to preserve temporal process.
        # GIF is palette-quantized by OptCuts itself, so extraction keeps native resolution
        # (no upscaling) to avoid magnifying blocky artifacts.
        extracted_gif_frames = self._export_frames_from_gifs(
            gif_paths=gif_paths,
            patch_dir=patch_dir,
            stride=stride,
            min_long_edge=0,
        )
        if gif_paths and self.config.optcuts_copy_raw_gif:
            for gif_idx, gif_path in enumerate(sorted(gif_paths)):
                gif_name = f"source_{gif_idx:02d}_{os.path.basename(gif_path)}"
                shutil.copy2(gif_path, os.path.join(patch_dir, gif_name))
        if extracted_gif_frames > 0:
            logger.warning(
                "No raster frame sequence found for patch %d. Exported frames from GIF at native "
                "resolution (palette quantization is produced by OptCuts). For viewer-level sharp "
                "process frames, OptCuts itself must output per-iteration PNG screenshots.",
                patch_index,
            )
            logger.info(
                "OptCuts GIF viewer frames exported for patch %d: %d image(s) -> %s",
                patch_index,
                extracted_gif_frames,
                patch_dir,
            )
            return

        if not png_paths:
            logger.info("OptCuts frame export enabled, but no PNG/GIF frames were found for patch %d.", patch_index)
            return

        frame_candidates = ordered_png_paths
        selected = frame_candidates[::stride]
        if not selected:
            selected = [frame_candidates[0]]

        # Different OptCuts subfolders can contain frames with the same basename
        # (e.g. multiple "0.png"). Copying by basename would overwrite files and
        # make it look like only one image was exported. Use deterministic unique names.
        for idx, src_path in enumerate(selected):
            base_name = os.path.basename(src_path)
            dst_name = f"{idx:04d}_{base_name}"
            dst_path = os.path.join(patch_dir, dst_name)
            self._copy_png_with_min_resolution(
                src_path=src_path,
                dst_path=dst_path,
                min_long_edge=max(0, int(self.config.optcuts_min_frame_long_edge)),
            )

        final_png = self._find_first_existing_file(
            [
                os.path.join(tmpdir, "output", "finalResult.png"),
                os.path.join(tmpdir, "finalResult.png"),
            ]
        )
        if final_png:
            self._copy_png_with_min_resolution(
                src_path=final_png,
                dst_path=os.path.join(patch_dir, "finalResult.png"),
                min_long_edge=max(0, int(self.config.optcuts_min_frame_long_edge)),
            )

        logger.info(
            "OptCuts frames exported for patch %d: %d image(s) -> %s",
            patch_index,
            len(selected),
            patch_dir,
        )
        logger.warning(
            "No viewer-like frame sequence found for patch %d. Exported fallback diagnostic PNGs; "
            "the current OptCuts build may only emit static analysis images.",
            patch_index,
        )

    @staticmethod
    def _export_frames_from_gifs(gif_paths: List[str], patch_dir: str, stride: int, min_long_edge: int) -> int:
        if not gif_paths:
            return 0

        def _gif_sort_key(path: str):
            base = os.path.basename(path).lower()
            return (0 if base == "anim.gif" else 1, base, path)

        frame_count = 0
        ordered_gifs = sorted(gif_paths, key=_gif_sort_key)
        for gif_idx, gif_path in enumerate(ordered_gifs):
            with Image.open(gif_path) as gif:
                total_frames = getattr(gif, "n_frames", 1)
                for frame_idx in range(0, total_frames, stride):
                    gif.seek(frame_idx)
                    frame = gif.convert("RGBA")
                    out_name = f"{frame_count:04d}_viewer_g{gif_idx:02d}_f{frame_idx:05d}.png"
                    OptCutsUVOptimizer._save_png_with_min_resolution(
                        image=frame,
                        dst_path=os.path.join(patch_dir, out_name),
                        min_long_edge=min_long_edge,
                    )
                    frame_count += 1
        return frame_count

    @staticmethod
    def _resize_image_if_needed(image: Image.Image, min_long_edge: int) -> Image.Image:
        if min_long_edge <= 0:
            return image
        w, h = image.size
        if w <= 0 or h <= 0:
            return image
        # Historical behavior only constrained the long edge, which allowed very narrow
        # frames such as 320x1600 to pass without resizing. Here we enforce that both
        # dimensions reach at least min_long_edge so width does not remain tiny.
        scale_w = float(min_long_edge) / float(w)
        scale_h = float(min_long_edge) / float(h)
        scale = max(1.0, scale_w, scale_h)
        if scale <= 1.0:
            return image
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _copy_png_with_min_resolution(src_path: str, dst_path: str, min_long_edge: int) -> None:
        with Image.open(src_path) as img:
            OptCutsUVOptimizer._save_png_with_min_resolution(
                image=img.convert("RGBA"),
                dst_path=dst_path,
                min_long_edge=min_long_edge,
            )

    @staticmethod
    def _save_png_with_min_resolution(image: Image.Image, dst_path: str, min_long_edge: int) -> None:
        out = OptCutsUVOptimizer._resize_image_if_needed(image, min_long_edge=min_long_edge)
        out.save(dst_path)
        if min_long_edge <= 0:
            return
        # Defensive check: make sure saved output really meets the configured minimum.
        with Image.open(dst_path) as saved:
            sw, sh = saved.size
        if sw >= min_long_edge and sh >= min_long_edge:
            return
        # Second pass should be rare, but guarantees final dimensions if first pass
        # was bypassed by any unexpected image backend behavior.
        corrected = OptCutsUVOptimizer._resize_image_if_needed(out, min_long_edge=min_long_edge)
        corrected.save(dst_path)

    @staticmethod
    def _find_first_existing_file(paths: List[str]) -> Optional[str]:
        for p in paths:
            if os.path.exists(p):
                return p
        return None

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
