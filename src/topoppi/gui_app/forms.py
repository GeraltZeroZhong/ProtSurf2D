"""Pure GUI form parsing helpers.

These functions intentionally do not import Tkinter so they can be unit-tested
without a display server.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Mapping

from topoppi.config import DEFAULT_GUI_CONFIG, DEFAULT_RUN_CONFIG
from topoppi.errors import ConfigurationError


@dataclass(frozen=True)
class SingleRunForm:
    path: str
    chain_a: str
    chain_b: str
    prolif: str
    cutoff: float
    res: float
    sigma: float
    min_points: int
    optcuts_bin: str
    save_optcuts_frames: bool
    optcuts_frame_stride: int
    optcuts_min_frame_long_edge: int
    optcuts_frames_dir: str
    output_dir: str
    auto_save: bool

    def to_params(self) -> dict[str, object]:
        return {
            "path": self.path,
            "chain_a": self.chain_a,
            "chain_b": self.chain_b,
            "prolif": self.prolif,
            "cutoff": self.cutoff,
            "res": self.res,
            "sigma": self.sigma,
            "min_points": self.min_points,
            "optcuts_bin": self.optcuts_bin,
            "save_optcuts_frames": self.save_optcuts_frames,
            "optcuts_frame_stride": self.optcuts_frame_stride,
            "optcuts_min_frame_long_edge": self.optcuts_min_frame_long_edge,
            "optcuts_frames_dir": self.optcuts_frames_dir,
            "output_dir": self.output_dir,
            "auto_save": self.auto_save,
        }


@dataclass(frozen=True)
class BenchmarkForm:
    folder: str
    chain_a: str
    chain_b: str
    cutoff: float
    res: float
    sigma: float
    patch_gap: float
    optcuts_bin: str
    output_root: str
    resume: bool
    run_mode: str
    max_workers: int | None

    def to_params(self) -> dict[str, object]:
        return {
            "folder": self.folder,
            "chain_a": self.chain_a,
            "chain_b": self.chain_b,
            "cutoff": self.cutoff,
            "res": self.res,
            "sigma": self.sigma,
            "patch_gap": self.patch_gap,
            "optcuts_bin": self.optcuts_bin,
            "output_root": self.output_root,
            "resume": self.resume,
            "run_mode": self.run_mode,
            "max_workers": self.max_workers,
        }


def parse_single_run_form(raw: Mapping[str, object]) -> SingleRunForm:
    path = _required_path(raw.get("path"), "Input structure")
    if os.path.isdir(path):
        raise ConfigurationError("Single analysis requires a structure file, not a folder.")
    if not os.path.isfile(path):
        raise ConfigurationError(f"Input structure is not a file: {path}")

    prolif = _optional_file(raw.get("prolif"), "ProLIF JSON")
    output_dir = _optional_dir(raw.get("output_dir"), "Output directory")

    chain_a = _required_text(raw.get("chain_a"), "Surface chain")
    chain_b = _required_text(raw.get("chain_b"), "Partner chain")
    _require_distinct_chains(chain_a, chain_b)

    return SingleRunForm(
        path=path,
        chain_a=chain_a,
        chain_b=chain_b,
        prolif=prolif,
        cutoff=_positive_float(raw.get("cutoff"), "Interface cutoff"),
        res=_positive_float(raw.get("res"), "Grid resolution"),
        sigma=_positive_float(raw.get("sigma"), "Surface sigma"),
        min_points=_positive_int(raw.get("min_points"), "Minimum interacting residues"),
        optcuts_bin=_text_or_default(raw.get("optcuts_bin"), DEFAULT_RUN_CONFIG.optcuts.optcuts_bin),
        save_optcuts_frames=bool(raw.get("save_optcuts_frames", False)),
        optcuts_frame_stride=_positive_int(
            raw.get("optcuts_frame_stride"),
            "OptCuts frame stride",
            default=DEFAULT_RUN_CONFIG.optcuts.optcuts_frame_stride,
        ),
        optcuts_min_frame_long_edge=_non_negative_int(
            raw.get("optcuts_min_frame_long_edge"),
            "OptCuts minimum frame size",
            default=DEFAULT_RUN_CONFIG.optcuts.optcuts_min_frame_long_edge,
        ),
        optcuts_frames_dir=str(raw.get("optcuts_frames_dir") or "").strip(),
        output_dir=output_dir,
        auto_save=bool(raw.get("auto_save", DEFAULT_GUI_CONFIG.auto_save_single_run)),
    )


def parse_benchmark_form(raw: Mapping[str, object]) -> BenchmarkForm:
    folder = _required_path(raw.get("folder"), "Benchmark folder")
    if not os.path.isdir(folder):
        raise ConfigurationError(f"Benchmark folder does not exist: {folder}")

    output_root = _optional_output_root(raw.get("output_root"), "Benchmark output folder")
    if not output_root:
        output_root = os.path.join(folder, DEFAULT_GUI_CONFIG.benchmark_output_folder)

    chain_a = _required_text(raw.get("chain_a"), "Surface chain")
    chain_b = _required_text(raw.get("chain_b"), "Partner chain")
    _require_distinct_chains(chain_a, chain_b)

    run_mode = str(raw.get("run_mode") or "resume").strip().lower()
    if run_mode not in {"resume", "new", "overwrite"}:
        raise ConfigurationError("Benchmark run mode must be resume, new, or overwrite.")

    return BenchmarkForm(
        folder=folder,
        chain_a=chain_a,
        chain_b=chain_b,
        cutoff=_positive_float(raw.get("cutoff"), "Interface cutoff"),
        res=_positive_float(raw.get("res"), "Grid resolution"),
        sigma=_positive_float(raw.get("sigma"), "Surface sigma"),
        patch_gap=DEFAULT_RUN_CONFIG.optcuts.patch_gap,
        optcuts_bin=_text_or_default(raw.get("optcuts_bin"), DEFAULT_RUN_CONFIG.optcuts.optcuts_bin),
        output_root=output_root,
        resume=(run_mode == "resume"),
        run_mode=run_mode,
        max_workers=_optional_positive_int(raw.get("max_workers"), "Benchmark workers"),
    )


def _required_path(value: object, label: str) -> str:
    text = _required_text(value, label)
    if not os.path.exists(text):
        raise ConfigurationError(f"{label} does not exist: {text}")
    return text


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ConfigurationError(f"{label} is required.")
    return text


def _require_distinct_chains(chain_a: str, chain_b: str) -> None:
    if chain_a == chain_b:
        raise ConfigurationError("Surface chain and partner chain must be different.")


def _text_or_default(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or str(default)


def _optional_file(value: object, label: str) -> str:
    text = str(value or "").strip()
    if text and not os.path.isfile(text):
        raise ConfigurationError(f"{label} does not exist: {text}")
    return text


def _optional_dir(value: object, label: str) -> str:
    text = str(value or "").strip()
    if text and not os.path.isdir(text):
        raise ConfigurationError(f"{label} does not exist: {text}")
    return text


def _optional_output_root(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if os.path.exists(text) and not os.path.isdir(text):
        raise ConfigurationError(f"{label} is not a folder: {text}")
    parent = os.path.dirname(os.path.abspath(text)) or os.getcwd()
    if not os.path.isdir(parent):
        raise ConfigurationError(f"{label} parent does not exist: {parent}")
    return text


def _positive_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be a number.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ConfigurationError(f"{label} must be > 0.")
    return parsed


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _positive_int(value, label)


def _positive_int(value: object, label: str, default: int | None = None) -> int:
    if (value is None or str(value).strip() == "") and default is not None:
        return int(default)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be an integer.") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{label} must be > 0.")
    return parsed


def _non_negative_int(value: object, label: str, default: int | None = None) -> int:
    if (value is None or str(value).strip() == "") and default is not None:
        return int(default)
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be an integer.") from exc
    if parsed < 0:
        raise ConfigurationError(f"{label} must be >= 0.")
    return parsed
