"""Central configuration objects for TopoPPI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

from topoppi.errors import ConfigurationError


@dataclass(frozen=True)
class SurfaceConfig:
    """Parameters for molecular surface generation."""

    grid_resolution: float = 2.0
    sigma: float = 1.0
    level: float = 0.1
    padding: float = 10.0
    max_voxels: int = 120_000_000
    closing_fraction: float = 0.9
    retry_attempts: int = 4
    retry_level_factor: float = 0.5
    min_vertices_per_atom: float = 0.5
    min_expected_vertices_cap: int = 500
    smoothing_iterations: int = 3

    def validate(self) -> None:
        _require_positive("surface.grid_resolution", self.grid_resolution)
        _require_positive("surface.sigma", self.sigma)
        _require_positive("surface.level", self.level)
        _require_positive("surface.padding", self.padding)
        _require_positive("surface.max_voxels", self.max_voxels)
        _require_positive("surface.retry_attempts", self.retry_attempts)
        _require_positive("surface.retry_level_factor", self.retry_level_factor)
        _require_positive("surface.min_vertices_per_atom", self.min_vertices_per_atom)
        _require_positive("surface.min_expected_vertices_cap", self.min_expected_vertices_cap)
        _require_non_negative("surface.smoothing_iterations", self.smoothing_iterations)
        if not 0.0 < self.closing_fraction <= 1.0:
            raise ConfigurationError("surface.closing_fraction must be in (0, 1].")


@dataclass(frozen=True)
class TopologyConfig:
    """Parameters for interface patch extraction and sanitation."""

    distance_cutoff: float = 9.0
    min_patch_vertices: int = 50
    degenerate_face_area: float = 1e-9
    max_edge_face_incidence: int = 2

    def validate(self) -> None:
        _require_positive("topology.distance_cutoff", self.distance_cutoff)
        _require_positive("topology.min_patch_vertices", self.min_patch_vertices)
        _require_positive("topology.degenerate_face_area", self.degenerate_face_area)
        _require_positive("topology.max_edge_face_incidence", self.max_edge_face_incidence)


@dataclass(frozen=True)
class ParameterizationConfig:
    """Parameters for mesh cleanup and UV parameterization."""

    method: str = "auto"
    min_vertices: int = 3
    min_face_area: float = 1e-5
    min_angle_deg: float = 2.0
    max_aspect_ratio: float = 50.0
    uv_epsilon: float = 1e-6
    expected_euler_characteristic: int = 1
    expected_boundary_loops: int = 1
    lscm_pin_a: tuple[float, float] = (0.0, 0.0)
    lscm_pin_b: tuple[float, float] = (1.0, 0.0)

    def validate(self) -> None:
        method = self.method.strip().lower()
        if method not in {"auto", "lscm", "harmonic", "spherical", "cylindrical"}:
            raise ConfigurationError(f"Unsupported parameterization method: {self.method}")
        _require_positive("parameterization.min_vertices", self.min_vertices)
        _require_positive("parameterization.min_face_area", self.min_face_area)
        _require_positive("parameterization.min_angle_deg", self.min_angle_deg)
        _require_positive("parameterization.max_aspect_ratio", self.max_aspect_ratio)
        _require_positive("parameterization.uv_epsilon", self.uv_epsilon)


@dataclass(frozen=True)
class OptCutsConfig:
    """Parameters for invoking OptCuts."""

    optcuts_bin: str = "OptCuts_bin"
    patch_gap: float = 0.08
    optcuts_mode: int = 10
    optcuts_headless_mode: int = 100
    optcuts_prog_mode: int = 1
    optcuts_method_type: int = 0
    optcuts_initial_cut_option: int = 0
    optcuts_lambda_init: float = 0.999
    optcuts_distortion_bound: float = 4.1
    optcuts_use_bijectivity: bool = True
    optcuts_quick_mode: bool = False
    optcuts_quick_distortion_bound: float = 6.0
    optcuts_quick_lambda_init: float = 0.95
    optcuts_quick_use_bijectivity: bool = False
    optcuts_output_tag: str = "patch"
    save_optcuts_frames: bool = False
    optcuts_frame_stride: int = 1
    optcuts_min_frame_long_edge: int = 3200
    optcuts_frames_dir: str = ""
    optcuts_copy_raw_gif: bool = True
    optcuts_env_var: str = "TOPOPPI_OPTCUTS_BIN"

    def validate(self) -> None:
        if not self.optcuts_bin.strip():
            raise ConfigurationError("optcuts.optcuts_bin is required.")
        _require_non_negative("optcuts.patch_gap", self.patch_gap)
        _require_positive("optcuts.optcuts_mode", self.optcuts_mode)
        _require_positive("optcuts.optcuts_headless_mode", self.optcuts_headless_mode)
        _require_non_negative("optcuts.optcuts_prog_mode", self.optcuts_prog_mode)
        _require_non_negative("optcuts.optcuts_method_type", self.optcuts_method_type)
        _require_non_negative("optcuts.optcuts_initial_cut_option", self.optcuts_initial_cut_option)
        _require_positive("optcuts.optcuts_lambda_init", self.optcuts_lambda_init)
        _require_positive("optcuts.optcuts_distortion_bound", self.optcuts_distortion_bound)
        _require_positive("optcuts.optcuts_quick_distortion_bound", self.optcuts_quick_distortion_bound)
        _require_positive("optcuts.optcuts_quick_lambda_init", self.optcuts_quick_lambda_init)
        _require_positive("optcuts.optcuts_frame_stride", self.optcuts_frame_stride)
        _require_non_negative("optcuts.optcuts_min_frame_long_edge", self.optcuts_min_frame_long_edge)
        if not self.optcuts_output_tag.strip():
            raise ConfigurationError("optcuts.optcuts_output_tag is required.")

    def for_headless(self) -> "OptCutsConfig":
        return replace(self, optcuts_mode=self.optcuts_headless_mode)


@dataclass(frozen=True)
class VisualizationConfig:
    """Parameters for plotting and residue interaction annotation."""

    show_plot: bool = False
    use_geometric_interaction_fallback: bool = True
    on_patch_distance: float = 3.0
    coarse_interaction_distance: float = 8.0
    partner_search_distance: float = 6.0
    vdw_distance: float = 6.0
    ionic_distance: float = 5.0
    strong_ionic_distance: float = 4.0
    hydrogen_bond_distance: float = 3.8
    pi_stack_distance: float = 4.5
    aromatic_distance: float = 5.5
    mesh_fill_alpha: float = 0.20
    mesh_line_alpha: float = 0.60
    label_offset: float = 0.04

    def validate(self) -> None:
        for name in (
            "on_patch_distance",
            "coarse_interaction_distance",
            "partner_search_distance",
            "vdw_distance",
            "ionic_distance",
            "strong_ionic_distance",
            "hydrogen_bond_distance",
            "pi_stack_distance",
            "aromatic_distance",
            "label_offset",
        ):
            _require_positive(f"visualization.{name}", getattr(self, name))
        _require_alpha("visualization.mesh_fill_alpha", self.mesh_fill_alpha)
        _require_alpha("visualization.mesh_line_alpha", self.mesh_line_alpha)


@dataclass(frozen=True)
class GUIConfig:
    """Defaults for the Tkinter GUI."""

    window_width: int = 1400
    window_height: int = 900
    min_window_width: int = 1180
    min_window_height: int = 760
    sidebar_width: int = 460
    ttk_theme: str = "clam"
    tk_scaling: float = 1.2
    font_family: str = "Segoe UI"
    font_fallbacks: tuple[str, ...] = ("Segoe UI", "Arial", "DejaVu Sans", "Noto Sans", "Liberation Sans")
    font_size: int = 10
    header_font_size: int = 16
    log_visible_lines: int = 7
    ui_poll_interval_ms: int = 80
    default_min_points: int = 10
    default_residue_color: str = "#d62728"
    auto_save_single_run: bool = True
    label_font_size: int = 9
    label_font_min_size: int = 5
    label_font_max_size: int = 20
    figure_dpi: int = 300
    benchmark_output_folder: str = "benchmark_results_resume"
    default_benchmark_run_mode: str = "resume"


@dataclass(frozen=True)
class TopoPPIRunConfig:
    """Configuration for one protein-protein interface map run."""

    pdb_file: str = ""
    chain_a: str = "A"
    chain_b: str = "B"
    output_file: str = "interface_map.png"
    prolif_file: Optional[str] = None
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    parameterization: ParameterizationConfig = field(default_factory=ParameterizationConfig)
    optcuts: OptCutsConfig = field(default_factory=OptCutsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def validate(self) -> None:
        input_path = Path(self.pdb_file)
        if not self.pdb_file:
            raise ConfigurationError("pdb_file is required.")
        if not input_path.exists():
            raise ConfigurationError(f"Input structure does not exist: {self.pdb_file}")
        if not input_path.is_file():
            raise ConfigurationError(f"Input structure is not a file: {self.pdb_file}")
        if not str(self.chain_a).strip():
            raise ConfigurationError("chain_a is required.")
        if not str(self.chain_b).strip():
            raise ConfigurationError("chain_b is required.")
        if self.prolif_file and not Path(self.prolif_file).exists():
            raise ConfigurationError(f"ProLIF JSON does not exist: {self.prolif_file}")
        if not str(self.output_file).strip():
            raise ConfigurationError("output_file is required.")
        self.surface.validate()
        self.topology.validate()
        self.parameterization.validate()
        self.optcuts.validate()
        self.visualization.validate()

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    input_folder: str
    output_root: str
    chain_a: str = "A"
    chain_b: str = "B"
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    parameterization: ParameterizationConfig = field(default_factory=ParameterizationConfig)
    optcuts: OptCutsConfig = field(default_factory=lambda: OptCutsConfig().for_headless())
    raster_size: int = 256
    max_workers: Optional[int] = None
    show_tqdm: bool = True
    resume: bool = True
    min_lscm_patch_vertices: int = 10
    min_lscm_patch_faces: int = 8
    min_chain_residues: int = 11
    checkpoint_filename: str = "benchmark_checkpoint.json"
    report_filename: str = "benchmark_report.json"
    summary_filename: str = "benchmark_summary.csv"

    def validate(self) -> None:
        input_path = Path(self.input_folder)
        if not input_path.exists() or not input_path.is_dir():
            raise ConfigurationError(f"Benchmark input folder does not exist: {self.input_folder}")
        if not self.output_root.strip():
            raise ConfigurationError("output_root is required.")
        if not self.chain_a.strip() or not self.chain_b.strip():
            raise ConfigurationError("benchmark chain IDs are required.")
        self.surface.validate()
        self.topology.validate()
        self.parameterization.validate()
        self.optcuts.validate()
        _require_positive("benchmark.raster_size", self.raster_size)
        _require_positive("benchmark.min_lscm_patch_vertices", self.min_lscm_patch_vertices)
        _require_positive("benchmark.min_lscm_patch_faces", self.min_lscm_patch_faces)
        _require_positive("benchmark.min_chain_residues", self.min_chain_residues)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_RUN_CONFIG = TopoPPIRunConfig()
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig(input_folder=".", output_root="benchmark_results")
DEFAULT_GUI_CONFIG = GUIConfig()


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigurationError(f"{name} must be > 0.")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ConfigurationError(f"{name} must be >= 0.")


def _require_alpha(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1].")
