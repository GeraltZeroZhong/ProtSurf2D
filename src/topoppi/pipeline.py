"""Reusable TopoPPI pipeline API."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import trimesh

from topoppi.config import TopoPPIRunConfig
from topoppi.errors import InputDataError, PipelineError
from topoppi.interactions.interaction_engine import generate_prolif_interactions
from topoppi.io.io_loader import PDBLoader
from topoppi.mesh.parameterization import Parameterizer
from topoppi.mesh.surface import SurfaceGenerator
from topoppi.mesh.topology import TopologyManager
from topoppi.optimization.optcuts import OptCutsUVOptimizer
from topoppi.visualization.visualizer import InterfaceVisualizer

logger = logging.getLogger("topoppi.pipeline")


@dataclass
class TopoPPIRunResult:
    """Result metadata for one TopoPPI pipeline run."""

    output_file: str
    prolif_file: Optional[str]
    patch_count: int
    valid_patch_count: int
    elapsed_sec: float
    optimizer_report: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_interface_mapping(config: TopoPPIRunConfig, log: Optional[logging.Logger] = None) -> TopoPPIRunResult:
    """Run the TopoPPI single-structure interface mapping pipeline."""

    log = log or logger
    config.validate()
    start_time = time.time()

    prolif_file = _resolve_prolif_file(config, log)
    coords_a, atoms_a, coords_b, atoms_b = _load_chain_data(config, log)
    mesh_a = _generate_surface(coords_a, config, log)
    patches = _extract_patches(mesh_a, coords_b, config, log)
    valid_patches = _parameterize_patches(patches, config, log)
    optimizer_report = _optimize_patches(valid_patches, config, log)
    _render_output(valid_patches, atoms_a, coords_a, atoms_b, coords_b, prolif_file, config, log)

    elapsed = time.time() - start_time
    log.info("Pipeline finished in %.2fs. Saved to %s", elapsed, config.output_file)
    return TopoPPIRunResult(
        output_file=str(config.output_file),
        prolif_file=prolif_file,
        patch_count=len(patches),
        valid_patch_count=len(valid_patches),
        elapsed_sec=float(elapsed),
        optimizer_report=optimizer_report,
    )


def _resolve_prolif_file(config: TopoPPIRunConfig, log: logging.Logger) -> Optional[str]:
    if config.prolif_file:
        return config.prolif_file
    generated_json = generate_prolif_interactions(config.pdb_file, config.chain_a, config.chain_b, log)
    return generated_json if generated_json else None


def _load_chain_data(config: TopoPPIRunConfig, log: logging.Logger):
    log.info("Loading %s...", config.pdb_file)
    try:
        loader = PDBLoader(config.pdb_file)
        coords_a, atoms_a = loader.get_chain_data(config.chain_a)
        coords_b, atoms_b = loader.get_chain_data(config.chain_b)
    except Exception as exc:
        raise InputDataError(f"Failed to load structure data: {exc}") from exc

    if len(coords_a) == 0:
        raise InputDataError(f"Chain {config.chain_a} has no standard protein atoms.")
    if len(coords_b) == 0:
        raise InputDataError(f"Chain {config.chain_b} has no standard protein atoms.")
    log.info("Loaded Chain %s: %d atoms", config.chain_a, len(coords_a))
    log.info("Loaded Chain %s: %d atoms", config.chain_b, len(coords_b))
    return coords_a, atoms_a, coords_b, atoms_b


def _generate_surface(coords_a, config: TopoPPIRunConfig, log: logging.Logger) -> trimesh.Trimesh:
    log.info("Generating SES surface for Chain %s...", config.chain_a)
    mesh_a = SurfaceGenerator(coords_a, config=config.surface).generate_mesh()
    if mesh_a is None or len(mesh_a.vertices) == 0:
        raise PipelineError("Failed to generate surface mesh.")
    return mesh_a


def _extract_patches(mesh_a: trimesh.Trimesh, coords_b, config: TopoPPIRunConfig, log: logging.Logger) -> List[trimesh.Trimesh]:
    log.info("Extracting interface patches...")
    patches = TopologyManager(mesh_a, coords_b, config=config.topology).get_interface_patches()
    if not patches:
        raise PipelineError("No interface patches found. Try increasing cutoff.")
    return list(patches)


def _parameterize_patches(patches: List[trimesh.Trimesh], config: TopoPPIRunConfig, log: logging.Logger) -> List[trimesh.Trimesh]:
    log.info("Parameterizing %d patches...", len(patches))
    valid_patches = []
    parameterizer = Parameterizer(config=config.parameterization)
    for idx, patch in enumerate(patches):
        log.info("Flattening patch %d (%d vertices)...", idx + 1, len(patch.vertices))
        uv = parameterizer.flatten_patch(patch)
        if uv is None:
            log.warning("Skipping patch %d due to parameterization failure.", idx + 1)
            continue
        patch.metadata["uv"] = uv
        valid_patches.append(patch)

    if not valid_patches:
        raise PipelineError("All patches failed to parameterize.")
    return valid_patches


def _optimize_patches(patches: List[trimesh.Trimesh], config: TopoPPIRunConfig, log: logging.Logger) -> Dict[str, object]:
    log.info("Running OptCuts UV optimization...")
    optimizer = OptCutsUVOptimizer(config.optcuts)
    optimizer.optimize_patches(patches)
    report = optimizer.get_last_report() if hasattr(optimizer, "get_last_report") else {}
    if report:
        log.info("Joint report: %s", report)
    return report


def _render_output(
    patches: List[trimesh.Trimesh],
    atoms_a,
    coords_a,
    atoms_b,
    coords_b,
    prolif_file: Optional[str],
    config: TopoPPIRunConfig,
    log: logging.Logger,
) -> None:
    log.info("Visualizing results...")
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz = InterfaceVisualizer(
        chain_A_atoms=atoms_a,
        chain_A_coords=coords_a,
        chain_B_coords=coords_b,
        chain_B_atoms=atoms_b,
        chain_a_id=config.chain_a,
        chain_b_id=config.chain_b,
        prolif_file=prolif_file,
        config=config.visualization,
    )
    viz.plot_patches(patches, output_file=str(output_path), show=config.visualization.show_plot)
