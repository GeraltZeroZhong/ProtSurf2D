from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    input_folder: str
    output_root: str
    chain_a: str
    chain_b: str
    cutoff: float
    res: float
    sigma: float
    patch_gap: float = 0.08
    optcuts_bin: str = "OptCuts_bin"
    optcuts_headless: bool = True
    optcuts_quick_mode: bool = False
    raster_size: int = 256
    max_workers: Optional[int] = None
    show_tqdm: bool = True
    resume: bool = True
    min_lscm_patch_vertices: int = 10
    min_lscm_patch_faces: int = 8
