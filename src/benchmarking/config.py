from dataclasses import dataclass
from typing import List, Optional


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
    raster_size: int = 256
    cutoff_sweep: Optional[List[float]] = None
    sigma_sweep: Optional[List[float]] = None
    res_sweep: Optional[List[float]] = None
