"""Backward-compatible import shim for UV optimizer.

Primary implementation now lives under src/optcuts/ for better modular management.
"""

from src.optcuts.joint_optimizer import OptCutsUVOptimizer, UVOptimizerConfig

__all__ = ["OptCutsUVOptimizer", "UVOptimizerConfig"]
