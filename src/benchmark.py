"""Benchmark compatibility entrypoint.

This module intentionally stays small and re-exports the benchmark public API.
Detailed logic lives in ``src/benchmarking`` submodules.
"""

from src.benchmarking import BenchmarkConfig, BenchmarkRunner

__all__ = ["BenchmarkConfig", "BenchmarkRunner"]
