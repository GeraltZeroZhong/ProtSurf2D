"""One-structure benchmark worker used for process-isolated measurement."""

from __future__ import annotations

import json
import sys
import traceback

from topoppi.benchmarking.runner import BenchmarkRunner
from topoppi.config import benchmark_config_from_dict
from topoppi.json_utils import dump_json_atomic


def run_worker(job_path: str, result_path: str) -> int:
    with open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    config = benchmark_config_from_dict(job["config"])
    runner = BenchmarkRunner(config=config, worker_mode=True)
    try:
        result = runner._run_single(
            str(job["pdb_path"]),
            str(job["chain_a"]),
            str(job["chain_b"]),
            job_metadata=dict(job["job_metadata"]),
        )
        payload = {"status": "ok", "result": result}
        exit_code = 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    dump_json_atomic(payload, result_path)
    return exit_code


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("Usage: python -m topoppi.benchmarking.worker JOB.json RESULT.json")
    return run_worker(args[0], args[1])


if __name__ == "__main__":
    raise SystemExit(main())
