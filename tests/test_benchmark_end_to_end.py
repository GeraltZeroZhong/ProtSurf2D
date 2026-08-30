import csv
import gzip
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from topoppi.benchmarking.runner import BenchmarkRunner
from topoppi.config import (
    BenchmarkConfig,
    OptCutsConfig,
    ParameterizationConfig,
    SurfaceConfig,
    TopologyConfig,
)
from topoppi.file_utils import sha256_file

ROOT = Path(__file__).parents[1]
TINY_PDB = ROOT / "tests" / "fixtures" / "tiny_complex.pdb"
OPTCUTS_BIN = ROOT / "tools" / "OptCuts" / "OptCuts_bin"
TINY_PDB_SHA256 = sha256_file(TINY_PDB)


@unittest.skipUnless(OPTCUTS_BIN.is_file(), "OptCuts binary is not available")
class BenchmarkEndToEndTests(unittest.TestCase):
    @staticmethod
    def _operational_config(
        tmp_name: str,
        method: str,
        weight: float,
        binary: Path = OPTCUTS_BIN,
    ) -> BenchmarkConfig:
        return BenchmarkConfig(
            input_folder=str(TINY_PDB.parent),
            output_root=str(Path(tmp_name) / "output"),
            execution_profile="operational_optcuts",
            optcuts_variants=(method,),
            include_topology_ablation=False,
            repetitions=1,
            max_workers=1,
            show_tqdm=False,
            resume=False,
            min_chain_residues=1,
            surface=SurfaceConfig(
                grid_resolution=0.5,
                sigma=0.8,
                level=0.02,
                padding=3.0,
                max_voxels=4_000_000,
                smoothing_iterations=0,
            ),
            topology=TopologyConfig(min_patch_vertices=3),
            parameterization=ParameterizationConfig(min_face_area=1e-10, slim_iterations=2),
            optcuts=replace(
                OptCutsConfig(),
                optcuts_bin=str(binary),
                optcuts_mode=OptCutsConfig().optcuts_headless_mode,
                residue_fragmentation_weight=weight,
                timeout_sec=60.0,
            ),
        )

    def test_operational_profile_executes_only_the_public_optcuts_path(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            config = self._operational_config(tmp_name, "optcuts_automatic", 0.0)

            result = BenchmarkRunner(config, worker_mode=True)._run_single(
                str(TINY_PDB),
                "A",
                "B",
                {"input_sha256": TINY_PDB_SHA256},
            )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["execution_domain"]["complete"])
        self.assertTrue(result["execution_domain"]["scientifically_usable"])
        self.assertTrue(result["execution_certificate"]["complete"])
        self.assertEqual(
            result["execution_certificate"]["certified_patch_count"],
            result["prepared_patch_count"],
        )
        self.assertFalse(result["comparison_domain"]["complete"])
        self.assertEqual(set(result["method_execution"]), {"optcuts_automatic"})
        self.assertNotIn("lscm_parameterization", result["timing"]["stages"])
        self.assertNotIn("residue_footprint_fragmentation", result)

    def test_operational_profile_measures_the_residue_aware_method(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            config = self._operational_config(
                tmp_name,
                "residue_aware_optcuts",
                1.0,
            )

            result = BenchmarkRunner(config, worker_mode=True)._run_single(
                str(TINY_PDB),
                "A",
                "B",
                {"input_sha256": TINY_PDB_SHA256},
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["operational_method"], "residue_aware_optcuts")
        self.assertEqual(set(result["method_execution"]), {"residue_aware_optcuts"})
        self.assertIn("contact_weight_preparation", result["timing"]["stages"])

    def test_isolated_worker_writes_a_strict_traceable_evidence_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            inputs = root / "inputs"
            inputs.mkdir()
            shutil.copy2(TINY_PDB, inputs / TINY_PDB.name)
            output_root = root / "output"
            config = BenchmarkConfig(
                input_folder=str(inputs),
                output_root=str(output_root),
                repetitions=1,
                warmup_runs=0,
                max_workers=1,
                show_tqdm=False,
                resume=False,
                min_chain_residues=1,
                include_topology_ablation=False,
                raster_size=64,
                bootstrap_iterations=20,
                per_face_sample_size_per_patch=8,
                surface=SurfaceConfig(
                    grid_resolution=0.5,
                    sigma=0.8,
                    level=0.02,
                    padding=3.0,
                    max_voxels=4_000_000,
                    smoothing_iterations=0,
                ),
                topology=TopologyConfig(min_patch_vertices=3),
                parameterization=ParameterizationConfig(min_face_area=1e-10, slim_iterations=2),
                optcuts=replace(
                    OptCutsConfig(),
                    optcuts_bin=str(OPTCUTS_BIN),
                    optcuts_mode=OptCutsConfig().optcuts_headless_mode,
                    timeout_sec=60.0,
                ),
            )

            self.assertTrue(BenchmarkRunner(config).preflight()["ready"])
            output = BenchmarkRunner(config).run()
            with (output_root / config.report_filename).open(encoding="utf-8") as handle:
                report = json.load(
                    handle,
                    parse_constant=lambda token: self.fail(f"non-standard JSON token: {token}"),
                )
            with (output_root / config.per_face_sample_filename).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                face_rows = list(csv.DictReader(handle))
            with gzip.open(
                output_root / config.provenance_filename,
                "rt",
                newline="",
                encoding="utf-8",
            ) as handle:
                provenance_rows = list(csv.DictReader(handle))
            with gzip.open(
                output_root / config.per_residue_filename,
                "rt",
                newline="",
                encoding="utf-8",
            ) as handle:
                residue_rows = list(csv.DictReader(handle))
            with gzip.open(
                output_root / config.optcuts_execution_filename,
                "rt",
                encoding="utf-8",
            ) as handle:
                execution_rows = [json.loads(line) for line in handle]

            record = report["files"][0]
            self.assertEqual(record["status"], "ok")
            self.assertTrue(record["comparison_domain"]["complete"])
            arm = record["independent_optcuts_arm_quality"]["optcuts_automatic"]
            self.assertTrue(arm["domain_complete"])
            self.assertIsNotNone(arm["quality"])
            self.assertIn("residue_footprint_fragmentation", arm["quality"])
            self.assertEqual(
                record["timing"]["end_to_end"]["cpu_scope"],
                "worker process plus waited-for child processes",
            )
            self.assertTrue(all(block["success"] == 1 for block in record["method_execution"].values()))
            self.assertEqual(len(record["worker_measurements"]), 1)
            self.assertEqual(len(face_rows), 8)
            self.assertGreater(len(provenance_rows), 0)
            self.assertGreater(len(residue_rows), 0)
            self.assertEqual(len(execution_rows), 1)
            self.assertIn("optcuts_lscm_initialized", execution_rows[0]["methods"])
            self.assertNotIn("executions", record["method_execution"]["optcuts_lscm_initialized"])
            self.assertNotIn(
                "residues",
                record["residue_footprint_fragmentation"]["methods"]["lscm"],
            )
            self.assertTrue((output_root / record["detail_artifact"]["path"]).is_file())
            self.assertEqual(output["summary"]["complete_comparison_structure_count"], 1)
            self.assertTrue((output_root / config.artifact_checksums_filename).is_file())


if __name__ == "__main__":
    unittest.main()
