import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
CPP_ROOT = ROOT / "tools" / "OptCuts" / "residue_aware"


@unittest.skipUnless(shutil.which("g++"), "g++ is not available")
class OptCutsFootprintCppTests(unittest.TestCase):
    def test_candidate_paths_are_evaluated_as_transactions(self):
        sidecar = """\
TOPOPPI_FOOTPRINT_V2
COUNTS 4 1 6 4
SOURCES 10 11 12 13
WEIGHTS 1
FACE 0 1 0 1
FACE 1 1 0 1
FACE 2 1 0 1
FACE 3 1 0 1
EDGE 0 0 1 0 1 0 1 0
EDGE 1 0 2 0 2 0 1 0
EDGE 2 1 2 0 3 0 1 0
EDGE 3 0 3 1 2 0 1 0
EDGE 4 1 3 1 3 0 1 0
EDGE 5 2 3 2 3 0 1 0
"""
        driver = textwrap.dedent(
            """\
            #include "ResidueFootprintEnergy.hpp"
            #include <cassert>
            #include <cmath>
            #include <vector>

            int main(int argc, char** argv) {
                assert(argc == 2);
                TopoPPI::ResidueFootprintEnergy energy =
                    TopoPPI::ResidueFootprintEnergy::load(argv[1]);
                assert(energy.residueCount() == 1);
                assert(energy.edgeCount() == 6);
                assert(energy.cycleRank() == 3);
                assert(energy.edgeId(1, 0) == 0);
                assert(energy.inputSourceVertices().size() == 4);
                assert(energy.inputSourceVertices()[0] == 10);
                assert(std::abs(energy.score()) < 1e-12);

                std::vector<TopoPPI::FootprintEdgeChange> one = {{0, true}};
                assert(std::abs(energy.candidateDelta(one)) < 1e-12);

                std::vector<TopoPPI::FootprintEdgeChange> isolate = {
                    {0, true}, {1, true}, {2, true}
                };
                assert(std::abs(energy.candidateDelta(isolate) - 0.375) < 1e-12);
                energy.commit(isolate);
                assert(std::abs(energy.score() - 0.375) < 1e-12);

                std::vector<TopoPPI::FootprintEdgeChange> reconnect = {{0, false}};
                assert(std::abs(energy.candidateDelta(reconnect) + 0.375) < 1e-12);
                energy.commit(reconnect);
                assert(std::abs(energy.score()) < 1e-12);

                energy.commit(isolate);
                std::vector<int> faces = {
                    0, 1, 2,
                    0, 1, 3,
                    0, 2, 3,
                    1, 2, 3
                };
                std::vector<int> sources = {0, 1, 2, 3};
                energy.synchronize(faces, sources);
                assert(std::abs(energy.score()) < 1e-12);
                return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            sidecar_path = temporary / "footprints.txt"
            driver_path = temporary / "driver.cpp"
            binary_path = temporary / "footprint_test"
            sidecar_path.write_text(sidecar, encoding="utf-8")
            driver_path.write_text(driver, encoding="utf-8")
            subprocess.run(
                [
                    shutil.which("g++"),
                    "-std=c++11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{CPP_ROOT}",
                    str(CPP_ROOT / "ResidueFootprintEnergy.cpp"),
                    str(driver_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary_path), str(sidecar_path)], check=True)

    def test_mass_weighting_and_natural_components_match_the_objective(self):
        sidecar = """\
TOPOPPI_FOOTPRINT_V2
COUNTS 6 3 2 4
SOURCES 0 1 2 3
WEIGHTS 2 4 10
FACE 0 1 0 1
FACE 1 1 0 3
FACE 2 1 1 2
FACE 3 1 1 2
FACE 4 1 2 1
FACE 5 1 2 9
EDGE 0 0 1 0 1 0 1 0
EDGE 1 2 3 2 3 0 1 1
"""
        driver = textwrap.dedent(
            """\
            #include "ResidueFootprintEnergy.hpp"
            #include <cassert>
            #include <cmath>
            #include <vector>

            int main(int argc, char** argv) {
                assert(argc == 2);
                TopoPPI::ResidueFootprintEnergy energy =
                    TopoPPI::ResidueFootprintEnergy::load(argv[1]);
                assert(energy.residueCount() == 3);
                assert(std::abs(energy.score()) < 1e-12);

                // Residue 0 is split into mass fractions 1/4 and 3/4:
                // F_0 = 1 - (1/4)^2 - (3/4)^2 = 3/8.  With objective
                // weight 2 and total weight 16, its atlas contribution is 3/64.
                const std::vector<TopoPPI::FootprintEdgeChange> cut0 = {{0, true}};
                assert(std::abs(energy.candidateDelta(cut0) - 3.0 / 64.0) < 1e-12);
                energy.commit(cut0);
                assert(std::abs(energy.score() - 3.0 / 64.0) < 1e-12);

                // Residue 1 is split evenly, adding 4*(1/2)/16 = 1/8.
                const std::vector<TopoPPI::FootprintEdgeChange> cut1 = {{1, true}};
                assert(std::abs(energy.candidateDelta(cut1) - 1.0 / 8.0) < 1e-12);
                energy.commit(cut1);
                assert(std::abs(energy.score() - 11.0 / 64.0) < 1e-12);

                // Residue 2 starts as two natural components and therefore
                // contributes zero despite carrying most of the objective weight.
                const std::vector<TopoPPI::FootprintEdgeChange> reconnect0 = {{0, false}};
                assert(std::abs(energy.candidateDelta(reconnect0) + 3.0 / 64.0) < 1e-12);
                energy.commit(reconnect0);
                assert(std::abs(energy.score() - 1.0 / 8.0) < 1e-12);
                return 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            sidecar_path = temporary / "footprints.txt"
            driver_path = temporary / "driver.cpp"
            binary_path = temporary / "footprint_test"
            sidecar_path.write_text(sidecar, encoding="utf-8")
            driver_path.write_text(driver, encoding="utf-8")
            subprocess.run(
                [
                    shutil.which("g++"),
                    "-std=c++11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    f"-I{CPP_ROOT}",
                    str(CPP_ROOT / "ResidueFootprintEnergy.cpp"),
                    str(driver_path),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary_path), str(sidecar_path)], check=True)


if __name__ == "__main__":
    unittest.main()
