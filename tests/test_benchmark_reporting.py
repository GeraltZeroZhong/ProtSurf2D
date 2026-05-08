import unittest

from topoppi.benchmarking.reporting import aggregate_results


class BenchmarkReportingTests(unittest.TestCase):
    def test_empty_aggregate_is_stable(self):
        self.assertEqual(aggregate_results([]), {"valid_structure_count": 0})

    def test_error_rows_are_excluded_from_valid_count(self):
        self.assertEqual(
            aggregate_results([{"pdb": "x.pdb", "patch_count": 0, "error": "failed"}]),
            {"valid_structure_count": 0},
        )
