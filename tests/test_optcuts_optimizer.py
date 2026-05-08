import tempfile
import unittest
from pathlib import Path

from topoppi.optimization.optcuts.joint_optimizer import OptCutsUVOptimizer


class OptCutsOptimizerTests(unittest.TestCase):
    def test_manual_obj_uv_parser(self):
        obj_text = """\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"""
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = Path(tmp) / "uv.obj"
            obj_path.write_text(obj_text, encoding="utf-8")

            uv = OptCutsUVOptimizer._read_uv_from_obj_manual(str(obj_path), expected_vertex_count=3)

        self.assertEqual(uv.shape, (3, 2))
        self.assertEqual(float(uv[1, 0]), 1.0)
