"""Each scored point's distance from the whole chip's roofline, in the replies (decided 2026-11-10)."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import roofline  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402

BOUNDS = {"cores": 3, "mem_bytes_per_cycle": 192, "points": {
    "op.matmul/medium/x": {"cycles": 21845.0, "binding": "fp_lane"},
    "op.add/tiny/y": {"cycles": 31.0, "binding": "mem_bytes"},
    "op.rope/small/z": {"cycles": 213.0, "binding": "mem_bytes"}}}


class View(unittest.TestCase):
    def test_every_point_by_number_in_the_sets_order_never_named(self):
        rows = [{"point": "op.add/tiny/y", "failure": None, "cycles": 1550, "base_cycles": 3100},
                {"point": "op.matmul/medium/x", "failure": None, "cycles": 187725, "base_cycles": 187725},
                {"point": "op.rope/small/z", "failure": "numeric-error", "cycles": None, "base_cycles": 82914}]
        got = roofline.view(BOUNDS, rows, ["op.matmul/medium/x", "op.add/tiny/y", "op.rope/small/z"])
        self.assertEqual(got, [{"n": 1, "binds": "compute", "eta": 0.1164, "eta_base": 0.1164},
                               {"n": 2, "binds": "memory", "eta": 0.02, "eta_base": 0.01},
                               {"n": 3, "binds": "memory", "eta": None, "eta_base": 0.0026}])
        self.assertNotIn("op.", json.dumps(got))
        self.assertIsNone(roofline.view(None, rows, ["op.add/tiny/y"]))


class Table(unittest.TestCase):
    def test_the_table_is_computed_once_per_base_and_kept(self):
        task = Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("compiler/",), evaluator_owned=(),
                    hide=(), instruction="", reference=None, wall_seconds=60,
                    evaluation={"profile": "p", "device": "d", "validation": ["v1", "v2"], "final": ["f"],
                                "board_validation": ["b1", "v1"]},
                    services={"score": {"requests": 1}}, path=HERE, sha256="")
        self.assertEqual(roofline.scored_points(task), ["v1", "v2", "b1"])      # never the final set
        calls = []

        def run(container, command, timeout):
            calls.append(json.loads(command[-1]))
            out = next(m.source for m in container.mounts if m.target == "/out")
            Path(out, "roofline.json").write_text(json.dumps(BOUNDS))
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            site = types.SimpleNamespace(state_dir=Path(tmp), images={"toolchain": "img"})
            with mock.patch("tritonchip_agent_harness.workspace.export"):
                first = roofline.table(site, Path(tmp), task, run=run)
                again = roofline.table(site, Path(tmp), task, run=run)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["points"], ["v1", "v2", "b1"])
        self.assertEqual(first["points"], BOUNDS["points"])
        self.assertEqual(again["version"], roofline.VERSION)

    def test_a_score_reply_lists_the_points(self):
        from tritonchip_agent_harness.services import Broker
        task = Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("compiler/",), evaluator_owned=(),
                    hide=(), instruction="", reference=None, wall_seconds=60,
                    evaluation={"profile": "p", "validation": ["op.add/tiny/y"], "final": ["f"]},
                    services={"score": {"requests": 1}}, path=HERE, sha256="")
        site = types.SimpleNamespace(fpga_worker=None, fpga_pipeline=None)
        b = Broker(site, HERE, task, Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "run")
        b.shared.roofline = BOUNDS
        r = {"aggregate": {"n": 1, "passed": 1, "speedup_geomean": 2.0, "fast_1": 1.0},
             "full": {"points": [{"point": "op.add/tiny/y", "failure": None, "cycles": 1550, "base_cycles": 3100}]}}
        with mock.patch("tritonchip_agent_harness.services.evaluator.measure_validation", return_value=r):
            visible, _ = b._run_score({"id": "s1", "containers": "c"}, None, Path(tempfile.mkdtemp()))
        self.assertEqual(visible["points"], [{"n": 1, "binds": "memory", "eta": 0.02, "eta_base": 0.01}])


if __name__ == "__main__":
    unittest.main()
