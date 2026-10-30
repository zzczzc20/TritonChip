"""A shipped task must start from an obtainable commit in a clean checkout."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.task import TaskError, load_task  # noqa: E402
from tritonchip_agent_harness.workspace import export, gate  # noqa: E402


class BuiltinTask(unittest.TestCase):
    def test_r39_v5_main_pilot_exports_and_gates(self):
        task = load_task(HERE / "tasks" / "r39-v5-main-pilot")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            result = export(ROOT, task.base_commit, task.hide, workspace)
            self.assertGreater(result["files"], 500)
            self.assertFalse((workspace / "archbench").exists())
            self.assertFalse((workspace / "tools" / "tritonchip_agent_harness").exists())
            self.assertIn("m.zero", (workspace / "compiler/python/tnca_cc/frontend.py").read_text())
            clean = gate(ROOT, task.base_commit, task.hide, task.design, workspace,
                         evaluator_owned=task.evaluator_owned)
            self.assertEqual(clean["verdict"], "PASS")
            self.assertFalse(clean["taken"])


class MulticoreTask(unittest.TestCase):
    def test_a_device_task_pins_its_bench(self):
        value = json.loads((HERE / "tasks" / "r39-v5-main-pilot" / "task.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instruction.md").write_text("Improve the machine.\n")
            path = root / "task.json"
            for evaluation, error in (
                    ({"device": "u280-3core-r40", "platform": None}, "needs evaluation.platform"),
                    ({"device": "../escape"}, "must name a description"),
                    ({"device": "u280-3core-r40", "drain_cycles": -1}, "non-negative whole number"),
                    ({"device": "u280-3core-r40", "drain_cycles": True}, "non-negative whole number")):
                with self.subTest(evaluation=evaluation):
                    path.write_text(json.dumps(dict(value, evaluation=dict(value["evaluation"], **evaluation))))
                    with self.assertRaisesRegex(TaskError, error):
                        load_task(path)
            path.write_text(json.dumps(dict(value, evaluation=dict(value["evaluation"], device="u280-3core-r40",
                                                                   drain_cycles=64))))
            self.assertEqual(load_task(path).evaluation["drain_cycles"], 64)


if __name__ == "__main__":
    unittest.main()
