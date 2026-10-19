"""A shipped task must start from an obtainable commit in a clean checkout."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.task import load_task  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
