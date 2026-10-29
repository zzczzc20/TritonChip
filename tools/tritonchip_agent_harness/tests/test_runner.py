"""What an agent's container is given from the host, and what a refreshed credential must look like."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import doctor  # noqa: E402
from tritonchip_agent_harness.runner import BRIDGE_SCRIPT, agent_mounts, codex_auth_ok  # noqa: E402


class Mounts(unittest.TestCase):
    def test_the_agent_gets_the_bridge_and_never_the_harness(self):
        mounts = agent_mounts(Path("/tmp/sock"), Path("/tmp/ws"))
        sources = [Path(m.source) for m in mounts]
        self.assertNotIn(doctor.HERE, sources, "the harness directory holds the task packages")
        self.assertFalse(any((HERE / "tasks") in s.parents or s == HERE / "tasks" for s in sources))
        self.assertEqual([(m.source, m.target) for m in mounts if m.target == BRIDGE_SCRIPT],
                         [(doctor.HERE / "bridge.py", BRIDGE_SCRIPT)])
        self.assertTrue(all(m.read_only for m in mounts if m.target != "/work"))


class CodexWriteBack(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.master, self.copy = d / "master.json", d / "copy.json"
        self.master.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"a": 1}, "last_refresh": "x"}))

    def test_a_refreshed_credential_of_the_same_shape_may_go_back(self):
        self.copy.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"a": 2}, "last_refresh": "y"}))
        self.assertIsNone(codex_auth_ok(self.copy, self.master))

    def test_anything_else_stays_in_the_sandbox(self):
        self.copy.write_text(json.dumps({"tokens": {}}))
        self.assertEqual(codex_auth_ok(self.copy, self.master), "not the same shape as the master")
        self.copy.write_text("not json")
        self.assertEqual(codex_auth_ok(self.copy, self.master), "not JSON")
        self.copy.unlink()
        os.symlink(self.master, self.copy)
        self.assertEqual(codex_auth_ok(self.copy, self.master), "not a regular file")


if __name__ == "__main__":
    unittest.main()
