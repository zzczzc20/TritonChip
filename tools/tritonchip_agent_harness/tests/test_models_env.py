"""The performance models' environment (task `models`, 2026-11-08): SystemC and the hook, only when asked."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import runner  # noqa: E402
from tritonchip_agent_harness.siteconfig import SiteError, load_site  # noqa: E402
from tritonchip_agent_harness.task import TaskError, load_task  # noqa: E402

EXAMPLE = HERE / "examples" / "site.json"


def systemc_install(root: Path) -> Path:
    (root / "lib" / "cmake" / "SystemCLanguage").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "include" / "systemc.h").write_text("// mock\n")
    return root


class ModelsEnvironment(unittest.TestCase):
    def test_the_site_names_a_systemc_installation_or_none(self):
        base = json.loads(EXAMPLE.read_text())
        d = Path(tempfile.mkdtemp())
        self.assertIsNone(load_site(EXAMPLE).systemc)
        base["systemc"] = str(systemc_install(d / "sc"))
        (d / "site.json").write_text(json.dumps(base))
        self.assertEqual(load_site(d / "site.json").systemc, d / "sc")
        base["systemc"] = str(d / "nothing")
        (d / "site.json").write_text(json.dumps(base))
        with self.assertRaisesRegex(SiteError, "not a SystemC installation"):
            load_site(d / "site.json")

    def test_a_task_asks_for_the_models_with_a_boolean(self):
        src = HERE / "tasks" / "r41-dp-formal" / "task.json"
        v = json.loads(src.read_text())
        d = Path(tempfile.mkdtemp())
        for f in src.parent.iterdir():
            if f.is_file() and f.name != "task.json":
                (d / f.name).write_bytes(f.read_bytes())
        if not (d / v["instruction"]).exists():
            (d / v["instruction"]).parent.mkdir(parents=True, exist_ok=True)
            (d / v["instruction"]).write_text("x")
        for value, ok in ((True, True), (False, True), ("yes", False)):
            v["models"] = value
            (d / "task.json").write_text(json.dumps(v))
            if ok:
                self.assertEqual(load_task(d / "task.json").models, value)
            else:
                with self.assertRaisesRegex(TaskError, "models must be"):
                    load_task(d / "task.json")

    def test_mounts_and_variables_only_for_a_task_that_asks(self):
        sc = systemc_install(Path(tempfile.mkdtemp()) / "sc")
        plain = runner.agent_mounts(Path("/s"), Path("/w"))
        self.assertNotIn(runner.SYSTEMC_DIR, [m.target for m in plain])
        mounts = {m.target: m for m in runner.agent_mounts(Path("/s"), Path("/w"), systemc=sc)}
        self.assertEqual(mounts[runner.SYSTEMC_DIR].source, sc)
        self.assertEqual(mounts[runner.MODELS_DIR].source, runner.MODELS_SOURCE)
        self.assertTrue(mounts[runner.SYSTEMC_DIR].read_only and mounts[runner.MODELS_DIR].read_only)
        env = runner.models_env(types.SimpleNamespace(systemc=sc))
        self.assertEqual(env["SystemCLanguage_DIR"], "/opt/systemc/lib/cmake/SystemCLanguage")
        self.assertEqual(env["TNCA_MODEL_HOOK"], runner.MODELS_DIR)
        with self.assertRaisesRegex(RuntimeError, "no `systemc`"):
            runner.models_env(types.SimpleNamespace(systemc=None))
        self.assertEqual(len(runner.models_manifest(types.SimpleNamespace(systemc=sc))["hook_sha256"]), 64)

    def test_the_hook_registers_the_model_backend_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            marker = model / "registered"
            (model / "tlm_backend.py").write_text(
                f"def register():\n    open({str(marker)!r}, 'w').write('yes')\n")
            env = {**os.environ, "PYTHONPATH": str(runner.MODELS_SOURCE), "TNCA_MODEL_DIR": str(model)}
            subprocess.run([sys.executable, "-c", "pass"], env={**env, "TNCA_BACKEND": "fpga-multicore-sim"},
                           check=True)
            self.assertFalse(marker.exists())
            subprocess.run([sys.executable, "-c", "pass"], env={**env, "TNCA_BACKEND": "systemc-tlm-grid"},
                           check=True)
            self.assertEqual(marker.read_text(), "yes")


if __name__ == "__main__":
    unittest.main()
