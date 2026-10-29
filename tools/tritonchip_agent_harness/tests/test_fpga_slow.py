"""The optional slow service keeps its pipeline and hidden-point evidence separate."""
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

from tritonchip_agent_harness import fpga_slow  # noqa: E402
from tritonchip_agent_harness.task import Task, TaskError, load_task, render  # noqa: E402


def task() -> Task:
    return Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("rtl/",),
                evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=20000,
                evaluation={"profile": "core-vrf-sync-v5", "validation": ["secret.validation"],
                            "final": ["secret.final"], "certify": {
                                "commit": "b" * 40, "paths": ["flow/fpga/"], "target": "u280", "clock_mhz": 100}},
                services={}, path=HERE, sha256="")


class SlowService(unittest.TestCase):
    def test_task_opt_in_requires_a_long_enough_agent_run(self):
        source = HERE / "tasks" / "r39-v5-main-pilot"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instruction.md").write_text("Improve the design.\n")
            value = json.loads((source / "task.json").read_text())
            value["services"]["fpga_slow"] = {"mode": "build", "minutes": 180,
                                                "job_minutes": 180}
            path = root / "task.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(TaskError, "wall budget is shorter"):
                load_task(path)
            value["budget"]["wall_seconds"] = 14400
            path.write_text(json.dumps(value))
            loaded = load_task(path)
            self.assertIn("tnca-service submit fpga-slow", render(loaded, "offline"))

    def test_manifest_exposes_only_validation_on_the_board(self):
        board = fpga_slow.manifest(task(), "l1", "c" * 40, board=True)
        self.assertEqual(board["board"]["archbench"]["points"], ["secret.validation"])
        self.assertNotIn("secret.final", json.dumps(board))
        build = fpga_slow.manifest(task(), "l1", "c" * 40, board=False)
        self.assertNotIn("board", build)

    def test_board_result_returns_only_an_aggregate_to_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"hosts": root / "hosts.json", "build_host": "build", "board_host": "card",
                      "allow_candidate_host_code": False}
            site = types.SimpleNamespace(fpga_pipeline=config)
            job = {"id": "l1", "remote_id": "agent-slow-1", "containers": "tnca-test"}
            sim = {"aggregate": {"n": 1, "passed": 1},
                   "full": {"points": {"secret.validation": {"status": "pass", "record_cycles": 42}}}}

            class Finished:
                def wait(self, timeout):
                    (root / "job" / "pipeline.json").write_text(json.dumps({
                        "status": "PASS", "run_id": "agent-slow-1", "build": {"status": "PASS",
                            "timing": {"wns_ns": 0.2}, "observed_clock_mhz": 100},
                        "board": {"status": "PASS"}}))
                    return 0

            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow.certify, "build_tree", return_value="c" * 40),
                  mock.patch.object(fpga_slow.certify, "fetch_board_records", return_value=[
                      {"point_id": "secret.validation", "status": "pass", "fallbacks": [],
                       "time": {"device": 42}}]),
                  mock.patch.object(fpga_slow.subprocess, "Popen", return_value=Finished()) as popen):
                visible, full = fpga_slow.run(site, root, task(), None, root / "job", job, 120,
                                              board=True)
            argv = popen.call_args.args[0]
            self.assertEqual(argv[argv.index("--run-id") + 1], "agent-slow-1")
            self.assertEqual(argv[argv.index("--board-host") + 1], "card")
            self.assertEqual(visible["board"], {"verdict": "certified", "validation_passed": 1,
                                                "validation_n": 1})
            self.assertNotIn("secret.validation", json.dumps(visible))
            self.assertNotIn("secret.final", json.dumps(visible))
            self.assertEqual(full["comparison"][0]["point"], "secret.validation")

    def test_failed_pipeline_cannot_be_reported_as_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"hosts": root / "hosts.json", "build_host": "build", "board_host": "card",
                   "allow_candidate_host_code": False}
            site = types.SimpleNamespace(fpga_pipeline=cfg)
            job = {"id": "l1", "remote_id": "agent-slow-2", "containers": "tnca-test"}
            sim = {"full": {"points": {"secret.validation": {"status": "pass", "record_cycles": 42}}}}

            class Finished:
                def wait(self, timeout):
                    (root / "job" / "pipeline.json").write_text(json.dumps({
                        "status": "BOARD_FAIL", "run_id": "agent-slow-2", "build": {"status": "PASS"},
                        "board": {"status": "BOARD_FAIL"}}))
                    return 0

            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow.certify, "build_tree", return_value="c" * 40),
                  mock.patch.object(fpga_slow.certify, "fetch_board_records", return_value=[
                      {"point_id": "secret.validation", "status": "pass", "fallbacks": [],
                       "time": {"device": 42}}]),
                  mock.patch.object(fpga_slow.subprocess, "Popen", return_value=Finished()),
                  mock.patch.object(fpga_slow, "cancel", return_value={})):
                visible, _ = fpga_slow.run(site, root, task(), None, root / "job", job, 120,
                                           board=True)
            self.assertEqual(visible["board"]["verdict"], "board-fail")

    def test_build_only_cancel_reconciles_no_card_host(self):
        from tritonchip_fpga_harness import hosts, pipeline

        remote_host = types.SimpleNamespace(python="python3", state_dir="/state",
                                            cli=lambda commit: f"/repo/{commit}/cli.py",
                                            checkout=lambda commit: f"/repo/{commit}")
        job = {"mode": "build", "remote_id": "agent-slow-3", "tree_commit": "c" * 40,
               "containers": "tnca-test"}
        cfg = {"hosts": Path("/hosts.json"), "build_host": "build", "board_host": "card"}
        with (mock.patch.object(fpga_slow.provider, "docker", return_value=types.SimpleNamespace(stdout="")),
              mock.patch.object(hosts, "load_hosts", return_value={}),
              mock.patch.object(hosts, "host_for", return_value=remote_host) as choose,
              mock.patch.object(pipeline, "remote", return_value=types.SimpleNamespace(
                  returncode=0, stdout=b"terminated")) as remote):
            result = fpga_slow.cancel(job, cfg)
        self.assertEqual(choose.call_count, 1)
        self.assertEqual(choose.call_args.args[2], "build")
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(result["build"]["exit"], 0)


if __name__ == "__main__":
    unittest.main()
