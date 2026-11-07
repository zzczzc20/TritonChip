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

from tritonchip_agent_harness import board, fpga_slow  # noqa: E402
from tritonchip_agent_harness.task import Task, TaskError, load_task, render  # noqa: E402


def task() -> Task:
    return Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("rtl/",),
                evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=20000,
                evaluation={"profile": "core-vrf-sync-v5", "validation": ["secret.validation"],
                            "final": ["secret.final"], "certify": {
                                "commit": "b" * 40, "paths": ["flow/fpga/"], "target": "u280", "clock_mhz": 100}},
                services={}, path=HERE, sha256="")


class SlowService(unittest.TestCase):
    def test_task_config_counts_flows_and_board_runs(self):
        source = HERE / "tasks" / "r39-v5-main-pilot"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instruction.md").write_text("Improve the design.\n")
            value = json.loads((source / "task.json").read_text())
            value["services"]["fpga_slow"] = {"mode": "board", "minutes": 180, "job_minutes": 180}
            path = root / "task.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(TaskError, "fpga_slow needs"):
                load_task(path)
            value["services"]["fpga_slow"] = {"mode": "board", "flows": 3, "board_runs": 6,
                                                "flow_minutes": 480, "board_minutes": 120}
            value["evaluation"]["platform"] = {"commit": "b" * 40, "paths": []}
            path.write_text(json.dumps(value))
            text = render(load_task(path), "offline")
            self.assertIn("tnca-service submit fpga-slow", text)
            self.assertIn("Board time is precious", text)
            self.assertIn("3 flows in this run", text)

    def test_manifest_exposes_only_validation_on_the_board(self):
        board = fpga_slow.manifest(task(), "l1", "c" * 40, board=True)
        self.assertEqual(board["board"]["archbench"]["points"], ["secret.validation"])
        self.assertNotIn("secret.final", json.dumps(board))
        build = fpga_slow.manifest(task(), "l1", "c" * 40, board=False)
        self.assertNotIn("board", build)

    def test_board_result_returns_only_an_aggregate_to_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "hosts.json", "build_host": "build",
                                                        "board_host": "card", "card": {"bdf": "0000:04:00.1"}})
            job = {"id": "l1", "remote_id": "agent-slow-1", "containers": "tnca-test", "kind": "software"}
            sim = {"aggregate": {"n": 1, "passed": 1},
                   "full": {"points": [{"point": "secret.validation", "failure": None, "done_cycles": 42}]}}
            card = {"session": {"status": "ok"}, "measured": {"checks": {"fp8": {"exit": 1, "log_tail": ["x"]}}},
                    "points": {"secret.validation": {"status": "pass", "cycles": 40, "record_cycles": 42,
                                                     "fallbacks": [], "observer": {"done_cycles": 42}}}}
            base = {"points": {"secret.validation": {"status": "pass", "cycles": 80}}}
            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow.board, "measure", return_value=card),
                  mock.patch.object(fpga_slow.board, "baseline", return_value=base)):
                visible, full = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                              image={"path": "/x", "sha256": "c" * 64}, flow_seconds=60,
                                              board_seconds=60)
            self.assertEqual(visible["board"]["validation"]["speedup_geomean"], 2.0)
            self.assertEqual(visible["board"]["counted_as_in_simulation"], 1)
            self.assertEqual(visible["board"]["checks"], {"fp8": "fail"})
            self.assertEqual(visible["charged"], "board_run")
            self.assertNotIn("secret.validation", json.dumps(visible))
            self.assertNotIn("secret.final", json.dumps(visible))

    def test_a_task_that_records_board_cycles_reports_their_ratio_in_aggregate(self):
        recording = task()
        recording.evaluation["certify"]["board_cycles"] = "record"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "hosts.json", "build_host": "build",
                                                        "board_host": "card", "card": {"bdf": "0000:04:00.1"}})
            job = {"id": "l1", "remote_id": "agent-slow-1", "containers": "tnca-test", "kind": "software"}
            sim = {"aggregate": {"n": 1, "passed": 1},
                   "full": {"points": [{"point": "secret.validation", "failure": None, "done_cycles": 42}]}}
            card = {"session": {"status": "ok"}, "measured": {"checks": {"fp8": {"exit": 1, "log_tail": ["x"]}}},
                    "points": {"secret.validation": {"status": "pass", "cycles": 40, "record_cycles": 42,
                                                     "fallbacks": [], "observer": {"done_cycles": 21}}}}
            base = {"points": {"secret.validation": {"status": "pass", "cycles": 80}}}
            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow.board, "measure", return_value=card),
                  mock.patch.object(fpga_slow.board, "baseline", return_value=base)):
                visible, full = fpga_slow.run(site, root, recording, None, root / "job", job, with_board=True,
                                              image={"path": "/x", "sha256": "c" * 64}, flow_seconds=60,
                                              board_seconds=60)
            self.assertEqual(visible["board"]["validation"]["speedup_geomean"], 2.0)
            self.assertNotIn("counted_as_in_simulation", visible["board"])
            self.assertEqual(visible["board"]["cycles_on_card"],
                             {"points": 1, "board_over_simulation_geomean": 0.5, "board_over_simulation_range": [0.5, 0.5]})
            self.assertEqual(visible["board"]["checks"], {"fp8": "fail"})
            self.assertEqual(visible["charged"], "board_run")
            self.assertNotIn("secret.validation", json.dumps(visible))
            self.assertNotIn("secret.final", json.dumps(visible))

    def test_a_snapshot_failing_in_simulation_is_refused_uncharged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "h", "build_host": "build"})
            job = {"id": "l1", "remote_id": "r", "containers": "c", "kind": "hardware"}
            sim = {"aggregate": {"n": 2, "passed": 1}, "failures": {"numeric-error": 1}}
            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow, "build") as build):
                visible, _ = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                           image=None, flow_seconds=60, board_seconds=60)
            build.assert_not_called()
            self.assertEqual((visible["status"], visible["charged"], job["outcome"]), ("refused", None, "refused"))

    def test_a_failed_flow_is_the_designs_or_the_infrastructures(self):
        self.assertEqual(fpga_slow.classify({"status": "PASS", "build": {"status": "PASS", "xclbin_sha256": "x"}}), "ok")
        self.assertEqual(fpga_slow.classify({"status": "BUILD_FAIL", "build": {"status": "TIMING_FAIL"}}), "design")
        self.assertEqual(fpga_slow.classify({"status": "BUILD_FAIL", "build": {
            "status": "BUILD_FAIL", "timing": {"metrics": {"wns_ns": -0.08}}}}), "design")
        self.assertEqual(fpga_slow.classify({"status": "BUILD_FAIL", "build": {
            "status": "BUILD_FAIL", "observed_clock_mhz": 91}}, 100), "design")
        self.assertEqual(fpga_slow.classify({"status": "BUILD_FAIL", "build": {
            "status": "BUILD_FAIL", "log_tail": ["ERROR: [Place 30-640] Place Check : This design requires more"]}}),
            "design")
        self.assertEqual(fpga_slow.classify({"status": "BUILD_FAIL", "build": {
            "status": "BUILD_FAIL", "log_tail": ["ERROR: license checkout failed"]}}), "infra")
        self.assertEqual(fpga_slow.classify({"status": "PIPELINE_FAIL"}), "infra")
        self.assertEqual(fpga_slow.classify({"status": "TIMEOUT", "build": {}}), "timeout")

    def test_a_launch_the_relay_could_not_count_rejects_the_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            observe = Path(tmp)
            (observe / "0.log").write_text("cmd op=1 seq=7 obs=10 done=10 status=0\nunfinished seq=6\nend launches=2\n")
            (observe / "1.log").write_text("cmd op=1 seq=9 obs=12 done=12 status=0\nend launches=1\n")
            rec = {"status": "pass", "time": {"device": 10}}
            got = board.scored(task(), ["a", "b"], {"records": {"a": rec, "b": dict(rec, time={"device": 12})}},
                               observe)
            self.assertEqual(got["a"]["status"], "unobserved-launch")
            self.assertEqual(got["b"]["status"], "pass")

    def test_a_container_written_link_is_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secret").write_text("key")
            (root / "check.log").symlink_to(root / "secret")
            self.assertIsNone(board.read_safe(root / "check.log"))
            (root / "plain").write_text("ok")
            self.assertEqual(board.read_safe(root / "plain"), b"ok")

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
