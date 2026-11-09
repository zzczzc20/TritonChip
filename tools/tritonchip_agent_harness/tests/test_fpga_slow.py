"""The optional slow service keeps its pipeline and hidden-point evidence separate."""
from __future__ import annotations

import dataclasses
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
            self.assertIn("One slow request at a time; the quick tier stays open while it runs.", text)
            value["services"]["fpga_slow"]["flows_at_once"] = 2
            path.write_text(json.dumps(value))
            text = render(load_task(path), "offline")
            self.assertIn("Up to 2 slow requests at a time, each flow on a build machine of its own", text)
            self.assertNotIn("One slow request at a time", text)
            value["services"]["fpga_slow"]["flows_at_once"] = 0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(TaskError, "flows_at_once must be an integer >= 1"):
                load_task(path)

    def test_manifest_exposes_only_validation_on_the_board(self):
        board = fpga_slow.manifest(task(), "l1", "c" * 40, board=True)
        self.assertEqual(board["board"]["archbench"]["points"], ["secret.validation"])
        self.assertNotIn("secret.final", json.dumps(board))
        build = fpga_slow.manifest(task(), "l1", "c" * 40, board=False)
        self.assertNotIn("board", build)

    def test_a_board_validation_set_is_what_the_card_measures(self):
        split = task()
        split.evaluation.update(board_validation=["secret.board"], board_final=["secret.board_final"])
        value = fpga_slow.manifest(split, "l1", "c" * 40, board=True)
        self.assertEqual(value["board"]["archbench"]["points"], ["secret.board"])
        for hidden in ("secret.validation", "secret.final", "secret.board_final"):
            self.assertNotIn(hidden, json.dumps(value["board"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "hosts.json", "build_host": "build",
                                                        "board_host": "card", "card": {"bdf": "0000:04:00.1"}})
            job = {"id": "l1", "remote_id": "agent-slow-1", "containers": "tnca-test", "kind": "software"}
            sim = {"aggregate": {"n": 1, "passed": 1},
                   "full": {"points": [{"point": "secret.validation", "failure": None, "done_cycles": 42}]}}
            card = {"session": {"status": "ok"}, "measured": {},
                    "points": {"secret.board": {"status": "pass", "cycles": 25, "record_cycles": 25,
                                                "fallbacks": [], "observer": {"done_cycles": 25}}}}
            base = {"points": {"secret.board": {"status": "pass", "cycles": 100}}}
            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow.board, "measure", return_value=card) as measure,
                  mock.patch.object(fpga_slow.board, "baseline", return_value=base)):
                visible, _ = fpga_slow.run(site, root, split, None, root / "job", job, with_board=True,
                                           image={"path": "/x", "sha256": "c" * 64}, flow_seconds=60,
                                           board_seconds=60)
            self.assertEqual(measure.call_args.args[5], ["secret.board"])
            self.assertEqual(visible["board"]["validation"]["speedup_geomean"], 4.0)
            # Simulation never ran the card's own points: nothing to compare the card's cycles with.
            self.assertNotIn("counted_as_in_simulation", visible["board"])
            self.assertNotIn("cycles_on_card", visible["board"])
            self.assertNotIn("secret.", json.dumps(visible))

    def test_point_sets_are_disjoint_lists_and_the_board_set_needs_the_card(self):
        source = HERE / "tasks" / "r39-v5-main-pilot"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instruction.md").write_text("Improve the design.\n")
            value = json.loads((source / "task.json").read_text())
            value["evaluation"]["platform"] = {"commit": "b" * 40, "paths": []}
            path = root / "task.json"

            def load(**ev):
                v = json.loads(json.dumps(value))
                v["evaluation"].update(ev)
                path.write_text(json.dumps(v))
                return load_task(path)

            v0 = value["evaluation"]["validation"][0]
            with self.assertRaisesRegex(TaskError, "board_validation needs services.fpga_slow in board mode"):
                load(board_validation=["op.x/large/fp16-a"])
            value["services"]["fpga_slow"] = {"mode": "board", "flows": 1, "board_runs": 2,
                                                "flow_minutes": 480, "board_minutes": 120}
            with self.assertRaisesRegex(TaskError, "in both evaluation.validation and evaluation.board_validation"):
                load(board_validation=[v0])
            with self.assertRaisesRegex(TaskError, "in both evaluation.board_validation and evaluation.board_final"):
                load(board_validation=["op.x/large/fp16-a"], board_final=["op.x/large/fp16-a"])
            with self.assertRaisesRegex(TaskError, "board_final must be a list of distinct point ids"):
                load(board_final=["op.x/large/fp16-a", "op.x/large/fp16-a"])
            with self.assertRaisesRegex(TaskError, "board_validation is empty"):
                load(board_validation=[])
            t = load(board_validation=["op.x/large/fp16-a"], board_final=["op.x/hbm/fp16-b"])
            self.assertEqual(fpga_slow.board_points(t), ["op.x/large/fp16-a"])
            self.assertIn("the hidden board validation set's aggregate", render(t, "offline"))
            self.assertNotIn("op.x/", render(t, "offline"))
            self.assertEqual(fpga_slow.board_points(load()), value["evaluation"]["validation"])

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
        # Software (no flow): the gate first, and nothing else when it does not pass.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "h", "build_host": "build"})
            job = {"id": "l1", "remote_id": "r", "containers": "c", "kind": "software"}
            sim = {"aggregate": {"n": 2, "passed": 1}, "failures": {"numeric-error": 1}}
            with (mock.patch.object(fpga_slow.evaluator, "measure_validation", return_value=sim),
                  mock.patch.object(fpga_slow, "build") as build, mock.patch.object(fpga_slow, "on_card") as card):
                visible, _ = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                           image={"path": "/i"}, flow_seconds=60, board_seconds=60)
            build.assert_not_called()
            card.assert_not_called()
            self.assertEqual((visible["status"], visible["charged"], job["outcome"]), ("refused", None, "refused"))

    @staticmethod
    def _stoppable_build(started, stopped):
        """A flow that runs until it is cancelled (or 5 s), as build() returns for a stopped flow."""
        def build(site, repo, task_, candidate, job_dir, job, seconds):
            started.set()
            stopped.wait(5)
            return {"outcome": "infra", "pipeline": {"status": "TIMEOUT"}, "attempts": [], "tree_commit": "t",
                    "manifest": {}, "routed": {}, "visible": {}, "run_id": "r", "build_host": "b"}
        return build

    @staticmethod
    def _cancel(stopped):
        def cancel(job, cfg):
            job["_cancel_requested"] = True
            stopped.set()
            return {"remote": "not-started"}
        return cancel

    def test_a_hardware_gate_runs_beside_its_flow_and_stops_it_when_it_fails(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "h", "build_host": "build"})
            job = {"id": "l1", "remote_id": "r", "containers": "c", "kind": "hardware"}
            started, stopped = threading.Event(), threading.Event()
            sim = {"aggregate": {"n": 2, "passed": 1}, "failures": {"numeric-error": 1}}

            def gate():
                # The flow is already running when the simulation finishes (D3).
                self.assertTrue(started.wait(5))
                return sim

            with (mock.patch.object(fpga_slow, "build", side_effect=self._stoppable_build(started, stopped)),
                  mock.patch.object(fpga_slow, "cancel", side_effect=self._cancel(stopped)) as cancel):
                visible, full = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                              image=None, flow_seconds=60, board_seconds=60, gate=gate)
            cancel.assert_called_once()
            self.assertEqual((visible["status"], visible["charged"], job["outcome"]), ("refused", None, "refused"))
            self.assertIn("flow was stopped", visible["why"])
            self.assertEqual(visible["failures"], {"numeric-error": 1})
            self.assertIs(full["simulation"], sim)
            self.assertTrue(job["gate_stopped_flow"])

    def test_a_gate_that_fails_on_the_infrastructure_stops_the_flow_uncharged(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "h", "build_host": "build"})
            job = {"id": "l1", "remote_id": "r", "containers": "c", "kind": "hardware"}
            started, stopped = threading.Event(), threading.Event()

            def gate():
                raise RuntimeError("the simulator host went away")

            with (mock.patch.object(fpga_slow, "build", side_effect=self._stoppable_build(started, stopped)),
                  mock.patch.object(fpga_slow, "cancel", side_effect=self._cancel(stopped))):
                visible, full = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                              image=None, flow_seconds=60, board_seconds=60, gate=gate)
            self.assertEqual((visible["status"], visible["charged"], job["outcome"]), ("not built", None, "infra"))
            self.assertIn("RuntimeError", full["gate_error"])
            self.assertNotIn("simulator host", json.dumps(visible))

    def test_a_flow_that_ends_first_waits_for_its_gate(self):
        import threading
        import time as _time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = types.SimpleNamespace(fpga_pipeline={"hosts": root / "h", "build_host": "build"})
            job = {"id": "l1", "remote_id": "r", "containers": "c", "kind": "hardware"}
            sim = {"aggregate": {"n": 2, "passed": 2}}
            ended = []

            def gate():
                _time.sleep(0.3)
                ended.append("gate")
                return sim

            def build(site_, repo, task_, candidate, job_dir, job_, seconds):
                ended.append("flow")
                return {"outcome": "design", "pipeline": {"status": "FAIL"}, "attempts": [], "tree_commit": "t",
                        "manifest": {}, "routed": {}, "visible": {"status": "FAIL"}, "run_id": "r", "build_host": "b"}

            with (mock.patch.object(fpga_slow, "build", side_effect=build),
                  mock.patch.object(fpga_slow, "cancel") as cancel):
                visible, full = fpga_slow.run(site, root, task(), None, root / "job", job, with_board=True,
                                              image=None, flow_seconds=60, board_seconds=60, gate=gate)
            cancel.assert_not_called()
            self.assertEqual(ended, ["flow", "gate"])
            self.assertEqual((visible["status"], visible["charged"], job["outcome"]), ("infeasible", "flow", "design"))
            self.assertIs(full["simulation"], sim)

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
               "containers": "tnca-test", "build_host": "second"}
        cfg = {"hosts": Path("/hosts.json"), "build_host": "build", "build_hosts": ["build", "second"],
               "board_host": "card"}
        with (mock.patch.object(fpga_slow.provider, "docker", return_value=types.SimpleNamespace(stdout="")),
              mock.patch.object(hosts, "load_hosts", return_value={}),
              mock.patch.object(hosts, "host_for", return_value=remote_host) as choose,
              mock.patch.object(pipeline, "remote", return_value=types.SimpleNamespace(
                  returncode=0, stdout=b"terminated")) as remote):
            result = fpga_slow.cancel(job, cfg)
            # A request that never got a machine ran nothing anywhere: nothing to reconcile.
            waiting = fpga_slow.cancel({**job, "build_host": None}, cfg)
        self.assertEqual(choose.call_count, 1)
        # Reconciled on the machine its flow ran on, not the site's first.
        self.assertEqual(choose.call_args.args[1:], ("second", "build"))
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(result["build"]["exit"], 0)
        self.assertEqual(waiting, {"remote": "not-started"})


def slow_task(**slow) -> Task:
    return dataclasses.replace(task(), wall_seconds=1440 * 60, services={"fpga_slow": {
        "mode": "board", "flows": 3, "board_runs": 2, "flow_minutes": 480, "board_minutes": 240, **slow}})


class Placement(unittest.TestCase):
    """Which build machine a flow runs on: one flow on each, the first free in the site's order."""

    def setUp(self):
        fpga_slow._HELD.clear()
        self.addCleanup(fpga_slow._HELD.clear)
        self.site = types.SimpleNamespace(fpga_pipeline={
            "hosts": Path("/hosts.json"), "build_host": "a", "build_hosts": ["a", "b"],
            "shared_slots": {"a": ["/home/owner/.slot.0", "/home/owner/.slot.1"]}})
        self.hosts = {n: types.SimpleNamespace(name=n, state_dir=f"/state/{n}") for n in ("a", "b")}
        self.busy = {}
        self.holds = []
        from tritonchip_fpga_harness import hosts
        for p in (mock.patch.object(hosts, "load_hosts", return_value=self.hosts),
                  mock.patch.object(hosts, "host_for", side_effect=lambda h, n, role: h[n]),
                  mock.patch.object(fpga_slow, "busy", side_effect=self.probe),
                  mock.patch.object(fpga_slow, "_hold_slot", side_effect=self.hold),
                  mock.patch.object(fpga_slow, "POLL_SECONDS", 0)):
            p.start()
            self.addCleanup(p.stop)

    def probe(self, cfg, host, processes=False):
        why = self.busy.get(host.name, [])
        return why() if callable(why) else list(why)

    def hold(self, cfg, host):
        self.holds.append(host.name)
        if host.name in cfg.get("shared_slots", {}):
            return ("connection", True) if host.name not in self.busy.get("slots-taken", ()) else (None, False)
        return None, True

    def test_one_flow_on_each_machine_in_the_sites_order(self):
        first, second, third = {"remote_id": "r1"}, {"remote_id": "r2"}, {"remote_id": "r3"}
        self.assertEqual(fpga_slow.place(self.site, first, 60), ("a", "connection"))
        self.assertEqual(fpga_slow.place(self.site, second, 60), ("b", None))
        # Both held by this process: a third waits, and ends when its cap does.
        self.assertEqual(fpga_slow.place(self.site, third, 0), (None, None))
        self.assertEqual(third["build_hosts_busy"], {"a": ["another flow of this run"],
                                                     "b": ["another flow of this run"]})
        fpga_slow._release("r1", "a")
        self.assertEqual(fpga_slow.place(self.site, third, 60)[0], "a")
        self.assertEqual((first["build_host"], second["build_host"], third["build_host"]), ("a", "b", "a"))
        self.assertNotIn("waiting_for_build_host", third)

    def test_a_machine_busy_from_outside_is_skipped_and_the_wait_recorded(self):
        self.busy["a"] = ["full.lock is held"]
        job = {"remote_id": "r1"}
        self.assertEqual(fpga_slow.place(self.site, job, 60)[0], "b")
        # Every machine busy: the request waits until one is free; the wait is its outside wait.
        rounds = iter([["full.lock is held"], ["full.lock is held"], []])
        self.busy.update(a=lambda: next(rounds, []), b=[".slot.0 is held"])
        later = {"remote_id": "r2"}
        self.assertEqual(fpga_slow.place(self.site, later, 60)[0], "a")
        self.assertIn("outside_wait_seconds", later)
        self.assertNotIn("waiting_for_build_host", later)

    def test_slots_taken_since_the_look_send_the_flow_to_the_next_machine(self):
        self.busy["slots-taken"] = ("a",)
        job = {"remote_id": "r1"}
        self.assertEqual(fpga_slow.place(self.site, job, 60), ("b", None))
        self.assertEqual(self.holds, ["a", "b"])
        self.assertNotIn("a", fpga_slow._HELD)

    def test_a_cancel_ends_the_wait(self):
        self.busy.update(a=["full.lock is held"], b=["full.lock is held"])
        job = {"remote_id": "r1", "_cancel_requested": True}
        self.assertEqual(fpga_slow.place(self.site, job, 3600), (None, None))
        self.assertEqual(fpga_slow._HELD, {})

    def test_the_run_must_have_room_for_its_flows(self):
        fit = fpga_slow.budget_fit(slow_task(flows_at_once=2))
        self.assertEqual((fit["rounds"], fit["minutes_needed"], fit["fits"]), (2, 1200.0, True))
        self.assertFalse(fpga_slow.budget_fit(slow_task())["fits"])   # 3 x 480 + 240 > 1440

    def test_preflight_needs_a_machine_per_flow_and_every_machine_idle(self):
        with self.assertRaisesRegex(fpga_slow.SlowError, "3 flows at once and the site has 2"):
            fpga_slow.preflight(self.site, slow_task(flows_at_once=3))
        self.busy["b"] = ["Vivado is running there"]
        with self.assertRaisesRegex(fpga_slow.SlowError, "b: Vivado is running there"):
            fpga_slow.preflight(self.site, slow_task(flows_at_once=2))
        self.busy.clear()
        got = fpga_slow.preflight(self.site, slow_task(flows_at_once=2))
        self.assertEqual((got["build_hosts"], got["busy"], got["budget"]["fits"]), (["a", "b"], {}, True))

    def test_a_retry_after_an_infrastructure_failure_takes_a_machine_again(self):
        places = iter([("a", "connection"), ("b", None)])
        records = iter([{"status": "BUILD_FAIL", "build": {"status": "BUILD_FAIL", "preflight_failed": True}},
                        {"status": "PASS", "run_id": "tnca-x-r1",
                         "build": {"status": "PASS", "xclbin_sha256": "d" * 64}}])

        class Proc:
            def __init__(self, argv, **kw):
                out = Path(argv[argv.index("--output") + 1])
                out.write_text(json.dumps(next(records)))
                self.host = argv[argv.index("--build-host") + 1]

            def wait(self, timeout=None):
                return 0

        seen, avoided = [], []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fpga_slow.certify, "build_tree", return_value="c" * 40), \
                mock.patch.object(fpga_slow, "place", side_effect=lambda s, j, sec, avoid=(): (
                    avoided.append(list(avoid)) or j.__setitem__("build_host", (p := next(places))[0]) or p)), \
                mock.patch.object(fpga_slow, "routed_timing", return_value={}), \
                mock.patch.object(fpga_slow, "_drop_slot", side_effect=lambda h: "held" if h else None), \
                mock.patch.object(fpga_slow.subprocess, "Popen", side_effect=lambda argv, **kw: (
                    seen.append(argv[argv.index("--build-host") + 1]) or Proc(argv, **kw))):
            job = {"id": "l1", "remote_id": "tnca-x"}
            got = fpga_slow.build(self.site, Path(tmp), task(), None, Path(tmp), job, 600)
        self.assertEqual(seen, ["a", "b"])
        # The retry never goes back to the machine that failed.
        self.assertEqual(avoided, [[], ["a"]])
        self.assertEqual([(a["build_host"], a["outcome"]) for a in got["attempts"]], [("a", "infra"), ("b", "ok")])
        self.assertEqual(got["attempts"][0]["shared_slot"], "held")
        self.assertEqual((got["outcome"], got["build_host"]), ("ok", "b"))

    def test_a_retry_never_takes_a_machine_that_failed_the_flow(self):
        self.assertEqual(fpga_slow.place(self.site, {"remote_id": "r1"}, 60, avoid=["a"]), ("b", None))
        self.assertEqual(fpga_slow.place(self.site, {"remote_id": "r2"}, 60, avoid=["a", "b"]), (None, None))

    def test_when_every_machine_failed_the_flow_is_not_tried_again_and_keeps_its_routed_timing(self):
        # RSCL, 2026-11-08: routed and met timing, then could not write the part's bitstream.
        places = iter([("a", None), ("b", None)])
        failed = {"status": "BUILD_FAIL", "build": {"status": "BUILD_FAIL",
                                                   "error": "Bitstream generation is not supported for part"}}
        timing = {"a": {"wns_ns": 0.007, "tns_ns": 0.0, "whs_ns": 0.01, "ths_ns": 0.0, "failing_endpoints": 0,
                        "clean": True}, "b": {}}

        class Proc:
            def __init__(self, argv, **kw):
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(failed))

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fpga_slow.certify, "build_tree", return_value="c" * 40), \
                mock.patch.object(fpga_slow, "place", side_effect=lambda s, j, sec, avoid=(): (
                    j.__setitem__("build_host", (p := next(places))[0]) or p)), \
                mock.patch.object(fpga_slow, "routed_timing", side_effect=lambda s, run_id, host: timing[host]), \
                mock.patch.object(fpga_slow, "_drop_slot", return_value=None), \
                mock.patch.object(fpga_slow.subprocess, "Popen", side_effect=Proc):
            got = fpga_slow.build(self.site, Path(tmp), task(), None, Path(tmp), {"id": "l1", "remote_id": "x"}, 600)
        self.assertEqual([(a.get("build_host"), a["outcome"]) for a in got["attempts"]],
                         [("a", "infra"), ("b", "infra"), (None, "infra")])
        self.assertEqual(got["attempts"][2]["status"], "NOT_RETRIED")
        self.assertEqual(got["outcome"], "infra")
        self.assertEqual(got["attempts"][0]["routed_timing"]["wns_ns"], 0.007)
        self.assertTrue(got["visible"]["routed_timing"]["clean"])
        self.assertEqual(got["visible"]["routed_timing"]["wns_ns"], 0.007)

    def test_the_routed_design_timing_summary_is_read_as_the_fpga_harness_reads_it(self):
        cols = ("    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)"
                "  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints"
                "  TPWS Total Endpoints  \n    -------      -------  ---------------------  -------------------"
                "      -------      -------  ---------------------  -------------------     --------     --------"
                "  ----------------------  --------------------  \n")
        report = ("| Design Timing Summary\n\n" + cols +
                  "     -0.176      -93.900                    819               584460        0.010        0.000"
                  "                      0               584428        0.000        0.000                       0"
                  "                212345  \n")
        got = fpga_slow.design_timing(report)
        self.assertEqual((got["wns_ns"], got["tns_ns"], got["failing_endpoints"], got["clean"]),
                         (-0.176, -93.9, 819, False))
        self.assertEqual(fpga_slow.design_timing("no table here"), {})


if __name__ == "__main__":
    unittest.main()


class BoardLaunchDeadline(unittest.TestCase):
    """R41-F2's base (2026-11-09): six board points ended as runtime faults at archbench.run's default
    deadline of 2e8 cycles; a board measurement gives each launch the pilot's deadline instead."""

    def test_board_points_get_the_pilots_deadline(self) -> None:
        task = load_task(HERE / "tasks" / "r41-dp-formal")
        env = board._env(task, 3, 600)
        self.assertIn(f"TNCA_DEADLINE={board.DEADLINE_CYCLES}", env.split())
        self.assertGreaterEqual(board.DEADLINE_CYCLES, 60 * 100_000_000)   # a minute of device time at 100 MHz
