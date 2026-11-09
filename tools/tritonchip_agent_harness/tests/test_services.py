"""The in-run services: budgets, one at a time, snapshots through the gate, and the socket the agent uses."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import json
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import service_cli  # noqa: E402  (a script; imported as a module here)
from tritonchip_agent_harness.services import Broker  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402
from tritonchip_agent_harness.workspace import export, git  # noqa: E402

FILES = {"compiler/a.py": "a = 1\n", "rtl/core/x.v": "module x; endmodule\n", "tb/t.v": "module t; endmodule\n"}


def make_repo():
    root = Path(tempfile.mkdtemp())
    for path, text in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git("init", "-q", str(root))
    git("-C", str(root), "add", "-A")
    git("-C", str(root), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    return root, git("-C", str(root), "rev-parse", "HEAD").decode().strip()


class Services(unittest.TestCase):
    def setUp(self):
        self.repo, base = make_repo()
        self.ws = Path(tempfile.mkdtemp()) / "ws"
        export(self.repo, base, (), self.ws)
        self.task = Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"),
                         evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=60,
                         evaluation={"profile": "core-sim", "validation": ["p"], "final": ["q"]},
                         services={"score": {"requests": 2}, "fpga_quick": {"minutes": 50, "job_minutes": 20},
                                   "fpga_slow": {"mode": "board", "flows": 3, "board_runs": 2, "flow_minutes": 180,
                                                 "board_minutes": 60, "base_image_sha256": "c" * 64}},
                         path=self.repo, sha256="")
        self.gate_open = threading.Event()
        self.seen = []

        def slow(kind):
            def run(job, candidate, job_dir):
                self.seen.append((kind, candidate is not None))
                self.gate_open.wait(5)
                return {"ok": kind}, {"full": kind}
            return run

        site = types.SimpleNamespace(fpga_worker={"ssh": "worker", "workdir": "/w", "vivado_settings": "/s"},
                                     fpga_pipeline={"hosts": Path("/nonexistent"), "build_host": "build",
                                                    "board_host": "card", "allow_candidate_host_code": False,
                                                    "card": {"bdf": "0000:04:00.1",
                                                             "images": {"c" * 64: "/images/base.xclbin"}}})
        self.broker = Broker(site, self.repo, self.task, self.ws, Path(tempfile.mkdtemp()) / "run",
                             runners={"score": slow("score"), "fpga-quick": slow("fpga"),
                                      "fpga-slow": slow("slow")})

    def tearDown(self):
        self.gate_open.set()
        self.broker.stop()

    def wait_done(self, job_id):
        for _ in range(100):
            if self.broker.handle({"op": "result", "id": job_id})["state"] == "done":
                return
            time.sleep(0.05)
        self.fail(f"{job_id} did not finish")

    def test_scoring_requests_are_counted_and_one_runs_at_a_time(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        first = self.broker.submit("score")
        self.assertEqual((first["id"], first["score_requests_left"], first["snapshot"]["taken"]),
                         ("s1", 1, ["compiler/a.py"]))
        self.assertIn("one at a time", self.broker.submit("score")["refused"])
        self.gate_open.set()
        self.wait_done("s1")
        self.assertEqual(self.broker.handle({"op": "result", "id": "s1"})["result"], {"ok": "score"})
        (self.ws / "compiler/a.py").write_text("a = 3\n")
        self.assertEqual(self.broker.submit("score")["id"], "s2")
        self.wait_done("s2")
        self.assertEqual(self.broker.submit("score")["refused"], "no scoring requests left")
        self.assertEqual(self.seen, [("score", True), ("score", True)])

    def test_refused_submissions_are_in_the_record(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        self.broker.submit("score")
        self.broker.submit("score")                          # one at a time
        (self.ws / "tb/t.v").write_text("module t2; endmodule\n")
        self.broker.fpga_left = 0
        self.broker.submit("fpga-quick")                     # no FPGA time
        refusals = self.broker.record()["refusals"]
        self.assertEqual([r["service"] for r in refusals], ["score", "fpga-quick"])
        self.assertIn("one at a time", refusals[0]["why"])
        self.assertTrue(all(isinstance(r["at"], float) for r in refusals))
        self.gate_open.set()
        self.wait_done("s1")

    def test_a_failed_score_says_what_it_may_and_keeps_the_rest(self):
        r = {"aggregate": {"n": 3, "passed": 1, "speedup_geomean": 1.1, "fast_1": 0.33},
             "failures": {"numeric-error": 1, "measurement-rejected": 1}, "device_build_errors": None,
             "stage_errors": None, "full": {"points": [{"point": "op.hidden_point", "failure": "numeric-error"}]}}
        with mock.patch("tritonchip_agent_harness.services.evaluator.measure_validation", return_value=r):
            visible, full = self.broker._run_score({"containers": "c"}, None, Path(tempfile.mkdtemp()))
        self.assertEqual(visible, {"validation": r["aggregate"],
                                   "failures": {"numeric-error": 1, "measurement-rejected": 1}})
        self.assertIs(full, r)
        self.assertNotIn("op.hidden_point", json.dumps(visible))

    def test_fpga_time_must_cover_a_whole_request(self):
        self.broker.fpga_left = 15 * 60
        self.assertIn("may take up to 20 min", self.broker.submit("fpga-quick")["refused"])
        self.broker.flows_left = self.broker.board_runs_left = 0
        self.assertIn("no Vivado flows and no board runs left", self.broker.submit("fpga-slow")["refused"])

    def test_quick_tier_time_is_charged_when_the_worker_call_raises(self):
        # fpga_quick.run() can raise after the job ran, when its remote clean-up times out.
        job, ticks = {"remote_id": "r1"}, [100.0]
        clock = types.SimpleNamespace(time=time.time, sleep=time.sleep,
                                      monotonic=lambda: ticks.pop(0) if ticks else 160.0)
        with mock.patch("tritonchip_agent_harness.services.fpga_quick.run",
                        side_effect=subprocess.TimeoutExpired("ssh", 30)), \
                mock.patch("tritonchip_agent_harness.services.time", clock):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.broker._run_fpga(job, None, Path(tempfile.mkdtemp()))
        self.assertEqual(job["charged_seconds"], 60.0)
        self.assertEqual(self.broker.fpga_left, 50 * 60 - 60.0)

    def test_a_software_request_is_a_board_run_and_the_quick_tier_stays_open(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        first = self.broker.submit("fpga-slow")
        self.assertEqual((first["id"], first["fpga_slow_board_runs_left"], first["fpga_slow_flows_left"]), ("l1", 1, 3))
        self.assertEqual(self.broker.jobs["l1"]["base_image"]["path"], "/images/base.xclbin")
        self.assertIn("one at a time", self.broker.submit("fpga-slow")["refused"])
        self.assertEqual(self.broker.submit("fpga-quick")["id"], "f2")
        self.gate_open.set()
        self.wait_done("l1")
        self.assertEqual(self.broker.handle({"op": "result", "id": "l1"})["result"], {"ok": "slow"})

    def test_a_hardware_request_needs_the_quick_tier_on_the_same_hardware(self):
        (self.ws / "rtl/core/x.v").write_text("module x; wire w; endmodule\n")
        self.assertIn("fpga-quick result", self.broker.submit("fpga-slow")["refused"])
        self.assertEqual(self.broker.submit("fpga-quick")["id"], "f1")
        self.gate_open.set()
        self.wait_done("f1")
        self.broker.jobs["f1"]["full"] = {"wns_ns": -0.2, "failing_endpoints": 9}
        self.assertIn("fpga-quick result", self.broker.submit("fpga-slow")["refused"])
        self.broker.jobs["f1"]["full"] = {"wns_ns": 0.1, "failing_endpoints": 0}
        reply = self.broker.submit("fpga-slow")
        self.assertEqual((reply["id"], reply["fpga_slow_flows_left"]), ("l2", 2))
        self.wait_done("l2")

    def two_at_once(self, runners=True):
        """The same broker, with a task that runs up to two flows at once."""
        import dataclasses
        slow = {**self.task.services["fpga_slow"], "flows_at_once": 2}
        task = dataclasses.replace(self.task, services={**self.task.services, "fpga_slow": slow})
        broker = Broker(self.broker.site, self.repo, task, self.ws, Path(tempfile.mkdtemp()) / "run",
                        runners=self.broker.runners if runners else None)
        self.addCleanup(broker.stop)
        return broker

    def test_up_to_flows_at_once_hardware_requests_run_and_never_the_same_hardware_twice(self):
        broker = self.two_at_once()
        (self.ws / "rtl/core/x.v").write_text("module x; wire a; endmodule\n")
        with mock.patch.object(broker, "_timing_met", return_value=True):
            self.assertEqual(broker.submit("fpga-slow")["id"], "l1")
            # The same machine again while its flow runs would build one image twice.
            self.assertIn("already building exactly this hardware", broker.submit("fpga-slow")["refused"])
            (self.ws / "rtl/core/x.v").write_text("module x; wire b; endmodule\n")
            second = broker.submit("fpga-slow")
            self.assertEqual((second["id"], second["fpga_slow_flows_left"]), ("l2", 1))
            (self.ws / "rtl/core/x.v").write_text("module x; wire c; endmodule\n")
            self.assertIn("at most 2 at once", broker.submit("fpga-slow")["refused"])
        self.assertEqual(len(broker.shared.flows), 2)
        self.gate_open.set()
        for job in ("l1", "l2"):
            for _ in range(100):
                if broker.handle({"op": "result", "id": job})["state"] == "done":
                    break
                time.sleep(0.05)
        self.assertEqual((broker.shared.flows, broker.busy), (set(), {}))

    def test_a_flow_waiting_for_a_build_machine_when_the_agent_stops_is_cancelled_and_not_charged(self):
        broker = self.two_at_once(runners=False)
        waiting = threading.Event()

        def run(site, repo, task, candidate, job_dir, job, **kw):
            job["waiting_for_build_host"] = True
            waiting.set()
            for _ in range(100):
                if job.get("_cancel_requested"):
                    break
                time.sleep(0.05)
            job["outcome"] = "infra"
            return {"status": "not built"}, {}

        (self.ws / "rtl/core/x.v").write_text("module x; wire a; endmodule\n")
        with mock.patch.object(broker, "_timing_met", return_value=True), \
                mock.patch("tritonchip_agent_harness.services.fpga_slow.run", side_effect=run), \
                mock.patch("tritonchip_agent_harness.services.fpga_slow.cancel", return_value={}):
            self.assertEqual(broker.submit("fpga-slow")["id"], "l1")
            self.assertTrue(waiting.wait(5))
            self.assertTrue(broker._summary(broker.jobs["l1"])["waiting_for_a_build_machine"])
            broker.stop()
        (job,) = broker.record()["requests"]
        self.assertEqual(job["state"], "cancelled")
        self.assertIn("waiting for a free build machine", job["note"])
        self.assertNotIn("outlived_run", job)
        self.assertEqual((broker.flows_left, job["charged"]), (3, None))

    def test_a_running_flow_says_within_how_many_minutes_it_ends(self):
        now = time.time()
        job = {"id": "l1", "service": "fpga-slow", "state": "running", "submitted": now - 60,
               "kind": "hardware", "flow_started": now - 30 * 60}
        # 180 min flow cap and 60 min card cap, half an hour in: 210 left; never which machine.
        self.assertEqual(self.broker._summary(job)["ends_within_minutes"], 210)
        self.assertEqual(self.broker._summary({**job, "flow_finished": now})["ends_within_minutes"], 60)
        self.assertNotIn("ends_within_minutes", self.broker._summary({**job, "flow_started": None}))
        self.assertNotIn("build_host", json.dumps(self.broker._summary({**job, "build_host": "a"})))

    def test_a_snapshot_with_forbidden_rtl_is_refused_and_not_charged(self):
        (self.ws / "rtl/core/x.v").write_text('module x; initial $system("id"); endmodule\n')
        reply = self.broker.submit("score")
        self.assertEqual(reply["refused"], "the snapshot cannot be measured")
        self.assertEqual((reply["rejected"], reply["score_requests_left"]), (["rtl/core/x.v: $system"], 2))

    def test_a_request_running_when_the_agent_stops_is_cancelled_without_a_result(self):
        self.assertEqual(self.broker.submit("fpga-quick")["id"], "f1")
        threading.Timer(0.3, self.gate_open.set).start()
        with mock.patch("tritonchip_agent_harness.services.fpga_quick.cancel") as cancel:
            self.broker.stop()
        cancel.assert_called_once()
        (job,) = self.broker.record()["requests"]
        self.assertEqual((job["state"], job.get("result"), job["full"]), ("cancelled", None, {"full": "fpga"}))
        self.assertIsNone(self.broker.handle({"op": "result", "id": "f1"})["result"])

    def test_the_agent_reaches_it_through_the_socket(self):
        sock_dir = self.broker.start()
        service_cli.SOCKET = str(sock_dir / "service.sock")
        self.assertEqual(service_cli.call({"op": "budget"})["score_requests_left"], 2)
        self.assertEqual(service_cli.call({"op": "submit", "service": "fpga-quick"})["id"], "f1")
        self.assertIn("no service", service_cli.call({"op": "submit", "service": "board"})["error"])
        self.gate_open.set()
        self.wait_done("f1")
        self.assertEqual(service_cli.call({"op": "result", "id": "f1"})["result"], {"ok": "fpga"})
        record = self.broker.record()
        self.assertEqual([(r["id"], r["full"]) for r in record["requests"]], [("f1", {"full": "fpga"})])


if __name__ == "__main__":
    unittest.main()


class RunSimulations(unittest.TestCase):
    """D1/D2/D6 (2026-11-09): one simulation per snapshot for the run, joined while it runs, within the slots."""

    def test_a_finished_result_is_reused_whether_or_not_it_passed(self):
        from tritonchip_agent_harness.services import Simulations
        sims, calls = Simulations(None), []
        failed = {"aggregate": {"n": 2, "passed": 1}}
        r1, how1 = sims.measure("k", "0:s1", lambda: calls.append(1) or failed)
        r2, how2 = sims.measure("k", "1:l2", lambda: calls.append(2) or {})
        self.assertEqual((r1, r2, calls), (failed, failed, [1]))
        self.assertEqual((how1["ran"], how2), (True, {"cached_from": "0:s1"}))

    def test_a_running_simulation_is_joined_and_the_joiner_takes_no_slot(self):
        from tritonchip_agent_harness.services import Simulations
        slots = threading.BoundedSemaphore(1)
        sims, go, out = Simulations(slots), threading.Event(), {}

        def source():
            go.wait(5)
            return {"aggregate": {"n": 1, "passed": 1}}

        t = threading.Thread(target=lambda: out.setdefault("a", sims.measure("k", "0:s1", source)))
        t.start()
        time.sleep(0.2)
        j = threading.Thread(target=lambda: out.setdefault("b", sims.measure("k", "2:l4", lambda: {"no": 1})))
        j.start()
        time.sleep(0.2)
        go.set()
        t.join(5)
        j.join(5)
        self.assertEqual(out["b"], ({"aggregate": {"n": 1, "passed": 1}}, {"joined": "0:s1"}))
        self.assertTrue(slots.acquire(blocking=False))   # the one slot is free again: nobody else held it

    def test_a_source_that_raised_or_was_cancelled_is_not_reused(self):
        from tritonchip_agent_harness.services import Simulations
        sims = Simulations(None)

        def broken():
            raise RuntimeError("host")

        with self.assertRaises(RuntimeError):
            sims.measure("k", "0:s1", broken)
        r, how = sims.measure("k", "1:s2", lambda: {"fresh": 1})
        self.assertEqual((r, how["ran"]), ({"fresh": 1}, True))
        stop = {"now": False}

        def measured_then_cancelled():
            stop["now"] = True
            return {"torn": 1}

        sims.measure("c", "0:s3", measured_then_cancelled, cancelled=lambda: stop["now"])
        self.assertEqual(sims.measure("c", "1:s4", lambda: {"own": 1})[0], {"own": 1})

    def test_a_score_and_a_gate_of_one_snapshot_simulate_once_and_both_take_slots(self):
        repo, base = make_repo()
        ws = Path(tempfile.mkdtemp()) / "ws"
        export(repo, base, (), ws)
        (ws / "compiler/a.py").write_text("a = 9\n")
        t = Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"),
                 evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=60,
                 evaluation={"profile": "core-sim", "validation": ["p"], "final": ["q"]},
                 services={"score": {"requests": 3}}, path=repo, sha256="")
        site = types.SimpleNamespace(fpga_worker=None, fpga_pipeline=None, evaluations=1)
        b = Broker(site, repo, t, ws, Path(tempfile.mkdtemp()) / "run")
        calls = []
        r = {"aggregate": {"n": 1, "passed": 1, "speedup_geomean": 1.2, "fast_1": 1.0}}
        with mock.patch("tritonchip_agent_harness.services.evaluator.measure_validation",
                        side_effect=lambda *a: calls.append(a[-1]) or r):
            job_dir = Path(tempfile.mkdtemp())
            from tritonchip_agent_harness.workspace import gate as take
            take(repo, base, (), t.design, ws, out=job_dir / "candidate")
            cand = job_dir / "candidate"
            visible, _ = b._run_score({"id": "s1", "containers": "c-s1"}, cand, job_dir)
            gated = b._gate({"id": "l2", "containers": "c-l2"}, cand, job_dir)
        self.assertEqual(calls, ["c-s1"])
        self.assertIs(gated, r)
        self.assertEqual(visible["validation"], r["aggregate"])
        self.assertTrue(b.shared.evaluations.acquire(blocking=False))


class Dedupe(unittest.TestCase):
    """D4: a scoring request of a snapshot the run already scored is not charged and is no new commit."""

    def setUp(self):
        from tritonchip_agent_harness.services import Shared
        self.repo, base = make_repo()
        self.task = Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"),
                         evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=60,
                         evaluation={"profile": "core-sim", "validation": ["p"], "final": ["q"]},
                         services={"score": {"requests": 2}}, path=self.repo, sha256="")
        site = types.SimpleNamespace(fpga_worker=None, fpga_pipeline=None)
        self.shared = Shared(site, self.task)
        self.brokers, self.calls = [], []
        self.result = {"aggregate": {"n": 1, "passed": 1, "speedup_geomean": 1.1, "fast_1": 1.0}}
        for i in range(2):
            ws = Path(tempfile.mkdtemp()) / f"ws{i}"
            export(self.repo, base, (), ws)
            (ws / "compiler/a.py").write_text("a = 7\n")
            self.brokers.append(Broker(site, self.repo, self.task, ws, Path(tempfile.mkdtemp()) / f"run{i}",
                                       island=i, shared=self.shared))
        patcher = mock.patch("tritonchip_agent_harness.services.evaluator.measure_validation",
                             side_effect=lambda *a: self.calls.append(a[-1]) or self.result)
        patcher.start()
        self.addCleanup(patcher.stop)

    def wait(self, b, job_id):
        for _ in range(200):
            reply = b.handle({"op": "result", "id": job_id})
            if reply["state"] in ("done", "failed", "cancelled"):
                return reply
            time.sleep(0.05)
        self.fail(f"{job_id} did not finish")

    def test_an_identical_snapshot_is_answered_with_the_earlier_result_uncharged(self):
        first = self.brokers[0].submit("score", message="x")
        self.wait(self.brokers[0], first["id"])
        again = self.brokers[1].submit("score", message="same")
        self.assertEqual((again["same_as"], again["score_requests_left"]), ("0:s1", 2))
        reply = self.wait(self.brokers[1], again["id"])
        self.assertEqual(reply["result"]["cached_from"], "0:s1")
        self.assertEqual(reply["result"]["validation"], self.result["aggregate"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.brokers[1].jobs[again["id"]]["charged"])
        # A changed snapshot is a new request again, and charged.
        (self.brokers[1].workspace / "compiler/a.py").write_text("a = 8\n")
        third = self.brokers[1].submit("score", message="new")
        self.assertNotIn("same_as", third)
        self.assertEqual(third["score_requests_left"], 1)
        self.wait(self.brokers[1], third["id"])
        self.assertEqual(len(self.calls), 2)

    def test_a_result_that_failed_on_the_host_is_measured_again(self):
        self.result = {"aggregate": {"n": 1, "passed": 0, "speedup_geomean": None, "fast_1": 0.0},
                       "failures": {"timeout": 1}}
        first = self.brokers[0].submit("score", message="x")
        self.wait(self.brokers[0], first["id"])
        self.assertTrue(self.brokers[0].jobs["s1"]["simulation"]["not_kept"])
        again = self.brokers[1].submit("score", message="same")
        self.assertNotIn("same_as", again)          # charged and measured afresh
        self.wait(self.brokers[1], again["id"])
        self.assertEqual(len(self.calls), 2)


class SlotWait(unittest.TestCase):
    def test_a_cancel_is_heard_while_waiting_for_a_slot(self):
        from tritonchip_agent_harness.services import Simulations
        slots = threading.BoundedSemaphore(1)
        slots.acquire()                               # every slot taken by someone else
        sims, stop = Simulations(slots), {"now": False}
        threading.Timer(0.2, lambda: stop.update(now=True)).start()
        with mock.patch.object(slots, "acquire", side_effect=lambda timeout=None: time.sleep(0.05) or False):
            r, how = sims.measure("k", "0:l1", lambda: {"never": 1}, cancelled=lambda: stop["now"])
        self.assertEqual((r, how.get("cancelled_waiting")), (None, True))
        self.assertFalse(sims.kept("k"))


class IslandPools(unittest.TestCase):
    """2026-11-09: with roles, each island's own slow-tier counts and quick minutes; the run's quick results."""

    def test_each_island_has_its_own_counts_and_any_islands_quick_result_counts(self):
        from tritonchip_agent_harness.services import Shared
        repo, base = make_repo()
        roles = [{"name": "software", "authors": ["software"], "flows": 0, "board_runs": 4, "fpga_quick_minutes": 0},
                 {"name": "hardware", "authors": ["hardware"], "flows": 2, "board_runs": 3, "fpga_quick_minutes": 90},
                 {"name": "both", "authors": ["hardware", "isa", "software"], "flows": 2, "board_runs": 3,
                  "fpga_quick_minutes": 90}]
        t = Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"),
                 evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=60,
                 evaluation={"profile": "core-sim", "validation": ["p"], "final": ["q"]},
                 services={"score": {"requests": 2}, "fpga_quick": {"minutes": 45, "job_minutes": 30},
                           "fpga_slow": {"mode": "board", "flows": 4, "board_runs": 10, "flow_minutes": 60,
                                         "board_minutes": 60}},
                 path=repo, sha256="", islands={"count": 3, "topology": "all", "roles": roles})
        site = types.SimpleNamespace(fpga_worker=None, fpga_pipeline=None)
        shared = Shared(site, t)
        b = [Broker(site, repo, t, Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / f"r{i}", island=i,
                    shared=shared) for i in range(3)]
        self.assertEqual([(x.budget()["fpga_slow_flows_left"], x.budget()["fpga_slow_board_runs_left"],
                           x.budget()["fpga_minutes_left"]) for x in b], [(0, 4, 0.0), (2, 3, 90.0), (2, 3, 90.0)])
        b[1].flows_left -= 1
        self.assertEqual((b[1].flows_left, b[2].flows_left), (1, 2))
        self.assertFalse(b[1]._timing_met("H"))
        b[2].jobs["f1"] = {"id": "f1", "service": "fpga-quick", "hardware": "H", "state": "done",
                           "full": {"wns_ns": 0.4, "failing_endpoints": 0}}
        self.assertTrue(b[1]._timing_met("H"))       # island 2's quick result on exactly this hardware
        self.assertEqual(len(b[0]._all_jobs()), 1)
