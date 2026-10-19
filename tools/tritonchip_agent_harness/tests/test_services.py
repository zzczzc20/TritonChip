"""The in-run services: budgets, one at a time, snapshots through the gate, and the socket the agent uses."""
from __future__ import annotations

import sys
import tempfile
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
                                   "fpga_slow": {"minutes": 360, "job_minutes": 180, "mode": "board"}},
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
                                                    "board_host": "card", "allow_candidate_host_code": False})
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
        self.assertEqual(self.broker.submit("score")["id"], "s2")
        self.wait_done("s2")
        self.assertEqual(self.broker.submit("score")["refused"], "no scoring requests left")
        self.assertEqual(self.seen, [("score", True), ("score", True)])

    def test_fpga_time_must_cover_a_whole_request(self):
        self.broker.fpga_left = 15 * 60
        self.assertIn("may take up to 20 min", self.broker.submit("fpga-quick")["refused"])
        self.broker.slow_left = 120 * 60
        self.assertIn("may take up to 180 min", self.broker.submit("fpga-slow")["refused"])

    def test_slow_request_is_agent_chosen_and_excludes_concurrent_fpga_work(self):
        first = self.broker.submit("fpga-slow")
        self.assertEqual((first["id"], first["fpga_slow_minutes_left"]), ("l1", 360.0))
        self.assertIn("another FPGA request", self.broker.submit("fpga-quick")["refused"])
        self.gate_open.set()
        self.wait_done("l1")
        self.assertEqual(self.broker.handle({"op": "result", "id": "l1"})["result"], {"ok": "slow"})

    def test_board_request_refuses_candidate_host_code_without_site_opt_in(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        reply = self.broker.submit("fpga-slow")
        self.assertIn("card host is not configured", reply["refused"])
        self.assertEqual(reply["fpga_slow_minutes_left"], 360.0)
        self.broker.site.fpga_pipeline["allow_candidate_host_code"] = True
        self.assertEqual(self.broker.submit("fpga-slow")["id"], "l1")
        self.gate_open.set()
        self.wait_done("l1")

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
