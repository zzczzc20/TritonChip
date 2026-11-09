"""The board as the score (decided 2026-11-10): family scores, the task's keys, `score` on the card, `flow`."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import families, fpga_slow, task as task_  # noqa: E402
from tritonchip_agent_harness.services import Broker, Shared  # noqa: E402
from tritonchip_agent_harness.task import Task, TaskError, board_points, render  # noqa: E402
from tritonchip_agent_harness.workspace import export  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_services import make_repo  # noqa: E402

LLAMA = {"llama": {"prefixes": ["model.llama_decode/", "model.llama_prefill/"], "weight": 2}}


class FamilyScores(unittest.TestCase):
    def test_a_family_is_its_operator_unless_the_task_names_one(self):
        self.assertEqual(families.family_of("op.matmul/small/fp16-m1"), "matmul")
        self.assertEqual(families.family_of("model.llama_decode/large/x", LLAMA), "llama")
        self.assertEqual(families.family_of("model.llama_prefill/hbm/x", LLAMA), "llama")
        self.assertEqual(families.family_of("model.llama_decode/large/x"), "llama_decode")

    def test_weighted_geometric_mean_of_family_geomeans_and_a_failure_is_half(self):
        rows = [{"point": "op.add/a", "speedup": 4.0, "failure": None},
                {"point": "op.add/b", "speedup": 1.0, "failure": None},          # add: 2.0
                {"point": "model.llama_decode/a", "speedup": 1.0, "failure": None},
                {"point": "model.llama_prefill/a", "speedup": 3.0, "failure": "timeout"}]   # llama: sqrt(0.5)
        got = families.scores(rows, LLAMA)
        self.assertAlmostEqual(got["families"]["add"], 2.0, places=4)
        self.assertAlmostEqual(got["families"]["llama"], 0.5 ** 0.5, places=4)
        # (2 ** 1 * (0.5 ** 0.5) ** 2) ** (1 / 3)
        self.assertAlmostEqual(got["score"], (2.0 * 0.5) ** (1 / 3), places=6)

    def test_a_family_counts_once_however_many_points_it_has(self):
        many = [{"point": f"op.add/{i}", "speedup": 2.0, "failure": None} for i in range(10)]
        one = [{"point": "op.mul/0", "speedup": 0.5, "failure": None}]
        self.assertAlmostEqual(families.scores(many + one)["score"], 1.0, places=6)


def board_task(repo, base, **ev):
    evaluation = {"profile": "core-sim", "validation": ["op.add/small/a"], "final": ["op.add/medium/held"],
                  "board_validation": ["op.matmul/large/b", "model.llama_decode/large/c"],
                  "board_final": ["op.matmul/hbm/held"], "families": LLAMA,
                  "gate_points": ["op.add/tiny/g"], "platform": {"commit": "p" * 40, "paths": []}}
    evaluation.update(ev)
    return Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"), evaluator_owned=(),
                hide=(), instruction="", reference=None, wall_seconds=60, evaluation=evaluation,
                services={"score": {"requests": 3, "on": "board"}, "fpga_quick": {"minutes": 50, "job_minutes": 20},
                          "fpga_slow": {"mode": "board", "flows": 2, "board_runs": 0, "flow_minutes": 180,
                                        "board_minutes": 60, "base_image_sha256": "c" * 64,
                                        "board_faster_than": 1.01}},
                path=repo, sha256="")


class TaskKeys(unittest.TestCase):
    def setUp(self):
        self.repo, self.base = make_repo()

    def check(self, services, ev):
        task_._check_board_score(Path("task.json"), services, ev)

    def test_board_points_are_the_small_set_then_the_cards_own(self):
        t = board_task(self.repo, self.base)
        self.assertEqual(board_points(t), ["op.add/small/a", "op.matmul/large/b", "model.llama_decode/large/c"])

    def test_the_board_score_needs_the_card_its_set_and_a_gate(self):
        t = board_task(self.repo, self.base)
        self.check(t.services, t.evaluation)
        with self.assertRaises(TaskError):
            self.check({**t.services, "fpga_slow": {**t.services["fpga_slow"], "mode": "build"}}, t.evaluation)
        with self.assertRaises(TaskError):
            self.check(t.services, {k: v for k, v in t.evaluation.items() if k != "gate_points"})
        with self.assertRaises(TaskError):
            self.check(t.services, {k: v for k, v in t.evaluation.items() if k != "board_validation"})
        with self.assertRaises(TaskError):
            self.check({"score": {"on": "card"}}, t.evaluation)

    def test_no_held_out_point_gates_or_checks_the_card(self):
        t = board_task(self.repo, self.base)
        with self.assertRaises(TaskError):
            self.check(t.services, {**t.evaluation, "gate_points": ["op.add/medium/held"]})
        with self.assertRaises(TaskError):
            self.check(t.services, {**t.evaluation, "health_point": "op.matmul/hbm/held"})
        with self.assertRaises(TaskError):
            self.check(t.services, {**t.evaluation, "families": {"x": {"prefixes": [], "weight": 1}}})
        with self.assertRaises(TaskError):
            self.check(t.services, {**t.evaluation, "families": {"x": {"prefixes": ["a/"], "weight": 0}}})

    def test_the_text_says_the_card_scores_and_flow_builds(self):
        t = board_task(self.repo, self.base)
        text = render(t, "offline")
        self.assertIn("the hidden board validation set on the real card", text)
        self.assertIn("`tnca-service submit flow`", text)
        self.assertIn("no simulated scoring service", text)
        self.assertIn("of weight 2", text)
        self.assertNotIn("submit fpga-slow", text)
        for held in ("op.add/medium/held", "op.matmul/hbm/held"):
            self.assertNotIn(held, text)


class BoardBroker(unittest.TestCase):
    def setUp(self):
        self.repo, base = make_repo()
        self.ws = Path(tempfile.mkdtemp()) / "ws"
        export(self.repo, base, (), self.ws)
        self.task = board_task(self.repo, base)
        self.site = types.SimpleNamespace(
            fpga_worker={"ssh": "worker"}, state_dir=Path(tempfile.mkdtemp()),
            fpga_pipeline={"hosts": Path("/nonexistent"), "build_host": "build", "board_host": "card",
                           "card": {"bdf": "0000:04:00.1", "images": {"c" * 64: "/images/base.xclbin"}}})
        self.broker = Broker(self.site, self.repo, self.task, self.ws, Path(tempfile.mkdtemp()) / "run")
        changed = mock.patch("tritonchip_agent_harness.services.evaluator.hardware_changed",
                             side_effect=lambda repo, task, cand: {"changed": bool(
                                 cand and (cand / "files" / "rtl").exists())})
        changed.start()
        self.addCleanup(changed.stop)
        self.calls = []

    def tearDown(self):
        self.broker.stop()

    def card(self, outcome="ok", score=1.25, health=None):
        def on_card(site, repo, task, candidate, image, sim, job_dir, job, seconds, base_xclbin=None, roofline=None,
                    health_=None, **kw):
            self.calls.append(image)
            visible = {"status": "measured", "validation": {"n": 3, "passed": 3, "speedup_geomean": 1.2,
                                                            "fast_1": 0.67, "score": score},
                       "families": {"add": 1.0, "matmul": 1.5, "llama": 1.2}}
            if outcome == "infra":
                visible = {"status": "not measured", "why": "x"}
            return {"outcome": outcome, "record": {"points": {}}, "visible": visible,
                    "score": {"points": []}, "health": health}
        return mock.patch("tritonchip_agent_harness.services.fpga_slow.on_card", side_effect=on_card)

    def wait(self, job_id):
        for _ in range(200):
            v = self.broker.handle({"op": "result", "id": job_id})
            if v["state"] in ("done", "failed", "cancelled"):
                return v
            time.sleep(0.02)
        self.fail(f"{job_id} did not finish")

    def test_a_software_snapshot_goes_straight_to_the_card_on_the_base_image(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        with self.card():
            got = self.broker.submit("score")
            self.assertEqual((got["id"], got["score_requests_left"]), ("s1", 2))
            r = self.wait("s1")
        self.assertEqual(r["result"]["validation"]["score"], 1.25)
        self.assertEqual(r["result"]["families"]["llama"], 1.2)
        self.assertEqual(self.calls[0]["kind"], "base")
        self.assertNotIn("fpga_slow_board_runs_left", got)

    def test_hardware_with_no_image_is_refused_uncharged_and_runs_once_built(self):
        (self.ws / "rtl/core/x.v").write_text("module y; endmodule\n")
        got = self.broker.submit("score")
        self.assertIn("submit flow", got["refused"])
        self.assertEqual(got["score_requests_left"], 3)
        hw = fpga_slow.hardware_digest(self.broker.dir / "s1" / "candidate")
        self.broker.shared.images[hw] = {"build_run": "r", "sha256": "d" * 64, "xclbin": "x.xclbin",
                                         "card": {"path": "/b/d.xclbin", "sha256": "d" * 64, "kind": "built"}}
        with self.card():
            got = self.broker.submit("score")
            self.assertNotIn("refused", got)
            self.wait(got["id"])
        self.assertEqual(self.calls[0]["kind"], "built")

    def test_an_infrastructure_failure_is_not_charged(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        with self.card(outcome="infra"):
            got = self.broker.submit("score")
            r = self.wait(got["id"])
        self.assertIsNone(r["result"]["charged"])
        self.assertEqual(self.broker.score_left, 3)

    def test_a_failed_health_check_closes_the_card_for_a_while(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        with self.card(outcome="infra", health={"ok": False, "why": "the known-answer point did not pass"}):
            self.wait(self.broker.submit("score")["id"])
        (self.ws / "compiler/a.py").write_text("a = 3\n")
        got = self.broker.submit("score")
        self.assertIn("health check", got["refused"])
        self.assertEqual(self.broker.score_left, 3)
        self.broker.shared.card_unhealthy["at"] -= Broker.UNHEALTHY_SECONDS + 1
        with self.card():
            self.assertNotIn("refused", self.broker.submit("score"))

    def test_the_same_snapshot_is_measured_once_and_charged_once(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        with self.card():
            self.wait(self.broker.submit("score")["id"])
            again = self.broker.submit("score")
            self.assertEqual(again["same_as"], "s1")
            r = self.wait(again["id"])
        self.assertEqual(r["result"]["validation"]["score"], 1.25)
        self.assertEqual(r["result"]["cached_from"], "s1")
        self.assertEqual((len(self.calls), self.broker.score_left), (1, 2))

    def test_the_slow_service_is_flow_and_score_here(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        self.assertIn("`flow` builds hardware", self.broker.submit("fpga-slow")["refused"])

    def test_a_flow_needs_a_changed_machine_and_its_quick_timing(self):
        (self.ws / "compiler/a.py").write_text("a = 2\n")
        self.assertIn("nothing to build", self.broker.submit("flow")["refused"])
        (self.ws / "rtl/core/x.v").write_text("module y; endmodule\n")
        self.assertIn("fpga-quick", self.broker.submit("flow")["refused"])
        self.assertEqual(self.broker.flows_left, 2)

    def test_a_flow_that_builds_registers_the_image_for_every_island(self):
        (self.ws / "rtl/core/x.v").write_text("module y; endmodule\n")
        hw = None

        def run(site, repo, task, candidate, job_dir, job, **kw):
            self.assertFalse(kw["with_board"])
            sim = kw["gate"]
            self.assertTrue(callable(sim))
            job["outcome"] = "ok"
            job["image"] = {"build_run": "r1", "sha256": "e" * 64, "xclbin": "x.xclbin"}
            return {"status": "done", "kind": "hardware", "build": {"status": "PASS"}, "charged": "flow"}, {}
        with mock.patch.object(self.broker, "_timing_met", return_value=True), \
                mock.patch("tritonchip_agent_harness.services.fpga_slow.run", side_effect=run):
            got = self.broker.submit("flow")
            self.assertEqual((got["id"], got["flows_left"]), ("b1", 1))
            r = self.wait("b1")
            hw = self.broker.jobs["b1"]["hardware"]
        self.assertIn("submit score", r["result"]["next"])
        self.assertEqual(self.broker.shared.images[hw]["sha256"], "e" * 64)
        self.assertEqual(self.broker.jobs["b1"]["charged"], "flow")

    def test_a_gate_that_stops_the_flow_gives_it_back(self):
        (self.ws / "rtl/core/x.v").write_text("module y; endmodule\n")

        def run(site, repo, task, candidate, job_dir, job, **kw):
            job["outcome"] = "refused"
            return {"status": "refused", "charged": None, "why": "x",
                    "validation": {"n": 1, "passed": 0, "speedup_geomean": None, "fast_1": 0.0}}, {}
        with mock.patch.object(self.broker, "_timing_met", return_value=True), \
                mock.patch("tritonchip_agent_harness.services.fpga_slow.run", side_effect=run):
            r = self.wait(self.broker.submit("flow")["id"])
        self.assertEqual(r["result"]["gate"], {"n": 1, "passed": 0})
        self.assertIn("RTL gate", r["result"]["why"])
        self.assertNotIn("validation", r["result"])
        self.assertEqual(self.broker.flows_left, 2)

    def test_the_flow_gate_runs_the_gate_points_only(self):
        seen = {}

        def measure(site, repo, task, candidate, job_dir, name):
            seen["validation"], seen["final"] = task.evaluation["validation"], task.evaluation["final"]
            return {"aggregate": {"n": 1, "passed": 1}}
        with mock.patch("tritonchip_agent_harness.services.evaluator.measure_validation", side_effect=measure):
            got = self.broker._flow_gate({"sim_key": "k", "containers": "c", "id": "b1"}, None,
                                         Path(tempfile.mkdtemp()))
        self.assertEqual(got["aggregate"]["passed"], 1)
        self.assertEqual(seen, {"validation": ["op.add/tiny/g"], "final": []})


class OnCardGuards(unittest.TestCase):
    """Each point's deadline from the base's cycles on the card, and the health check after the points."""

    def test_deadlines_and_health_from_the_base_and_a_failed_check_is_the_infrastructures(self):
        repo, base = make_repo()
        t = board_task(repo, base, health_point="op.add/small/a")
        site = types.SimpleNamespace(fpga_pipeline={"card": {}})
        cycles = {"op.add/small/a": 1000, "op.matmul/large/b": 50000, "model.llama_decode/large/c": 70000}
        baseline = {"points": {p: {"cycles": c, "status": "pass"} for p, c in cycles.items()}}
        seen = {}

        def measure(site, repo, task, candidate, image, points, job_dir, name, tag, seconds, **kw):
            seen.update(kw)
            return {"points": {p: {"status": "pass", "cycles": c} for p, c in cycles.items()},
                    "session": {"status": "COMPLETE"}, "measured": {},
                    "health": {"ok": False, "why": "the known-answer point did not pass"}}
        with mock.patch("tritonchip_agent_harness.fpga_slow.board.baseline", return_value=baseline), \
                mock.patch("tritonchip_agent_harness.fpga_slow.board.measure", side_effect=measure), \
                mock.patch("tritonchip_agent_harness.fpga_slow.board.status", return_value="ok"):
            got = fpga_slow.on_card(site, repo, t, None, {"kind": "base"}, None, Path(tempfile.mkdtemp()),
                                    {"containers": "c", "remote_id": "r"}, 60,
                                    health={"point": "op.add/small/a", "tree": "/t"})
        self.assertEqual(seen["deadlines"], {p: 20 * c for p, c in cycles.items()})
        self.assertEqual(seen["health"]["expect"], 1000)
        self.assertEqual(got["outcome"], "infra")
        self.assertIn("health check", got["visible"]["why"])


if __name__ == "__main__":
    unittest.main()
