"""Which machine each part of a run held, and from when to when: requests, the slow tier's two stages, the
sandbox, the final evaluation, and the list a run's record keeps."""
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

from tritonchip_agent_harness import evaluator, loop, occupancy, runner  # noqa: E402
from tritonchip_agent_harness.services import Broker  # noqa: E402
from tritonchip_agent_harness.siteconfig import Account  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402
from tritonchip_agent_harness.workspace import export, git  # noqa: E402

# The site's own labels: the quick tier's worker by its ssh alias, the pipeline's hosts by name.
SITE = types.SimpleNamespace(cpus=8.0, memory="16g", pids=4096, upstream_proxy=None,
                             images={"agent": "tritonchip-agent:dev", "toolchain": "tritonchip-toolchain:dev"},
                             fpga_worker={"ssh": "fpga-build", "workdir": "/w", "vivado_settings": "/s", "threads": 6},
                             fpga_pipeline={"hosts": Path("/nonexistent"), "build_host": "builder",
                                            "board_host": "card", "allow_candidate_host_code": False})
# The step names the FPGA harness's pipeline records (tritonchip_fpga_harness/tests/test_split_hosts.py pins them).
STEPS = ("sync:builder", "sync:card", "doctor:builder", "doctor:card", "build:submit", "build", "export",
         "transfer", "board:measure", "board")


def make_repo() -> tuple:
    root = Path(tempfile.mkdtemp())
    for path, text in {"compiler/a.py": "a = 1\n", "rtl/core/x.v": "module x; endmodule\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git("init", "-q", str(root))
    git("-C", str(root), "add", "-A")
    git("-C", str(root), "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
    return root, git("-C", str(root), "rev-parse", "HEAD").decode().strip()


def make_task(base: str, services: dict = None) -> Task:
    # An observed evaluation (a platform): every point's device has a container of its own, limited to 1 CPU.
    return Task(task_id="t", kind="cross-layer", base_commit=base, design=("compiler/", "rtl/"), evaluator_owned=(),
                hide=(), instruction="Improve it.\n", reference=None, wall_seconds=600,
                evaluation={"profile": "core-sim", "validation": ["v.one", "v.two"], "final": ["f.one"], "jobs": 1,
                            "timeout_seconds": 60, "platform": {"commit": "c" * 40, "paths": ["tb/"]}},
                services=services or {}, path=Path("/t"), sha256="t" * 64)


def pipeline_record(t0: float, board: bool = True) -> dict:
    """A pipeline record as the FPGA harness writes it: each step's start and elapsed time."""
    names = STEPS if board else STEPS[:1] + STEPS[2:3] + STEPS[4:6]
    seconds = dict(zip(STEPS, (20.0, 25.0, 5.0, 8.0, 3.0, 6000.0, 30.0, 90.0, 2.0, 1800.0)))
    steps, t = [], t0
    for name in names:
        steps.append({"name": name, "started_at": t, "ok": True, "elapsed_s": seconds[name]})
        t += seconds[name]
    return {"status": "PASS", "run_id": "agent-slow-1", "started_at": t0, "steps": steps,
            "build": {"status": "PASS", "elapsed_s": 5990.0, "host": "builder"}}


class Requests(unittest.TestCase):
    def test_each_request_is_one_entry_on_its_machine_with_its_queue(self):
        repo, base = make_repo()
        ws = Path(tempfile.mkdtemp()) / "ws"
        export(repo, base, (), ws)
        task = make_task(base, {"score": {"requests": 2}, "fpga_quick": {"minutes": 50, "job_minutes": 20}})
        broker = Broker(SITE, repo, task, ws, Path(tempfile.mkdtemp()) / "run",
                        runners={"score": lambda job, c, d: ({}, {}), "fpga-quick": lambda job, c, d: ({}, {})})
        self.assertEqual(broker.submit("score")["id"], "s1")
        self.assertEqual(broker.submit("fpga-quick")["id"], "f2")
        for job in list(broker.jobs.values()):
            job["thread"].join(5)
        requests = broker.record()["requests"]
        entries = occupancy.services(SITE, task, requests)
        self.assertEqual([(e["role"], e["id"], e["host"], e["reserved"]) for e in entries],
                         [("score", "s1", None, {"cpus": 9.0}),        # 8 for the host, 1 device at a time (jobs 1)
                          ("fpga-quick", "f2", "fpga-build", {"cpus": 6})])
        for e, job in zip(entries, requests):
            self.assertEqual((e["started"], e["finished"]), (job["started"], job["finished"]))
            self.assertEqual(e["queued_seconds"], round(job["started"] - job["submitted"], 1))
        # The queue is the wait from submission to start.
        waited = dict(requests[0], submitted=requests[0]["started"] - 42.0)
        self.assertEqual(occupancy.services(SITE, task, [waited])[0]["queued_seconds"], 42.0)
        # A request cancelled before it started held nothing.
        self.assertEqual(occupancy.services(SITE, task, [{"id": "s3", "service": "score", "state": "cancelled",
                                                          "submitted": 1.0}]), [])


class SlowTier(unittest.TestCase):
    def setUp(self):
        self.task = make_task("a" * 40)

    def job(self, full: dict, mode: str = "board") -> dict:
        return {"id": "l1", "service": "fpga-slow", "mode": mode, "state": "done", "submitted": 900.0,
                "started": 940.0, "finished": 8100.0, "full": full}

    def test_the_link_and_the_board_stage_are_two_entries_on_their_hosts(self):
        build, board = occupancy.services(SITE, self.task, [self.job({"pipeline": pipeline_record(1000.0)})])
        # build:submit starts after both syncs and doctors (58 s); the board stage after export and transfer.
        self.assertEqual((build["role"], build["host"], build["started"], build["finished"], build["seconds"]),
                         ("fpga-slow-build", "builder", 1058.0, 7061.0, 6003.0))
        self.assertEqual((board["role"], board["host"], board["started"], board["finished"], board["seconds"]),
                         ("fpga-slow-board", "card", 7181.0, 8983.0, 1802.0))
        self.assertEqual((build["reserved"], board["reserved"]), (None, {"cards": 1}))
        self.assertEqual((build["id"], board["id"], build["queued_seconds"]), ("l1", "l1", 0.0))
        self.assertNotIn("covers", build)

    def test_a_build_only_request_has_no_board_entry(self):
        entries = occupancy.services(SITE, self.task, [self.job({"pipeline": pipeline_record(1000.0, board=False)},
                                                                mode="build")])
        self.assertEqual([e["role"] for e in entries], ["fpga-slow-build"])

    def test_without_the_pipelines_step_timing_the_whole_request_is_one_entry(self):
        for full in ({"simulation": {}, "error": "the slow-tier budget expired during simulation"},
                     {"pipeline": {"status": "TIMEOUT", "error": "controller timeout; no pipeline record"}}):
            (e,) = occupancy.services(SITE, self.task, [self.job(full)])
            self.assertEqual((e["role"], e["host"], e["seconds"], e["queued_seconds"], e["reserved"]),
                             ("fpga-slow", None, 7160.0, 40.0, None))
            self.assertIn("whole request", e["covers"])


CLAUDE_RESULT = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "result": "OK",
                 "usage": {"input_tokens": 9, "cache_creation_input_tokens": 26000, "cache_read_input_tokens": 0,
                           "output_tokens": 5},
                 "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 9, "outputTokens": 5, "cacheReadInputTokens": 0,
                                                      "cacheCreationInputTokens": 26000}}}


class Sandbox(unittest.TestCase):
    def test_the_agents_container_is_timed_from_its_start_to_its_removal(self):
        run_dir = Path(tempfile.mkdtemp()) / "t-claude-1"
        run_dir.mkdir()
        account = Account(label="claude", cli="claude-code", auth="subscription", credential=Path("/c.env"),
                          hosts=("api.anthropic.com",), models={"main": "claude-sonnet-5-5"}, options={})
        cli_ran = []

        def docker(*args, **kw):
            if args[:1] == ("exec",) and "--version" in args:
                return types.SimpleNamespace(returncode=0, stdout="2.1.287 (Claude Code)\n", stderr="")
            if args[:1] == ("exec",) and "-p" in args:
                cli_ran.append(time.time())
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(CLAUDE_RESULT) + "\n", stderr="")
            return types.SimpleNamespace(returncode=1, stdout="" if kw.get("text", True) else b"", stderr="")

        with mock.patch.object(runner.provider, "docker", side_effect=docker), \
                mock.patch.object(runner.provider, "remove"), \
                mock.patch.object(runner.provider, "image_id", return_value="sha256:1"), \
                mock.patch.object(runner.doctor, "start_gateway", return_value=(run_dir, run_dir)), \
                mock.patch.object(runner.doctor, "stop_gateway"), \
                mock.patch.object(runner.doctor, "read_log", return_value=[]), \
                mock.patch.object(runner.time, "sleep"):
            result = runner.run_agent(SITE, account, "prompt", "offline", run_dir, None, timeout=60)
        box = result["container"]
        self.assertTrue(box["started"] <= cli_ran[0] <= box["finished"])
        self.assertEqual(result["tokens"]["cache_write"], 26000)
        (e,) = occupancy.of_run(SITE, make_task("a" * 40), result, None)
        self.assertEqual((e["role"], e["host"], e["started"], e["finished"], e["queued_seconds"], e["reserved"]),
                         ("sandbox", None, box["started"], box["finished"], 0.0, {"cpus": 8.0}))
        self.assertNotIn("id", e)


class RunRecord(unittest.TestCase):
    def setUp(self):
        self.repo, base = make_repo()
        self.site = types.SimpleNamespace(**vars(SITE), repository=self.repo, state_dir=Path(tempfile.mkdtemp()))
        self.task = make_task(base, {"score": {"requests": 2}})
        self.evaluated = []

    def evaluate(self, site, repo, task, candidate, run_dir):
        self.evaluated.append(time.time())
        rows = {"points": [], "n": 0, "passed": 0, "speedup_geomean": None, "fast_1": None}
        return {"valid": True, "validation": rows, "final": rows}

    def run_task(self, agent: str) -> dict:
        with mock.patch.object(loop.evaluator, "evaluate", side_effect=self.evaluate):
            record = loop.run_task(self.site, self.task, agent)
        return json.loads((Path(record["run_dir"]) / "run.json").read_text())

    def test_the_final_evaluation_is_timed(self):
        saved = self.run_task("noop")
        (e,) = saved["occupancy"]
        self.assertEqual((e["role"], e["host"], e["queued_seconds"]), ("final-evaluation", None, 0.0))
        self.assertTrue(e["started"] <= self.evaluated[0] <= e["finished"])
        self.assertEqual(e["reserved"], {"cpus": 9.0}, "8 for the host, 1 for the one device that runs at a time")

    def test_reserved_counts_the_devices_that_run_together(self):
        # The evaluator runs the points' device containers, 1 CPU each, `jobs` at a time.
        task = types.SimpleNamespace(evaluation={"jobs": 2, "platform": {"commit": "c" * 40}})
        self.assertEqual(evaluator.reserved(SITE, task, ["a", "b", "c"]), {"cpus": 10.0})
        self.assertEqual(evaluator.reserved(SITE, task, ["a"]), {"cpus": 9.0})
        task.evaluation.pop("platform")
        self.assertEqual(evaluator.reserved(SITE, task, ["a", "b", "c"]), {"cpus": 8.0})

    def test_an_agent_run_lists_its_sandbox_its_requests_and_the_evaluation(self):
        def run_agent(site, account, prompt, mode, run_dir, workspace, timeout, services=None):
            self.assertIsNotNone(services, "the task's services get a broker")
            return {"label": "claude", "cli": "claude-code", "stop_reason": "exited",
                    "container": {"started": 100.0, "finished": 700.0},
                    "services": {"requests": [
                        {"id": "s1", "service": "score", "state": "done", "submitted": 200.0, "started": 200.5,
                         "finished": 248.0},
                        {"id": "l2", "service": "fpga-slow", "state": "done", "submitted": 300.0, "started": 300.0,
                         "finished": 650.0, "full": {"pipeline": pipeline_record(310.0)}}]}}

        self.site.account = lambda label: Account(label="claude", cli="claude-code", auth="subscription",
                                                  credential=Path("/c.env"), hosts=("api.anthropic.com",),
                                                  models={"main": "claude-sonnet-5-5"}, options={})
        with mock.patch.object(loop.runner, "run_agent", side_effect=run_agent):
            saved = self.run_task("claude")
        self.assertEqual([(e["role"], e.get("id")) for e in saved["occupancy"]],
                         [("sandbox", None), ("score", "s1"), ("fpga-slow-build", "l2"), ("fpga-slow-board", "l2"),
                          ("final-evaluation", None)])
        self.assertEqual(saved["occupancy"][1]["queued_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
