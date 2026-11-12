"""The operator's stop (SIGINT or SIGTERM) still leaves a run record: run 6180 was stopped and left none."""
from __future__ import annotations

import _thread
import json
import os
import signal
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import islands, loop  # noqa: E402
from tritonchip_agent_harness.task import Task  # noqa: E402


def task() -> Task:
    cfg = {"count": 1, "topology": "ring", "migration_interval": 3, "migration_rate": 0.3,
           "features": ["layer"], "max_idle_turns": 3, "max_restarts": 2}
    return Task(task_id="t", kind="cross-layer", base_commit="a" * 40, design=("compiler/",),
                evaluator_owned=(), hide=(), instruction="", reference=None, wall_seconds=600,
                evaluation={"profile": "p", "validation": ["v"], "final": ["f"]},
                services={"score": {"requests": 5}}, path=HERE, sha256="", islands=cfg)


class FakeBroker:
    def __init__(self, *args, **kwargs):
        self.jobs = {}

    def record(self):
        return {"left": {}, "requests": [
            {"id": "l1", "service": "fpga-slow", "state": "running", "remote_id": "tnca-x", "build_host": "rscl"},
            {"id": "s1", "service": "score", "state": "done"}]}


class OperatorStop(unittest.TestCase):
    def test_sigterm_and_sigint_are_the_operators_stop_while_a_run_runs(self):
        before = signal.getsignal(signal.SIGTERM)
        for sig, name in ((signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")):
            with self.assertRaises(loop.OperatorStop) as got:
                with loop.operator_signals():
                    os.kill(os.getpid(), sig)
                    for _ in range(100):   # the handler runs at the next bytecode boundary
                        pass
            self.assertEqual(got.exception.how, name)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_islands_hand_up_what_they_had_when_stopped(self):
        stopped = threading.Event()

        def agent(*args, **kwargs):
            stopped.wait(10)
            return {"stop_reason": "operator"}

        def stop_agents(*args):
            stopped.set()

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(islands, "export", return_value={"files": 1}), \
                mock.patch.object(islands, "render", return_value="prompt"), \
                mock.patch.object(islands.runner, "run_agent", side_effect=agent), \
                mock.patch.object(islands, "_stop_agents", side_effect=stop_agents):
            timer = threading.Timer(0.3, _thread.interrupt_main)
            timer.start()
            account = types.SimpleNamespace(cli="codex")
            with self.assertRaises(islands.Stopped) as got:
                islands.run(None, task(), account, "offline", Path(tmp), Path(tmp), [], FakeBroker, None)
            timer.cancel()
        partial = got.exception.partial
        self.assertEqual(got.exception.how, "SIGINT")
        self.assertEqual(partial["islands"][0]["agent_result"], {"stop_reason": "operator"})
        self.assertIsNotNone(partial["archive"])

    def test_an_island_that_does_not_end_in_time_still_hands_up_its_requests(self):
        # The stopped R41-F2 launch (2026-11-09): its island had not ended within the wait, and the record had
        # `islands: []`. Its broker's requests and its turns are there all along.
        def agent(*args, **kwargs):
            threading.Event().wait(3)        # ignores the stop for longer than the stop waits
            return {"stop_reason": "operator"}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(islands, "STOP_JOIN_SECONDS", 0.2), \
                mock.patch.object(islands, "export", return_value={"files": 1}), \
                mock.patch.object(islands, "render", return_value="prompt"), \
                mock.patch.object(islands.runner, "run_agent", side_effect=agent), \
                mock.patch.object(islands, "_stop_agents"):
            # A real signal: it interrupts the main thread's wait on the island, as the operator's SIGTERM did.
            timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
            timer.start()
            with self.assertRaises(islands.Stopped) as got:
                islands.run(None, task(), types.SimpleNamespace(cli="codex"), "offline", Path(tmp), Path(tmp), [],
                            FakeBroker, None)
            timer.cancel()
        row = got.exception.partial["islands"][0]
        self.assertEqual(row["agent_result"]["stop_reason"], "operator")
        self.assertIn("had not ended", row["agent_result"]["incomplete"])
        self.assertIsInstance(row["broker"], FakeBroker)
        self.assertEqual(row["workspace"], {"files": 1})
        self.assertEqual([r["island"] for r in loop._island_rows(got.exception.partial["islands"])], [0])
        self.assertEqual(loop._island_rows([row])[0]["agent_result"]["services"]["requests"][1]["id"], "s1")

    def test_a_stopped_island_run_writes_its_record_and_lists_the_flows_left_running(self):
        rows = [{"island": 0, "agent_result": {"stop_reason": "operator"}, "broker": FakeBroker(),
                 "commits": 0}]
        archive = types.SimpleNamespace(record=lambda: {"commits": []})
        stop = islands.Stopped({"islands": rows, "archive": archive}, "SIGTERM")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(loop.evaluator, "baseline"), \
                mock.patch("tritonchip_agent_harness.services.Shared"), \
                mock.patch.object(islands, "run", side_effect=stop), \
                mock.patch.object(loop.occupancy, "of_run", return_value=[{"role": "agent"}]):
            run_dir = Path(tmp)
            record = {"run_id": "r"}
            with self.assertRaises(KeyboardInterrupt):
                loop._islands(None, task(), None, "offline", [], record, run_dir, run_dir)
            written = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(written["validity"], "stopped")
        self.assertEqual((written["stopped"]["how"], written["stopped"]["stage"]), ("SIGTERM", "islands"))
        self.assertEqual(written["stopped"]["slow_requests_left_running"],
                         [{"island": 0, "id": "l1", "remote_id": "tnca-x", "build_host": "rscl"}])
        self.assertEqual(written["islands"]["islands"][0]["agent_result"]["services"]["requests"][1]["id"], "s1")
        self.assertEqual(written["archive"], {"commits": []})
        self.assertEqual(written["occupancy"], [{"role": "agent", "island": 0}])
        self.assertNotIn("champions", written)


if __name__ == "__main__":
    unittest.main()
