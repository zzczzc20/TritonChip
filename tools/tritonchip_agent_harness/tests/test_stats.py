"""stats: runs grouped by agent framework, their tokens and occupancy summed, and nothing about a point."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import cli, stats  # noqa: E402
from tritonchip_agent_harness.occupancy import entry  # noqa: E402

SECRET = ("hidden.point-one.validation", "hidden.point-two.final")


def evaluation() -> dict:
    rows = [{"point": p, "status": "pass", "failure": None, "base_cycles": 100, "cycles": 90, "speedup": 1.11,
             "done_cycles": 90} for p in SECRET]
    return {"valid": True, "validation": {"points": rows[:1], "n": 1, "passed": 1, "speedup_geomean": 1.11,
                                          "fast_1": 1.0},
            "final": {"points": rows[1:], "n": 1, "passed": 1, "speedup_geomean": 1.11, "fast_1": 1.0}}


def counts(input_uncached, cache_write, cache_read, output, reasoning) -> dict:
    total = input_uncached + cache_write + cache_read + output
    return {"input_uncached": input_uncached, "cache_write": cache_write, "cache_read": cache_read, "output": output,
            "reasoning": reasoning, "total": total, "by_model": None}


def claude(**extra) -> dict:
    return {"label": "claude", "cli": "claude-code", "requested_model": "claude-sonnet-5-5", **extra}


def at(t0: float, seconds: float) -> tuple:
    return t0, t0 + seconds


def write_run(root: Path, name: str, agent_result: dict, occupancy=None, certification=None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    run = {"schema": stats.RUN_SCHEMA, "run_id": name, "agent_result": agent_result, "evaluation": evaluation(),
           "validity": "valid"}
    if occupancy is not None:
        run["occupancy"] = occupancy
    (d / "run.json").write_text(json.dumps(run))
    if certification is not None:
        (d / "certification.json").write_text(json.dumps({"verdict": "certified", "occupancy": certification,
                                                          "points": [{"point": p} for p in SECRET]}))
    return d


class Stats(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "runs"
        r1 = write_run(self.root, "t-claude-1", claude(tokens=counts(2330, 91519, 3868404, 22835, 6120)), [
            entry("sandbox", None, *at(0, 640.0), reserved={"cpus": 8.0}),
            entry("score", None, *at(100, 48.0), queued=0.5, reserved={"cpus": 10.0}, request_id="s1"),
            entry("final-evaluation", None, *at(700, 300.0), reserved={"cpus": 11.0})],
            certification=[entry("certify-link", "builder", *at(2000, 6000.0)),
                           entry("certify-board", "card", *at(8100, 1800.0), reserved={"cards": 1})])
        # Inside a run directory nothing is searched: a workspace may hold files of any name.
        (r1 / "workspace" / "evidence").mkdir(parents=True)
        (r1 / "workspace" / "evidence" / "run.json").write_text("{}")
        write_run(self.root, "t-claude-2", claude(tokens=counts(50, 20000, 400000, 5000, 1000)), [
            entry("sandbox", None, *at(0, 360.0), reserved={"cpus": 8.0}),
            entry("score", None, *at(50, 50.0), queued=1.5, request_id="s1"),
            entry("score", None, *at(150, 46.0), request_id="s2"),
            entry("final-evaluation", None, *at(400, 310.0))])
        # A record from before `tokens`: normalised from the CLI's raw report.
        write_run(self.root, "t-claude-old", claude(usage={"input_tokens": 100, "output_tokens": 2000},
                                                    model_usage={"claude-sonnet-5-5": {
                                                        "inputTokens": 100, "outputTokens": 2000,
                                                        "cacheReadInputTokens": 50000,
                                                        "cacheCreationInputTokens": 1000}}))
        write_run(self.root, "t-codex-1", {"label": "codex", "cli": "codex", "requested_model": "gpt-6.1-sol",
                                           "tokens": counts(70045, 0, 1418496, 7800, 2071)}, [
            entry("sandbox", None, *at(0, 170.0)), entry("fpga-quick", "fpga-build", *at(20, 710.0), request_id="f1"),
            entry("final-evaluation", None, *at(900, 290.0))])
        write_run(self.root, "t-noop-1", {"pseudo": "noop"}, [entry("final-evaluation", None, *at(0, 295.0))])
        (self.root / "t-broken").mkdir()
        (self.root / "t-broken" / "run.json").write_text("{not json")

    def test_runs_are_grouped_by_framework_and_summed(self):
        value = stats.summarise([self.root])
        self.assertEqual((value["runs"], value["skipped"]), (5, 1))
        noop, cc, codex = value["groups"]
        self.assertEqual([(g["cli"], g["model"], g["label"], g["runs"]) for g in value["groups"]],
                         [(None, None, "noop", 1), ("claude-code", "claude-sonnet-5-5", "claude", 3),
                          ("codex", "gpt-6.1-sol", "codex", 1)])
        self.assertEqual(cc["tokens"]["input_uncached"], {"total": 2480, "mean": 826.7, "runs": 3})
        self.assertEqual(cc["tokens"]["reasoning"], {"total": 7120, "mean": 3560.0, "runs": 2},
                         "the old record's model reported no thinking: it is not counted as 0")
        self.assertEqual(cc["tokens"]["total"]["total"], 3985088 + 425050 + 53100)
        self.assertEqual(cc["occupancy"]["sandbox"], {"entries": 2, "seconds": {"total": 1000.0, "mean": 500.0},
                                                      "queued_seconds": {"total": 0.0, "mean": 0.0}})
        self.assertEqual(cc["occupancy"]["score"], {"entries": 3, "seconds": {"total": 144.0, "mean": 48.0},
                                                    "queued_seconds": {"total": 2.0, "mean": 0.7}})
        self.assertEqual(cc["occupancy"]["certify-link"]["seconds"], {"total": 6000.0, "mean": 6000.0})
        self.assertEqual(sorted(codex["occupancy"]), ["final-evaluation", "fpga-quick", "sandbox"])
        self.assertEqual(codex["tokens"]["cache_read"], {"total": 1418496, "mean": 1418496.0, "runs": 1})
        self.assertEqual(noop["tokens"]["total"], {"total": None, "mean": None, "runs": 0})
        self.assertEqual(list(noop["occupancy"]), ["final-evaluation"])

    def test_the_command_prints_json_with_nothing_about_a_point(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["stats", str(self.root / "t-claude-1"), str(self.root / "t-codex-1")]), 0)
        text = out.getvalue()
        self.assertEqual([g["label"] for g in json.loads(text)["groups"]], ["claude", "codex"])
        for secret in (*SECRET, "hidden.", "points", "speedup", "cycles", "t-claude-1"):
            self.assertNotIn(secret, text)

    def test_a_path_that_does_not_exist_is_an_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["stats", str(self.root / "missing")]), 2)
        self.assertIn("does not exist", err.getvalue())


if __name__ == "__main__":
    unittest.main()
