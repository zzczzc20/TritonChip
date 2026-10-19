"""Scoring a set of points: each point on its own, failures counted, never hidden."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

import json  # noqa: E402
import tempfile  # noqa: E402

from tritonchip_agent_harness.evaluator import observed_point, read_observer, rtl_check, score  # noqa: E402

BASE = {p: {"status": "pass", "cycles": 1000} for p in ("a", "b", "c", "d")}


class Score(unittest.TestCase):
    def test_a_wrong_point_fails_alone(self):
        got = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "fail-numerics", "cycles": 10},
               "c": {"status": "pass", "cycles": 2000}, "d": {"status": "pass", "cycles": 1000}}
        s = score(BASE, got, ["a", "b", "c", "d"])
        self.assertEqual((s["n"], s["passed"]), (4, 3))
        self.assertEqual([r["failure"] for r in s["points"]], [None, "fail-numerics", None, None])
        self.assertAlmostEqual(s["speedup_geomean"], 1.0)     # 2.0, 0.5, 1.0 over the three that passed
        self.assertAlmostEqual(s["fast_1"], 0.25)             # only a beat the base, out of all four

    def test_fallback_and_missing_are_failures(self):
        got = {"a": {"status": "pass", "cycles": 500, "fallbacks": ["aten.mm"]},
               "c": {"status": "pass", "cycles": None}}
        s = score(BASE, got, ["a", "b", "c"])
        self.assertEqual([r["failure"] for r in s["points"]], ["fallback", "missing", "no-cycles"])
        self.assertEqual(s["passed"], 0)
        self.assertIsNone(s["speedup_geomean"])
        self.assertEqual(s["fast_1"], 0.0)

    def test_breaking_a_slow_point_does_not_raise_fast_1(self):
        slow = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "pass", "cycles": 4000}}
        broken = {"a": {"status": "pass", "cycles": 500}, "b": {"status": "fail-numerics"}}
        s_slow, s_broken = score(BASE, slow, ["a", "b"]), score(BASE, broken, ["a", "b"])
        self.assertGreater(s_broken["speedup_geomean"], s_slow["speedup_geomean"])
        self.assertEqual(s_broken["fast_1"], s_slow["fast_1"])


def candidate(files: dict) -> Path:
    d = Path(tempfile.mkdtemp())
    for path, text in files.items():
        (d / "files" / path).parent.mkdir(parents=True, exist_ok=True)
        (d / "files" / path).write_text(text)
    (d / "candidate.json").write_text(json.dumps({"taken": [{"path": p, "status": "modified"} for p in files]}))
    return d


class RtlCheck(unittest.TestCase):
    def test_plain_rtl_passes(self):
        c = candidate({"rtl/core/x.v": 'module x; initial if ($value$plusargs("t=%d", t)) $display("hi"); endmodule\n',
                       "compiler/a.py": "fopen = 1\n"})
        self.assertEqual(rtl_check(c), [])

    def test_file_and_system_tasks_and_outside_includes_are_refused(self):
        c = candidate({"rtl/core/x.v": 'module x; integer f; initial begin f = $fopen("/obs/observe.log"); '
                                       '$system("id"); end\n`include "../../etc/passwd"\nendmodule\n'})
        self.assertEqual(rtl_check(c), ['rtl/core/x.v: $fopen', 'rtl/core/x.v: $system',
                                        'rtl/core/x.v: `include "../../etc/passwd"'])

    def test_a_task_named_only_in_a_comment_is_not_refused(self):
        self.assertEqual(rtl_check(candidate({"rtl/core/x.v": "// never $fopen here\n/* nor $system */ module x; endmodule\n"})), [])


LOG = ("cmd op=1 seq=5 obs=107 done=101 status=0 inside=28 after_done=0 outside=0 extra=0\n"
       "cmd op=1 seq=9 obs=207 done=201 status=0 inside=40 after_done=0 outside=0 extra=0\n"
       "cmd op=2 seq=10 obs=2 done=0 status=0 inside=0 after_done=0 outside=0 extra=0\n"
       "end outside=0 extra=0\n")


class Observer(unittest.TestCase):
    def log(self, text):
        p = Path(tempfile.mkdtemp()) / "observe.log"
        p.write_text(text)
        return read_observer(p)

    def test_the_score_is_the_platforms_count(self):
        obs = self.log(LOG)
        self.assertEqual((obs["launches"], obs["cycles"], obs["done_cycles"], obs["ended"]), (2, 314, 302, True))
        point = observed_point({"status": "pass", "time": {"device": 302}}, obs)
        self.assertEqual((point["status"], point["cycles"]), ("pass", 314))

    def test_a_record_the_device_never_reported_fails(self):
        point = observed_point({"status": "pass", "time": {"device": 150}}, self.log(LOG))
        self.assertEqual(point["status"], "record-disagrees")

    def test_activity_outside_a_launch_fails(self):
        bad = LOG.replace("end outside=0", "end outside=3")
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(bad))["status"],
                         "activity-outside-launch")
        late = LOG.replace("after_done=0", "after_done=1", 1)
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(late))["status"],
                         "activity-outside-launch")

    def test_no_observation_fails(self):
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, None)["status"], "no-observation")
        cut = LOG.replace("end outside=0 extra=0\n", "")
        self.assertEqual(observed_point({"status": "pass", "time": {"device": 302}}, self.log(cut))["status"],
                         "no-observation")

    def test_a_failed_point_keeps_its_own_status(self):
        self.assertEqual(observed_point({"status": "fail-numerics", "time": {"device": 302}}, self.log(LOG))["status"],
                         "fail-numerics")


if __name__ == "__main__":
    unittest.main()
