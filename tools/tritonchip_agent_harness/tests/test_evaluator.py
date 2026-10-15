"""Scoring a set of points: each point on its own, failures counted, never hidden."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.evaluator import score  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
