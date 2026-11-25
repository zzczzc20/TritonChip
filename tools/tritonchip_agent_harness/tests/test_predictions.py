"""Predictions (decided 2026-11-23): what an agent expects of a snapshot on the card, set beside the card."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness import predictions  # noqa: E402

POINTS = ["op.matmul/hbm/a", "op.copy/hbm/b", "op.add/small/c"]
STANDINS = {"op.matmul/membound/x": "op.matmul/hbm/a"}


class Predictions(unittest.TestCase):
    def test_a_prediction_is_checked_and_stand_ins_count_for_their_point(self):
        got = predictions.check({"op.matmul/membound/x": 1.3, "op.add/small/c": 0.9, "score": 1.1}, POINTS, STANDINS)
        self.assertEqual(got, {"op.matmul/hbm/a": 1.3, "op.add/small/c": 0.9, "score": 1.1})
        for bad in ({}, {"op.nope/x": 1.2}, {"op.add/small/c": -1}, {"op.add/small/c": True},
                    {"op.matmul/membound/x": 1.2, "op.matmul/hbm/a": 1.3}, [1.2]):
            with self.assertRaises(predictions.PredictionError, msg=repr(bad)):
                predictions.check(bad, POINTS, STANDINS)

    def test_each_expectation_beside_the_card(self):
        full = {"score": 1.2, "points": [
            {"point": "op.matmul/hbm/a", "failure": None, "speedup": 1.5},
            {"point": "op.copy/hbm/b", "failure": None, "speedup": 0.95},
            {"point": "op.add/small/c", "failure": "timeout", "speedup": None}]}
        g = predictions.grade({"op.matmul/hbm/a": 1.2, "op.copy/hbm/b": 1.1, "op.add/small/c": 1.0, "score": 1.3},
                              full)
        rows = {r["point"]: r for r in g["points"]}
        self.assertEqual(rows["op.matmul/hbm/a"]["measured_over_expected"], 1.25)
        self.assertEqual(rows["op.add/small/c"]["failure"], "timeout")
        self.assertEqual(g["summary"]["measured"], 2)
        self.assertEqual(g["summary"]["direction_agreed"], "1/2")          # copy: expected faster, was slower
        self.assertEqual(g["summary"]["score"], {"expected": 1.3, "measured": 1.2})
        self.assertAlmostEqual(g["summary"]["median_abs_error_pct"], (25.0 + 13.64) / 2, places=1)


if __name__ == "__main__":
    unittest.main()
