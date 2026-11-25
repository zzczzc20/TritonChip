"""The routed timing summary's per-clock tables, read from a real R41 link (R41-F2's flow l8, 2026-11-09).

l8 failed on its kernel clock (WNS -0.352, TNS -457.657, 4062 endpoints). Its TNS is wider than the
`TNS(ns)` heading, so it starts before that column's dash run; cutting the rows at the dash runs put its
sign into the WNS cell, the tables were lost, and the hbm_aclk exception fell back to the design row.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.adapters.u50 import U50Adapter  # noqa: E402

L8 = (Path(__file__).parent / "data" / "r41_l8_timing_summary_tables.rpt").read_text()


def retyped(text: str, row_start: str, values: dict) -> str:
    """`text` with the numbers of the first line starting `row_start` (indentation aside) replaced, each keeping
    its right edge (the report's alignment): {old token: new token}."""
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if line.strip().startswith(row_start):
            for old, new in values.items():
                m = re.search(rf"(?<!\S){re.escape(old)}(?!\S)", line)
                assert m, (row_start, old)
                start, end = m.start(), m.end()
                line = line[:start] + " " * (end - start) + line[end:]        # the old token gone
                assert line[end - len(new):end].strip() == "", "no room on the left"
                line = line[:end - len(new)] + new + line[end:]
            lines[n] = line
            return "\n".join(lines) + "\n"
    raise AssertionError(row_start)


class Tables(unittest.TestCase):
    def test_the_l8_tables_are_read_with_a_wide_tns(self):
        tables = U50Adapter._clock_tables(L8)
        self.assertIsNotNone(tables)
        self.assertEqual({k: len(v) for k, v in tables.items()}, {"intra": 49, "inter": 31, "other": 28})
        kernel = next(r for r in tables["intra"] if r["clock"] == "clk_kernel_00_unbuffered_net")
        self.assertEqual((kernel["wns_ns"], kernel["tns_ns"], kernel["tns_failing_endpoints"]),
                         (-0.352, -457.657, 4062))
        hbm = next(r for r in tables["intra"] if r["clock"] == "hbm_aclk")
        self.assertEqual((hbm["wns_ns"], hbm["whs_ns"], hbm["wpws_ns"]), (0.029, 0.011, 0.001))
        # A row with only some columns keeps them in place.
        tx = next(r for r in tables["intra"] if r["clock"] == "GTYE4_CHANNEL_TXOUTCLK[0]_4")
        self.assertEqual(tx, {"clock": "GTYE4_CHANNEL_TXOUTCLK[0]_4", "wpws_ns": 0.71, "tpws_ns": 0.0,
                              "tpws_failing_endpoints": 0, "tpws_total_endpoints": 1})
        crossing = next(r for r in tables["inter"] if r.get("from_clock") == "hbm_aclk"
                        and r.get("to_clock") == "clk_kernel_00_unbuffered_net")
        self.assertEqual(crossing["wns_ns"], 0.504)

    def test_l8_fails_on_its_kernel_clock_with_hbm_exempt(self):
        verdict = U50Adapter._exempt_timing(U50Adapter._clock_tables(L8), U50Adapter._design_timing_summary(L8),
                                            ("hbm_aclk",))
        self.assertFalse(verdict["ok"])
        self.assertEqual([(v["table"], v["clock"]) for v in verdict["judged"]["violations"]],
                         [("intra", "clk_kernel_00_unbuffered_net")])
        self.assertEqual(verdict["unaccounted"], [])

    def test_a_wide_hbm_only_failure_is_excused(self):
        # The kernel clock closes and only hbm_aclk misses, by a TNS wider than its heading: the case the
        # exception exists for, which the old parser judged design-wide.
        text = retyped(L8, "clk_kernel_00_unbuffered_net", {"-0.352": "0.352", "-457.657": "0.000", "4062": "0"})
        text = retyped(text, "hbm_aclk", {"0.029": "-0.300", "0.000": "-150.123", "0": "812"})
        text = retyped(text, "-0.352", {"-0.352": "-0.300", "-457.657": "-150.123", "4062": "812"})   # design row
        tables = U50Adapter._clock_tables(text)
        self.assertIsNotNone(tables)
        verdict = U50Adapter._exempt_timing(tables, U50Adapter._design_timing_summary(text), ("hbm_aclk",))
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["applied"])
        self.assertEqual([r["clock"] for r in verdict["exempt_failing"]], ["hbm_aclk"])

    def test_a_cell_that_is_not_a_number_still_loses_the_tables(self):
        text = retyped(L8, "hbm_aclk", {"0.029": "x.xxx"})
        self.assertIsNone(U50Adapter._clock_tables(text))


if __name__ == "__main__":
    unittest.main()
