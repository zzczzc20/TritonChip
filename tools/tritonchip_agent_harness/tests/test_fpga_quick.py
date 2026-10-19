"""The quick tier's reading of Vivado's reports, and the parameters it passes for the scored profile."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.fpga_quick import effective_params, parse_paths, parse_summary, parse_util  # noqa: E402

SUMMARY = """Design Timing Summary
| ----------------------
    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)
    -------      -------  ---------------------  -------------------      -------
      3.906        0.000                      0                42327        0.052
"""
UTIL = """| CLB LUTs*               | 49564 |     0 |          0 |   1303680 |  3.80 |
| CLB Registers           | 12770 |     0 |          0 |   2607360 |  0.49 |
| Block RAM Tile |    4 |     0 |          0 |      2016 |  0.20 |
| URAM           |    0 |     0 |          0 |       960 |  0.00 |
| DSPs           |   32 |     0 |          0 |      9024 |  0.35 |
"""
PATHS = """Slack (MET) :             3.906ns  (required time - arrival time)
  Source:                 g_mac[0].u_mac/u_fma/p0_sub_671_reg[3]/C
  Destination:            g_mac[0].u_mac/u_fma/p1_sh_mag_reg[7]/D
  Data Path Delay:        6.076ns  (logic 2.1ns (34%)  route 3.9ns (66%))
  Logic Levels:           35  (CARRY8=8 LUT3=4)
"""


class Reports(unittest.TestCase):
    def test_the_numbers_come_out(self):
        self.assertEqual(parse_summary(SUMMARY), {"wns_ns": 3.906, "tns_ns": 0.0, "failing_endpoints": 0,
                                                  "total_endpoints": 42327})
        self.assertEqual(parse_util(UTIL), {"lut": 49564, "ff": 12770, "bram": 4, "uram": 0, "dsp": 32})
        self.assertEqual(parse_paths(PATHS), [{"slack_ns": 3.906, "start": "g_mac[0].u_mac/u_fma/p0_sub_671_reg[3]/C",
                                               "end": "g_mac[0].u_mac/u_fma/p1_sh_mag_reg[7]/D", "delay_ns": 6.076,
                                               "levels": 35}])


class Parameters(unittest.TestCase):
    def test_the_scored_profile_overrides_the_tops_own_defaults(self):
        t = Path(tempfile.mkdtemp())
        (t / "rtl/common").mkdir(parents=True)
        (t / "rtl/fpga").mkdir(parents=True)
        (t / "spec").mkdir()
        (t / "rtl/common/tnca_uarch_defaults.vh").write_text(
            "`ifndef TNCA_IF_NUM_RCTX\n`define TNCA_IF_NUM_RCTX 2\n`endif\n"
            "`define TNCA_UARCH_USE_BANKED_VRF 0\n`define TNCA_UARCH_PHYS_W $clog2(PHYS_VREGS)\n"
            "`define TNCA_UARCH_MTX_ABUF 0\n")
        (t / "rtl/fpga/tnca_fpga_top.v").write_text(
            "module tnca_fpga_top #(\n    parameter NUM_RCTX = 2,\n    parameter USE_BANKED_VRF = 1,\n"
            "    parameter MTX_ABUF = 0,\n    parameter CORE_ID = 0\n) ();\nendmodule\n")
        (t / "spec/r1_profiles.json").write_text(json.dumps({"profiles": {"p": {
            "elaboration": {"NUM_RCTX": 2, "CORE_ID": 3},
            "implementation": {"rtl_parameters": {"MTX_ABUF": 1, "NEW_ONE": 4}}}}}))
        self.assertEqual(effective_params(t, "p"), {"MTX_ABUF": 1, "NUM_RCTX": 2, "USE_BANKED_VRF": 0})


if __name__ == "__main__":
    unittest.main()
