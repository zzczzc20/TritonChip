"""The quick tier's reading of Vivado's reports, and the parameters it passes for the scored profile."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from unittest import mock  # noqa: E402

from tritonchip_agent_harness import fpga_quick  # noqa: E402
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


def device_tree() -> Path:
    t = Path(tempfile.mkdtemp())
    (t / "rtl/common").mkdir(parents=True)
    (t / "rtl/fpga").mkdir(parents=True)
    (t / "spec").mkdir()
    (t / "rtl/common/tnca_uarch_defaults.vh").write_text("`define TNCA_UARCH_MTX_ABUF 0\n")
    (t / "rtl/fpga/tnca_fpga_top.v").write_text("module tnca_fpga_top #(parameter MTX_ABUF = 0) ();\nendmodule\n")
    (t / "spec/r1_profiles.json").write_text(json.dumps({"profiles": {"p": {}}}))
    return t


class Provenance(unittest.TestCase):
    # Estimates from two Vivado releases, or two workers, are not one series:
    # every result says which worker ran it and which Vivado its log names.
    WORKER = {"ssh": "build-box", "workdir": "/w", "vivado_settings": "/opt/Xilinx/Vivado/2023.2/settings64.sh"}

    def run_job(self, banner):
        job = Path(tempfile.mkdtemp())
        reports = job / "ooc" / "tnca_fpga_top"
        reports.mkdir(parents=True)
        if banner is not None:
            (reports / "vivado.log").write_text(f"#-----\n****** {banner}\n  **** SW Build 1 on Mon\n")
        (reports / "tnca_fpga_top_summary.rpt").write_text(SUMMARY)
        (reports / "tnca_fpga_top_util.rpt").write_text(UTIL)
        with mock.patch.object(fpga_quick.subprocess, "run",
                               return_value=mock.Mock(returncode=0, stdout="", stderr="")):
            return fpga_quick.run(self.WORKER, device_tree(), "p", 10.0, 60, job, "tnca-job")

    def test_the_worker_and_the_vivado_release_are_recorded(self):
        r = self.run_job("Vivado v2023.2 (64-bit)")
        self.assertEqual((r["status"], r["worker"], r["vivado_version"]), ("ok", "build-box", "2023.2"))
        self.assertEqual(r["vivado_settings"], self.WORKER["vivado_settings"])
        self.assertEqual(self.run_job("Vivado v2022.2.2 (64-bit)")["vivado_version"], "2022.2.2")

    def test_no_log_is_no_version_not_a_guess(self):
        self.assertIsNone(self.run_job(None)["vivado_version"])

    def test_the_agent_is_not_shown_where_it_ran(self):
        from tritonchip_agent_harness.services import FPGA_VISIBLE
        self.assertFalse({"worker", "vivado_settings", "vivado_version"} & set(FPGA_VISIBLE))


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


class Sandboxes(unittest.TestCase):
    """The worker's command: bubblewrap by default; `unshare` for a worker without it."""
    WORKER = {"ssh": "w", "workdir": "/jobs", "vivado_settings": "/x/settings64.sh", "threads": 6}

    def test_bwrap_is_the_default(self):
        from tritonchip_agent_harness.fpga_quick import _sandboxed
        cmd = _sandboxed(self.WORKER, "/jobs/j1", 10.0, {"NUM_RCTX": 4}, 1800)
        self.assertIn("bwrap --unshare-all", cmd)
        self.assertNotIn("unshare -Urn", cmd)
        self.assertIn("maxThreads 6", cmd)
        self.assertIn("OOC_SV=1 OOC_GENERICS=NUM_RCTX=4", cmd)

    def test_unshare_keeps_the_job_offline_and_niced(self):
        from tritonchip_agent_harness.fpga_quick import _sandboxed
        cmd = _sandboxed(dict(self.WORKER, sandbox="unshare"), "/jobs/j1", 10.0, {"NUM_RCTX": 4}, 1800)
        self.assertNotIn("bwrap", cmd)
        self.assertIn("timeout --kill-after=30 1800 nice -n 10 unshare -Urn /bin/bash -c", cmd)
        self.assertIn("cd /jobs/j1 && source /x/settings64.sh", cmd)
        self.assertIn("vivado -mode batch", cmd)


if __name__ == "__main__":
    unittest.main()
