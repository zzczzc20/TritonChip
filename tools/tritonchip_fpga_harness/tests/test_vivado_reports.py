"""Timing and DRC acceptance against what a Vitis link really leaves behind.

The layouts here are Vivado 2023.2's, as found in a U280 link's output: the
"Design Timing Summary" of ``*_timing_summary_routed.rpt`` and the DRC lines
of ``prj.runs/impl_1/runme.log``.  A Vitis link writes no DRC report file.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tritonchip_fpga_harness.adapters.u280 import U280FlowAdapter
from tritonchip_fpga_harness.adapters.u50 import U50Adapter
from tritonchip_fpga_harness.core import load_profile

COLUMNS = (
    "    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)"
    "  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  \n"
    "    -------      -------  ---------------------  -------------------      -------      -------"
    "  ---------------------  -------------------     --------     --------  ----------------------  --------------------  \n"
)


def routed_report(design_row: str) -> str:
    """A timing summary report: the design row, then the kernel clock's own row."""
    return (
        "------------------------------------------------------------------------------------------------\n"
        "| Design Timing Summary\n"
        "| ---------------------\n"
        "------------------------------------------------------------------------------------------------\n\n"
        + COLUMNS + design_row + "\n\n\n"
        "All user specified timing constraints are met.\n\n"
        "------------------------------------------------------------------------------------------------\n"
        "| Intra Clock Table\n"
        "| -----------------\n"
        "------------------------------------------------------------------------------------------------\n\n"
        "Clock         " + COLUMNS.splitlines()[0].strip() + "\n"
        "-----         " + COLUMNS.splitlines()[1].strip() + "\n"
        # Clean on its own, whatever the design row says: 0.177 / 0.010, 167090 endpoints.
        "ap_clk              0.177        0.000                      0               167090        0.010        0.000"
        "                      0               167090        3.725        0.000                       0                 71304  \n"
    )


CLEAN = "      0.016        0.000                      0               584460        0.006        0.000                      0               582791        0.000        0.000                       0                241423  "
HOLD = "      0.016        0.000                      0               584460       -0.025       -0.035                      3               582791        0.000        0.000                       0                241423  "
SETUP = "     -0.683    -2642.759                   7019               584460        0.006        0.000                      0               582791        0.000        0.000                       0                241423  "

IMPLEMENTATION_LOG = (
    "Running DRC as a precondition to command opt_design\n"
    "INFO: [Project 1-461] DRC finished with 0 Errors, 216 Warnings\n"
    "Running DRC as a precondition to command place_design\n"
    "INFO: [Vivado_Tcl 4-198] DRC finished with 0 Errors, 217 Warnings\n"
    "Running DRC as a precondition to command route_design\n"
    "INFO: [Vivado_Tcl 4-198] DRC finished with 0 Errors\n"
    "INFO: [Route 35-416] Intermediate Timing Summary | WNS=-0.614 | TNS=-30.520| WHS=-0.813 | THS=-75.633|\n"
    "INFO: [Route 35-57] Estimated Timing Summary | WNS=0.016  | TNS=0.000  | WHS=0.006  | THS=0.000  |\n"
    "Running DRC as a precondition to command write_bitstream\n"
    "INFO: [Vivado 12-3199] DRC finished with 0 Errors, 625 Warnings\n"
)


class VitisLinkOutput(unittest.TestCase):
    """validate_artifacts on a directory laid out like a real U280 link's."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="tritonchip-harness-vivado-")
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.impl = self.out / "tmp/link/vivado/vpl/prj/prj.runs/impl_1"
        self.reports = self.out / "tmp/reports/link/imp"
        self.impl.mkdir(parents=True)
        self.reports.mkdir(parents=True)
        (self.out / "tnca_u50.xo").write_bytes(b"xo")
        (self.out / "tnca_u280.xclbin").write_bytes(b"xclbin")
        (self.out / "xclbin.info").write_text("Name: DATA_CLK\nFrequency: 100 MHz\n", encoding="utf-8")
        (self.out / "link.log").write_text(
            "VPP_RC=0\nclock name 'kernel_00' (clock ID '0') is being mapped to clock name 'DATA_CLK'\nLINK_OK\n",
            encoding="utf-8")
        self.adapter = U280FlowAdapter()._impl(Path("/nonexistent/source"), load_profile(HERE, "u280"))

    def link(self, design_row: str = CLEAN, log: str = IMPLEMENTATION_LOG) -> dict[str, object]:
        for path in (self.reports / "impl_1_hw_bb_locked_timing_summary_routed.rpt",
                     self.impl / "hw_bb_locked_timing_summary_routed.rpt"):
            path.write_text(routed_report(design_row), encoding="utf-8")
        (self.impl / "runme.log").write_text(log, encoding="utf-8")
        return self.adapter.validate_artifacts(self.out).as_dict()

    def failed(self, validation: dict[str, object]) -> set[str]:
        return {c["name"] for c in validation["checks"] if not c["ok"]}

    def test_a_clean_link_passes_on_the_design_row_and_the_logged_drc(self) -> None:
        validation = self.link()
        self.assertTrue(validation["ok"], self.failed(validation))
        timing = validation["timing"]
        self.assertEqual(timing["scope"], "design")
        self.assertEqual(timing["metrics"]["wns_ns"], 0.016)
        self.assertEqual(timing["metrics"]["whs_ns"], 0.006)
        self.assertEqual(timing["metrics"]["ths_ns"], 0.0)
        self.assertEqual(timing["metrics"]["failing_endpoints"], 0)
        self.assertEqual(timing["metrics"]["tns_total_endpoints"], 584460)

    def test_a_hold_violation_fails_though_the_kernel_clocks_own_row_is_clean(self) -> None:
        validation = self.link(HOLD)
        self.assertEqual(self.failed(validation), {"timing_clean"})
        metrics = validation["timing"]["metrics"]
        self.assertEqual(metrics["whs_ns"], -0.025)
        self.assertEqual(metrics["ths_ns"], -0.035)
        self.assertEqual(metrics["failing_endpoints"], 3)

    def test_a_setup_violation_on_a_crossing_fails(self) -> None:
        validation = self.link(SETUP)
        self.assertEqual(self.failed(validation), {"timing_clean"})
        self.assertEqual(validation["timing"]["metrics"]["wns_ns"], -0.683)
        self.assertEqual(validation["timing"]["metrics"]["failing_endpoints"], 7019)

    def test_drc_is_read_from_the_implementation_log(self) -> None:
        validation = self.link()
        self.assertNotIn("drc_report", self.failed(validation))
        report = U50Adapter._drc_report(self.out)
        self.assertEqual(report["source"], "implementation-log")
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["checks"], 4)
        self.assertTrue(report["bitstream_precondition"])
        self.assertTrue(report["report"].endswith("impl_1/runme.log"))

    def test_a_drc_error_at_any_precondition_fails(self) -> None:
        log = IMPLEMENTATION_LOG.replace("DRC finished with 0 Errors, 217 Warnings", "DRC finished with 2 Errors, 217 Warnings")
        validation = self.link(log=log)
        self.assertEqual(self.failed(validation), {"drc_clean"})

    def test_a_link_with_no_drc_verdict_anywhere_is_not_accepted(self) -> None:
        validation = self.link(log="INFO: nothing about DRC here\n")
        self.assertEqual(self.failed(validation), {"drc_report"})

    def test_a_log_named_runme_outside_an_implementation_run_is_not_drc_evidence(self) -> None:
        self.link(log="INFO: nothing about DRC here\n")
        synth = self.out / "tmp/link/vivado/vpl/prj/prj.runs/synth_1"
        synth.mkdir(parents=True)
        (synth / "runme.log").write_text("DRC finished with 0 Errors\n", encoding="utf-8")
        self.assertIsNone(U50Adapter._drc_report(self.out)["report"])


class DesignTimingSummary(unittest.TestCase):
    def test_per_clock_table_alone_is_not_a_design_summary(self) -> None:
        table = routed_report(CLEAN).split("| Intra Clock Table")[1]
        self.assertIsNone(U50Adapter._design_timing_summary(table))

    def test_header_without_a_numeric_row_is_not_a_summary(self) -> None:
        self.assertIsNone(U50Adapter._design_timing_summary(COLUMNS + "  (no data)\n"))
        self.assertIsNone(U50Adapter._design_timing_summary("no table at all\n"))

    def test_pulse_width_violation_counts(self) -> None:
        row = CLEAN.replace("0.000        0.000                       0                241423",
                            "-0.120       -0.480                       4                241423")
        metrics = U50Adapter._design_timing_summary(COLUMNS + row + "\n")
        self.assertEqual(metrics["wpws_ns"], -0.120)
        self.assertEqual(metrics["failing_endpoints"], 4)


if __name__ == "__main__":
    unittest.main()
