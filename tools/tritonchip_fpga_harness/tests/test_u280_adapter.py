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
from tritonchip_fpga_harness.core import load_profile


class U280AdapterTests(unittest.TestCase):
    def test_pr5_command_and_environment_contract(self) -> None:
        profile = load_profile(HERE, "u280-r38-pr5")
        impl = U280FlowAdapter()._impl(Path("/tmp/pr5-u280-source"), profile)
        self.assertEqual(impl.config.expected_ap_clk_mhz, 100)
        self.assertEqual(impl.config.expected_device_divider, 1)
        self.assertEqual(impl.config.xo_name, "tnca_u50.xo")
        self.assertEqual(impl.config.xclbin_name, "tnca_u280.xclbin")
        self.assertEqual(impl.config.environment["TNCA_BOARD"], "u280")
        self.assertEqual(impl.config.environment["VPP_JOBS"], "8")
        self.assertEqual(impl.config.required_board_options, "--connectivity.slr tnca_u50_1:SLR1")
        self.assertTrue(str(impl.config.package_script).endswith("flow/fpga/package_xo.tcl"))
        self.assertTrue(str(impl.config.link_script).endswith("flow/fpga/link.sh"))
        self.assertEqual(impl.package_command(Path("/tmp/build"))[-1], "xcu280-fsvh2892-2L-e")

    def test_u280_profile_allows_only_advisory_scaling_rule(self) -> None:
        profile = load_profile(HERE, "u280-r38-pr5")
        self.assertEqual(profile.data["allowed_auto_frequency_rules"], ["AUTO-FREQ-SCALING-08"])

    def test_u280_artifact_accepts_advisory_but_rejects_clock_change(self) -> None:
        profile = load_profile(HERE, "u280-r38-pr5")
        impl = U280FlowAdapter()._impl(Path("/tmp/pr5-u280-source"), profile)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "tnca_u50.xo").write_bytes(b"xo")
            (out / "tnca_u280.xclbin").write_bytes(b"xclbin")
            (out / "xclbin.info").write_text("Name: DATA_CLK\nFrequency: 100 MHz\n", encoding="utf-8")
            (out / "dr_timing_summary.rpt").write_text(
                "Clock WNS TNS WHS THS Failing Endpoints\n"
                "ap_clk 0.127 0.0 0.011 0.0 0\n", encoding="utf-8"
            )
            (out / "drc.rpt").write_text("Number of Errors: 0\n", encoding="utf-8")
            log = (
                "VPP_RC=0\n"
                "clock name 'kernel_00' (clock ID '0') is being mapped to clock name 'DATA_CLK'\n"
                "AUTO-FREQ-SCALING-08\nU280_LINK_OK\n"
            )
            (out / "link.log").write_text(log, encoding="utf-8")
            self.assertTrue(impl.validate_artifacts(out).ok)
            (out / "link.log").write_text(log.replace("AUTO-FREQ-SCALING-08", "AUTO-FREQ-SCALING-04"), encoding="utf-8")
            self.assertFalse(impl.validate_artifacts(out).ok)

    def test_u280_full_validation_requires_routed_timing_and_drc(self) -> None:
        profile = load_profile(HERE, "u280-r38-pr5")
        impl = U280FlowAdapter()._impl(Path("/tmp/pr5-u280-source"), profile)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "tnca_u50.xo").write_bytes(b"xo")
            (out / "tnca_u280.xclbin").write_bytes(b"xclbin")
            (out / "xclbin.info").write_text("Name: DATA_CLK\nFrequency: 100 MHz\n", encoding="utf-8")
            (out / "link.log").write_text("VPP_RC=0\nLINK_OK\n", encoding="utf-8")
            result = impl.validate_artifacts(out)
            self.assertFalse(result.ok)
            failed = {check.name for check in result.checks if not check.ok}
            self.assertIn("routed_timing_report", failed)
            self.assertIn("drc_report", failed)


if __name__ == "__main__":
    unittest.main()
