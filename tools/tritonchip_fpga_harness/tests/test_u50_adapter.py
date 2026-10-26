"""Offline contract tests for the U50 adapter.

These tests deliberately never invoke Vivado or Vitis.  They exercise the
acceptance boundary that a worker can run on any host and use tiny fixture
artifacts to prove that a missing xclbin cannot pass on an exit code alone.
"""
from __future__ import annotations

import tempfile
import re
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tritonchip_fpga_harness.adapters.u50 import CommandResult, U50Adapter, U50Config, flow_clock


class U50ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        # Resolve the checkout from this fixture rather than the caller's
        # current directory so the test is safe under unittest discovery and
        # when invoked from a CI workspace wrapper.
        link_text = (REPO / "flow/fpga/link.sh").read_text(encoding="utf-8")
        clock = flow_clock(link_text)[0]
        wrapper_text = (REPO / "rtl/board/tnca_u50_xo.v").read_text(encoding="utf-8")
        divider_match = re.search(r"parameter\s+DEV_DIV\s*=\s*(\d+)|BUFGCE_DIVIDE\s*\((\d+)\)", wrapper_text)
        divider = int(next(value for value in divider_match.groups() if value is not None))
        self.clock = clock
        self.divider = divider
        self.adapter = U50Adapter(U50Config(REPO, expected_ap_clk_mhz=clock, expected_device_divider=divider))

    def _write_good(self, out: Path) -> None:
        (out / "tnca_u50.xo").write_bytes(b"fixture-xo")
        (out / "tnca_u50.xclbin").write_bytes(b"fixture-xclbin")
        (out / "link.log").write_text(
            "VPP_RC=0\n"
            "clock name 'ulp_ucs/aclk_kernel_00' (clock ID '0') is being mapped "
            "to clock name 'DATA_CLK'\n"
            "LINK_OK\n",
            encoding="utf-8",
        )
        (out / "xclbin.info").write_text(
            "Scalable Clocks\n"
            "---------------\n"
            "   Name: DATA_CLK\n"
            f"   Frequency: {self.clock} MHz\n",
            encoding="utf-8",
        )

    def test_good_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = self.adapter.validate_artifacts(Path(td))
            self.assertFalse(result.ok)
            out = Path(td)
            self._write_good(out)
            result = self.adapter.validate_artifacts(out)
            self.assertTrue(result.ok, result.as_dict())
            self.assertEqual(result.observed_clock_mhz, self.clock)
            self.assertEqual(result.mapped_clock_name, "DATA_CLK")
            self.assertEqual(result.artifacts["xclbin"]["size"], len(b"fixture-xclbin"))

    def test_zero_rc_without_xclbin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self._write_good(out)
            (out / "tnca_u50.xclbin").unlink()
            result = self.adapter.validate_artifacts(out)
            self.assertFalse(result.ok)
            names = {check.name for check in result.checks if not check.ok}
            self.assertIn("xclbin", names)
            self.assertIn("missing_xclbin_after_zero_rc", names)

    def test_dry_run_has_no_vendor_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "u50-run"
            result = self.adapter.run(out, mode="full", dry_run=True)
            self.assertTrue(result.preflight.ok, result.preflight.as_dict())
            self.assertTrue(result.dry_run)
            self.assertEqual([Path(c.argv[0]).name for c in result.commands], ["vivado", "link.sh"])
            self.assertFalse(result.artifacts.ok)
            self.assertFalse(out.exists(), "dry-run must not create build artifacts")

    @patch("tools.tritonchip_fpga_harness.adapters.u50._existing_vendor_processes", return_value=["123 vivado"])
    def test_live_preflight_blocks_existing_vendor_process(self, _existing) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = self.adapter.preflight(Path(td) / "run", dry_run=False, require_vendor_tools=False)
        self.assertFalse(result.ok)
        self.assertIn("existing_vendor_processes", {c.name for c in result.checks if not c.ok})

    def test_flow_and_divider_contracts_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = self.adapter.preflight(Path(td) / "run", dry_run=True)
        self.assertTrue(result.ok, result.as_dict())
        names = {check.name for check in result.checks}
        self.assertIn("flow_clock_contract", names)
        self.assertIn("device_divider_contract", names)

    def test_a_target_clock_reaches_the_flow(self) -> None:
        # A target names its own clock (R40's three-core image runs at 90 MHz).
        # The flow takes it from TNCA_KERNEL_MHZ, so the preflight accepts a
        # clock other than the flow's default and every command carries it.
        seen: list[dict[str, str] | None] = []

        def executor(argv, *, cwd, env, log_path, timeout_s):
            seen.append(env)
            return CommandResult(tuple(argv), 1, "", "", str(log_path), False)

        adapter = U50Adapter(U50Config(REPO, expected_ap_clk_mhz=90, expected_device_divider=self.divider),
                             executor=executor)
        with tempfile.TemporaryDirectory() as td:
            result = adapter.preflight(Path(td) / "run", dry_run=True)
            self.assertTrue(result.ok, result.as_dict())
            adapter.run(Path(td) / "run", mode="package", require_vendor_tools=False)
        self.assertEqual(adapter.command_environment()["TNCA_KERNEL_MHZ"], "90")
        self.assertTrue(seen)
        self.assertEqual(seen[0]["TNCA_KERNEL_MHZ"], "90")

    def test_a_fixed_flow_clock_must_match_the_target(self) -> None:
        self.assertEqual(flow_clock("EXPECT_MHZ=100\n"), (100, False))
        self.assertEqual(flow_clock("EXPECT_MHZ=${TNCA_KERNEL_MHZ:-100}\n"), (100, True))
        self.assertEqual(flow_clock("no clock here\n"), (None, False))


if __name__ == "__main__":
    unittest.main()
