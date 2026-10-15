#!/usr/bin/env python3
"""Low-cost checks for the existing U50 vendor-flow boundary.

These checks deliberately do not require Vivado/Vitis or a license.  They
exercise the part of the adapter that must be safe on every host: syntax
validation and fail-closed preflight before any vendor process is launched.
Run from the repository root with::

    python tools/tritonchip_fpga_harness/tests/test_u50_flow_guard.py

The harness's own lifecycle tests are next to this file.  This one is a
regression check on the flow script the harness wraps.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
LINK = REPO / "flow" / "fpga" / "link.sh"


class U50FlowGuardTests(unittest.TestCase):
    def test_link_script_is_shell_valid(self) -> None:
        proc = subprocess.run(
            ["bash", "-n", str(LINK)],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_xo_is_rejected_before_vendor_launch(self) -> None:
        # link.sh checks for the package output before sourcing Vitis or
        # resolving v++.  An empty temporary directory therefore gives a
        # deterministic, license-free preflight test.
        with tempfile.TemporaryDirectory(prefix="tnca-u50-preflight-") as td:
            env = os.environ.copy()
            # If a future wrapper adds a mock vendor launcher, this marker lets
            # the test prove that it was not called on a failed preflight.
            env["TNCA_HARNESS_PREVENT_VENDOR_LAUNCH"] = "1"
            proc = subprocess.run(
                ["bash", str(LINK), td],
                cwd=REPO,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertFalse((Path(td) / "tnca_u50.xclbin").exists())
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("no ", proc.stderr)
        self.assertIn("tnca_u50.xo", proc.stderr)
        self.assertNotIn("v++", proc.stderr)
        self.assertNotIn("v++", proc.stdout)

    def test_u50_clock_contract_is_visible_in_flow(self) -> None:
        text = LINK.read_text(encoding="utf-8")
        expected_clock = re.search(r"(?m)^EXPECT_MHZ=(\d+)$", text)
        self.assertIsNotNone(expected_clock)
        self.assertIn(f"EXPECT_MHZ={expected_clock.group(1)}", text)
        self.assertIn("--kernel_frequency 0:$EXPECT_MHZ", text)
        # The board wrapper divider is part of the flow contract; the harness
        # must retain it in the run manifest instead of inferring the device
        # clock from the kernel clock alone.
        wrapper = REPO / "rtl" / "board" / "tnca_u50_xo.v"
        wrapper_text = wrapper.read_text(encoding="utf-8")
        self.assertRegex(wrapper_text, r"(?:parameter\s+DEV_DIV\s*=\s*\d+|BUFGCE_DIVIDE\s*\(\s*\d+\s*\))")


if __name__ == "__main__":
    unittest.main(verbosity=2)
