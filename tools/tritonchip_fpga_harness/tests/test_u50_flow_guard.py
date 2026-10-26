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

    def test_kernel_clock_is_checked_before_vendor_launch(self) -> None:
        # A target's clock reaches link.sh as TNCA_KERNEL_MHZ. A malformed or
        # out-of-range one stops before anything else; a valid one goes on to
        # the next preflight (here, the missing XO).
        for value, expected in (("abc", "must be a whole number of MHz"),
                                ("9", "outside the platform's 10..650 MHz"),
                                ("90", "tnca_u50.xo")):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory(prefix="tnca-clock-") as td:
                    env = dict(os.environ, TNCA_KERNEL_MHZ=value)
                    env.pop("TNCA_BOARD", None)
                    proc = subprocess.run(["bash", str(LINK), td], cwd=REPO, env=env,
                                          text=True, capture_output=True, check=False)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn(expected, proc.stderr)

    def test_packaged_variant_file_is_parsed(self) -> None:
        # package_xo.tcl writes the variant it packaged next to the XO. The
        # on-chip build used to write an empty line, which link.sh refused as
        # an unknown variant: every default (single-core U280, U50) build
        # stopped before v++. A known variant must reach the next preflight
        # (the missing XO); an unknown one must not.
        cases = [
            ("\n", None, "no "),
            ("", None, "no "),
            ("none\n", None, "no "),
            ("hbm\n", None, "no "),
            ("multicore-hbm\n", "u280", "no "),
            ("multicore-hbm\n", "u50", "multicore-hbm requires TNCA_BOARD=u280"),
            ("bogus\n", None, "unknown packaged variant bogus"),
        ]
        for content, board, expected in cases:
            with self.subTest(content=content, board=board):
                with tempfile.TemporaryDirectory(prefix="tnca-variant-") as td:
                    (Path(td) / "tnca_variant.txt").write_text(content, encoding="utf-8")
                    env = os.environ.copy()
                    env.pop("TNCA_BOARD", None)
                    if board:
                        env["TNCA_BOARD"] = board
                    proc = subprocess.run(
                        ["bash", str(LINK), td],
                        cwd=REPO,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn(expected, proc.stderr)
                if expected == "no ":
                    self.assertNotIn("unknown packaged variant", proc.stderr)
                    self.assertIn("tnca_u50.xo", proc.stderr)
                self.assertNotIn("v++", proc.stdout)

    def test_a_recheck_takes_the_clock_its_link_asked_for(self) -> None:
        # LINK_RECHECK=1 re-runs the checks on a finished link. A 90 MHz link
        # rechecked without TNCA_KERNEL_MHZ must be judged at 90, as its log
        # records, and one rechecked at another clock is refused. The vendor
        # tools are stand-ins: v++ is never run on a recheck, and xclbinutil
        # reports the clock the link left.
        with tempfile.TemporaryDirectory(prefix="tnca-recheck-") as td:
            root = Path(td)
            vitis, platforms, out = root / "vitis", root / "platforms", root / "out"
            (vitis / "bin").mkdir(parents=True)
            (platforms / "xilinx_u280_gen3x16_xdma_1_202211_1").mkdir(parents=True)
            out.mkdir()
            (vitis / "settings64.sh").write_text(f'export PATH="{vitis}/bin:$PATH"\n', encoding="utf-8")
            for tool, body in (("v++", "exit 0"),
                               ("xclbinutil", "printf 'Scalable Clocks\\n   Name: DATA_CLK\\n   Frequency: 90 MHz\\n"
                                              "System Clocks\\n'")):
                (vitis / "bin" / tool).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
                (vitis / "bin" / tool).chmod(0o755)
            (out / "tnca_u50.xo").write_bytes(b"xo")
            (out / "tnca_variant.txt").write_text("multicore-hbm\n", encoding="utf-8")
            (out / "tnca_u280.xclbin").write_bytes(b"xclbin")
            (out / "link.log").write_text(
                "=== LINK START ===\nEXPECT_MHZ=90\n---\nINFO: [v++ 60-991] clock name 'ulp_ucs/aclk_kernel_00' "
                "(clock ID '0') is being mapped to clock name 'DATA_CLK' in the xclbin\nVPP_RC=0\n", encoding="utf-8")
            env = dict(os.environ, LINK_RECHECK="1", TNCA_BOARD="u280", XILINX_VITIS=str(vitis),
                       XILINX_PLATFORMS=str(platforms))
            env.pop("TNCA_KERNEL_MHZ", None)
            ok = subprocess.run(["bash", str(LINK), str(out)], cwd=REPO, env=env, text=True,
                                capture_output=True, check=False)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertIn("kernel clock 90 MHz, the three-core device on it undivided", ok.stdout)
            other = subprocess.run(["bash", str(LINK), str(out)], cwd=REPO, env=dict(env, TNCA_KERNEL_MHZ="100"),
                                   text=True, capture_output=True, check=False)
            self.assertEqual(other.returncode, 2, other.stdout + other.stderr)
            self.assertIn("that link asked for 90 MHz", other.stderr)

    def test_u50_clock_contract_is_visible_in_flow(self) -> None:
        text = LINK.read_text(encoding="utf-8")
        # The clock is the target's (TNCA_KERNEL_MHZ), 100 MHz when unset.
        expected_clock = re.search(r"(?m)^EXPECT_MHZ=\$\{TNCA_KERNEL_MHZ:-(\d+)\}$", text)
        self.assertIsNotNone(expected_clock)
        self.assertEqual(expected_clock.group(1), "100")
        self.assertIn("--kernel_frequency 0:$EXPECT_MHZ", text)
        package = (REPO / "flow" / "fpga" / "package_xo.tcl").read_text(encoding="utf-8")
        self.assertIn("::env(TNCA_KERNEL_MHZ)", package)
        # The board wrapper divider is part of the flow contract; the harness
        # must retain it in the run manifest instead of inferring the device
        # clock from the kernel clock alone.
        wrapper = REPO / "rtl" / "board" / "tnca_u50_xo.v"
        wrapper_text = wrapper.read_text(encoding="utf-8")
        self.assertRegex(wrapper_text, r"(?:parameter\s+DEV_DIV\s*=\s*\d+|BUFGCE_DIVIDE\s*\(\s*\d+\s*\))")


if __name__ == "__main__":
    unittest.main(verbosity=2)
