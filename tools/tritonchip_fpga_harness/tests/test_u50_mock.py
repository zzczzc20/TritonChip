#!/usr/bin/env python3
"""Mock end-to-end checks for the U50 adapter.

The adapter must be testable without a Vivado license.  The executor below
stands in for the two vendor commands, writes the same small set of outputs
that the real flow is expected to produce, and lets the adapter perform its
normal artifact checks.  No files are written into the repository checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tritonchip_fpga_harness.adapters.u50 import (
    CommandResult,
    U50Adapter,
    U50Config,
    execute_command,
)


def _fake_tool(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def _command_result(argv: tuple[str, ...], log_path: Path, rc: int = 0) -> CommandResult:
    return CommandResult(
        argv=argv,
        returncode=rc,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        log_path=str(log_path),
    )


class U50MockFlowTests(unittest.TestCase):
    def test_owned_process_resource_sample_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tnca-harness-monitor-") as td:
            result = execute_command(
                ("/bin/sh", "-c", "sleep 0.05"),
                cwd=Path(td),
                env=None,
                log_path=Path(td) / "command.log",
                timeout_s=5,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("sample_count", result.resource)
        self.assertIn("peak_owned_processes", result.resource)

    def _config(self, root: Path) -> tuple[U50Config, Path]:
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for tool in ("vivado", "v++", "xclbinutil"):
            _fake_tool(bin_dir, tool)
        vitis = root / "vitis"
        (vitis / "settings64.sh").parent.mkdir(parents=True)
        (vitis / "settings64.sh").write_text("# mock\n", encoding="utf-8")
        platforms = root / "platforms"
        (platforms / "xilinx_u50_gen3x16_xdma_5_202210_1").mkdir(parents=True)
        link_text = (REPO / "flow/fpga/link.sh").read_text(encoding="utf-8")
        clock = int(re.search(r"(?m)^EXPECT_MHZ=(\d+)$", link_text).group(1))
        wrapper_text = (REPO / "rtl/board/tnca_u50_xo.v").read_text(encoding="utf-8")
        divider_match = re.search(r"parameter\s+DEV_DIV\s*=\s*(\d+)|BUFGCE_DIVIDE\s*\((\d+)\)", wrapper_text)
        divider = int(next(value for value in divider_match.groups() if value is not None))
        config = U50Config(
            repo=REPO,
            vivado="vivado",
            vitis_root=vitis,
            platforms_root=platforms,
            expected_ap_clk_mhz=clock,
            expected_device_divider=divider,
        )
        return config, bin_dir

    def test_dry_run_records_commands_without_fabricating_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tnca-harness-mock-") as td:
            root = Path(td)
            config, bin_dir = self._config(root)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
            try:
                build = root / "run"  # outside the source checkout
                result = U50Adapter(config).run(build, mode="full", dry_run=True)
            finally:
                os.environ["PATH"] = old_path

        self.assertTrue(result.preflight.ok)
        self.assertTrue(result.dry_run)
        self.assertEqual(len(result.commands), 2)
        self.assertEqual(result.commands[0].argv[0], "vivado")
        self.assertTrue(result.commands[1].argv[0].endswith("flow/fpga/link.sh"))
        # A plan is not a build: no .xo/.xclbin must be reported as valid.
        self.assertIsNotNone(result.artifacts)
        self.assertFalse(result.artifacts.ok)

    def test_mock_package_and_link_pass_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tnca-harness-mock-") as td:
            root = Path(td)
            config, bin_dir = self._config(root)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
            build = root / "run"

            def executor(argv, *, cwd, env, log_path, timeout_s):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if argv[0] == "vivado":
                    (build / "tnca_u50.xo").write_bytes(b"mock-xo")
                else:
                    (build / "tnca_u50.xclbin").write_bytes(b"mock-xclbin")
                    (build / "link.log").write_text(
                        "clock name 'ulp_ucs/aclk_kernel_00' (clock ID '0') "
                        "is being mapped to clock name 'DATA_CLK'\n"
                        "VPP_RC=0\nLINK_OK\n",
                        encoding="utf-8",
                    )
                    (build / "xclbin.info").write_text(
                        f"Name: DATA_CLK\nFrequency: {config.expected_ap_clk_mhz} MHz\n",
                        encoding="utf-8",
                    )
                return _command_result(tuple(str(x) for x in argv), log_path)

            try:
                result = U50Adapter(config, executor=executor).run(build, mode="full")
            finally:
                os.environ["PATH"] = old_path

        self.assertTrue(result.preflight.ok)
        self.assertEqual(len(result.commands), 2)
        self.assertIsNotNone(result.artifacts)
        self.assertTrue(result.artifacts.ok, result.artifacts.as_dict())
        self.assertEqual(result.artifacts.observed_clock_mhz, config.expected_ap_clk_mhz)
        self.assertEqual(result.artifacts.mapped_clock_name, "DATA_CLK")
        self.assertRegex(result.artifacts.artifacts["xclbin"]["sha256"], r"^[0-9a-f]{64}$")

    def test_zero_rc_without_xclbin_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tnca-harness-mock-") as td:
            root = Path(td)
            config, bin_dir = self._config(root)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
            build = root / "run"

            def executor(argv, *, cwd, env, log_path, timeout_s):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if argv[0] == "vivado":
                    (build / "tnca_u50.xo").write_bytes(b"mock-xo")
                else:
                    # Reproduce the known wrapper failure mode: VPP_RC=0 but
                    # no xclbin.  The adapter must not trust rc alone.
                    (build / "link.log").write_text(
                        "VPP_RC=0\nLINK_OK\n", encoding="utf-8"
                    )
                return _command_result(tuple(str(x) for x in argv), log_path)

            try:
                result = U50Adapter(config, executor=executor).run(build, mode="full")
            finally:
                os.environ["PATH"] = old_path

        self.assertIsNotNone(result.artifacts)
        self.assertFalse(result.artifacts.ok)
        checks = {c.name: c for c in result.artifacts.checks}
        self.assertFalse(checks["xclbin"].ok)
        self.assertFalse(checks["missing_xclbin_after_zero_rc"].ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
