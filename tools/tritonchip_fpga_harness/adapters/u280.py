"""U280 adapter for the board-selectable TritonChip FPGA flow.

The PR5 flow keeps the RTL/package top named ``tnca_u50`` for compatibility,
but selects the U280 platform, SLR placement and output xclbin through
``TNCA_BOARD=u280``.  This adapter makes those choices explicit in the run
manifest and reuses the U50 adapter's fail-closed artifact checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .u50 import U50Adapter, U50Config


class U280FlowAdapter:
    name = "u280"

    def _impl(self, root: Path, profile: Any) -> U50Adapter:
        data = profile.data
        expected_artifacts = list(data["expected_artifacts"])
        environment = {
            "TNCA_BOARD": "u280",
            "XILINX_VITIS": str(data.get("vitis_root", "/tools/Xilinx/Vitis/2023.2")),
            "XILINX_PLATFORMS": str(data.get("platforms_root", "/opt/xilinx/platforms")),
        }
        environment.update({str(k): str(v) for k, v in data.get("environment", {}).items()})
        return U50Adapter(
            U50Config(
                repo=root,
                vitis_root=Path(environment["XILINX_VITIS"]),
                platforms_root=Path(environment["XILINX_PLATFORMS"]),
                platform=str(data["platform"]),
                expected_ap_clk_mhz=int(profile.expected_clock),
                expected_device_divider=profile.expected_device_divider,
                require_routed_timing=bool(data.get("require_routed_timing", False)),
                require_drc=bool(data.get("require_drc", False)),
                board_name="u280",
                package_part=str(data["part"]),
                required_board_options=str(data.get("board_vpp_opts", "")) or None,
                xo_name=str(expected_artifacts[0]),
                xclbin_name=str(expected_artifacts[1]),
                allowed_auto_frequency_rules=tuple(data.get("allowed_auto_frequency_rules", [])),
                environment=environment,
                package_script=root / str(data["package_script"]),
                link_script=root / str(data["link_script"]),
            )
        )

    def doctor(self, root: Path, profile: Any) -> list[dict[str, Any]]:
        impl = self._impl(root, profile)
        result = impl.preflight(root.parent / ".tritonchip-fpga-harness-doctor", stage="full", dry_run=False)
        return [{"name": c.name, "ok": c.ok, "detail": c.detail, "severity": c.severity} for c in result.checks]

    def run(self, root: Path, run_dir: Path, profile: Any, mode: str, timeout_s: int | None) -> list[dict[str, Any]]:
        if mode == "ooc":
            raise RuntimeError("the current U280 adapter supports full implementation only")
        build = run_dir / "artifacts"
        impl = self._impl(root, profile)
        result = impl.run(build, mode="full", timeout_s=timeout_s)
        stages: list[dict[str, Any]] = []
        for command in result.commands:
            name = "package" if command.argv and command.argv[0] == impl.config.vivado else "link"
            stages.append({
                "name": name,
                "returncode": command.returncode,
                "log": str(Path(command.log_path).relative_to(run_dir)),
                "timed_out": command.timed_out,
                "resource": command.resource,
            })
        from ..schema import dump_json
        dump_json(run_dir / "preflight.json", result.preflight.as_dict())
        if result.artifacts is not None:
            dump_json(run_dir / "artifact_validation.json", result.artifacts.as_dict())
        return stages

    def validate_existing(self, root: Path, profile: Any, build_dir: Path) -> dict[str, Any]:
        return self._impl(root, profile).validate_artifacts(build_dir, require_xclbin=True).as_dict()


__all__ = ["U280FlowAdapter"]
