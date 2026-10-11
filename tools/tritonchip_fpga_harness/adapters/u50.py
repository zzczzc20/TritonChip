"""U50 adapter for TritonChip-FPGA-Harness.

The adapter is intentionally independent from the repository's historical
``flow/fpga`` scripts.  It *calls* those scripts and verifies their outputs;
it does not duplicate Vivado/Vitis implementation logic.  This gives an RTL
candidate a stable interface while keeping vendor-specific details in the
existing flow.

No command is executed on import.  The public methods are useful to both the
harness worker and an offline/mock verifier.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


ADAPTER_VERSION = "u50-adapter-v0.2"


class CommandExecutor(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None,
        log_path: Path,
        timeout_s: float | None,
    ) -> "CommandResult": ...


@dataclass(frozen=True)
class U50Config:
    """Resolved paths and fixed U50 flow identity.

    ``repo`` must be the source snapshot under test, not a separate working
    tree.  The adapter records its commit in the manifest so a run cannot be
    confused with a later RTL change.
    """

    repo: Path
    vivado: str = "vivado"
    vitis_root: Path = Path("/home/x4/Software/Xilinx/Vitis/2022.2")
    platforms_root: Path = Path("/opt/xilinx/platforms")
    platform: str = "xilinx_u50_gen3x16_xdma_5_202210_1"
    expected_ap_clk_mhz: int = 150
    expected_device_divider: int = 4
    board_name: str | None = None
    package_part: str | None = None
    required_board_options: str | None = None
    xo_name: str = "tnca_u50.xo"
    xclbin_name: str = "tnca_u50.xclbin"
    allowed_auto_frequency_rules: tuple[str, ...] = ()
    require_routed_timing: bool = False
    require_drc: bool = False
    environment: dict[str, str] = field(default_factory=dict)
    package_script: Path | None = None
    link_script: Path | None = None

    def resolved(self) -> "U50Config":
        repo = self.repo.expanduser().resolve()
        package = self.package_script or repo / "flow/fpga/package_xo.tcl"
        link = self.link_script or repo / "flow/fpga/link.sh"
        return U50Config(
            repo=repo,
            vivado=self.vivado,
            vitis_root=self.vitis_root.expanduser().resolve(),
            platforms_root=self.platforms_root.expanduser().resolve(),
            platform=self.platform,
            expected_ap_clk_mhz=self.expected_ap_clk_mhz,
            expected_device_divider=self.expected_device_divider,
            board_name=self.board_name,
            package_part=self.package_part,
            required_board_options=self.required_board_options,
            xo_name=self.xo_name,
            xclbin_name=self.xclbin_name,
            allowed_auto_frequency_rules=tuple(self.allowed_auto_frequency_rules),
            require_routed_timing=self.require_routed_timing,
            require_drc=self.require_drc,
            environment=dict(self.environment),
            package_script=Path(package).expanduser().resolve(),
            link_script=Path(link).expanduser().resolve(),
        )

    @property
    def platform_path(self) -> Path:
        return self.platforms_root / self.platform

    @property
    def settings64(self) -> Path:
        return self.vitis_root / "settings64.sh"


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    started_at: str
    finished_at: str
    log_path: str
    timed_out: bool = False
    resource: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "error"


@dataclass
class PreflightResult:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
            "warnings": list(self.warnings),
        }


@dataclass
class ArtifactValidation:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    observed_clock_mhz: int | None = None
    mapped_clock_name: str | None = None
    timing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(c) for c in self.checks],
            "artifacts": self.artifacts,
            "observed_clock_mhz": self.observed_clock_mhz,
            "mapped_clock_name": self.mapped_clock_name,
            "timing": self.timing,
        }


@dataclass
class U50Run:
    run_id: str
    mode: str
    build_dir: str
    source_commit: str | None
    adapter_version: str
    preflight: PreflightResult
    commands: list[CommandResult] = field(default_factory=list)
    artifacts: ArtifactValidation | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "build_dir": self.build_dir,
            "source_commit": self.source_commit,
            "adapter_version": self.adapter_version,
            "dry_run": self.dry_run,
            "preflight": self.preflight.as_dict(),
            "commands": [asdict(c) for c in self.commands],
            "artifacts": self.artifacts.as_dict() if self.artifacts else None,
        }


class U50Adapter:
    """Call and verify the repository's U50 package/link flow."""

    def __init__(
        self,
        config: U50Config,
        *,
        executor: CommandExecutor | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config.resolved()
        self._executor = executor or execute_command
        self._clock = clock or time.time

    def source_commit(self) -> str | None:
        """Return the exact source SHA, or ``None`` for a non-git snapshot."""
        try:
            p = subprocess.run(
                ["git", "-C", str(self.config.repo), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    def package_command(self, build_dir: Path) -> tuple[str, ...]:
        c = self.config
        command = (
            c.vivado,
            "-mode",
            "batch",
            "-source",
            str(c.package_script),
            "-tclargs",
            str(c.repo),
            str(build_dir),
        )
        return command + ((c.package_part,) if c.package_part else ())

    def link_command(self, build_dir: Path) -> tuple[str, ...]:
        return (str(self.config.link_script), str(build_dir))

    def preflight(
        self,
        build_dir: Path,
        *,
        stage: str = "full",
        dry_run: bool = False,
        require_vendor_tools: bool = True,
    ) -> PreflightResult:
        """Check all cheap gates before an expensive implementation run.

        ``dry_run`` keeps missing Vivado/Vitis/platform entries as warnings,
        while still failing on malformed repository paths and invalid target
        parameters.  A real run fails closed before spawning a vendor process.
        """
        c = self.config
        b = Path(build_dir).expanduser().resolve()
        checks: list[Check] = []
        warnings: list[str] = []

        def add(name: str, ok: bool, detail: str, severity: str = "error") -> None:
            checks.append(Check(name, ok, detail, severity))

        add("repo", c.repo.is_dir(), str(c.repo))
        add("package_script", stage == "link" or bool(c.package_script and c.package_script.is_file()), str(c.package_script))
        add("link_script", stage == "package" or bool(c.link_script and c.link_script.is_file()), str(c.link_script))
        add("build_dir", b != c.repo and b.is_absolute(), str(b))
        if b.exists():
            add("build_dir_writable", os.access(b, os.W_OK), str(b))
        else:
            parent = b.parent
            add("build_parent_writable", parent.is_dir() and os.access(parent, os.W_OK), str(parent))

        if stage not in {"package", "link", "full"}:
            add("stage", False, f"unsupported stage {stage!r}")
        else:
            add("stage", True, stage)
        add("clock_positive", c.expected_ap_clk_mhz > 0, str(c.expected_ap_clk_mhz))
        add("divider_range", 1 <= c.expected_device_divider <= 8, str(c.expected_device_divider))

        # The profile is a contract for the checked-in flow, not a free-form
        # label.  Catch a stale profile before spending hours in Vivado.
        flow_text = self._read_text(c.link_script or Path(""))
        flow_match = re.search(r"(?m)^EXPECT_MHZ=(\d+)\s*$", flow_text)
        add(
            "flow_clock_contract",
            flow_match is not None and int(flow_match.group(1)) == c.expected_ap_clk_mhz,
            (f"EXPECT_MHZ={flow_match.group(1)}" if flow_match else "EXPECT_MHZ missing")
            + f", expected {c.expected_ap_clk_mhz}",
        )
        package_text = self._read_text(c.package_script or Path(""))
        package_match = re.search(r"(?m)^\s*set\s+AP_CLK_HZ\s+(\d+)\s*$", package_text)
        add(
            "package_clock_contract",
            package_match is not None and int(package_match.group(1)) == c.expected_ap_clk_mhz * 1_000_000,
            (f"AP_CLK_HZ={package_match.group(1)}" if package_match else "AP_CLK_HZ missing")
            + f", expected {c.expected_ap_clk_mhz * 1_000_000}",
        )
        xdc = c.repo / "flow/fpga/ooc.xdc"
        xdc_text = self._read_text(xdc)
        period_match = re.search(r"(?m)^\s*create_clock\s+-period\s+([0-9.]+)", xdc_text)
        expected_period = 1000.0 / c.expected_ap_clk_mhz
        period_ok = period_match is not None and abs(float(period_match.group(1)) - expected_period) < 0.002
        add(
            "ooc_clock_contract",
            period_ok,
            (f"period={period_match.group(1)}" if period_match else "create_clock missing")
            + f", expected {expected_period:.3f}",
        )
        wrapper = c.repo / "rtl/board/tnca_u50_xo.v"
        wrapper_text = self._read_text(wrapper)
        div_match = re.search(
            r"(?:parameter\s+DEV_DIV\s*=|BUFGCE_DIVIDE\s*\()\s*([0-9]+)",
            wrapper_text,
        )
        add(
            "device_divider_contract",
            div_match is not None and int(div_match.group(1)) == c.expected_device_divider,
            (f"DEV_DIV={div_match.group(1)}" if div_match else "device divider missing")
            + f", expected {c.expected_device_divider}",
        )
        if c.board_name:
            boards = c.repo / "flow/fpga/boards.sh"
            boards_text = self._read_text(boards)
            board_ok = re.search(rf"(?m)^\s*{re.escape(c.board_name)}\)\s*$", boards_text) is not None
            add("board_contract", board_ok, f"{c.board_name} in {boards}")
            if c.required_board_options:
                options_ok = c.required_board_options in boards_text
                add(
                    "board_options_contract",
                    options_ok,
                    f"required {c.required_board_options!r} in {boards}",
                )

        if stage in {"package", "full"}:
            vivado_path = shutil.which(c.vivado)
            ok = vivado_path is not None
            if not ok and (dry_run or not require_vendor_tools):
                warnings.append(f"vivado unavailable (planning mode): {c.vivado}")
            add("vivado", ok or dry_run or not require_vendor_tools, vivado_path or c.vivado,
                "warning" if (dry_run or not require_vendor_tools) and not ok else "error")

        if stage in {"link", "full"}:
            settings_ok = c.settings64.is_file()
            platform_ok = c.platform_path.is_dir()
            if not settings_ok and (dry_run or not require_vendor_tools):
                warnings.append(f"Vitis settings unavailable (planning mode): {c.settings64}")
            if not platform_ok and (dry_run or not require_vendor_tools):
                board_label = c.board_name.upper() if c.board_name else "U50"
                warnings.append(f"{board_label} platform unavailable (planning mode): {c.platform_path}")
            add("vitis_settings", settings_ok or dry_run or not require_vendor_tools, str(c.settings64),
                "warning" if (dry_run or not require_vendor_tools) and not settings_ok else "error")
            add("platform", platform_ok or dry_run or not require_vendor_tools, str(c.platform_path),
                "warning" if (dry_run or not require_vendor_tools) and not platform_ok else "error")
            # v++ and xclbinutil are provided after settings64 is sourced by
            # link.sh; checking the host PATH here catches obvious local
            # mistakes, but does not replace the script's post-source check.
            for tool in ("v++", "xclbinutil"):
                found = shutil.which(tool)
                if not found and (dry_run or not require_vendor_tools):
                    warnings.append(f"{tool} unavailable before settings64 (planning mode)")
                add(tool, bool(found) or dry_run or not require_vendor_tools, found or tool,
                    "warning" if (dry_run or not require_vendor_tools) and not found else "error")

        if stage == "full":
            existing = _existing_vendor_processes()
            allow_existing = os.environ.get("FPGA_HARNESS_ALLOW_EXISTING_VENDOR_PROCS") == "1"
            add(
                "existing_vendor_processes",
                not existing or allow_existing or dry_run,
                "none" if not existing else "; ".join(existing[:8]),
                "warning" if existing and (allow_existing or dry_run) else "error",
            )

        # The build output must never be inside the source checkout: a stale
        # XO/XCLBIN there can make a candidate appear to have been built.
        try:
            b.relative_to(c.repo)
            inside_repo = True
        except ValueError:
            inside_repo = False
        add("build_outside_repo", not inside_repo, str(b))

        return PreflightResult(ok=all(x.ok for x in checks if x.severity == "error"), checks=checks, warnings=warnings)

    def run(
        self,
        build_dir: Path,
        *,
        mode: str = "full",
        dry_run: bool = False,
        timeout_s: float | None = None,
        require_vendor_tools: bool = True,
        run_id: str | None = None,
    ) -> U50Run:
        """Execute package/link, or return a complete deterministic dry-run plan."""
        b = Path(build_dir).expanduser().resolve()
        if mode not in {"package", "link", "full"}:
            raise ValueError(f"unsupported mode: {mode}")
        rid = run_id or f"u50-{int(self._clock())}"
        pf = self.preflight(
            b,
            stage=mode,
            dry_run=dry_run,
            require_vendor_tools=require_vendor_tools,
        )
        result = U50Run(rid, mode, str(b), self.source_commit(), ADAPTER_VERSION, pf, dry_run=dry_run)
        if not pf.ok:
            result.artifacts = self.validate_artifacts(b, require_xclbin=(mode != "package"))
            return result

        commands: list[tuple[str, ...]] = []
        if mode in {"package", "full"}:
            commands.append(self.package_command(b))
        if mode in {"link", "full"}:
            commands.append(self.link_command(b))
        if dry_run:
            # Record planned commands in a stable shape without fabricating
            # success or artifact files.
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock()))
            result.commands = [CommandResult(tuple(x), 0, now, now, "", False) for x in commands]
            result.artifacts = self.validate_artifacts(b, require_xclbin=(mode != "package"))
            return result

        b.mkdir(parents=True, exist_ok=True)
        for i, argv in enumerate(commands):
            log = b / ("package.log" if argv[0] == self.config.vivado else "adapter-link.log")
            cr = self._executor(
                argv,
                cwd=self.config.repo,
                env=self.config.environment or None,
                log_path=log,
                timeout_s=timeout_s,
            )
            result.commands.append(cr)
            # The next stage must never consume a missing/stale XO. Validate
            # package output before invoking link.
            if i == 0 and mode == "full":
                package_check = self.validate_artifacts(b, require_xclbin=False)
                if not package_check.ok:
                    result.artifacts = package_check
                    return result
            if cr.returncode != 0 or cr.timed_out:
                result.artifacts = self.validate_artifacts(b, require_xclbin=(mode == "link" or i > 0))
                return result

        result.artifacts = self.validate_artifacts(b, require_xclbin=(mode in {"full", "link"}))
        return result

    def validate_artifacts(self, build_dir: Path, *, require_xclbin: bool = True) -> ArtifactValidation:
        """Verify output identity and fail closed on a missing artifact.

        This intentionally does not trust a shell exit code.  In particular,
        an implementation wrapper returning zero while omitting the xclbin is
        represented as ``missing_xclbin`` and fails validation.
        """
        c = self.config
        b = Path(build_dir).expanduser().resolve()
        checks: list[Check] = []
        artifacts: dict[str, dict[str, Any]] = {}

        def add(name: str, ok: bool, detail: str) -> None:
            checks.append(Check(name, ok, detail))

        xo = b / c.xo_name
        xclbin = b / c.xclbin_name
        link_log = b / "link.log"
        info = b / "xclbin.info"
        self._record_file(artifacts, "xo", xo)
        self._record_file(artifacts, "xclbin", xclbin)
        self._record_file(artifacts, "link_log", link_log)
        self._record_file(artifacts, "xclbin_info", info)

        add("xo", xo.is_file() and xo.stat().st_size > 0, str(xo))
        if require_xclbin:
            add("xclbin", xclbin.is_file() and xclbin.stat().st_size > 0, str(xclbin))
        else:
            add("xclbin", True, "not required for package stage")

        log_text = self._read_text(link_log)
        if require_xclbin and link_log.is_file():
            rc_match = re.findall(r"VPP_RC=(-?\d+)", log_text)
            if rc_match:
                vpp_rc = int(rc_match[-1])
                add("vpp_rc", vpp_rc == 0, str(vpp_rc))
                # Explicitly reject the known wrapper bug: rc 0 + absent file.
                if vpp_rc == 0 and not xclbin.is_file():
                    add("missing_xclbin_after_zero_rc", False, "link.log says VPP_RC=0 but tnca_u50.xclbin is absent")
            else:
                add("vpp_rc", False, "link.log has no VPP_RC marker")
            add("link_ok_marker", "LINK_OK" in log_text, "LINK_OK" if "LINK_OK" in log_text else "missing LINK_OK")
            scaling_rules = sorted(set(re.findall(r"AUTO-FREQ-SCALING(?:-\d+)?", log_text)))
            bad_scaling = [rule for rule in scaling_rules if rule not in c.allowed_auto_frequency_rules]
            add(
                "no_auto_frequency_scaling",
                not bad_scaling,
                "clean" if not scaling_rules else f"rules={scaling_rules}, rejected={bad_scaling}",
            )
        elif require_xclbin:
            add("link_log", False, f"missing {link_log}")

        observed, mapped = self._parse_clock(log_text, self._read_text(info), c.expected_ap_clk_mhz)
        if require_xclbin:
            add("clock_info", info.is_file() and observed is not None, f"{mapped or 'unknown'}={observed if observed is not None else 'unknown'} MHz")
            add("expected_clock", observed == c.expected_ap_clk_mhz, f"got {observed if observed is not None else 'unknown'} MHz, expected {c.expected_ap_clk_mhz} MHz")

        if require_xclbin and xclbin.is_file() and xclbin.stat().st_size == 0:
            add("nonempty_xclbin", False, str(xclbin))

        timing = self._timing_reports(b)
        if require_xclbin and self.config.require_routed_timing:
            add("routed_timing_report", timing.get("report") is not None, str(timing.get("report") or "missing"))
            if timing.get("report") is not None:
                metrics = timing.get("metrics", {})
                for key in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "failing_endpoints"):
                    add(f"timing_{key}", key in metrics, f"{key}={metrics.get(key, 'missing')}")
                if all(key in metrics for key in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "failing_endpoints")):
                    add(
                        "timing_clean",
                        metrics["wns_ns"] >= 0 and metrics["tns_ns"] >= 0 and metrics["whs_ns"] >= 0
                        and metrics["ths_ns"] >= 0 and metrics.get("wpws_ns", 0) >= 0
                        and metrics["failing_endpoints"] == 0,
                        json.dumps(metrics, sort_keys=True),
                    )
        if require_xclbin and self.config.require_drc:
            drc = self._drc_report(b)
            add("drc_report", drc.get("report") is not None, str(drc.get("report") or "missing"))
            if drc.get("report") is not None:
                add("drc_clean", drc.get("errors") == 0, f"errors={drc.get('errors', 'unknown')}")

        return ArtifactValidation(
            ok=all(c.ok for c in checks),
            checks=checks,
            artifacts=artifacts,
            observed_clock_mhz=observed,
            mapped_clock_name=mapped,
            timing=timing,
        )

    # Vivado's "Design Timing Summary" row, by position.  The table is the
    # first in a timing summary report whose header starts with WNS(ns); the
    # per-clock tables below it start with "Clock" and have the same columns
    # shifted by one.
    _DESIGN_SUMMARY = (
        "wns_ns", "tns_ns", "tns_failing_endpoints", "tns_total_endpoints",
        "whs_ns", "ths_ns", "ths_failing_endpoints", "ths_total_endpoints",
        "wpws_ns", "tpws_ns", "tpws_failing_endpoints", "tpws_total_endpoints",
    )

    @staticmethod
    def _design_timing_summary(text: str) -> dict[str, float | int] | None:
        """The whole design's worst slacks and failing endpoints, or None.

        This is the row to accept or reject a build on: it covers every clock
        and every crossing, setup, hold and pulse width.  A per-clock row for
        the kernel clock alone says nothing about a path into the shell.
        """
        lines = text.splitlines()
        for index, line in enumerate(lines):
            header = line.split()
            if not header or header[0] != "WNS(ns)" or "WHS(ns)" not in header or "THS(ns)" not in header:
                continue
            for row in lines[index + 1:index + 6]:
                fields = row.split()
                if not fields or set(row.strip()) <= {"-", " "}:
                    continue
                try:
                    values = [float(item) for item in fields]
                except ValueError:
                    return None
                if len(values) < 8:
                    return None
                metrics: dict[str, float | int] = {}
                for name, value in zip(U50Adapter._DESIGN_SUMMARY, values):
                    metrics[name] = int(value) if name.endswith("_endpoints") else value
                metrics["failing_endpoints"] = sum(
                    int(metrics.get(key, 0))
                    for key in ("tns_failing_endpoints", "ths_failing_endpoints", "tpws_failing_endpoints")
                )
                return metrics
            return None
        return None

    @staticmethod
    def _timing_reports(build_dir: Path) -> dict[str, Any]:
        candidates = sorted(
            path for path in build_dir.rglob("*") if path.is_file()
            and ("timing" in path.name.lower() or path.name.lower().endswith("timing_summary.rpt"))
            and path.suffix.lower() in {".rpt", ".txt", ".log"}
        )
        # Prefer the routed implementation report over pre-route summaries.
        candidates.sort(key=lambda p: ("routed" not in p.name.lower(), len(str(p))))
        for path in candidates:
            text = U50Adapter._read_text(path)
            design = U50Adapter._design_timing_summary(text)
            if design is not None:
                return {"report": str(path), "sha256": sha256(path), "metrics": design, "scope": "design"}
            # Not a Vivado timing summary.  The forms below are for a flow that
            # writes its own digest; they must never be tried on a Vivado
            # report, whose per-clock rows have other columns in these places.
            metrics: dict[str, float | int] = {}
            for key, pattern in {
                "wns_ns": r"\bWNS(?:\s*\(ns\))?\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)",
                "tns_ns": r"\bTNS(?:\s*\(ns\))?\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)",
                "whs_ns": r"\bWHS(?:\s*\(ns\))?\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)",
                "ths_ns": r"\bTHS(?:\s*\(ns\))?\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)",
            }.items():
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    metrics[key] = float(match.group(1))
            failed = re.search(r"(?:Failing\s+Endpoints|Violating\s+Endpoints|Endpoints\s+with\s+timing\s+violations)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
            if failed:
                metrics["failing_endpoints"] = int(failed.group(1))
            if "failing_endpoints" not in metrics:
                count = re.search(r"Number\s+of\s+Failing\s+Endpoints\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
                if count:
                    metrics["failing_endpoints"] = int(count.group(1))
            if "failing_endpoints" not in metrics:
                for line in text.splitlines():
                    fields = line.split()
                    # Exactly "<clock> WNS TNS WHS THS failing": a digest row.
                    if len(fields) == 6 and fields[0].lower() in {"ap_clk", "data_clk", "kernel_00"}:
                        try:
                            values = [float(item) for item in fields[1:6]]
                        except ValueError:
                            continue
                        if all(key not in metrics for key in ("wns_ns", "tns_ns", "whs_ns", "ths_ns")):
                            metrics.update({"wns_ns": values[0], "tns_ns": values[1], "whs_ns": values[2], "ths_ns": values[3]})
                        metrics["failing_endpoints"] = int(values[4])
                        break
            if metrics:
                return {"report": str(path), "sha256": sha256(path), "metrics": metrics, "scope": "digest"}
        return {"report": None, "metrics": {}}

    @staticmethod
    def _drc_report(build_dir: Path) -> dict[str, Any]:
        candidates = sorted(
            path for path in build_dir.rglob("*") if path.is_file()
            and "drc" in path.name.lower() and path.suffix.lower() in {".rpt", ".txt", ".log"}
        )
        for path in candidates:
            text = U50Adapter._read_text(path)
            match = re.search(r"(?:Number\s+of\s+Errors|Errors)\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
            if match:
                return {"report": str(path), "sha256": sha256(path), "errors": int(match.group(1)), "source": "drc-report"}
        # A Vitis link keeps no DRC report.  Vivado runs DRC as a precondition
        # to opt, place, route and write_bitstream and writes each verdict in
        # the implementation run's log; that is the evidence there is.
        for path in sorted(build_dir.rglob("runme.log")):
            if not any(part.startswith("impl_") for part in path.parts):
                continue
            text = U50Adapter._read_text(path)
            counts = [int(n) for n in re.findall(r"DRC finished with (\d+) Errors", text)]
            if counts:
                return {
                    "report": str(path), "sha256": sha256(path), "errors": max(counts),
                    "source": "implementation-log", "checks": len(counts),
                    "bitstream_precondition": "precondition to command write_bitstream" in text,
                }
        return {"report": None, "errors": None}

    @staticmethod
    def _record_file(dst: dict[str, dict[str, Any]], key: str, path: Path) -> None:
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            entry["size"] = path.stat().st_size
            entry["sha256"] = sha256(path)
        dst[key] = entry

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _parse_clock(log: str, info: str, expected: int) -> tuple[int | None, str | None]:
        # Existing link.sh obtains the xclbin clock name from this mapping line.
        mapped_match = re.findall(
            r"clock name '[^']*' \(clock ID '0'\) is being mapped to clock name '([^']+)'",
            log,
        )
        mapped = mapped_match[-1] if mapped_match else None
        if mapped:
            pattern = rf"Name:\s*{re.escape(mapped)}\s*.*?Frequency:\s*(\d+)\s*MHz"
            m = re.search(pattern, info, flags=re.S)
            if m:
                return int(m.group(1)), mapped
        # Mock fixtures and older xclbinutil output can omit the mapping line.
        # Only accept an unambiguous single frequency; never guess among clocks.
        matches = re.findall(r"Frequency:\s*(\d+)\s*MHz", info)
        if len(matches) == 1:
            return int(matches[0]), mapped
        return None, mapped


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _proc_boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return value or None
    except OSError:
        return None


def _proc_starttime(pid: int) -> str | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii").split()
        return fields[21]
    except (OSError, IndexError):
        return None


def _host_mem_available_kb() -> int | None:
    """Read Linux MemAvailable without adding a psutil dependency."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _owned_process_stats(root_pid: int) -> dict[str, Any]:
    """Return a best-effort snapshot of a process group and its descendants."""
    pids: set[int] = {root_pid}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        entries = []
    parents: dict[int, int] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in pids and pid not in pids:
                pids.add(pid)
                changed = True
    rss_values: list[int] = []
    for pid in pids:
        try:
            for line in (proc_root / str(pid) / "status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    rss_values.append(int(line.split()[1]))
                    break
        except (OSError, ValueError, IndexError):
            continue
    return {
        "process_count": len(pids),
        "rss_kb": sum(rss_values) if rss_values else None,
        "mem_available_kb": _host_mem_available_kb(),
    }


def _resource_summary(samples: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
    try:
        free_before = shutil.disk_usage(work_dir).free
    except OSError:
        free_before = None
    peak_rss = [s["rss_kb"] for s in samples if s.get("rss_kb") is not None]
    process_counts = [s["process_count"] for s in samples if s.get("process_count") is not None]
    return {
        "sample_interval_s": 2,
        "sample_count": len(samples),
        "peak_owned_rss_kb": max(peak_rss) if peak_rss else None,
        "peak_owned_processes": max(process_counts) if process_counts else None,
        "mem_available_kb_min": min(
            (s["mem_available_kb"] for s in samples if s.get("mem_available_kb") is not None),
            default=None,
        ),
        "disk_free_bytes_after": free_before,
        "samples": samples,
    }


def _existing_vendor_processes() -> list[str]:
    """Find same-user Vivado/Vitis processes before admitting a full run."""
    found: list[str] = []
    try:
        ps = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,comm="],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return found
    for line in ps.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].lower() in {"vivado", "vpl", "v++", "loader"}:
            found.append(" ".join(fields[:2]))
    return found


def execute_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    log_path: Path,
    timeout_s: float | None,
) -> CommandResult:
    """Run one vendor command with a tee-free, append-only log."""
    started = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    samples: list[dict[str, Any]] = []
    stop_monitor = threading.Event()
    monitor_thread: threading.Thread | None = None
    rc = 1
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND=" + json.dumps(list(argv)) + "\n")
            log.flush()
            # A Vitis link starts a process tree.  Start a new session so a
            # timeout can terminate the complete tree rather than leaving a
            # Vivado child holding the full-implementation lease.
            p = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=(dict(os.environ, **dict(env)) if env is not None else None),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # The parent harness may be killed by an SSH disconnect.  Keep a
            # durable identity record so a later `reconcile` can distinguish
            # this process from a reused PID and terminate the whole group.
            active = log_path.parent.parent / "active_process.json"
            active_payload = {
                "pid": p.pid,
                "pgid": p.pid,
                "pid_starttime": _proc_starttime(p.pid),
                "boot_id": _proc_boot_id(),
                "argv": [str(x) for x in argv],
                "log": str(log_path),
                "started_at": start_iso,
            }
            try:
                active.write_text(json.dumps(active_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except OSError:
                active = None
            def monitor() -> None:
                while not stop_monitor.is_set():
                    sample = _owned_process_stats(p.pid)
                    sample["timestamp"] = time.time()
                    samples.append(sample)
                    stop_monitor.wait(2)
            monitor_thread = threading.Thread(target=monitor, name="fpga-harness-monitor", daemon=True)
            monitor_thread.start()
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                rc = 124
                log.write(f"HARNESS_TIMEOUT after {timeout_s}s\n")
            else:
                rc = p.returncode
            stop_monitor.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=3)
            log.write(f"HARNESS_RC={rc}\n")
            if active is not None:
                try:
                    active.unlink()
                except FileNotFoundError:
                    pass
    except OSError as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"EXEC_ERROR={exc}\n")
    finished = time.time()
    return CommandResult(
        tuple(str(x) for x in argv),
        rc,
        start_iso,
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        str(log_path),
        timed_out,
        _resource_summary(samples, log_path.parent),
    )


__all__ = [
    "ADAPTER_VERSION",
    "ArtifactValidation",
    "Check",
    "CommandResult",
    "PreflightResult",
    "U50Adapter",
    "U50Config",
    "U50Run",
    "execute_command",
    "sha256",
]
