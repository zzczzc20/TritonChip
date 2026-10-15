"""Board programming, conformance, and ArchBench evidence collection.

The implementation flow produces an XCLBIN; it does not prove that the
XCLBIN can be loaded or that a TNCA kernel completes on the target board.
This module is the deliberately separate post-link stage.  All commands are
argv lists, all output stays under the run directory, and a benchmark result
is accepted only when it is explicitly marked ``measured-board``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters.u50 import CommandResult, execute_command
from .schema import dump_json


class MeasurementError(RuntimeError):
    """The board or benchmark evidence did not satisfy the harness contract.

    ``status`` is the result status the failure maps to.  ``partial`` is the
    record of the stages that did run, so a board stage that stopped half way
    still says how far it got instead of leaving only log files behind.
    """

    def __init__(self, message: str, *, status: str = "BOARD_FAIL",
                 partial: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.partial = partial


def _replace(value: str, substitutions: Mapping[str, str]) -> str:
    for key, replacement in substitutions.items():
        value = value.replace(key, replacement)
    return value


def _argv(value: Any, substitutions: Mapping[str, str], name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise MeasurementError(f"{name} must be a non-empty argv list")
    return [_replace(x, substitutions) for x in value]


def _relative(path: str, run_dir: Path) -> Path:
    out = Path(path)
    if out.is_absolute() or ".." in out.parts:
        raise MeasurementError(f"measurement path must be relative to run directory: {path}")
    return run_dir / out


def _command_record(result: CommandResult, run_dir: Path, name: str) -> dict[str, Any]:
    try:
        log = str(Path(result.log_path).relative_to(run_dir))
    except ValueError:
        log = result.log_path
    return {
        "name": name,
        "argv": list(result.argv),
        "returncode": result.returncode,
        "log": log,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "timed_out": result.timed_out,
        "resource": result.resource,
    }


def _run(
    argv: Sequence[str],
    *,
    root: Path,
    run_dir: Path,
    log_name: str,
    env: Mapping[str, str],
    timeout_s: float | None,
) -> tuple[CommandResult, dict[str, Any]]:
    log = run_dir / "logs" / "board" / f"{log_name}.log"
    result = execute_command(argv, cwd=root, env=dict(env), log_path=log, timeout_s=timeout_s)
    return result, _command_record(result, run_dir, log_name)


def _default_conformance(
    *, root: Path, run_dir: Path, backend: str, profile: str, xclbin: Path, result_file: Path
) -> list[str]:
    # conformance.py defaults to core-vrf-sync-v1 whatever image is loaded.
    # Name the profile the manifest names, so conformance and ArchBench drive
    # the card as the same machine.
    return [
        sys.executable,
        str(root / "tools/r36/conformance.py"),
        "--backends",
        backend,
        "--profile",
        profile,
        "--xclbin",
        str(xclbin),
        "--require-measured",
        "--json",
        str(result_file),
    ]


def archbench_python(config: Mapping[str, Any]) -> str:
    """The interpreter ArchBench runs under.

    ArchBench needs PyTorch, Triton and the installed ``tnca`` package
    (spec/r38/torch.lock.json), which the repository keeps in an environment
    of its own and names ``TORCH_PYTHON``.  The harness itself needs only the
    standard library, so the two are usually different interpreters.
    """
    arch = config.get("archbench")
    chosen = arch.get("python") if isinstance(arch, dict) else None
    return str(chosen or os.environ.get("TORCH_PYTHON") or sys.executable)


def _xrt_root() -> Path:
    return Path(os.environ.get("XILINX_XRT", "/opt/xilinx/xrt"))


def _tool_dirs(profile: Any) -> list[str]:
    """Directories the board commands need on PATH, most specific first."""
    dirs: list[str] = []
    vitis_root = profile.data.get("vitis_root")
    if vitis_root:
        dirs.append(str(Path(vitis_root) / "bin"))
    xrt_bin = _xrt_root() / "bin"
    if xrt_bin.is_dir():
        dirs.append(str(xrt_bin))
    return dirs


_ARCHBENCH_PROBE = (
    "import json, torch, triton, triton.backends; "
    "print('HARNESS-PROBE ' + json.dumps({'torch': torch.__version__, 'triton': triton.__version__, "
    "'tnca_backend': 'cpu' in triton.backends.backends}))"
)


def board_preflight(*, root: Path, profile: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check, cheaply, that the board stage this manifest asks for can start.

    A link takes hours and the board stage follows it; an interpreter without
    PyTorch or a host without XRT should be found before the link, not after.
    Only the steps the harness would run itself are checked: one the manifest
    replaces with its own command is the manifest's business.
    """
    config = manifest.get("board")
    if not isinstance(config, dict):
        return []
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail), "severity": "error"})

    program = config.get("program", True)
    if program and not (isinstance(program, dict) and program.get("command")):
        script = config.get("program_script") or profile.data.get("program_script")
        path = root / str(script) if script else None
        add("board_program_script", bool(path and path.is_file() and os.access(path, os.X_OK)),
            path if path else "the target profile names no program_script")
        xbutil = Path(os.environ.get("XRT_BIN") or _xrt_root() / "bin") / "xbutil"
        add("board_xbutil", xbutil.is_file(), xbutil)
        search = os.pathsep.join(_tool_dirs(profile) + [os.environ.get("PATH", "")])
        xclbinutil = shutil.which("xclbinutil", path=search)
        add("board_xclbinutil", xclbinutil is not None, xclbinutil or "xclbinutil is in neither Vitis, XRT nor PATH")

    conformance = config.get("conformance", True)
    own_conformance = bool(conformance) and not (isinstance(conformance, dict) and conformance.get("command"))
    arch = config.get("archbench")
    own_archbench = bool(arch) and not (isinstance(arch, dict) and arch.get("command"))
    if own_conformance:
        tool = root / "tools/r36/conformance.py"
        add("board_conformance_tool", tool.is_file(), tool)
    if own_conformance or own_archbench:
        library = _xrt_root() / "lib" / "libxrt_core.so"
        add("board_xrt_library", library.is_file(), library)
    if own_archbench:
        runner = root / "archbench/run.py"
        add("board_archbench_runner", runner.is_file(), runner)
        python = archbench_python(config)
        try:
            probe = subprocess.run([python, "-c", _ARCHBENCH_PROBE], capture_output=True, text=True,
                                   timeout=300, check=False)
            found = [line for line in probe.stdout.splitlines() if line.startswith("HARNESS-PROBE ")]
            if probe.returncode == 0 and found:
                seen = json.loads(found[-1][len("HARNESS-PROBE "):])
                add("board_archbench_python", bool(seen.get("tnca_backend")),
                    f"{python}: torch {seen.get('torch')}, triton {seen.get('triton')}, "
                    + ("tnca is its Triton cpu backend" if seen.get("tnca_backend")
                       else "but the tnca package is not installed there (no triton.backends 'cpu')"))
            else:
                reason = (probe.stderr.strip().splitlines() or ["no output"])[-1]
                add("board_archbench_python", False,
                    f"{python}: {reason} -- ArchBench needs PyTorch, Triton and tnca "
                    "(spec/r38/torch.lock.json); set TORCH_PYTHON or board.archbench.python")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            add("board_archbench_python", False, f"{python}: {exc}")
    return checks


def _default_archbench(
    *,
    root: Path,
    python: str,
    backend: str,
    profile: str,
    xclbin: Path,
    tiers: list[str],
    output: Path,
    jobs: int,
    counters: bool,
) -> list[str]:
    # archbench.run accepts a JSON object for --backend-args.  Building this
    # token here, rather than asking a shell to quote it, keeps paths with
    # spaces and remote worker wrappers deterministic.
    return [
        python,
        "-m",
        "archbench.run",
        "--platform",
        "tnca",
        "--backend",
        backend,
        "--profile",
        profile,
        "--backend-args",
        json.dumps({"xclbin": str(xclbin)}, separators=(",", ":")),
        "--tier",
        *tiers,
        "--jobs",
        str(jobs),
        "--out",
        str(output),
    ] + (["--counters"] if counters else [])


_PARAMETER = re.compile(r"(?m)^\s*parameter\s+(?:integer\s+)?([A-Za-z_]\w*)\s*=\s*(\d+)\b")

# Outermost first.  Each level passes the parameters it declares down to the
# next, so the first module to declare a name decides its value in the image.
_IMAGE_RTL = ("rtl/board/tnca_u50_xo.v", "rtl/fpga/tnca_u50_top.v", "rtl/fpga/tnca_fpga_top.v")


def _declared_parameters(profile: Mapping[str, Any]) -> dict[str, int]:
    declared: dict[str, int] = {}
    for section in (profile.get("elaboration"), (profile.get("implementation") or {}).get("rtl_parameters")):
        if isinstance(section, dict):
            declared.update({str(k): v for k, v in section.items() if isinstance(v, int) and not isinstance(v, bool)})
    return declared


def image_profile_contract(root: Path, arch_profile: str) -> dict[str, Any] | None:
    """Compare a device profile's parameters with the image this checkout builds.

    The package/link flow takes no profile argument: the image is whatever the
    RTL parameter defaults elaborate to.  A manifest that names another
    profile would label the measurement as a machine that was not on the card.

    Returns ``None`` when the checkout has no profile table or board RTL to
    read (the contract is then unknown, not passed).  ``mismatches`` are
    parameters the named profile declares and the RTL defaults contradict.
    ``more_specific`` names profiles that also match and pin down more of the
    switches -- the usual sign that an older profile name was reused for a
    newer image.
    """
    try:
        profiles = json.loads((root / "spec/r1_profiles.json").read_text(encoding="utf-8"))["profiles"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(profiles, dict) or not isinstance(profiles.get(arch_profile), dict):
        return None
    defaults: dict[str, int] = {}
    for relative in _IMAGE_RTL:
        try:
            text = (root / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, value in _PARAMETER.findall(text):
            defaults.setdefault(name, int(value))
    if not defaults:
        return None

    def compare(name: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
        checked: dict[str, int] = {}
        wrong: list[dict[str, Any]] = []
        for key, want in sorted(_declared_parameters(profiles[name]).items()):
            if key not in defaults:
                continue
            checked[key] = want
            if defaults[key] != want:
                wrong.append({"parameter": key, "profile": want, "rtl_default": defaults[key]})
        return checked, wrong

    checked, mismatches = compare(arch_profile)
    more_specific = []
    for other in sorted(profiles):
        if other == arch_profile or not isinstance(profiles[other], dict):
            continue
        other_checked, other_wrong = compare(other)
        if not other_wrong and set(other_checked) > set(checked):
            more_specific.append(other)
    return {
        "profile": arch_profile,
        "checked": checked,
        "mismatches": mismatches,
        "more_specific": more_specific,
        "sources": ["spec/r1_profiles.json", *_IMAGE_RTL],
    }


def _is_conformance_pass(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.is_file():
        return False, {"error": f"missing conformance result: {path}"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, {"error": f"invalid conformance JSON: {exc}"}
    rows = value if isinstance(value, list) else [value]
    if not rows or not all(isinstance(row, dict) for row in rows):
        return False, {"error": "conformance result must be a JSON object or list"}
    passed = all(row.get("passed") is True for row in rows)
    return passed, {"rows": rows, "passed": passed}


def summarize_archbench(
    path: Path,
    *,
    backend: str,
    expected_clock_hz: float,
    require_counters: bool,
) -> dict[str, Any]:
    """Validate and summarize raw ``archbench-torch-record/1`` JSONL.

    ``time.device`` is the ArchBench point latency: the sum of each launch's
    completion ``done_cycles``.  Counter totals are retained separately so a
    caller can explain the latency without confusing ``perf_cycles`` with the
    shell completion count.
    """
    if not path.is_file():
        raise MeasurementError(f"missing ArchBench JSONL: {path}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MeasurementError(f"invalid ArchBench JSON at line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise MeasurementError(f"ArchBench line {line_no} is not an object")
        records.append(row)
    if not records:
        raise MeasurementError("ArchBench output is empty")

    failures: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    cycles_total = 0
    retired_total = 0
    counter_totals: dict[str, int] = {}
    cycles_by_tier: dict[str, int] = {}
    for row in records:
        point_id = row.get("point_id")
        platform = row.get("platform") or {}
        timing = row.get("time") or {}
        status = row.get("status")
        problems = list(row.get("problems") or [])
        if not isinstance(platform, dict):
            problems.append("platform is not an object")
            platform = {}
        if not isinstance(timing, dict):
            problems.append("time is not an object")
            timing = {}
        if platform.get("backend") != backend:
            problems.append(f"backend {platform.get('backend')!r} != {backend!r}")
        if row.get("evidence_level") != "measured-board":
            problems.append(f"evidence_level {row.get('evidence_level')!r} is not measured-board")
        cycles = timing.get("device")
        if timing.get("unit") != "cycles" or not isinstance(cycles, int) or cycles <= 0:
            problems.append("missing positive device cycle count")
            cycles = 0
        clock = timing.get("clock_hz")
        if clock is None and expected_clock_hz:
            problems.append("missing measured clock_hz")
        elif expected_clock_hz:
            try:
                clock_delta = abs(float(clock) - expected_clock_hz)
            except (TypeError, ValueError):
                problems.append(f"clock_hz {clock!r} is not numeric")
            else:
                if clock_delta > max(1.0, expected_clock_hz * 0.01):
                    problems.append(f"clock_hz {clock!r} does not match {expected_clock_hz:g}")
        counters = row.get("counters") or {}
        if require_counters:
            if not isinstance(counters, dict) or not counters:
                problems.append("counters were requested but the record has none")
            else:
                done_cycles = counters.get("done_cycles")
                done_retired = counters.get("done_retired")
                if not isinstance(done_cycles, int) or done_cycles <= 0:
                    problems.append("counter record has no positive done_cycles")
                if not isinstance(done_retired, int) or done_retired < 0:
                    problems.append("counter record has no non-negative done_retired")
                # ArchBench's device latency is defined as the sum of the
                # completion counters.  Catch a stale or host-timed record
                # before it can be used in an A/B speedup calculation.
                if isinstance(done_cycles, int) and done_cycles > 0 and cycles and done_cycles != cycles:
                    problems.append(f"done_cycles {done_cycles} != time.device {cycles}")
        if status != "pass":
            problems.append(f"status={status!r}")
        if problems:
            failures.append({"point_id": point_id, "problems": problems})
        point = {
            "point_id": point_id,
            "family": row.get("family"),
            "tier": row.get("tier"),
            "status": status,
            "device_cycles": cycles,
            "device_seconds": timing.get("device_seconds"),
            "launches": timing.get("launches"),
            "per_launch": timing.get("per_launch", []),
            "done_cycles": counters.get("done_cycles"),
            "done_retired": counters.get("done_retired"),
        }
        points.append(point)
        cycles_total += cycles
        tier = str(row.get("tier") or "unknown")
        cycles_by_tier[tier] = cycles_by_tier.get(tier, 0) + cycles
        for name, value in counters.items():
            if isinstance(value, int):
                counter_totals[name] = counter_totals.get(name, 0) + value
        if isinstance(counters.get("done_retired"), int):
            retired_total += counters["done_retired"]

    summary = {
        "schema_version": 1,
        "schema": "tritonchip-fpga-harness-archbench/1",
        "source": str(path),
        "backend": backend,
        "evidence_level": "measured-board" if not failures else "invalid",
        "records": len(records),
        "passed": sum(row.get("status") == "pass" for row in records),
        "failed": len(failures),
        "status": "PASS" if not failures else "ARCHBENCH_FAIL",
        "cycles_total": cycles_total,
        "cycles_by_tier": cycles_by_tier,
        "done_cycles_total": counter_totals.get("done_cycles"),
        "done_retired_total": retired_total,
        "counter_totals": counter_totals,
        "points": points,
        "failures": failures,
    }
    return summary


def measure_board(
    *,
    root: Path,
    run_dir: Path,
    profile: Any,
    manifest: dict[str, Any],
    timeout_s: int | None,
    software: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Program a board, run conformance, and optionally run ArchBench.

    ``software`` is the identity of the checkout whose runtime, conformance
    suite and ArchBench drive the card; it is recorded with the measurement
    because that checkout need not be the commit the image was built from.

    Every exit, pass or fail, writes ``board_measurement.json``.  A failure
    raises :class:`MeasurementError` carrying the same record.
    """
    config = manifest.get("board")
    if not isinstance(config, dict):
        raise MeasurementError("board measurement requires a manifest.board object")
    artifacts = run_dir / "artifacts"
    expected = list(profile.data["expected_artifacts"])
    xclbin = artifacts / expected[-1]
    if not xclbin.is_file() or xclbin.stat().st_size == 0:
        raise MeasurementError(f"cannot measure without a non-empty xclbin: {xclbin}")

    board = str(profile.data.get("board", "unknown"))
    backend = str(config.get("backend", f"fpga-{board}"))
    expected_backend = f"fpga-{board}"
    if backend != expected_backend:
        raise MeasurementError(
            f"board backend {backend!r} does not match profile board {board!r} "
            f"(expected {expected_backend!r})"
        )
    arch_profile = config.get("profile")
    if not isinstance(arch_profile, str) or not arch_profile:
        # No default: the profile is a claim about which machine is on the
        # card, and a guessed one would be recorded with every cycle count.
        raise MeasurementError("board.profile is required: name the device profile of the image")
    substitutions = {
        "{repo}": str(root.resolve()),
        "{run_dir}": str(run_dir.resolve()),
        "{xclbin}": str(xclbin.resolve()),
        "{backend}": backend,
        "{profile}": arch_profile,
        "{board}": board,
        "{clock_mhz}": str(profile.expected_clock),
    }
    env = {
        "TNCA_BOARD": board,
        "TNCA_HARNESS_RUN_DIR": str(run_dir.resolve()),
        "TNCA_HARNESS_XCLBIN": str(xclbin.resolve()),
    }
    # Vendor commands are executed without a shell so that argv is recorded
    # exactly.  Add the tool directories explicitly instead of relying on a
    # user's interactive shell having sourced settings64.sh or XRT's setup.sh:
    # the programmer needs xclbinutil, and both installations ship one.
    vitis_root = profile.data.get("vitis_root")
    if vitis_root:
        env["XILINX_VITIS"] = str(vitis_root)
        env["XILINX_PLATFORMS"] = str(profile.data.get("platforms_root", ""))
    tool_dirs = _tool_dirs(profile)
    if tool_dirs:
        env["PATH"] = os.pathsep.join(tool_dirs + [os.environ.get("PATH", "")])
    for key, value in profile.data.get("environment", {}).items():
        env[str(key)] = str(value)
    if config.get("bdf"):
        # flow/fpga/program.sh reads BDF; without it the card is the one
        # flow/fpga/boards.sh records for TNCA_BOARD.  TNCA_BDF is kept for
        # manifest-supplied commands.
        env["BDF"] = str(config["bdf"])
        env["TNCA_BDF"] = str(config["bdf"])

    stages: list[dict[str, Any]] = []
    loaded_image: str | None = None
    replaced_image: str | None = None
    reconfigured: bool | None = None
    conformance_summary: dict[str, Any] | None = None
    arch_summary: dict[str, Any] | None = None
    contract = image_profile_contract(root, arch_profile)
    xclbin_sha256 = _sha256(xclbin)

    def record(status: str, error: str | None = None, failed_stage: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "schema": "tritonchip-fpga-harness-board-measurement/1",
            "status": status,
            "board": board,
            "backend": backend,
            "bdf": config.get("bdf"),
            "programmed": bool(config.get("program", True)),
            "replaced_xclbin_uuid": replaced_image,
            "loaded_xclbin_uuid": loaded_image,
            # True: the load put a different image on the card.  False: the
            # card already held this one and XRT downloaded nothing.  None:
            # nothing was loaded, or the programmer's log does not say.
            "reconfigured": reconfigured,
            "xclbin": str(xclbin.relative_to(run_dir)),
            "xclbin_sha256": xclbin_sha256,
            "profile": arch_profile,
            "profile_hash": _profile_hash(profile),
            "image_profile": contract,
            "clock_hz": float(profile.data.get("device_clock_mhz", 0)) * 1e6,
            "software": software,
            "archbench_python": archbench_python(config) if config.get("archbench") else None,
            "stages": stages,
            "conformance": conformance_summary,
            "archbench": arch_summary,
        }
        if error is not None:
            value["error"] = error
            value["failed_stage"] = failed_stage
        dump_json(run_dir / "board_measurement.json", value)
        return value

    def fail(message: str, stage: str, status: str = "BOARD_FAIL") -> MeasurementError:
        return MeasurementError(message, status=status, partial=record(status, message, stage))

    if contract and contract["mismatches"]:
        detail = ", ".join(
            f"{m['parameter']}: profile {m['profile']}, RTL default {m['rtl_default']}" for m in contract["mismatches"]
        )
        raise fail(
            f"board profile {arch_profile!r} is not the image this checkout builds ({detail})",
            "image_profile",
        )

    program_cfg = config.get("program", True)
    if program_cfg:
        if isinstance(program_cfg, dict) and program_cfg.get("command"):
            check_command = program_cfg.get("check_command")
            go_command = program_cfg["command"]
        else:
            script = config.get("program_script") or profile.data.get("program_script")
            if not script:
                raise fail(f"{board} has no program_script; set board.program.command explicitly", "program_check")
            check_command = [str(root / str(script)), "--check", "{xclbin}"]
            go_command = [str(root / str(script)), "--go", "{xclbin}"]
        if check_command:
            checked, stage = _run(
                _argv(check_command, substitutions, "board.program.check_command"),
                root=root, run_dir=run_dir, log_name="program_check", env=env, timeout_s=timeout_s,
            )
            stages.append(stage)
            if checked.returncode != 0:
                raise fail("board programming preflight failed", "program_check")
        loaded, stage = _run(
            _argv(go_command, substitutions, "board.program.command"),
            root=root, run_dir=run_dir, log_name="program", env=env, timeout_s=timeout_s,
        )
        stages.append(stage)
        if loaded.returncode != 0:
            raise fail("board programming failed", "program")
        # A load that succeeds is not always a load that happened.  XRT does
        # not download an image the card already holds: it returns success,
        # the device keeps its state, and its UUID is the file's as before.
        # The flow's programmer logs what it replaced and what the device
        # reports afterwards; the two being the same is that case.
        try:
            text = Path(loaded.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        before = re.findall(r"REPLACING=(\S+)", text)
        after = re.findall(r"device now reports xclbin (\S+)", text)
        if before and before[-1] != "<none>":
            replaced_image = before[-1].lower()
        if after:
            loaded_image = after[-1].lower()
        if before and after:
            reconfigured = before[-1].lower() != after[-1].lower()

    else:
        # Nothing is loaded, so the measurement is of whatever the card holds.
        # The runtime refuses a card whose image is not this file; the
        # programmer's read-only report puts that image's identity on record.
        script = config.get("program_script") or profile.data.get("program_script")
        if script and (root / str(script)).is_file():
            reported, stage = _run([str(root / str(script)), "--status"], root=root, run_dir=run_dir,
                                   log_name="loaded_image", env=env, timeout_s=120)
            stages.append(stage)
            try:
                text = Path(reported.log_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            found = re.search(r"(?m)^loaded\s+([0-9a-fA-F-]{8,})\s*$", text)
            loaded_image = found.group(1).lower() if found else None

    conformance_cfg = config.get("conformance", True)
    if conformance_cfg:
        conformance_cfg = conformance_cfg if isinstance(conformance_cfg, dict) else {}
        result_name = str(conformance_cfg.get("result_file", "board/conformance.json"))
        result_path = _relative(result_name, run_dir)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            # A verdict left by an earlier attempt must not answer for this one.
            result_path.unlink()
        command = conformance_cfg.get("command") or _default_conformance(
            root=root, run_dir=run_dir, backend=backend, profile=arch_profile,
            xclbin=xclbin, result_file=result_path,
        )
        finished, stage = _run(
            _argv(command, substitutions, "board.conformance.command"),
            root=root, run_dir=run_dir, log_name="conformance", env=env, timeout_s=timeout_s,
        )
        stages.append(stage)
        passed, conformance_summary = _is_conformance_pass(result_path)
        conformance_summary["result_file"] = str(result_path.relative_to(run_dir)) if result_path.is_file() else None
        if finished.returncode != 0 or not passed:
            raise fail("board conformance failed", "conformance")

    arch_cfg = config.get("archbench")
    if arch_cfg:
        arch_cfg = arch_cfg if isinstance(arch_cfg, dict) else {}
        tiers = arch_cfg.get("tiers", ["tiny", "small"])
        if not isinstance(tiers, list) or not tiers or not all(isinstance(x, str) and x for x in tiers):
            raise fail("board.archbench.tiers must be a non-empty string list", "archbench", "ARCHBENCH_FAIL")
        output_name = str(arch_cfg.get("result_file", "board/archbench.jsonl"))
        output_path = _relative(output_name, run_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        command = arch_cfg.get("command") or _default_archbench(
            root=root, python=archbench_python(config), backend=backend, profile=arch_profile, xclbin=xclbin,
            tiers=tiers, output=output_path, jobs=int(arch_cfg.get("jobs", 1)),
            counters=bool(arch_cfg.get("counters", True)),
        )
        finished, stage = _run(
            _argv(command, substitutions, "board.archbench.command"),
            root=root, run_dir=run_dir, log_name="archbench", env=env,
            timeout_s=arch_cfg.get("timeout_s", timeout_s),
        )
        stages.append(stage)
        try:
            arch_summary = summarize_archbench(
                output_path, backend=backend,
                expected_clock_hz=profile.data.get("device_clock_mhz", 0) * 1e6,
                require_counters=bool(arch_cfg.get("counters", True)),
            )
        except MeasurementError as exc:
            raise fail(str(exc), "archbench", "ARCHBENCH_FAIL") from exc
        dump_json(run_dir / "archbench_summary.json", arch_summary)
        if finished.returncode != 0 or arch_summary["status"] != "PASS":
            raise fail("ArchBench measurement failed", "archbench", "ARCHBENCH_FAIL")

    return record("PASS")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _profile_hash(profile: Any) -> str:
    """Hash the exact board profile used for programming and measurement."""
    payload = json.dumps(profile.data, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
