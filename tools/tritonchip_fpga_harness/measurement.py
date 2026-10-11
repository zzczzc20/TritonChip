"""Board programming, conformance, and ArchBench evidence collection.

The implementation flow produces an XCLBIN; it does not prove that the
XCLBIN can be loaded or that a TNCA kernel completes on the target board.
This module is the deliberately separate post-link stage.  All commands are
argv lists, all output stays under the run directory, and a benchmark result
is accepted only when it is explicitly marked ``measured-board``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters.u50 import CommandResult, execute_command
from .schema import dump_json


class MeasurementError(RuntimeError):
    """The board or benchmark evidence did not satisfy the harness contract."""


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
    *, root: Path, run_dir: Path, backend: str, xclbin: Path, result_file: Path
) -> list[str]:
    return [
        sys.executable,
        str(root / "tools/r36/conformance.py"),
        "--backends",
        backend,
        "--xclbin",
        str(xclbin),
        "--require-measured",
        "--json",
        str(result_file),
    ]


def _default_archbench(
    *,
    root: Path,
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
        sys.executable,
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
) -> dict[str, Any]:
    """Program a board, run conformance, and optionally run ArchBench."""
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
    arch_profile = str(config.get("profile", "core-vrf-sync-v1"))
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
    # exactly.  Add the profile's tool directories explicitly instead of
    # relying on a user's interactive shell having sourced settings64.sh.
    vitis_root = profile.data.get("vitis_root")
    if vitis_root:
        env["XILINX_VITIS"] = str(vitis_root)
        env["XILINX_PLATFORMS"] = str(profile.data.get("platforms_root", ""))
        env["PATH"] = os.pathsep.join([str(Path(vitis_root) / "bin"), os.environ.get("PATH", "")])
    for key, value in profile.data.get("environment", {}).items():
        env[str(key)] = str(value)
    if config.get("bdf"):
        env["TNCA_BDF"] = str(config["bdf"])
    stages: list[dict[str, Any]] = []

    program_cfg = config.get("program", True)
    if program_cfg:
        if isinstance(program_cfg, dict) and program_cfg.get("command"):
            check_command = program_cfg.get("check_command")
            go_command = program_cfg["command"]
        else:
            script = config.get("program_script") or profile.data.get("program_script")
            if not script:
                raise MeasurementError(
                    f"{board} has no program_script; set board.program.command explicitly"
                )
            check_command = [str(root / str(script)), "--check", "{xclbin}"]
            go_command = [str(root / str(script)), "--go", "{xclbin}"]
        if check_command:
            checked, record = _run(
                _argv(check_command, substitutions, "board.program.check_command"),
                root=root, run_dir=run_dir, log_name="program_check", env=env, timeout_s=timeout_s,
            )
            stages.append(record)
            if checked.returncode != 0:
                raise MeasurementError("board programming preflight failed")
        loaded, record = _run(
            _argv(go_command, substitutions, "board.program.command"),
            root=root, run_dir=run_dir, log_name="program", env=env, timeout_s=timeout_s,
        )
        stages.append(record)
        if loaded.returncode != 0:
            raise MeasurementError("board programming failed")

    conformance_summary: dict[str, Any] | None = None
    conformance_cfg = config.get("conformance", True)
    if conformance_cfg:
        conformance_cfg = conformance_cfg if isinstance(conformance_cfg, dict) else {}
        result_name = str(conformance_cfg.get("result_file", "board/conformance.json"))
        result_path = _relative(result_name, run_dir)
        command = conformance_cfg.get("command") or _default_conformance(
            root=root, run_dir=run_dir, backend=backend, xclbin=xclbin, result_file=result_path
        )
        finished, record = _run(
            _argv(command, substitutions, "board.conformance.command"),
            root=root, run_dir=run_dir, log_name="conformance", env=env, timeout_s=timeout_s,
        )
        stages.append(record)
        passed, conformance_summary = _is_conformance_pass(result_path)
        conformance_summary["result_file"] = str(result_path.relative_to(run_dir)) if result_path.is_file() else None
        if finished.returncode != 0 or not passed:
            raise MeasurementError("board conformance failed")

    arch_summary: dict[str, Any] | None = None
    arch_cfg = config.get("archbench")
    if arch_cfg:
        arch_cfg = arch_cfg if isinstance(arch_cfg, dict) else {}
        tiers = arch_cfg.get("tiers", ["tiny", "small"])
        if not isinstance(tiers, list) or not tiers or not all(isinstance(x, str) and x for x in tiers):
            raise MeasurementError("board.archbench.tiers must be a non-empty string list")
        output_name = str(arch_cfg.get("result_file", "board/archbench.jsonl"))
        output_path = _relative(output_name, run_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        command = arch_cfg.get("command") or _default_archbench(
            root=root, backend=backend, profile=arch_profile, xclbin=xclbin,
            tiers=tiers, output=output_path, jobs=int(arch_cfg.get("jobs", 1)),
            counters=bool(arch_cfg.get("counters", True)),
        )
        finished, record = _run(
            _argv(command, substitutions, "board.archbench.command"),
            root=root, run_dir=run_dir, log_name="archbench", env=env,
            timeout_s=arch_cfg.get("timeout_s", timeout_s),
        )
        stages.append(record)
        arch_summary = summarize_archbench(
            output_path, backend=backend,
            expected_clock_hz=profile.data.get("device_clock_mhz", 0) * 1e6,
            require_counters=bool(arch_cfg.get("counters", True)),
        )
        dump_json(run_dir / "archbench_summary.json", arch_summary)
        if finished.returncode != 0 or arch_summary["status"] != "PASS":
            raise MeasurementError("ArchBench measurement failed")

    result = {
        "schema_version": 1,
        "schema": "tritonchip-fpga-harness-board-measurement/1",
        "status": "PASS",
        "board": board,
        "backend": backend,
        "xclbin": str(xclbin.relative_to(run_dir)),
        "xclbin_sha256": _sha256(xclbin),
        "profile": arch_profile,
        "profile_hash": _profile_hash(profile),
        "clock_hz": float(profile.data.get("device_clock_mhz", 0)) * 1e6,
        "stages": stages,
        "conformance": conformance_summary,
        "archbench": arch_summary,
    }
    dump_json(run_dir / "board_measurement.json", result)
    return result


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _profile_hash(profile: Any) -> str:
    """Hash the exact board profile used for programming and measurement."""
    import hashlib

    payload = json.dumps(profile.data, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()

