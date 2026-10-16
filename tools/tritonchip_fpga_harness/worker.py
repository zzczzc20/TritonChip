"""SSH worker and dispatch protocol for remote FPGA Harness hosts.

The controller sends one JSON request on stdin.  The remote process reads that
request, invokes the existing Harness API (``doctor``, ``submit``, ``status``,
``export`` or ``measure``), and emits one JSON response on stdout.  ``submit``
and ``measure`` can be detached: the worker starts the Harness CLI in a session
of its own and answers at once, so an hours-long link does not depend on one
SSH connection staying up; the controller then polls ``status``.  Commands are assembled as argv vectors and every
value that becomes an SSH/remote argument is checked before it reaches
``subprocess.run``.  Manifests and logs therefore never need shell quoting or a
temporary file in the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from .schema import WORKER_OPERATIONS, SchemaError, dump_json, load_json, validate_manifest, validate_worker_request


class DispatchError(RuntimeError):
    """A controller or remote worker protocol error."""


# OpenSSH receives a command vector but executes the remote command through the
# user's login shell.  Reject shell syntax and whitespace in path/option values
# so the controller never relies on shell escaping.  The manifest itself is
# data on stdin and is intentionally not restricted by this check.
_SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")


def _safe_arg(value: str, name: str) -> str:
    # A leading '-' would make a value an option of ssh or of the remote program.
    if not isinstance(value, str) or not value or value.startswith("-") or not _SAFE_ARG.fullmatch(value):
        raise DispatchError(
            f"{name} contains unsupported characters; SSH arguments must be "
            "single shell-safe argv tokens"
        )
    return value


def _board_for_target(target: str) -> str:
    # A target is named for its card, with an optional variant after a dash.
    board = target.lower().split("-", 1)[0]
    if board in {"u50", "u280"}:
        return board
    raise DispatchError(
        f"cannot infer FPGA server board from target {target!r}; pass --board u50|u280"
    )


_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def ssh_prefix(
    *,
    host: str,
    port: int | None = None,
    identity_file: str | None = None,
    connect_timeout: int | None = None,
    options: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    ssh_binary: str = "ssh",
) -> list[str]:
    """``ssh [opts] host [env K=V ...]``: the part every remote command shares.

    ``env`` is how a host's settings reach the remote process (a build host's
    Vitis root, a card host's ArchBench interpreter): ``env`` is a program, so
    no shell syntax is involved.  Names and values are checked like any other
    argument.
    """
    for value, name in ((ssh_binary, "ssh_binary"), (host, "host")):
        _safe_arg(value, name)
    if identity_file is not None:
        _safe_arg(identity_file, "identity_file")
    if port is not None and (not isinstance(port, int) or not 1 <= port <= 65535):
        raise DispatchError("SSH port must be between 1 and 65535")
    if connect_timeout is not None and (not isinstance(connect_timeout, int) or connect_timeout <= 0):
        raise DispatchError("SSH connect timeout must be positive")
    argv = [ssh_binary, "-T"]
    if port is not None:
        argv.extend(["-p", str(port)])
    if identity_file is not None:
        argv.extend(["-i", identity_file])
    if connect_timeout is not None:
        argv.extend(["-o", f"ConnectTimeout={connect_timeout}"])
    for option in options:
        argv.extend(["-o", _safe_arg(option, "ssh option")])
    # '--' ends ssh's options: OpenSSH otherwise reads options that follow the
    # destination too, and the remote argv's first word would be one.
    argv.extend(["--", host])
    if env:
        argv.append("env")
        for key, value in sorted(env.items()):
            if not _ENV_NAME.fullmatch(str(key)):
                raise DispatchError(f"environment name {key!r} is not a plain variable name")
            argv.append(_safe_arg(f"{key}={value}", f"environment {key}"))
    return argv


def build_ssh_argv(
    *,
    host: str,
    remote_repo: str,
    operation: str,
    adapter: str,
    remote_python: str = "python3",
    remote_cli: str = "tools/tritonchip_fpga_harness/cli.py",
    port: int | None = None,
    identity_file: str | None = None,
    connect_timeout: int | None = None,
    ssh_binary: str = "ssh",
    options: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Build the fixed remote worker argv without a shell command string."""

    if operation not in WORKER_OPERATIONS:
        raise DispatchError(f"unsupported dispatch operation {operation!r}")
    if adapter not in {"u50", "u280", "mock"}:
        raise DispatchError(f"remote FPGA adapter must be u50, u280 or mock, got {adapter!r}")
    for value, name in (
        (remote_repo, "remote_repo"),
        (remote_python, "remote_python"),
        (remote_cli, "remote_cli"),
    ):
        _safe_arg(value, name)
    argv = ssh_prefix(host=host, port=port, identity_file=identity_file, connect_timeout=connect_timeout,
                      options=options, env=env, ssh_binary=ssh_binary)
    argv.extend(
        [
            remote_python,
            remote_cli,
            "--repo",
            remote_repo,
            "worker",
            "--stdin",
        ]
    )
    return argv


def _log_summary(text: str, *, limit: int = 20) -> dict[str, Any]:
    raw = text.encode("utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(lines),
        "tail": lines[-limit:],
    }


def summarize_run_logs(state_dir: Path, run_id: str | None) -> dict[str, Any]:
    """Return hashes and short tails for logs generated on the worker host."""

    if not run_id:
        return {"run_id": None, "files": []}
    run_dir = state_dir / "runs" / run_id
    files: list[dict[str, Any]] = []
    if run_dir.is_dir():
        for path in sorted((run_dir / "logs").rglob("*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                item = _log_summary(text)
                item["path"] = str(path.relative_to(run_dir))
                files.append(item)
            except OSError as exc:
                files.append({"path": str(path.relative_to(run_dir)), "error": str(exc)})
    return {"run_id": run_id, "files": files}


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def call_worker(request: dict[str, Any], argv: list[str], *, ssh_timeout: int | None = None) -> dict[str, Any]:
    """Send one validated request over ``argv`` and return structured evidence.

    A zero SSH exit status is not a result: ``ok`` is the remote's own answer.
    """
    validate_worker_request(request)
    operation = request["operation"]
    board = request["board"]
    payload = _canonical(request)
    started = time.time()
    try:
        completed = subprocess.run(
            argv,
            input=payload,
            text=True,
            capture_output=True,
            timeout=ssh_timeout,
            check=False,
            shell=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "operation": operation,
            "board": board,
            "error": f"SSH timed out after {ssh_timeout}s",
            "argv": argv,
            "transport": {"returncode": None, "elapsed_s": time.time() - started,
                           "stdout": _log_summary(_text(exc.stdout)),
                           "stderr": _log_summary(_text(exc.stderr))},
        }
    transport = {
        "returncode": returncode,
        "elapsed_s": time.time() - started,
        "stdout": _log_summary(stdout),
        "stderr": _log_summary(stderr),
    }
    remote: dict[str, Any] | None = None
    protocol_error: str | None = None
    if stdout.strip():
        try:
            remote_value = json.loads(stdout)
            if isinstance(remote_value, dict):
                remote = remote_value
            else:
                protocol_error = "remote worker returned a non-object JSON value"
        except json.JSONDecodeError as exc:
            protocol_error = f"remote worker returned invalid JSON: {exc}"
    elif returncode == 0:
        protocol_error = "remote worker returned no JSON"
    value: dict[str, Any] = {
        "schema_version": 1,
        "ok": bool(returncode == 0 and remote and remote.get("ok") is True and not protocol_error),
        "operation": operation,
        "board": board,
        "remote": remote,
        "transport": transport,
        "argv": argv,
    }
    if protocol_error:
        value["error"] = protocol_error
    if returncode != 0 and not remote:
        value["error"] = f"ssh exited with status {returncode}"
    return value


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def dispatch_manifest(
    *,
    manifest: dict[str, Any],
    host: str,
    remote_repo: str,
    operation: str,
    board: str | None = None,
    adapter: str | None = None,
    remote_python: str = "python3",
    remote_cli: str = "tools/tritonchip_fpga_harness/cli.py",
    state_dir: str | None = None,
    allow_dirty: bool = False,
    skip_board: bool = False,
    timeout: int | None = None,
    run_id: str | None = None,
    port: int | None = None,
    identity_file: str | None = None,
    connect_timeout: int | None = None,
    ssh_binary: str = "ssh",
    ssh_timeout: int | None = None,
) -> dict[str, Any]:
    """Send a manifest to a U50/U280 worker and return structured evidence."""

    validate_manifest(manifest)
    if board is None:
        board = _board_for_target(str(manifest["target"]))
    if board not in {"u50", "u280"}:
        raise DispatchError("board must be u50 or u280")
    inferred = _board_for_target(str(manifest["target"]))
    if inferred != board:
        raise DispatchError(
            f"target {manifest['target']!r} belongs to {inferred}; cannot dispatch to {board}"
        )
    adapter = adapter or board
    if adapter != board:
        raise DispatchError(f"adapter {adapter!r} does not match board {board!r}")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise DispatchError("remote Harness timeout must be positive")
    if ssh_timeout is not None and (not isinstance(ssh_timeout, int) or ssh_timeout <= 0):
        raise DispatchError("SSH timeout must be positive")

    request: dict[str, Any] = {
        "schema_version": 1,
        "operation": operation,
        "board": board,
        "adapter": adapter,
        "manifest": manifest,
        "allow_dirty": bool(allow_dirty),
        "skip_board": bool(skip_board),
    }
    for key, value in (("state_dir", state_dir), ("timeout", timeout), ("run_id", run_id)):
        if value is not None:
            request[key] = value
    validate_worker_request(request)
    argv = build_ssh_argv(
        host=host,
        remote_repo=remote_repo,
        operation=operation,
        adapter=adapter,
        remote_python=remote_python,
        remote_cli=remote_cli,
        port=port,
        identity_file=identity_file,
        connect_timeout=connect_timeout,
        ssh_binary=ssh_binary,
    )
    return call_worker(request, argv, ssh_timeout=ssh_timeout)


def _worker_error(message: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "error": message}


# A run is in one of these while something is still working on it.
LIVE_STATUSES = ("PENDING", "RUNNING", "MEASURING", "STARTING")


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def check_run_id(run_id: str) -> str:
    """A run id names a directory under ``runs/``: a plain name, never a path."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError(f"run id {run_id!r} is not a plain name")
    return run_id


def _attempts(state: Path, run_id: str, operation: str) -> list[tuple[int, Path]]:
    found = []
    for path in (state / "requests").glob(f"{run_id}.{operation}.*.json"):
        number = path.name[len(f"{run_id}.{operation}."):-len(".json")]
        if number.isdigit():
            found.append((int(number), path))
    return sorted(found)


def _latest(state: Path, run_id: str, operation: str) -> tuple[str | None, dict[str, Any] | None]:
    attempts = _attempts(state, run_id, operation)
    if not attempts:
        return None, None
    number, path = attempts[-1]
    return f"{run_id}.{operation}.{number}", load_json(path)


def _spawn(package_dir: Path, root: Path, state: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Start ``submit`` or ``measure`` in a session of its own and return at once.

    The child is the Harness CLI of this checkout, with the request's options
    as arguments; it inherits this process's environment (the host settings
    the controller passed through ``env``).  Each attempt has its own record
    and log in ``<state>/requests/``, named ``<run>.<operation>.<n>``; the
    name is the attempt's id, which a measurement writes into the run's record
    so that ``status`` can tell its verdict from an earlier one.  While the
    latest attempt's child is alive, a new request attaches to it instead of
    starting another.
    """
    from .core import _process_identity

    operation = request["operation"]
    run_id = check_run_id(request.get("run_id") or (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]))
    requests = state / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    attempt, latest = _latest(state, run_id, operation)
    if latest and _alive(latest):
        return {"run_id": run_id, "detached": True, "attached": True, "attempt": attempt,
                "pid": latest.get("pid"), "requested_at": latest.get("requested_at"), "log": latest.get("log")}
    number = (_attempts(state, run_id, operation) or [(0, None)])[-1][0] + 1
    attempt = f"{run_id}.{operation}.{number}"
    record_path, log_path = requests / f"{attempt}.json", requests / f"{attempt}.log"
    cli = str(package_dir / "cli.py")
    argv = [sys.executable, cli, "--repo", str(root)]
    if operation == "submit":
        manifest_path = requests / f"{attempt}.manifest.json"
        dump_json(manifest_path, request["manifest"])
        argv += ["submit", str(manifest_path), "--state-dir", str(state), "--run-id", run_id,
                 "--adapter", request["adapter"]]
        if request.get("skip_board"):
            argv.append("--skip-board")
        if request.get("allow_dirty"):
            argv.append("--allow-dirty")
    else:
        argv += ["measure", run_id, "--state-dir", str(state), "--request", attempt]
        if isinstance(request["manifest"].get("board"), dict):
            # The board stage the controller asked for, not the one the run
            # was submitted with; measure checks it names the same candidate.
            manifest_path = requests / f"{attempt}.manifest.json"
            dump_json(manifest_path, request["manifest"])
            argv += ["--manifest", str(manifest_path)]
        if request.get("allow_source_mismatch"):
            argv.append("--allow-source-mismatch")
    if request.get("timeout"):
        argv += ["--timeout", str(request["timeout"])]
    requested_at = time.time()
    with log_path.open("ab") as log:
        child = subprocess.Popen(argv, cwd=str(root), stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = {**_process_identity(child.pid), "argv": argv, "log": str(log_path),
              "operation": operation, "attempt": attempt, "requested_at": requested_at}
    dump_json(record_path, record)
    return {"run_id": run_id, "detached": True, "attached": False, "attempt": attempt, "pid": child.pid,
            "requested_at": requested_at, "log": str(log_path)}


def _alive(spawn: Mapping[str, Any] | None) -> bool:
    """Whether the detached child is still running.

    On Linux the PID's start time and the boot ID rule out a reused PID.
    Where /proc is absent (a macOS test machine) the record has no start time,
    and the PID's existence is the best answer there is.
    """
    from .core import _same_process

    if not spawn:
        return False
    if spawn.get("pid_starttime") is not None:
        return _same_process(dict(spawn))
    try:
        os.kill(int(spawn["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (KeyError, ValueError, OSError):
        return False
    return True


def _tail(path: Path | None, lines: int = 30) -> list[str]:
    if path is None:
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def remote_status(state: Path, run_id: str, *, operation: str = "submit") -> dict[str, Any]:
    """Where a run stands on this host, for the latest detached ``operation`` on it.

    For a build, the run's own record answers once it exists. For a
    measurement, the record answers only once it names this attempt
    (``measure_request``, written by the measurement itself on this host's
    clock): until then an imported build's PASS is not the board stage's, and
    the run is ``STARTING`` while the child lives or ``FAILED_TO_START``, with
    its log, once it is gone.
    """
    from .core import status as run_status

    check_run_id(run_id)
    attempt, spawn = _latest(state, run_id, operation)
    log = Path(spawn["log"]) if spawn and spawn.get("log") else None
    alive = _alive(spawn)
    run_path = state / "runs" / run_id / "run.json"
    value: dict[str, Any] = {"run_id": run_id, "operation": operation, "attempt": attempt, "spawn_alive": alive}

    def not_yet() -> dict[str, Any]:
        if alive:
            value["status"] = "STARTING"
        elif spawn:
            value.update(status="FAILED_TO_START", log_tail=_tail(log))
        else:
            value["status"] = "UNKNOWN"
        return value

    if not run_path.is_file():
        return not_yet()
    if operation == "measure":
        if load_json(run_path).get("measure_request") != attempt or attempt is None:
            return not_yet()
    run = run_status(state, run_id)
    current = str(run.get("status"))
    value["status"] = current
    manifest = run.get("manifest") or {}
    value["run"] = {"candidate_id": manifest.get("candidate_id"), "target": manifest.get("target"),
                    "mode": manifest.get("mode"), "commit": (run.get("source") or {}).get("commit"),
                    "adapter": run.get("adapter")}
    if current in ("ORPHANED",) and log:
        value["log_tail"] = _tail(log)
    if current not in LIVE_STATUSES:
        value["result"] = run.get("result")
    return value


def worker_main(
    *,
    package_dir: Path,
    root: Path,
    stream: TextIO | None = None,
    output: TextIO | None = None,
) -> int:
    """Run one stdin request on a target host (the remote half of dispatch)."""

    stream = stream or sys.stdin
    output = output or sys.stdout
    try:
        request = json.load(stream)
        validate_worker_request(request)
        operation = request["operation"]
        manifest = request["manifest"]
        adapter = request["adapter"]
        state = Path(request.get("state_dir") or (root.parent / ".tritonchip-fpga-harness"))
        run_id: str | None = None
        # Imports stay here so importing the controller remains cheap and the
        # worker has the same implementation as the local Harness CLI.
        from .core import doctor, export_evidence, load_profile, measure_existing, source_identity, submit
        from .measurement import board_preflight

        if operation == "doctor":
            if request.get("scope") == "source":
                # Only which commit this checkout is: for a controller that
                # attaches to a build already running here, whose vendor
                # processes a full doctor would count against it.
                result = {"schema_version": 1, "target": manifest["target"], "scope": "source", "checks": [], "ok": True}
            elif request.get("scope") == "board":
                # A card host needs no Vivado: only what the board stage runs.
                checks = board_preflight(root=root, profile=load_profile(package_dir, manifest["target"]),
                                         manifest=manifest)
                result = {"schema_version": 1, "target": manifest["target"], "scope": "board",
                          "checks": checks, "ok": all(c["ok"] for c in checks)}
            else:
                result = doctor(package_dir, root, manifest["target"], adapter)
                result["scope"] = "build"
            source = source_identity(root)
            result["source"] = {"commit": source["commit"], "dirty": source["dirty"],
                                "untracked_files": source["untracked_files"][:20]}
            ok = bool(result.get("ok"))
            state_for_logs = None
        elif operation in ("submit", "measure") and request.get("detach"):
            result = _spawn(package_dir, root, state, request)
            run_id, ok, state_for_logs = result["run_id"], True, None
        elif operation == "submit":
            result = submit(
                package_dir,
                root,
                manifest,
                state,
                adapter,
                bool(request.get("allow_dirty", False)),
                request.get("timeout"),
                request.get("run_id"),
                bool(request.get("skip_board", False)),
            )
            run_id = result.get("run_id")
            ok = result.get("status") == "PASS"
            state_for_logs = state
        elif operation == "measure":
            run_id = check_run_id(request["run_id"])
            result = measure_existing(package_dir, root, state, run_id, request.get("timeout"),
                                      board_manifest=manifest if isinstance(manifest.get("board"), dict) else None,
                                      allow_source_mismatch=bool(request.get("allow_source_mismatch", False)))
            ok = result.get("status") == "PASS"
            state_for_logs = state
        elif operation == "status":
            run_id = request["run_id"]
            result = remote_status(state, run_id, operation=request.get("step", "submit"))
            ok, state_for_logs = result["status"] != "UNKNOWN", None
        else:  # export
            run_id = check_run_id(request["run_id"])
            packet = export_evidence(state, run_id, state / "outgoing", light=request.get("light", True))
            listed = load_json(packet / "evidence-manifest.json")["files"]
            result = {"run_id": run_id, "packet": str(packet), "parent": str(packet.parent),
                      "files": len(listed), "bytes": sum((packet / name).stat().st_size for name in listed),
                      "manifest_sha256": hashlib.sha256((packet / "evidence-manifest.json").read_bytes()).hexdigest()}
            ok, state_for_logs = True, None
        response = {
            "schema_version": 1,
            "ok": ok,
            "operation": operation,
            "board": request["board"],
            "result": result,
            "remote_log": summarize_run_logs(state_for_logs, run_id) if state_for_logs else {"run_id": None, "files": []},
            "worker_time": time.time(),
        }
        output.write(_canonical(response))
        output.flush()
        return 0 if ok else 1
    except (SchemaError, OSError, ValueError, RuntimeError) as exc:
        output.write(_canonical(_worker_error(str(exc))))
        output.flush()
        return 2
