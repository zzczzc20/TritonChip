"""SSH worker and dispatch protocol for remote FPGA Harness hosts.

The controller sends one JSON request on stdin.  The remote process reads that
request, invokes the existing Harness ``doctor`` or ``submit`` API, and emits
one JSON response on stdout.  Commands are assembled as argv vectors and every
value that becomes an SSH/remote argument is checked before it reaches
``subprocess.run``.  Manifests and logs therefore never need shell quoting or a
temporary file in the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from .schema import SchemaError, validate_manifest, validate_worker_request


class DispatchError(RuntimeError):
    """A controller or remote worker protocol error."""


# OpenSSH receives a command vector but executes the remote command through the
# user's login shell.  Reject shell syntax and whitespace in path/option values
# so the controller never relies on shell escaping.  The manifest itself is
# data on stdin and is intentionally not restricted by this check.
_SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")


def _safe_arg(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_ARG.fullmatch(value):
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
) -> list[str]:
    """Build the fixed remote worker argv without a shell command string."""

    if operation not in {"doctor", "submit"}:
        raise DispatchError(f"unsupported dispatch operation {operation!r}")
    if adapter not in {"u50", "u280"}:
        raise DispatchError(f"remote FPGA adapter must be u50 or u280, got {adapter!r}")
    for value, name in (
        (ssh_binary, "ssh_binary"),
        (host, "host"),
        (remote_repo, "remote_repo"),
        (remote_python, "remote_python"),
        (remote_cli, "remote_cli"),
    ):
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
    argv.extend(
        [
            host,
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
                           "stdout": _log_summary(exc.stdout or ""),
                           "stderr": _log_summary(exc.stderr or "")},
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


def _worker_error(message: str) -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "error": message}


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
        # Imports stay here so importing the controller remains cheap and the
        # worker has the same implementation as the local Harness CLI.
        from .core import doctor, submit

        if operation == "doctor":
            result = doctor(package_dir, root, manifest["target"], adapter)
            run_id = None
            state = None
            ok = bool(result.get("ok"))
        else:
            state = Path(request.get("state_dir") or (root.parent / ".tritonchip-fpga-harness"))
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
        response = {
            "schema_version": 1,
            "ok": ok,
            "operation": operation,
            "board": request["board"],
            "result": result,
            "remote_log": summarize_run_logs(state, run_id) if state else {"run_id": None, "files": []},
        }
        output.write(_canonical(response))
        output.flush()
        return 0 if ok else 1
    except (SchemaError, OSError, ValueError, RuntimeError) as exc:
        output.write(_canonical(_worker_error(str(exc))))
        output.flush()
        return 2

