"""A run split over two servers: the Vivado flow on one, the card on another.

The controller -- any machine whose SSH configuration reaches both servers --
does what one ``submit`` does on a single host, with each half where it runs
best (hosts.py says which server is which)::

    sync      a clean checkout of the commit on each server, from a git bundle
    doctor    the build server for the flow; the card server for the board stage
    build     `submit --skip-board`, detached on the build server; polled
    export    a light packet of that run: the image, its records, the routed reports
    transfer  the packet as a tar stream, build server -> controller -> card server,
              where `import` checks every file's hash before installing the run
    measure   `measure RUN_ID`, detached on the card server; polled

The servers never talk to each other, and neither needs the repository's
remote. Each step is recorded with its timing and outcome; the record is the
result. A build or measurement survives the controller: ``--build-run`` picks
up a build that is running or done, and a failed transfer or board stage can be
repeated without building again.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .hosts import Host, host_for
from .schema import validate_manifest
from .worker import LIVE_STATUSES, _board_for_target, build_ssh_argv, call_worker, ssh_prefix

SSH_OPTIONS = ("BatchMode=yes", "ServerAliveInterval=30", "ServerAliveCountMax=6")
CONNECT_TIMEOUT = 20
# A status poll that fails is retried; this many in a row ends the wait.
MAX_POLL_FAILURES = 10


class PipelineError(RuntimeError):
    """A step of a split run could not be completed."""


def _log(message: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _ssh(host: Host, ssh_binary: str, *, env: bool) -> list[str]:
    return ssh_prefix(host=host.ssh, port=host.port, identity_file=host.identity_file,
                      connect_timeout=CONNECT_TIMEOUT, options=SSH_OPTIONS,
                      env=host.remote_env() if env else None, ssh_binary=ssh_binary)


def remote(host: Host, argv: list[str], *, ssh_binary: str = "ssh", data: bytes | None = None,
           timeout: int | None = 600, check: bool = True) -> subprocess.CompletedProcess:
    """Run one plain command on ``host`` (git, tar, mkdir, dd), without a shell string."""
    full = _ssh(host, ssh_binary, env=False) + list(argv)
    done = subprocess.run(full, input=data, capture_output=True, timeout=timeout, check=False)
    if check and done.returncode != 0:
        detail = done.stderr.decode("utf-8", errors="replace").strip().splitlines()[-3:]
        raise PipelineError(f"{host.name}: {' '.join(argv[:4])} failed ({done.returncode}): {' | '.join(detail)}")
    return done


def _out(done: subprocess.CompletedProcess) -> str:
    return done.stdout.decode("utf-8", errors="replace").strip()


def resolve_commit(local_repo: Path, wanted: str | None) -> str:
    done = subprocess.run(["git", "-C", str(local_repo), "rev-parse", "--verify", f"{wanted or 'HEAD'}^{{commit}}"],
                          capture_output=True, text=True, check=False)
    if done.returncode:
        raise PipelineError(f"commit {wanted or 'HEAD'!r} is not in {local_repo}: {done.stderr.strip()}")
    return done.stdout.strip()


def sync(host: Host, commit: str, local_repo: Path, *, ssh_binary: str = "ssh") -> dict[str, Any]:
    """Make ``<workspace>/checkouts/<commit>`` a clean checkout of ``commit`` on ``host``.

    A checkout that is already there is used if it is at the commit and clean,
    and refused otherwise; nothing on a server is ever deleted. A missing one
    is created as a worktree of the server's bare mirror, after the controller
    has sent the commit as a git bundle (the mirror is fetched from the bundle
    only when it does not have the commit yet).
    """
    checkout = host.checkout(commit)
    present = remote(host, ["git", "-C", checkout, "rev-parse", "HEAD"], ssh_binary=ssh_binary, check=False)
    if present.returncode == 0:
        head = _out(present)
        if head != commit:
            raise PipelineError(f"{host.name}: {checkout} is at {head}, not {commit}; it is left as it is")
        changes = _out(remote(host, ["git", "-C", checkout, "status", "--porcelain", "--untracked-files=no"],
                              ssh_binary=ssh_binary))
        if changes:
            raise PipelineError(f"{host.name}: {checkout} has local changes; it is left as it is")
        return {"host": host.name, "checkout": checkout, "action": "present"}
    remote(host, ["mkdir", "-p", host.incoming, f"{host.workspace}/checkouts"], ssh_binary=ssh_binary)
    remote(host, ["git", "init", "--bare", "-q", host.mirror], ssh_binary=ssh_binary)
    has = remote(host, ["git", "-C", host.mirror, "cat-file", "-e", f"{commit}^{{commit}}"],
                 ssh_binary=ssh_binary, check=False)
    sent = 0
    if has.returncode != 0:
        ref = f"refs/tritonchip-fpga-harness/sync/{commit}"
        subprocess.run(["git", "-C", str(local_repo), "update-ref", ref, commit], check=True, capture_output=True)
        try:
            bundle = subprocess.run(["git", "-C", str(local_repo), "bundle", "create", "-", ref],
                                    capture_output=True, check=True).stdout
        finally:
            subprocess.run(["git", "-C", str(local_repo), "update-ref", "-d", ref], check=False, capture_output=True)
        path = f"{host.incoming}/{commit}.bundle"
        remote(host, ["dd", f"of={path}", "bs=1048576"], ssh_binary=ssh_binary, data=bundle)
        try:
            remote(host, ["git", "-C", host.mirror, "fetch", "-q", path, f"{ref}:{ref}"], ssh_binary=ssh_binary)
        finally:
            remote(host, ["rm", "-f", path], ssh_binary=ssh_binary, check=False)
        sent = len(bundle)
    remote(host, ["git", "-C", host.mirror, "worktree", "add", "-q", "--detach", checkout, commit],
           ssh_binary=ssh_binary)
    head = _out(remote(host, ["git", "-C", checkout, "rev-parse", "HEAD"], ssh_binary=ssh_binary))
    if head != commit:
        raise PipelineError(f"{host.name}: the new checkout is at {head}, not {commit}")
    return {"host": host.name, "checkout": checkout, "action": "created", "bundle_bytes": sent}


def worker(host: Host, commit: str, request: dict[str, Any], *, ssh_binary: str = "ssh",
           timeout: int | None = 900) -> dict[str, Any]:
    """One worker request to ``host``, run by the harness of the commit's checkout there."""
    argv = build_ssh_argv(host=host.ssh, remote_repo=host.checkout(commit), operation=request["operation"],
                          adapter=request["adapter"], remote_python=host.python, remote_cli=host.cli(commit),
                          port=host.port, identity_file=host.identity_file, connect_timeout=CONNECT_TIMEOUT,
                          options=SSH_OPTIONS, env=host.remote_env(), ssh_binary=ssh_binary)
    return call_worker(dict(request, state_dir=host.state_dir), argv, ssh_timeout=timeout)


def _remote_value(reply: dict[str, Any], what: str) -> dict[str, Any]:
    remote_value = reply.get("remote") or {}
    if not reply.get("ok"):
        error = reply.get("error") or remote_value.get("error") or (remote_value.get("result") or {}).get("error")
        raise PipelineError(f"{what}: {error or 'the worker said no'}")
    return remote_value.get("result") or {}


def poll(host: Host, commit: str, base: dict[str, Any], run_id: str, *, since: float | None = None,
         poll_s: float = 60.0, deadline: float | None = None, ssh_binary: str = "ssh",
         sleep: Callable[[float], None] = time.sleep, label: str = "") -> dict[str, Any]:
    """Ask ``status`` until the run leaves the live states; transient failures are retried."""
    failures = 0
    last = None
    while True:
        request = dict(base, operation="status", run_id=run_id)
        if since is not None:
            request["since"] = since
        reply = worker(host, commit, request, ssh_binary=ssh_binary, timeout=120)
        value = (reply.get("remote") or {}).get("result")
        if reply.get("ok") and isinstance(value, dict):
            failures = 0
            current = value.get("status")
            if current != last:
                _log(f"{label} {run_id} on {host.name}: {current}")
                last = current
            if current not in LIVE_STATUSES:
                return value
        else:
            failures += 1
            _log(f"{label} status poll on {host.name} failed ({failures}/{MAX_POLL_FAILURES}): "
                 f"{reply.get('error') or (reply.get('remote') or {}).get('error')}")
            if failures >= MAX_POLL_FAILURES:
                raise PipelineError(f"{host.name} did not answer {MAX_POLL_FAILURES} status polls in a row; "
                                    f"run {run_id} may still be running there")
        if deadline is not None and time.time() > deadline:
            raise PipelineError(f"run {run_id} on {host.name} was still {last} at the controller's deadline")
        sleep(poll_s)


def transfer(source: Host, packet_parent: str, run_id: str, destination: Host, commit: str, *,
             ssh_binary: str = "ssh", timeout: int | None = 1800) -> dict[str, Any]:
    """Stream a packet from the build server into `import` on the card server."""
    send = _ssh(source, ssh_binary, env=False) + ["tar", "-C", packet_parent, "-cf", "-", run_id]
    receive = _ssh(destination, ssh_binary, env=True) + [
        destination.python, destination.cli(commit), "--repo", destination.checkout(commit),
        "import", "--state-dir", destination.state_dir, "--from-tar", "-",
    ]
    started = time.time()
    sender = subprocess.Popen(send, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    receiver = subprocess.Popen(receive, stdin=sender.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert sender.stdout is not None
    sender.stdout.close()
    try:
        out, err = receiver.communicate(timeout=timeout)
        send_err = sender.stderr.read() if sender.stderr else b""
        sender.wait(timeout=60)
    except subprocess.TimeoutExpired:
        sender.kill()
        receiver.kill()
        raise PipelineError(f"the transfer of {run_id} did not finish in {timeout}s")
    if sender.returncode != 0:
        raise PipelineError(f"{source.name}: tar of the packet failed: "
                            f"{send_err.decode('utf-8', errors='replace').strip()[-300:]}")
    text = out.decode("utf-8", errors="replace")
    try:
        value = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        value = {}
    if receiver.returncode != 0 or not value.get("ok"):
        raise PipelineError(f"{destination.name}: import refused the packet: "
                            f"{err.decode('utf-8', errors='replace').strip()[-400:] or text[-400:]}")
    value["elapsed_s"] = round(time.time() - started, 1)
    return value


def _build_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    result = value.get("result") or {}
    validation = result.get("artifact_validation") or {}
    artifacts = result.get("artifacts") or {}
    return {
        "status": value.get("status"),
        "elapsed_s": result.get("elapsed_s"),
        "observed_clock_mhz": validation.get("observed_clock_mhz"),
        "timing": validation.get("timing"),
        "xclbin_sha256": next((digest for name, digest in artifacts.items() if name.endswith(".xclbin")), None),
        "error": result.get("error"),
        "preflight_failed": result.get("preflight_failed"),
        "log_tail": value.get("log_tail"),
    }


def _board_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    result = value.get("result") or {}
    measurement = result.get("measurement") or {}
    archbench = measurement.get("archbench") or {}
    return {
        "status": value.get("status"),
        "failed_stage": measurement.get("failed_stage"),
        "error": result.get("measurement_error") or measurement.get("error"),
        "loaded_xclbin_uuid": measurement.get("loaded_xclbin_uuid"),
        "replaced_xclbin_uuid": measurement.get("replaced_xclbin_uuid"),
        "reconfigured": measurement.get("reconfigured"),
        "xclbin_sha256": measurement.get("xclbin_sha256"),
        "software": measurement.get("software"),
        "conformance": measurement.get("conformance"),
        "archbench": {key: archbench.get(key) for key in ("status", "records", "passed", "failed", "total_device_cycles")
                      if key in archbench} or None,
        "log_tail": value.get("log_tail"),
    }


def run_pipeline(
    *,
    local_repo: Path,
    hosts: Mapping[str, Host],
    manifest: dict[str, Any],
    build_host: str,
    board_host: str | None = None,
    adapter: str | None = None,
    timeout: int | None = None,
    poll_s: float = 60.0,
    skip_board: bool = False,
    build_run: str | None = None,
    do_sync: bool = True,
    ssh_binary: str = "ssh",
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    validate_manifest(manifest)
    builder = host_for(hosts, build_host, "build")
    want_board = bool(manifest.get("board")) and manifest.get("mode") == "full" and not skip_board
    card = host_for(hosts, board_host, "board") if board_host and want_board else None
    if want_board and card is None:
        raise PipelineError("the manifest has a board block: name a --board-host, or pass --skip-board")
    commit = resolve_commit(local_repo, manifest.get("source_commit") if manifest.get("source_commit") != "HEAD" else None)
    manifest = dict(manifest, source_commit=commit)
    board = _board_for_target(str(manifest["target"]))
    adapter = adapter or board
    base = {"schema_version": 1, "board": board, "adapter": adapter, "manifest": manifest}
    record: dict[str, Any] = {
        "schema_version": 1,
        "schema": "tritonchip-fpga-harness-pipeline/1",
        "candidate_id": manifest["candidate_id"],
        "target": manifest["target"],
        "commit": commit,
        "adapter": adapter,
        "hosts": {"build": builder.summary(), "board": card.summary() if card else None},
        "started_at": time.time(),
        "steps": [],
        "status": "RUNNING",
    }

    def step(name: str, action: Callable[[], Any]) -> Any:
        started = time.time()
        _log(f"{name} ...")
        entry: dict[str, Any] = {"name": name, "started_at": started}
        record["steps"].append(entry)
        try:
            value = action()
        except Exception as exc:
            entry.update(ok=False, elapsed_s=round(time.time() - started, 1), error=str(exc))
            raise
        entry.update(ok=True, elapsed_s=round(time.time() - started, 1))
        if isinstance(value, dict):
            entry["detail"] = value
        return value

    def finish(status: str, error: str | None = None) -> dict[str, Any]:
        record.update(status=status, ok=status == "PASS", finished_at=time.time(),
                      elapsed_s=round(time.time() - record["started_at"], 1))
        if error:
            record["error"] = error
        _log(f"done: {status}" + (f" ({error})" if error else ""))
        return record

    deadline = None if timeout is None else time.time() + timeout + 3600
    try:
        servers = [builder] + ([card] if card and card.name != builder.name else [])
        if do_sync:
            for server in servers:
                step(f"sync:{server.name}", lambda server=server: sync(server, commit, local_repo, ssh_binary=ssh_binary))

        def doctor(server: Host, scope: str) -> dict[str, Any]:
            value = _remote_value(worker(server, commit, dict(base, operation="doctor", scope=scope),
                                         ssh_binary=ssh_binary), f"doctor ({scope}) on {server.name}")
            source = value.get("source") or {}
            if source.get("commit") != commit or source.get("dirty"):
                raise PipelineError(f"{server.name}: the checkout is {source.get('commit')} "
                                    f"(dirty: {source.get('dirty')}), not a clean {commit}")
            failed = [c for c in value.get("checks", []) if not c.get("ok") and c.get("severity", "error") == "error"]
            if failed:
                raise PipelineError(f"{server.name} is not ready ({scope}): "
                                    + "; ".join(f"{c['name']} ({c.get('detail')})" for c in failed[:6]))
            return {"scope": scope, "checks": len(value.get("checks", [])),
                    "tool_roots": (value.get("environment") or {}).get("tool_roots")}

        step(f"doctor:{builder.name}", lambda: doctor(builder, "build"))
        if card:
            step(f"doctor:{card.name}", lambda: doctor(card, "board"))

        if build_run:
            run_id = build_run
        else:
            request = dict(base, operation="submit", detach=True, skip_board=True)
            if timeout:
                request["timeout"] = timeout
            started = step("build:submit", lambda: _remote_value(
                worker(builder, commit, request, ssh_binary=ssh_binary), f"submit on {builder.name}"))
            run_id = started["run_id"]
        record["run_id"] = run_id
        built = step("build", lambda: poll(builder, commit, base, run_id, poll_s=poll_s, deadline=deadline,
                                           ssh_binary=ssh_binary, sleep=sleep, label="build"))
        record["build"] = dict(_build_summary(built), run_id=run_id, host=builder.name)
        if built.get("status") != "PASS":
            return finish(str(built.get("status")), "the build did not pass")
        if not want_board:
            return finish("PASS")

        packet = step("export", lambda: _remote_value(
            worker(builder, commit, dict(base, operation="export", run_id=run_id, light=True), ssh_binary=ssh_binary),
            f"export on {builder.name}"))
        assert card is not None
        imported = step("transfer", lambda: transfer(builder, packet["parent"], run_id, card, commit,
                                                     ssh_binary=ssh_binary))
        if imported.get("xclbin_sha256") != record["build"]["xclbin_sha256"]:
            raise PipelineError("the card server installed an image whose hash is not the build's")
        request = dict(base, operation="measure", run_id=run_id, detach=True)
        if timeout:
            request["timeout"] = timeout
        asked = step("board:measure", lambda: _remote_value(
            worker(card, commit, request, ssh_binary=ssh_binary), f"measure on {card.name}"))
        measured = step("board", lambda: poll(card, commit, base, run_id, since=float(asked["requested_at"]),
                                              poll_s=poll_s, deadline=deadline, ssh_binary=ssh_binary,
                                              sleep=sleep, label="board"))
        record["board"] = dict(_board_summary(measured), run_id=run_id, host=card.name)
        return finish(str(measured.get("status")))
    except (PipelineError, OSError, subprocess.SubprocessError, ValueError) as exc:
        return finish("PIPELINE_FAIL", str(exc))


__all__ = ["PipelineError", "poll", "remote", "resolve_commit", "run_pipeline", "sync", "transfer", "worker"]
