"""Run one agent once: the gateway, a container with no network, the CLI, and what it left behind.

Used by smoke runs (an empty workspace, a trivial prompt) and task runs (an
exported TNCA workspace, the task's prompt) alike, so a smoke pass means the
task run's launch path works.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional

from . import adapters, doctor, provider
from .provider import AGENT_HOME, AgentContainer, Mount
from .siteconfig import Account, Site

BINARY = {"claude-code": "claude", "codex": "codex", "opencode": "opencode"}
# The one harness file an agent's container gets: the bridge to the gateway's
# socket. Never the harness directory, which holds the task packages (their
# point lists and reference patches) and the evaluator.
BRIDGE_SCRIPT = "/opt/tnca-bridge/bridge.py"
# A Codex credential the CLI refreshed is copied back only if it still looks like one.
AUTH_JSON_LIMIT = 64 * 1024
# Where each CLI keeps its transcript inside the container (collected at the end).
TRANSCRIPTS = {"claude-code": ".claude-run/projects", "opencode": ".local/share/opencode"}


def agent_mounts(socket_dir: Path, workspace: Path, extra=()) -> list:
    """Everything a run's agent container sees from the host, and nothing else."""
    return [Mount(doctor.HERE / "bridge.py", BRIDGE_SCRIPT), Mount(socket_dir, doctor.SOCKET_DIR),
            Mount(workspace, adapters.WORKDIR, read_only=False), *extra]


def codex_auth_ok(copy: Path, master: Path) -> Optional[str]:
    """Why the sandbox's auth.json must not replace the operator's, or None if it may.

    The copy sat in a directory the agent could write, so it is taken back only
    if it is a regular file of plausible size whose JSON has exactly the
    master's top-level keys.
    """
    import os
    import stat
    try:
        st = os.lstat(copy)
    except OSError:
        return "missing"
    if not stat.S_ISREG(st.st_mode):
        return "not a regular file"
    if st.st_size > AUTH_JSON_LIMIT:
        return "too large"
    try:
        new, old = json.loads(copy.read_text(encoding="utf-8")), json.loads(master.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "not JSON"
    if not isinstance(new, dict) or not isinstance(old, dict) or set(new) != set(old):
        return "not the same shape as the master"
    return None


def proxy_env() -> dict:
    env = {k: f"http://{doctor.BRIDGE}" for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")}
    env.update(NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
    return env


def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _exec(name: str, command: list, timeout: float, workdir: str = adapters.WORKDIR):
    return provider.docker("exec", "-w", workdir, name, *command, check=False, timeout=timeout)


def _collect(name: str, account: Account, dest: Path) -> None:
    """Copy the CLI's own transcript out of the container's (tmpfs) home, without any credential."""
    sub = TRANSCRIPTS.get(account.cli)
    if not sub:
        return
    proc = provider.docker("exec", name, "tar", "-C", AGENT_HOME, "-cf", "-", sub, check=False, timeout=120,
                           text=False)
    if proc.returncode != 0 or not proc.stdout:
        return
    # The agent controls what is in there: take regular files only, by name
    # checked here, so a crafted archive cannot write outside `dest`.
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if (not member.isfile() or member.name.startswith("/") or ".." in parts
                    or parts[-1] in ("auth.json", ".credentials.json")):
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                target.write_bytes(src.read())


def run_agent(site: Site, account: Account, prompt: str, mode: str, run_dir: Path,
              workspace: Optional[Path], timeout: float) -> dict:
    run_id = run_dir.name
    gw, agent = f"tnca-gw-{run_id}", f"tnca-agent-{run_id}"
    result: dict = {"label": account.label, "cli": account.cli, "auth": account.auth, "mode": mode,
                    "requested_model": account.models["main"], "credential_in_sandbox": True,
                    "images": {k: provider.image_id(v) for k, v in site.images.items()}}
    out_dir = run_dir / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    if workspace is None:
        workspace = run_dir / "empty-workspace"
        workspace.mkdir(exist_ok=True)

    original = None
    launch = adapters.launch(account, prompt, mode, run_dir, proxy_env())
    if account.cli == "codex":
        shutil.copy2(account.credential / "auth.json", run_dir / "codex-home" / "auth.json")
        original = _sha(run_dir / "codex-home" / "auth.json")
    result["switches"] = launch.switches

    open_route = None
    if mode == "open":
        open_route = "upstream" if site.upstream_proxy else "direct"
    try:
        socket_dir, log = doctor.start_gateway(site, run_dir, gw, list(account.hosts), open_route=open_route)
        mounts = agent_mounts(socket_dir, workspace, launch.mounts)
        c = AgentContainer(name=agent, image=site.images["agent"], cpus=site.cpus, memory=site.memory,
                           pids=site.pids, mounts=mounts, env=launch.env, env_files=launch.env_files,
                           labels={"tritonchip-agent-harness": "agent", "run": run_id}, workdir=adapters.WORKDIR)
        provider.docker(*c.run_args(["python3", BRIDGE_SCRIPT, "--listen", doctor.BRIDGE,
                                     "--socket", doctor.SOCKET], detach=True, remove=False))
        time.sleep(1)
        version = _exec(agent, [BINARY[account.cli], "--version"], timeout=60)
        result["cli_version"] = (version.stdout.strip().splitlines() or [None])[-1]
        start = time.time()
        try:
            proc = _exec(agent, launch.command, timeout=timeout)
            result["stop_reason"] = "exited"
            result["exit"] = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
        except Exception as exc:  # subprocess.TimeoutExpired, chiefly
            provider.docker("exec", agent, "pkill", "-TERM", "-f", BINARY[account.cli], check=False, timeout=30)
            result["stop_reason"] = "timeout" if "Timeout" in type(exc).__name__ else f"error: {type(exc).__name__}"
            result["exit"] = None
            stdout = (getattr(exc, "stdout", None) or b"")
            stderr = (getattr(exc, "stderr", None) or b"")
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        result["seconds"] = round(time.time() - start, 1)
        (out_dir / "stdout.jsonl").write_text(stdout or "", encoding="utf-8")
        (out_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
        exporter = (lambda sid: _exec(agent, ["opencode", "export", sid], timeout=120).stdout) \
            if account.cli == "opencode" else None
        result.update(adapters.parse(account, stdout or "", run_dir, exporter))
        _collect(agent, account, out_dir / "transcript")
    finally:
        provider.remove(agent)
        doctor.stop_gateway(gw)

    entries = doctor.read_log(run_dir / "gateway-log" / "tunnels.jsonl")
    tunnels = [e for e in entries if e.get("event") == "tunnel"]
    result["gateway"] = {
        "tunnels": {h: {"count": sum(1 for t in tunnels if t["host"] == h),
                        "bytes_up": sum(t["bytes_up"] for t in tunnels if t["host"] == h),
                        "bytes_down": sum(t["bytes_down"] for t in tunnels if t["host"] == h),
                        "listed": h in account.hosts}
                    for h in sorted({t["host"] for t in tunnels})},
        "refused": sorted({e.get("host") for e in entries if e.get("event") == "refused" and e.get("host")}),
    }

    if account.cli == "codex":
        copy = run_dir / "codex-home" / "auth.json"
        result["credential_refreshed"] = _sha(copy) != original
        if result["credential_refreshed"]:
            master = account.credential / "auth.json"
            refused = None if _sha(master) == original else "the master changed during the run"
            refused = refused or codex_auth_ok(copy, master)
            result["credential_written_back"] = refused is None
            result["credential_write_back_refused"] = refused
            if refused is None:
                shutil.copy2(copy, master)
        copy.unlink(missing_ok=True)
    return result


__all__ = ["BRIDGE_SCRIPT", "agent_mounts", "codex_auth_ok", "proxy_env", "run_agent"]
