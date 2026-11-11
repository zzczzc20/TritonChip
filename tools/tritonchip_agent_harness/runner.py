"""Run one agent once: the gateway, a container with no network, the CLI, and what it left behind.

Used by smoke runs (an empty workspace, a trivial prompt) and task runs (an
exported TNCA workspace, the task's prompt) alike, so a smoke pass means the
task run's launch path works.
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional

from . import adapters, doctor, provider, roofline
from . import skills as skills_
from .provider import AGENT_HOME, AgentContainer, Mount
from .siteconfig import Account, Site

BINARY = {"claude-code": "claude", "codex": "codex", "opencode": "opencode"}
# The harness files an agent's container gets: the bridge to the gateway's
# socket, and the skills directory (below). Never the harness directory
# itself, which holds the task packages (their point lists and reference
# patches) and the evaluator.
BRIDGE_SCRIPT = "/opt/tnca-bridge/bridge.py"
# The skills a run gets (skills.py): the selected ones are staged into the run's
# directory, that directory alone is mounted read-only at adapters.SKILLS_DIR, and
# the CLI's own skills directory links to what is mounted, so every CLI in every
# run discovers the same skills from the same bytes. The run record names them.
SKILLS_SOURCE = skills_.SOURCE
# The agent's side of the in-run services (services.py), and where their socket appears.
SERVICE_CLI = "/usr/local/bin/tnca-service"
SERVICE_DIR = "/run/tnca-service"
# The performance models' environment (task `models`, 2026-11-08): the site's SystemC installation and the
# harness's model hook (models/sitecustomize.py), both read-only. The models themselves are the agent's
# tools/performance_harness/systemc; their environment is the harness's, as the skills' tools are not.
SYSTEMC_DIR = "/opt/systemc"
MODELS_DIR = "/opt/tnca-models"
MODELS_SOURCE = Path(__file__).resolve().parent / "models"
# A Codex credential the CLI refreshed is copied back only if it still looks like one.
AUTH_JSON_LIMIT = 64 * 1024
# Where each CLI keeps its transcript inside the container (collected at the end).
TRANSCRIPTS = {"claude-code": ".claude-run/projects", "opencode": ".local/share/opencode"}


def agent_mounts(socket_dir: Path, workspace: Path, extra=(), skills: Optional[Path] = None,
                 systemc: Optional[Path] = None) -> list:
    """Everything a run's agent container sees from the host, and nothing else.

    `skills`: the run's staged skills directory (skills.stage), mounted read-only; None mounts no skill.
    `systemc`: the site's SystemC installation when the task asks for the models; with it, the model hook.
    """
    mounts = [Mount(doctor.HERE / "bridge.py", BRIDGE_SCRIPT), Mount(socket_dir, doctor.SOCKET_DIR),
              Mount(workspace, adapters.WORKDIR, read_only=False)]
    if skills is not None:
        mounts.append(Mount(skills, adapters.SKILLS_DIR))
    if systemc is not None:
        mounts += [Mount(systemc, SYSTEMC_DIR), Mount(MODELS_SOURCE, MODELS_DIR)]
    return mounts + list(extra)


def models_env(site: Site) -> dict:
    """The model environment's variables: where SystemC is (CMake's SystemCLanguage_DIR, the loader's path)
    and where the hook is. Raises when the site has no SystemC installation."""
    if site.systemc is None:
        raise RuntimeError("the task asks for the performance models and site.json names no `systemc`")
    return {"SYSTEMC_HOME": SYSTEMC_DIR, "SystemCLanguage_DIR": f"{SYSTEMC_DIR}/lib/cmake/SystemCLanguage",
            "TNCA_MODEL_HOOK": MODELS_DIR}


def models_manifest(site: Site) -> dict:
    """For the run record: the SystemC installation mounted and the hook's hash."""
    h = hashlib.sha256()
    for f in sorted(MODELS_SOURCE.rglob("*.py")):
        h.update(f.relative_to(MODELS_SOURCE).as_posix().encode() + b"\0" + f.read_bytes())
    return {"systemc": str(site.systemc), "hook_sha256": h.hexdigest()}


def link_skills(name: str, skills_dir: str, timeout: float = 60) -> dict:
    """Link every mounted skill into the CLI's skills directory, inside the running container.

    A link, not a copy: the source stays read-only, and the CLI's own listing
    is what the record relies on. Returns what the directory then holds.
    """
    script = (f"mkdir -p {shlex.quote(skills_dir)} && for s in {adapters.SKILLS_DIR}/*/; do "
              f"ln -s \"${{s%/}}\" {shlex.quote(skills_dir)}/; done && ls -1 {shlex.quote(skills_dir)}")
    proc = _exec(name, ["sh", "-c", script], timeout=timeout)
    return {"dir": skills_dir, "linked": proc.stdout.split() if proc.returncode == 0 else [],
            "error": (proc.stderr or "").strip()[:300] if proc.returncode != 0 else None}


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


def _turn(agent: str, account: Account, command: list, timeout: float) -> tuple:
    """One CLI invocation in the running sandbox: its output, and how it ended."""
    started = time.time()
    try:
        proc = _exec(agent, command, timeout=max(1.0, timeout))
        record = {"stop_reason": "exited", "exit": proc.returncode}
        stdout, stderr = proc.stdout, proc.stderr
    except Exception as exc:  # subprocess.TimeoutExpired, chiefly
        provider.docker("exec", agent, "pkill", "-TERM", "-f", BINARY[account.cli], check=False, timeout=30)
        record = {"stop_reason": "timeout" if "Timeout" in type(exc).__name__ else f"error: {type(exc).__name__}",
                  "exit": None}
        stdout = (getattr(exc, "stdout", None) or b"")
        stderr = (getattr(exc, "stderr", None) or b"")
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    record["seconds"] = round(time.time() - started, 1)
    return stdout, stderr, record


def run_agent(site: Site, account: Account, prompt: str, mode: str, run_dir: Path,
              workspace: Optional[Path], timeout: float, services=None, skills: Optional[list] = None,
              turns=None, name: Optional[str] = None, models: bool = False) -> dict:
    """`services`, when given, is a started-on-demand services.Broker the agent reaches with `tnca-service`.

    `turns`, an island's controller: called after each CLI run with that turn's record and the seconds left;
    it returns the next prompt (the same session), `(prompt, True)` for a new session, or None to stop.
    `name`: the sandbox's name suffix when several run at once (islands); the run's id otherwise.

    `skills`: the names of the skills this run gets (skills.select), staged and mounted; none by default.
    `models`: the task asks for the performance models' environment (SYSTEMC_DIR, MODELS_DIR).
    """
    selected = skills_.select(skills or [])
    model_env = models_env(site) if models else {}
    run_id = name or run_dir.name
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
    container_started = None
    try:
        socket_dir, log = doctor.start_gateway(site, run_dir, gw, list(account.hosts), open_route=open_route)
        extra, env = list(launch.mounts), {**dict(launch.env), **model_env}
        if services is not None:
            extra += [Mount(services.start(), SERVICE_DIR), Mount(doctor.HERE / "service_cli.py", SERVICE_CLI)]
            env["TNCA_SERVICE"] = f"{SERVICE_DIR}/service.sock"
            # The card's roof, for the test-case skill's eta (roofline.agent_roof: the machine, never a point).
            roof = roofline.agent_roof(getattr(getattr(services, "shared", None), "roofline", None))
            if roof:
                env[roofline.ENV] = json.dumps(roof, sort_keys=True, separators=(",", ":"))
                result["chip_roof"] = roof
        staged = skills_.stage(selected, run_dir / "skills") if selected else None
        mounts = agent_mounts(socket_dir, workspace, extra, skills=staged,
                              systemc=site.systemc if models else None)
        if models:
            result["models"] = models_manifest(site)
        c = AgentContainer(name=agent, image=site.images["agent"], cpus=site.cpus, memory=site.memory,
                           pids=site.pids, mounts=mounts, env=env, env_files=launch.env_files,
                           labels={"tritonchip-agent-harness": "agent", "run": run_id}, workdir=adapters.WORKDIR)
        container_started = time.time()
        provider.docker(*c.run_args(["python3", BRIDGE_SCRIPT, "--listen", doctor.BRIDGE,
                                     "--socket", doctor.SOCKET], detach=True, remove=False))
        time.sleep(1)
        version = _exec(agent, [BINARY[account.cli], "--version"], timeout=60)
        result["cli_version"] = (version.stdout.strip().splitlines() or [None])[-1]
        result["skills"] = {"provided": skills_.manifest(selected), "catalog": skills_.version(),
                            **(link_skills(agent, launch.skills_dir) if selected else {})}
        start = time.time()
        command, turn, session, turns_log, total = launch.command, 0, None, [], {}
        running = None       # Codex: the run's usage so far, after the last turn
        by_session: dict = {}  # Claude Code: each session's usage so far, after its last turn
        while True:
            left = timeout - (time.time() - start)
            stdout, stderr, record = _turn(agent, account, command, left)
            suffix = "" if turn == 0 else f"-{turn}"
            (out_dir / f"stdout{suffix}.jsonl").write_text(stdout or "", encoding="utf-8")
            (out_dir / f"stderr{suffix}.txt").write_text(stderr or "", encoding="utf-8")
            exporter = (lambda sid: _exec(agent, ["opencode", "export", sid], timeout=120).stdout) \
                if account.cli == "opencode" else None
            parsed = adapters.parse(account, stdout or "", run_dir, exporter)
            if account.cli == "codex":
                # Codex's usage is the run's running total (adapters.codex_rollouts): this turn's is the difference,
                # and the run's is the last.
                turn_tokens = adapters.token_delta(parsed.get("tokens"), running)
                running = parsed.get("tokens") or running
                total = dict(running or {})
            elif account.cli == "claude-code":
                # Claude Code's result counts its session so far, a resumed one included (R41-F5's smoke,
                # 2026-11-11: a 4-second resumed turn reported its session's 4.1M tokens): a turn's share is the
                # difference from the same session's last count, and the run's is each session's last, added up.
                sid = adapters.session_id(account, stdout or "")
                now = parsed.get("tokens")
                turn_tokens = adapters.token_delta(now, by_session.get(sid)) if sid else now
                if now is not None:
                    by_session[sid or f"turn-{turn}"] = now
                total = adapters.token_sum(by_session.values())
            else:
                turn_tokens = parsed.get("tokens")
                for k, v in (turn_tokens or {}).items():
                    if isinstance(v, (int, float)):
                        total[k] = total.get(k, 0) + v
            record.update(turn=turn, tool_calls=adapters.tool_calls(account, stdout or ""),
                          throttled=adapters.throttle(account, stdout or "", stderr or ""),
                          tokens=turn_tokens)
            session = adapters.session_id(account, stdout or "") or session
            record["session"] = session
            turns_log.append(record)
            result.update(parsed)
            result.update(stop_reason=record["stop_reason"], exit=record["exit"], throttled=record["throttled"])
            # One CLI run, or an island's turns (report section 8.4 D2/D7): the controller says whether, and
            # with what prompt, the agent goes on; the same session when the CLI gave one, else a new one.
            nxt = turns(record, timeout - (time.time() - start)) if turns is not None else None
            if nxt is None or timeout - (time.time() - start) < 30:
                break
            turn += 1
            prompt_next, fresh = nxt if isinstance(nxt, tuple) else (nxt, False)
            use = None if fresh else session
            command = adapters.launch(account, prompt_next, mode, run_dir, proxy_env(), session=use).command
            record["next"] = "resume" if use else "new-session"
        result["seconds"] = round(time.time() - start, 1)
        if turns is not None:
            result["turns"] = turns_log
            result["tokens"] = total or result.get("tokens")
        _collect(agent, account, out_dir / "transcript")
    finally:
        provider.remove(agent)
        if container_started is not None:
            # The sandbox's occupancy of this host: from `docker run` to the container's removal.
            result["container"] = {"started": container_started, "finished": time.time()}
        doctor.stop_gateway(gw)
        if services is not None:
            services.stop()
            result["services"] = services.record()

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


__all__ = ["BRIDGE_SCRIPT", "MODELS_DIR", "SERVICE_CLI", "SKILLS_SOURCE", "SYSTEMC_DIR", "agent_mounts",
           "codex_auth_ok", "link_skills", "models_env", "models_manifest", "proxy_env", "run_agent"]
