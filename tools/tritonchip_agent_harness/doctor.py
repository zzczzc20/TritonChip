"""doctor: can this host give an agent run the guarantees, measured rather than assumed.

It starts containers through the same provider code a run uses and asks
canary.py, inside them, what an agent there could reach. A host passes only if
every guarantee holds; anything else is a reason, in the report, to refuse
formal runs here.

    1. the daemon and the host class (rootful, or rootless on cgroup v2: formal; else development only)
    2. both images present
    3. every account's credential in place, from its metadata
    4. isolation inside an agent's container (canary checks)
    5. nothing carries over from one run to the next (a second canary looks for the first one's marker)
    6. the memory limit stops a process (a canary allocates past it)
    7. the gateway: a listed host through it, an unlisted one refused, nothing direct
"""

from __future__ import annotations

import getpass
import json
import platform
import secrets
import shutil
import time
from pathlib import Path
from typing import Optional

from . import provider
from .provider import AgentContainer, Mount
from .siteconfig import Site, credential_status

HERE = Path(__file__).resolve().parent
SOCKET_DIR = "/run/tnca-gateway"
SOCKET = f"{SOCKET_DIR}/gateway.sock"
BRIDGE = "127.0.0.1:3128"


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)


def _cpu_max(cpus: float) -> str:
    return f"{int(round(cpus * 100000))} 100000"


def _canary(site: Site, name: str, work: Optional[Path], env: dict, args: list, *, memory="512m",
            cpus=1.0, pids=256, extra_mounts: Optional[list] = None, shell: Optional[str] = None):
    mounts = [Mount(HERE, "/opt/tnca-harness")]
    if work is not None:
        mounts.append(Mount(work, "/work", read_only=False))
    mounts += extra_mounts or []
    c = AgentContainer(name=name, image=site.images["toolchain"], cpus=cpus, memory=memory, pids=pids,
                       mounts=mounts, env=env, labels={"tritonchip-agent-harness": "doctor"})
    command = ["sh", "-c", shell] if shell else ["python3", "/opt/tnca-harness/canary.py", *args]
    proc = provider.docker(*c.run_args(command, detach=False, remove=False), check=False, timeout=300)
    state = provider.inspect_state(name)
    provider.remove(name)
    return proc, state


def _parse(proc) -> list:
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return [{"check": "canary ran", "ok": False,
                 "detail": {"exit": proc.returncode, "stderr": proc.stderr.strip()[-400:]}}]


def run(site: Site) -> dict:
    stamp = _stamp()
    root = site.state_dir / "doctor" / stamp
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema": "tritonchip-agent-doctor/1", "host": platform.node(), "stamp": stamp,
                    "site": str(site.path), "checks": []}
    add = report["checks"].append

    # 1. daemon
    d = provider.info()
    report["provider"] = d
    formal = (not d["rootless"]) or d["cgroup_version"] == "2"
    report["host_class"] = "formal" if formal else "development-only"
    add({"check": "resource limits are enforceable here", "ok": formal,
         "detail": f"rootless={d['rootless']} cgroup v{d['cgroup_version']}"})

    # 2. images
    report["images"] = {role: {"ref": ref, "id": provider.image_id(ref)} for role, ref in site.images.items()}
    for role, img in report["images"].items():
        add({"check": f"{role} image present", "ok": img["id"] is not None, "detail": img["ref"]})

    # 3. credentials, by metadata
    report["credentials"] = [credential_status(a) for a in site.accounts.values()]
    for c in report["credentials"]:
        add({"check": f"credential for {c['label']}", "ok": c["ok"], "detail": c["problems"] or "in place"})

    if report["images"]["toolchain"]["id"] is None:
        report["ok"] = False
        return _finish(report, root)

    # 4 and 5. isolation, and nothing left over from another run
    marker = f"canary-marker-{stamp}"
    expect = {"CANARY_WORK": "/work", "CANARY_HOST_USER": getpass.getuser(),
              "CANARY_MOUNTS": "/work,/opt/tnca-harness", "CANARY_MEMORY": str(provider.memory_bytes("512m")),
              "CANARY_CPU": _cpu_max(1.0), "CANARY_PIDS": "256"}
    work_a, work_b = root / "work-a", root / "work-b"
    work_a.mkdir()
    work_b.mkdir()
    proc, _ = _canary(site, f"tnca-doctor-{stamp}-a", work_a, {**expect, "CANARY_LEAVE": marker}, ["checks"])
    for c in _parse(proc):
        add({**c, "check": "isolation: " + c["check"]})
    proc, _ = _canary(site, f"tnca-doctor-{stamp}-b", work_b, {**expect, "CANARY_LOOK_FOR": marker}, ["checks"])
    carried = [c for c in _parse(proc) if c["check"] in ("nothing from another run", "canary ran")]
    for c in carried:
        add({**c, "check": "next run: " + c["check"]})

    # 6. the memory limit stops a process
    proc, state = _canary(site, f"tnca-doctor-{stamp}-hog", None, {}, ["hog", "1024"], memory="256m")
    stopped = state.get("OOMKilled") or proc.returncode == 137
    add({"check": "memory limit stops a process", "ok": bool(stopped),
         "detail": {"exit": proc.returncode, "oom_killed": state.get("OOMKilled"),
                    "last_line": (proc.stdout.strip().splitlines() or [""])[-1]}})

    # 7. the gateway
    if site.routes:
        add_all = _gateway_checks(site, root, stamp)
        for c in add_all:
            add(c)

    report["ok"] = all(c["ok"] for c in report["checks"])
    return _finish(report, root)


def start_gateway(site: Site, run_dir: Path, name: str, hosts: list, open_route: Optional[str] = None) -> tuple:
    """Start the sidecar for `hosts`; return (socket dir, log file). The caller stops it.

    With `open_route` the gateway runs in open mode: unlisted hosts go out that way, logged.
    """
    socket_dir, log_dir = run_dir / "gateway-socket", run_dir / "gateway-log"
    socket_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    command = ["python3", "/opt/tnca-harness/gateway.py", "--socket", SOCKET,
               "--log", "/var/log/tnca-gateway/tunnels.jsonl"]
    for h in hosts:
        command += ["--route", f"{h}={site.routes[h]}"]
    if site.upstream_proxy:
        command += ["--upstream", site.upstream_proxy]
    if open_route:
        command += ["--open", open_route]
    provider.docker(*provider.gateway_run_args(name, site.images["toolchain"], socket_dir, log_dir, HERE, command))
    sock = socket_dir / "gateway.sock"
    for _ in range(100):
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        logs = provider.docker("logs", name, check=False)
        provider.remove(name)
        raise provider.ProviderError(f"gateway did not open its socket: {logs.stderr.strip()[-300:]}")
    return socket_dir, log_dir / "tunnels.jsonl"


def stop_gateway(name: str) -> None:
    provider.docker("stop", "-t", "5", name, check=False)
    provider.remove(name)


def read_log(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _gateway_checks(site: Site, root: Path, stamp: str) -> list:
    allowed = sorted(site.routes)[0]
    denied = "example.com" if "example.com" not in site.routes else "example.org"
    gw = f"tnca-doctor-{stamp}-gw"
    out = []
    try:
        socket_dir, log = start_gateway(site, root, gw, sorted(site.routes))
    except provider.ProviderError as exc:
        return [{"check": "gateway starts", "ok": False, "detail": str(exc)}]
    try:
        shell = (f"python3 /opt/tnca-harness/bridge.py --listen {BRIDGE} --socket {SOCKET} & sleep 1; "
                 f"python3 /opt/tnca-harness/canary.py net --via {BRIDGE} --allowed {allowed} --denied {denied}")
        proc, _ = _canary(site, f"tnca-doctor-{stamp}-net", None, {}, [], shell=shell,
                          extra_mounts=[Mount(socket_dir, SOCKET_DIR)])
        out += [{**c, "check": "gateway: " + c["check"]} for c in _parse(proc)]
    finally:
        stop_gateway(gw)
    entries = read_log(log)
    tunnels = [e for e in entries if e.get("event") == "tunnel" and e.get("host") == allowed]
    refused = [e for e in entries if e.get("event") == "refused" and e.get("host") == denied]
    out.append({"check": "gateway: logged the tunnel and the refusal, outside the agent's container",
                "ok": bool(tunnels) and bool(refused),
                "detail": {"tunnels": len(tunnels), "refused": len(refused),
                           "bytes": [(t["bytes_up"], t["bytes_down"]) for t in tunnels]}})
    return out


def _finish(report: dict, root: Path) -> dict:
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The scratch workspaces were the canaries'; the report is what is kept.
    for sub in ("work-a", "work-b"):
        shutil.rmtree(root / sub, ignore_errors=True)
    report["report_path"] = str(root / "report.json")
    return report


__all__ = ["read_log", "run", "start_gateway", "stop_gateway"]
