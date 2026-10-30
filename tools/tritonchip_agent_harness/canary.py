#!/usr/bin/env python3
"""Run inside a container exactly as an agent's would be, and report what it can reach.

doctor.py starts it and reads its JSON. Each check states what an agent must
NOT be able to do (or must be able to do) and whether that held, so a result
is a measurement of this host and this provider rather than a reading of their
configuration.

    canary.py checks               isolation properties; expectations come from CANARY_* variables
    canary.py hog <MiB>            allocate until killed; the caller expects the memory limit to stop it
    canary.py net --via H:P --allowed HOST --denied HOST
                                   egress through the bridge: listed host yes, others no, direct never

Standard library only.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
from pathlib import Path

# What Docker gives every container in /dev. Anything else is a host device.
STANDARD_DEV = {"console", "core", "fd", "full", "mqueue", "null", "ptmx", "pts", "random", "shm",
                "stderr", "stdin", "stdout", "tty", "urandom", "zero", "init"}
# Docker bind-mounts these itself (docker-init because of --init; on a merged-/usr
# image such as Ubuntu 24.04 it shows up under /usr/sbin).
DOCKER_FILES = {"/etc/hostname", "/etc/hosts", "/etc/resolv.conf", "/sbin/docker-init", "/usr/sbin/docker-init"}


def check(name: str, ok: bool, detail="") -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def fails(fn) -> tuple:
    try:
        fn()
    except Exception as exc:  # the point is that it raises
        return True, type(exc).__name__
    return False, "succeeded"


def status_field(name: str) -> str:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def bind_mounts() -> list:
    """Mount points whose source is a host path (not a pseudo or tmpfs mount)."""
    points = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, _, right = line.partition(" - ")
        fstype = right.split()[0] if right else ""
        point = left.split()[4]
        if fstype in ("proc", "sysfs", "tmpfs", "devpts", "mqueue", "cgroup", "cgroup2", "overlay", "shm"):
            continue
        if point.startswith(("/proc", "/sys", "/dev")):
            continue
        points.append(point)
    return sorted(set(points))


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def checks() -> list:
    env = os.environ
    out = []
    out.append(check("not root", os.getuid() != 0 and os.getgid() != 0, f"uid={os.getuid()} gid={os.getgid()}"))
    out.append(check("no capabilities", int(status_field("CapEff") or "1", 16) == 0, status_field("CapEff")))
    out.append(check("no new privileges", status_field("NoNewPrivs") == "1", status_field("NoNewPrivs")))
    ok, why = fails(lambda: Path("/etc/.canary").write_text("x"))
    out.append(check("root filesystem read-only", ok, why))
    ok, why = fails(lambda: Path("/tmp/.canary").write_text("x"))
    out.append(check("/tmp writable", not ok, why))
    if env.get("CANARY_WORK"):
        ok, why = fails(lambda: (Path(env["CANARY_WORK"]) / ".canary").write_text("x"))
        out.append(check("workspace writable", not ok, why))
    host_user = env.get("CANARY_HOST_USER", "")
    if host_user:
        out.append(check("host home invisible", not Path("/home", host_user).exists(), f"/home/{host_user}"))
    out.append(check("no ~/.ssh", not (Path.home() / ".ssh").exists(), str(Path.home() / ".ssh")))
    socks = [p for p in ("/var/run/docker.sock", "/run/docker.sock") if Path(p).exists()]
    out.append(check("no Docker socket", not socks, socks))
    extra_dev = sorted(set(os.listdir("/dev")) - STANDARD_DEV)
    out.append(check("no host devices", not extra_dev, extra_dev))
    ok, why = fails(lambda: socket.getaddrinfo("example.com", 443))
    out.append(check("no DNS", ok, why))
    ok, why = fails(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3).close())
    out.append(check("no direct TCP", ok, why))
    ifaces = [l.split(":")[0].strip() for l in read("/proc/net/dev").splitlines()[2:]]
    out.append(check("loopback is the only interface", ifaces == ["lo"], ifaces))
    expected = set(filter(None, env.get("CANARY_MOUNTS", "").split(","))) | DOCKER_FILES
    mounts = bind_mounts()
    out.append(check("only the expected host mounts", set(mounts) <= expected,
                     {"seen": mounts, "unexpected": sorted(set(mounts) - expected)}))
    if env.get("CANARY_MEMORY"):
        out.append(check("memory limit in place", read("/sys/fs/cgroup/memory.max") == env["CANARY_MEMORY"],
                         read("/sys/fs/cgroup/memory.max")))
    if env.get("CANARY_CPU"):
        out.append(check("cpu limit in place", read("/sys/fs/cgroup/cpu.max") == env["CANARY_CPU"],
                         read("/sys/fs/cgroup/cpu.max")))
    if env.get("CANARY_PIDS"):
        out.append(check("process limit in place", read("/sys/fs/cgroup/pids.max") == env["CANARY_PIDS"],
                         read("/sys/fs/cgroup/pids.max")))
    if env.get("CANARY_LEAVE"):
        Path("/tmp", env["CANARY_LEAVE"]).write_text("left by a previous run")
        if env.get("CANARY_WORK"):
            (Path(env["CANARY_WORK"]) / env["CANARY_LEAVE"]).write_text("left by a previous run")
    if env.get("CANARY_LOOK_FOR"):
        name = env["CANARY_LOOK_FOR"]
        seen = [str(p) for p in (Path("/tmp", name), Path(env.get("CANARY_WORK", "/nonexistent"), name)) if p.exists()]
        out.append(check("nothing from another run", not seen, seen))
    return out


def hog(mib: int) -> int:
    blocks = []
    for i in range(mib):
        blocks.append(bytearray(1024 * 1024))  # touched: bytearray zero-fills
        if i % 64 == 0:
            print(f"allocated {i} MiB", flush=True)
    print(f"allocated {mib} MiB and was not stopped", flush=True)
    return 0


def connect_via(proxy: tuple, host: str) -> tuple:
    """CONNECT host:443 through the proxy; then a TLS handshake if it was allowed."""
    s = socket.create_connection(proxy, timeout=10)
    s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = s.recv(4096)
        if not chunk:
            break
        head += chunk
    status = head.split(b"\r\n", 1)[0].decode("latin-1")
    code = status.split()[1] if len(status.split()) > 1 else "?"
    handshake = None
    if code == "200":
        try:
            tls = ssl.create_default_context().wrap_socket(s, server_hostname=host)
            handshake = tls.version()
            tls.close()
        except (ssl.SSLError, OSError) as exc:
            handshake = f"failed: {type(exc).__name__}"
    else:
        s.close()
    return code, handshake


def net(via: str, allowed: str, denied: str) -> list:
    host, _, port = via.rpartition(":")
    proxy = (host, int(port))
    out = []
    code, tls = connect_via(proxy, allowed)
    out.append(check(f"listed host {allowed} reachable through the gateway",
                     code == "200" and tls and not str(tls).startswith("failed"), {"status": code, "tls": tls}))
    code, _ = connect_via(proxy, denied)
    out.append(check(f"unlisted host {denied} refused by the gateway", code == "403", {"status": code}))
    ok, why = fails(lambda: socket.getaddrinfo(allowed, 443))
    out.append(check("no DNS of its own", ok, why))
    return out


def main(argv: list) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "checks":
        print(json.dumps(checks()))
        return 0
    if argv[0] == "hog":
        return hog(int(argv[1]))
    if argv[0] == "net":
        opts = dict(zip(argv[1::2], argv[2::2]))
        print(json.dumps(net(opts["--via"], opts["--allowed"], opts["--denied"])))
        return 0
    print(f"canary: unknown mode {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
