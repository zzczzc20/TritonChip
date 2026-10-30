"""The Docker provider: the only place a container is started.

An agent's container always gets the same hardening, and a caller cannot
remove any of it: there is no parameter for it. What a caller chooses is the
image, the mounts, the environment and the limits. The gateway is the one
container started with a network, and it is trusted harness code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The user an agent runs as inside the image (tools/.../image/Containerfile).
AGENT_UID = 1000
AGENT_HOME = "/home/tnca"

# Applied to every agent container, unconditionally.
HARDENING = (
    "--network", "none",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--read-only",
    "--user", f"{AGENT_UID}:{AGENT_UID}",
    "--init",
)


class ProviderError(RuntimeError):
    """Docker is missing, refused a request, or did something unexpected."""


def docker(*args: str, check: bool = True, timeout: Optional[float] = None,
           input_text: Optional[str] = None, text: bool = True) -> subprocess.CompletedProcess:
    if shutil.which("docker") is None:
        raise ProviderError("docker is not on PATH")
    proc = subprocess.run(["docker", *args], capture_output=True, text=text, timeout=timeout, input=input_text)
    if check and proc.returncode != 0:
        err = proc.stderr if text else proc.stderr.decode(errors="replace")
        raise ProviderError(f"docker {' '.join(args[:3])} … failed ({proc.returncode}): {err.strip()[:400]}")
    return proc


def info() -> dict:
    """What the daemon says about itself: version, cgroup version, rootless."""
    out = docker("info", "--format", "{{json .}}").stdout
    data = json.loads(out)
    options = data.get("SecurityOptions") or []
    return {
        "server_version": data.get("ServerVersion"),
        "cgroup_version": str(data.get("CgroupVersion") or ""),
        "cgroup_driver": data.get("CgroupDriver"),
        "rootless": any("rootless" in o for o in options),
        "security_options": options,
        "kernel": data.get("KernelVersion"),
        "os": data.get("OperatingSystem"),
    }


def image_id(ref: str) -> Optional[str]:
    proc = docker("image", "inspect", "--format", "{{.Id}}", ref, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def memory_bytes(text: str) -> int:
    return int(text[:-1]) * {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[text[-1]]


@dataclass
class Mount:
    source: Path
    target: str
    read_only: bool = True

    def arg(self) -> str:
        return f"type=bind,source={self.source},target={self.target}" + (",readonly" if self.read_only else "")


@dataclass
class AgentContainer:
    """One agent container's request. Everything not listed here is the hardening."""
    name: str
    image: str
    cpus: float
    memory: str
    pids: int
    mounts: list[Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_files: list[Path] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    tmp_size: str = "4g"
    workdir: str = "/tmp"

    def run_args(self, command: list[str], *, detach: bool, remove: bool) -> list[str]:
        args = ["run"]
        if detach:
            args.append("-d")
        if remove:
            args.append("--rm")
        args += ["--name", self.name, *HARDENING,
                 "--cpus", f"{self.cpus:g}", "--memory", self.memory, "--memory-swap", self.memory,
                 "--pids-limit", str(self.pids),
                 # A writable /tmp and a home the agent's user owns; both vanish
                 # with the container, so nothing carries over to another run.
                 "--tmpfs", f"/tmp:rw,exec,nosuid,size={self.tmp_size}",
                 "--tmpfs", f"{AGENT_HOME}:rw,exec,nosuid,uid={AGENT_UID},gid={AGENT_UID},mode=0700",
                 "--workdir", self.workdir]
        for k, v in sorted(self.labels.items()):
            args += ["--label", f"{k}={v}"]
        for m in self.mounts:
            args += ["--mount", m.arg()]
        for f in self.env_files:
            # Docker reads the file; the value never appears on a command line.
            args += ["--env-file", str(f)]
        for k, v in sorted(self.env.items()):
            args += ["-e", f"{k}={v}"]
        return args + [self.image, *command]


def gateway_run_args(name: str, image: str, socket_dir: Path, log_dir: Path, harness_dir: Path,
                     command: list[str]) -> list[str]:
    """The gateway: trusted, on the host's network so it can reach an upstream proxy on localhost.

    It still drops every capability and runs read-only as the unprivileged
    user. It sees the socket directory it serves and its own log directory;
    the agent's container sees the socket directory only.
    """
    return ["run", "-d", "--rm", "--name", name, "--network", "host",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--user", f"{AGENT_UID}:{AGENT_UID}", "--init", "--pids-limit", "256", "--memory", "512m",
            "--label", "tritonchip-agent-harness=gateway",
            "--mount", Mount(socket_dir, "/run/tnca-gateway", read_only=False).arg(),
            "--mount", Mount(log_dir, "/var/log/tnca-gateway", read_only=False).arg(),
            "--mount", Mount(harness_dir, "/opt/tnca-harness").arg(),
            image, *command]


def remove(name: str) -> None:
    docker("rm", "-f", name, check=False)


def inspect_state(name: str) -> dict:
    proc = docker("inspect", "--format", "{{json .State}}", name, check=False)
    return json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {}


__all__ = ["AGENT_HOME", "AGENT_UID", "AgentContainer", "HARDENING", "Mount", "ProviderError",
           "docker", "gateway_run_args", "image_id", "info", "inspect_state", "memory_bytes", "remove"]
