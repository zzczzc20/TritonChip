"""The servers a split run uses: one builds, another holds the card.

The Vivado flow and the card need not be on one machine. A hosts file, kept by
whoever operates the servers and never in the repository, names each server
and what it does::

    {
      "schema_version": 1,
      "hosts": {
        "build": {
          "ssh": "build-worker",
          "roles": ["build"],
          "python": "/opt/python3.9/bin/python3",
          "workspace": "/data/fpga/tritonchip",
          "state_dir": "/data/fpga/tritonchip-harness",
          "vitis_root": "/opt/Xilinx/Vitis/2022.2"
        },
        "card": {
          "ssh": "card-host",
          "roles": ["board"],
          "workspace": "/home/fpga/tritonchip",
          "state_dir": "/home/fpga/tritonchip-harness",
          "environment": {"TORCH_PYTHON": "/home/fpga/venv/bin/python"}
        }
      }
    }

- ``ssh`` is a destination the controller's own SSH configuration resolves
  (an alias with its user, key and jump host); the harness adds no secrets.
- ``workspace`` holds a bare mirror of the repository (``mirror.git``) and one
  clean checkout per commit (``checkouts/<sha>``), which ``sync`` creates from
  a git bundle the controller sends: the servers need no access to the
  repository's remote, and never to each other.
- ``state_dir`` is the harness state directory on that server.
- ``vitis_root`` and ``platforms_root`` override the target's installation
  paths on that server (toolroots.py); ``environment`` passes further
  variables (an ArchBench interpreter, ``XILINX_XRT``) to every remote command.

Every value is checked to be one shell-safe token: OpenSSH runs the remote
command through the login shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .schema import SchemaError, load_json
from .toolroots import PLATFORMS_ENV, VITIS_ENV

ROLES = ("build", "board")
_SAFE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_FIELDS = {"ssh", "roles", "python", "workspace", "state_dir", "vitis_root", "platforms_root",
           "environment", "port", "identity_file", "description"}


@dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    roles: tuple[str, ...]
    workspace: str
    state_dir: str
    python: str = "python3"
    vitis_root: str | None = None
    platforms_root: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    port: int | None = None
    identity_file: str | None = None

    @property
    def mirror(self) -> str:
        return f"{self.workspace}/mirror.git"

    @property
    def incoming(self) -> str:
        return f"{self.workspace}/incoming"

    def checkout(self, commit: str) -> str:
        return f"{self.workspace}/checkouts/{commit}"

    def cli(self, commit: str) -> str:
        return f"{self.checkout(commit)}/tools/tritonchip_fpga_harness/cli.py"

    def remote_env(self) -> dict[str, str]:
        """The variables every remote command on this host runs with."""
        env = dict(self.environment)
        if self.vitis_root:
            env[VITIS_ENV] = self.vitis_root
        if self.platforms_root:
            env[PLATFORMS_ENV] = self.platforms_root
        return env

    def summary(self) -> dict[str, Any]:
        """What a pipeline record says about the host (no paths beyond the harness's own)."""
        return {"name": self.name, "ssh": self.ssh, "roles": list(self.roles),
                "vitis_root": self.vitis_root, "platforms_root": self.platforms_root,
                "environment": sorted(self.environment)}


def _token(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE.fullmatch(value):
        raise SchemaError(f"{where} must be one shell-safe token, got {value!r}")
    return value


def _absolute(value: Any, where: str) -> str:
    text = _token(value, where)
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts:
        raise SchemaError(f"{where} must be an absolute path without '..', got {value!r}")
    return text.rstrip("/") or "/"


def parse_hosts(value: Mapping[str, Any]) -> dict[str, Host]:
    if not isinstance(value, Mapping) or value.get("schema_version", 1) != 1:
        raise SchemaError("hosts file must be an object with schema_version 1")
    unknown = sorted(set(value) - {"schema_version", "hosts", "description"})
    if unknown:
        raise SchemaError("hosts file has unsupported fields: " + ", ".join(unknown))
    entries = value.get("hosts")
    if not isinstance(entries, Mapping) or not entries:
        raise SchemaError("hosts file needs a non-empty 'hosts' object")
    hosts: dict[str, Host] = {}
    for name, entry in entries.items():
        where = f"hosts.{name}"
        if not _NAME.fullmatch(str(name)):
            raise SchemaError(f"{where}: a host name is letters, digits, '-' and '_'")
        if not isinstance(entry, Mapping):
            raise SchemaError(f"{where} must be an object")
        extra = sorted(set(entry) - _FIELDS)
        if extra:
            raise SchemaError(f"{where} has unsupported fields: " + ", ".join(extra))
        roles = entry.get("roles")
        if not isinstance(roles, list) or not roles or any(role not in ROLES for role in roles):
            raise SchemaError(f"{where}.roles must list one or more of: " + ", ".join(ROLES))
        environment = entry.get("environment", {})
        if not isinstance(environment, Mapping):
            raise SchemaError(f"{where}.environment must be an object")
        env: dict[str, str] = {}
        for key, val in environment.items():
            if not _ENV_NAME.fullmatch(str(key)):
                raise SchemaError(f"{where}.environment: {key!r} is not a plain variable name")
            env[str(key)] = _token(val, f"{where}.environment.{key}")
        port = entry.get("port")
        if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535):
            raise SchemaError(f"{where}.port must be between 1 and 65535")
        python = _token(entry.get("python", "python3"), f"{where}.python")
        hosts[str(name)] = Host(
            name=str(name),
            ssh=_token(entry.get("ssh"), f"{where}.ssh"),
            roles=tuple(dict.fromkeys(roles)),
            workspace=_absolute(entry.get("workspace"), f"{where}.workspace"),
            state_dir=_absolute(entry.get("state_dir"), f"{where}.state_dir"),
            python=python,
            vitis_root=_absolute(entry["vitis_root"], f"{where}.vitis_root") if entry.get("vitis_root") else None,
            platforms_root=_absolute(entry["platforms_root"], f"{where}.platforms_root") if entry.get("platforms_root") else None,
            environment=env,
            port=port,
            identity_file=_token(entry["identity_file"], f"{where}.identity_file") if entry.get("identity_file") else None,
        )
    return hosts


def load_hosts(path: Path) -> dict[str, Host]:
    return parse_hosts(load_json(path))


def host_for(hosts: Mapping[str, Host], name: str, role: str) -> Host:
    if name not in hosts:
        raise SchemaError(f"no host {name!r} in the hosts file; it has: {', '.join(sorted(hosts))}")
    host = hosts[name]
    if role not in host.roles:
        raise SchemaError(f"host {name!r} does not have the {role!r} role (it has {', '.join(host.roles)})")
    return host


__all__ = ["Host", "ROLES", "host_for", "load_hosts", "parse_hosts"]
