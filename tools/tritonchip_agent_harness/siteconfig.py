"""The site configuration: what one operator's machine provides.

It lives outside the repository, by default at
~/.config/tritonchip-agent-harness/site.json, because it names this
operator's accounts and where their credentials are. The repository ships only
examples/site.json, which has paths and labels but no credentials.

The harness reads a credential's *metadata* -- that the file exists, its mode,
its size, the names of the variables in it and the lengths of their values --
and passes the file itself to Docker (`--env-file`, or a copied home). It never
reads a value into Python.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA = "tritonchip-agent-site/1"
DEFAULT_PATH = Path("~/.config/tritonchip-agent-harness/site.json")

CLIS = ("claude-code", "codex", "opencode")
AUTHS = ("subscription", "api-key")
ROUTES = ("direct", "upstream")

_LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_HOST = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
_MEMORY = re.compile(r"^[1-9][0-9]*[kmg]$")


class SiteError(ValueError):
    """The site configuration is malformed, or names something that is not there."""


@dataclass(frozen=True)
class Account:
    label: str
    cli: str
    auth: str
    credential: Path
    hosts: tuple[str, ...]
    models: dict[str, str]
    options: dict[str, Any]


@dataclass(frozen=True)
class Site:
    path: Path
    provider: str
    images: dict[str, str]
    cpus: float
    memory: str
    pids: int
    state_dir: Path
    upstream_proxy: Optional[str]
    routes: dict[str, str]
    accounts: dict[str, Account]
    repository: Optional[Path] = None  # the TritonChip clone workspaces are exported from

    def account(self, label: str) -> Account:
        if label not in self.accounts:
            raise SiteError(f"no account {label!r} in {self.path}; there are: {', '.join(sorted(self.accounts))}")
        return self.accounts[label]


def _keys(where: str, value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, dict):
        raise SiteError(f"{where} must be an object")
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise SiteError(f"{where}: missing {', '.join(missing)}")
    if unknown:
        raise SiteError(f"{where}: unknown {', '.join(unknown)}")


def _path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SiteError(f"{where} must be a path")
    return Path(os.path.expanduser(value))


def load_site(path: Optional[Path] = None) -> Site:
    path = Path(os.path.expanduser(str(path or DEFAULT_PATH)))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteError(f"cannot read site configuration {path}: {exc}") from exc
    _keys(str(path), value, {"schema", "provider", "images", "resources", "state_dir", "network", "accounts"},
          {"repository"})
    if value["schema"] != SCHEMA:
        raise SiteError(f"{path}: schema must be {SCHEMA}")
    if value["provider"] != "docker":
        raise SiteError(f"{path}: provider must be 'docker' (the only one implemented)")

    images = value["images"]
    _keys("images", images, {"toolchain", "agent"})
    if not all(isinstance(v, str) and v for v in images.values()):
        raise SiteError("images: each must be an image reference")

    res = value["resources"]
    _keys("resources", res, {"cpus", "memory", "pids"})
    if not isinstance(res["cpus"], (int, float)) or isinstance(res["cpus"], bool) or res["cpus"] <= 0:
        raise SiteError("resources.cpus must be a positive number")
    if not isinstance(res["memory"], str) or not _MEMORY.match(res["memory"]):
        raise SiteError("resources.memory must look like 16g, 512m")
    if not isinstance(res["pids"], int) or isinstance(res["pids"], bool) or res["pids"] < 64:
        raise SiteError("resources.pids must be an integer of at least 64")

    net = value["network"]
    _keys("network", net, {"routes"}, {"upstream_proxy"})
    upstream = net.get("upstream_proxy")
    if upstream is not None and not (isinstance(upstream, str) and re.match(r"^http://[^/@]+:[0-9]+/?$", upstream)):
        # No credentials in the URL: a proxy that needs them is out of scope here.
        raise SiteError("network.upstream_proxy must be http://host:port, with no credentials in it")
    routes = net["routes"]
    if not isinstance(routes, dict) or not routes:
        raise SiteError("network.routes must map host names to 'direct' or 'upstream'")
    for host, route in routes.items():
        if not isinstance(host, str) or not _HOST.match(host):
            raise SiteError(f"network.routes: {host!r} is not a host name")
        if route not in ROUTES:
            raise SiteError(f"network.routes.{host} must be 'direct' or 'upstream'")
        if route == "upstream" and not upstream:
            raise SiteError(f"network.routes.{host} goes upstream, but no network.upstream_proxy is set")

    accounts: dict[str, Account] = {}
    if not isinstance(value["accounts"], list) or not value["accounts"]:
        raise SiteError("accounts must be a non-empty list")
    for i, a in enumerate(value["accounts"]):
        where = f"accounts[{i}]"
        _keys(where, a, {"label", "cli", "auth", "credential", "hosts", "models"}, {"options"})
        if not isinstance(a["label"], str) or not _LABEL.match(a["label"]):
            raise SiteError(f"{where}.label must match {_LABEL.pattern}")
        if a["label"] in accounts:
            raise SiteError(f"{where}: label {a['label']!r} used twice")
        if a["cli"] not in CLIS:
            raise SiteError(f"{where}.cli must be one of {', '.join(CLIS)}")
        if a["auth"] not in AUTHS:
            raise SiteError(f"{where}.auth must be one of {', '.join(AUTHS)}")
        hosts = a["hosts"]
        if not isinstance(hosts, list) or not hosts or not all(h in routes for h in hosts):
            raise SiteError(f"{where}.hosts must be a non-empty list of hosts that network.routes names")
        models = a["models"]
        _keys(f"{where}.models", models, {"main"}, {"small"})
        if not all(isinstance(m, str) and m for m in models.values()):
            raise SiteError(f"{where}.models: each must be a model id")
        options = a.get("options", {})
        if not isinstance(options, dict):
            raise SiteError(f"{where}.options must be an object")
        accounts[a["label"]] = Account(
            label=a["label"], cli=a["cli"], auth=a["auth"],
            credential=_path(a["credential"], f"{where}.credential"),
            hosts=tuple(hosts), models=dict(models), options=dict(options),
        )

    return Site(
        path=path, provider=value["provider"], images=dict(images),
        cpus=float(res["cpus"]), memory=res["memory"], pids=res["pids"],
        state_dir=_path(value["state_dir"], "state_dir"),
        upstream_proxy=upstream.rstrip("/") if upstream else None,
        routes=dict(routes), accounts=accounts,
        repository=_path(value["repository"], "repository") if value.get("repository") else None,
    )


def _private(path: Path) -> list[str]:
    """Problems with a credential's file mode: anyone but the owner may read it."""
    st = path.stat()
    problems = []
    if st.st_uid != os.getuid():
        problems.append(f"{path} is not owned by this user")
    if stat.S_IMODE(st.st_mode) & 0o077:
        problems.append(f"{path} is mode {stat.S_IMODE(st.st_mode):04o}; it must be readable by its owner only (0600)")
    return problems


def env_file_shape(path: Path) -> dict[str, int]:
    """The variable names in an env file and the length of each value. Values are not kept."""
    shape: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, sep, rest = line.partition("=")
        if not sep or not re.match(r"^[A-Z_][A-Z0-9_]*$", name):
            shape["<malformed line>"] = -1
            continue
        shape[name] = len(rest)
    return shape


# The variable each CLI's env-file credential must define.
ENV_VARIABLE = {"claude-code": "CLAUDE_CODE_OAUTH_TOKEN", "opencode": "GLM_API_KEY"}


def credential_status(account: Account) -> dict[str, Any]:
    """Whether an account's credential is in place, from its metadata alone."""
    path = account.credential
    status: dict[str, Any] = {"label": account.label, "cli": account.cli, "auth": account.auth, "path": str(path)}
    problems: list[str] = []
    if account.cli == "codex":
        auth = path / "auth.json"
        if not path.is_dir() or not auth.is_file():
            problems.append(f"{auth} does not exist; log in with `codex login --device-auth` and CODEX_HOME={path}")
        else:
            problems += _private(auth)
            status["size"] = auth.stat().st_size
            try:
                status["fields"] = sorted(json.loads(auth.read_text(encoding="utf-8")).keys())
            except (OSError, json.JSONDecodeError):
                problems.append(f"{auth} is not JSON")
    else:
        if not path.is_file():
            problems.append(f"{path} does not exist")
        else:
            problems += _private(path)
            shape = env_file_shape(path)
            status["variables"] = shape
            wanted = ENV_VARIABLE[account.cli]
            if shape.get(wanted, 0) <= 0:
                problems.append(f"{path} does not define {wanted}")
            if "<malformed line>" in shape:
                problems.append(f"{path} has a line that is not NAME=value")
    status["ok"] = not problems
    status["problems"] = problems
    return status


__all__ = ["Account", "Site", "SiteError", "credential_status", "env_file_shape", "load_site"]
