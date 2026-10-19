"""How each CLI is launched, and how its output is read. One place, used by smoke runs and task runs alike.

The line drawn for every CLI is the same:

- What belongs to the agent system stays as its vendor ships it -- subagents,
  context compaction, reasoning settings -- because the comparison is between
  agent systems as their vendors pair them.
- What leaks the environment is switched off -- user configuration, memory
  carried between runs, telemetry, plugins and skills fetched from the
  network, update checks -- and in `offline` mode every web tool too.

Each switch is listed here, so a run record can name the configuration it ran under.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .provider import AGENT_HOME, Mount
from .siteconfig import Account

MODES = ("offline", "open")
CODEX_HOME = "/run/tnca-codex-home"
WORKDIR = "/work"


@dataclass
class Launch:
    command: list
    env: dict = field(default_factory=dict)
    env_files: list = field(default_factory=list)
    mounts: list = field(default_factory=list)
    switches: dict = field(default_factory=dict)  # what was turned off or on, for the run record


def codex_config(mode: str) -> str:
    """CODEX_HOME/config.toml for one run. auth.json is copied in beside it."""
    return "\n".join([
        'cli_auth_credentials_store = "file"',
        "check_for_update_on_startup = false",
        f'web_search = "{"disabled" if mode == "offline" else "live"}"',
        "",
        "[analytics]",
        "enabled = false",
        "",
        "[otel]",
        'metrics_exporter = "none"',
        "",
        "[features]",
        "remote_plugin = false",
        "apps = false",
        "skill_mcp_dependency_install = false",
        "memories = false",
        "",
        "[memories]",
        "generate_memories = false",
        "use_memories = false",
        "",
    ])


def opencode_config(account: Account, mode: str) -> dict:
    prov, base = account.options["provider"], account.options["base_url"]
    main = account.models["main"]
    small = account.models.get("small", main)
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False, "share": "disabled", "mcp": {},
        "permission": {"webfetch": "deny" if mode == "offline" else "allow"},
        "model": f"{prov}/{main}", "small_model": f"{prov}/{small}",
        "provider": {prov: {"options": {"baseURL": base, "apiKey": "{env:GLM_API_KEY}"},
                            "models": {m: {} for m in sorted(set(account.models.values()))}}},
    }


def launch(account: Account, prompt: str, mode: str, run_dir: Path, proxy_env: dict) -> Launch:
    if mode not in MODES:
        raise ValueError(f"network mode must be one of {MODES}")
    main = account.models["main"]
    env = dict(proxy_env)
    if account.cli == "claude-code":
        command = ["claude", "-p", prompt, "--model", main, "--output-format", "stream-json", "--verbose",
                   "--dangerously-skip-permissions", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        if mode == "offline":
            command += ["--disallowedTools", "WebSearch,WebFetch"]
        env.update(CLAUDE_CONFIG_DIR=f"{AGENT_HOME}/.claude-run", DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        return Launch(command, env, [account.credential], [], {
            "config_dir": "fresh per run", "mcp": "none (strict)", "updates": "off",
            "nonessential_traffic": "off", "web_tools": "off" if mode == "offline" else "on"})
    if account.cli == "codex":
        home = run_dir / "codex-home"
        home.mkdir(mode=0o700, exist_ok=True)
        (home / "config.toml").write_text(codex_config(mode), encoding="utf-8")
        command = ["sh", "-c", "exec codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "
                               f"-C {WORKDIR} -m {shlex.quote(main)} {shlex.quote(prompt)} < /dev/null"]
        env.update(CODEX_HOME=CODEX_HOME)
        return Launch(command, env, [], [Mount(home, CODEX_HOME, read_only=False)], {
            "codex_home": "fresh per run, auth.json copied", "updates": "off", "analytics": "off",
            "otel_metrics": "none", "remote_plugin": "off", "apps": "off", "skill_dependency_install": "off",
            "memories": "off", "web_search": "disabled" if mode == "offline" else "live"})
    prov = account.options["provider"]
    command = ["opencode", "run", "--format", "json", "--auto", "--dir", WORKDIR, "-m", f"{prov}/{main}", prompt]
    env.update(OPENCODE_CONFIG_CONTENT=json.dumps(opencode_config(account, mode), separators=(",", ":")),
               OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_MODELS_FETCH="1",
               OPENCODE_DISABLE_DEFAULT_PLUGINS="1", OPENCODE_DISABLE_LSP_DOWNLOAD="1",
               OPENCODE_DISABLE_CLAUDE_CODE="1",
               XDG_CONFIG_HOME=f"{AGENT_HOME}/.config", XDG_DATA_HOME=f"{AGENT_HOME}/.local/share",
               XDG_CACHE_HOME=f"{AGENT_HOME}/.cache", XDG_STATE_HOME=f"{AGENT_HOME}/.local/state")
    return Launch(command, env, [account.credential], [], {
        "config": "inline only", "updates": "off", "models_fetch": "off", "default_plugins": "off",
        "lsp_download": "off", "reads_.claude": "off", "share": "off",
        "webfetch": "deny" if mode == "offline" else "allow"})


def _events(out: str) -> list:
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def parse(account: Account, out: str, run_dir: Path, export_session=None) -> dict:
    """What the CLI answered last, which models served it, and the usage it reported."""
    r: dict = {"text": None, "served_models": [], "usage": None, "turns": None}
    events = _events(out)
    if account.cli == "claude-code":
        final = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if final is None:
            return {**r, "error": out[-300:]}
        r.update(text=final.get("result"), served_models=sorted((final.get("modelUsage") or {}).keys()),
                 usage=final.get("usage"), turns=final.get("num_turns"), is_error=final.get("is_error"),
                 model_usage=final.get("modelUsage"))
        return r
    if account.cli == "codex":
        usage = {}
        for e in events:
            if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message":
                r["text"] = e["item"].get("text")
            if e.get("type") == "turn.completed":
                for k, v in (e.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + (v or 0)
        r["usage"] = usage or None
        models = set()
        for rollout in (run_dir / "codex-home" / "sessions").rglob("rollout-*.jsonl"):
            for line in rollout.read_text(encoding="utf-8").splitlines():
                try:
                    p = json.loads(line).get("payload") or {}
                except json.JSONDecodeError:
                    continue
                if isinstance(p, dict) and p.get("model"):
                    models.add(p["model"])
        r["served_models"] = sorted(models)
        return r
    session, usage = None, {}
    for e in events:
        session = session or e.get("sessionID")
        part = e.get("part") or {}
        if part.get("type") == "text" and part.get("text"):
            r["text"] = part["text"]
        tokens = part.get("tokens")
        if tokens:
            for k, v in tokens.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        usage[f"{k}.{kk}"] = usage.get(f"{k}.{kk}", 0) + (vv or 0)
                else:
                    usage[k] = usage.get(k, 0) + (v or 0)
        if e.get("type") == "error":
            r["error"] = e.get("error")
    r["usage"] = usage or None
    if session and export_session is not None:
        try:
            msgs = json.loads(export_session(session)).get("messages", [])
            r["served_models"] = sorted({f"{(m.get('info') or {}).get('providerID')}/{(m.get('info') or {}).get('modelID')}"
                                         for m in msgs if (m.get("info") or {}).get("role") == "assistant"})
        except (json.JSONDecodeError, TypeError):
            pass
    return r


__all__ = ["CODEX_HOME", "Launch", "MODES", "WORKDIR", "codex_config", "launch", "opencode_config", "parse"]
