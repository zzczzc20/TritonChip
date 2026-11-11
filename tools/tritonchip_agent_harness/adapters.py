"""How each CLI is launched, and how its output is read. One place, used by smoke runs and task runs alike.

The line drawn for every CLI is the same:

- What belongs to the agent system stays as its vendor ships it -- subagents,
  context compaction, reasoning settings -- because the comparison is between
  agent systems as their vendors pair them.
- What leaks the environment is switched off -- user configuration, memory
  carried between runs, telemetry, plugins and skills fetched from the
  network, update checks -- and in `offline` mode every web tool too.
- What the harness itself provides is the same for every CLI: its skills
  (`SKILLS_DIR`, mounted read-only) are linked into each CLI's own skills
  directory, so each CLI discovers them the way its vendor made it discover
  skills, from one byte-identical source. The run record names them by hash.

Each switch is listed here, so a run record can name the configuration it ran under.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .provider import AGENT_HOME, Mount
from .siteconfig import Account

MODES = ("offline", "open")
CODEX_HOME = "/run/tnca-codex-home"
WORKDIR = "/work"
# The harness's skills, one subdirectory each, as the agent's container sees them.
SKILLS_DIR = "/opt/tnca-skills"


@dataclass
class Launch:
    command: list
    env: dict = field(default_factory=dict)
    env_files: list = field(default_factory=list)
    mounts: list = field(default_factory=list)
    switches: dict = field(default_factory=dict)  # what was turned off or on, for the run record
    skills_dir: str = ""  # where this CLI looks for user skills; SKILLS_DIR's entries are linked there


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


def launch(account: Account, prompt: str, mode: str, run_dir: Path, proxy_env: dict,
           session: Optional[str] = None) -> Launch:
    """The CLI's command and environment. `session`: continue that session of the CLI (an island's next turn,
    report section 8.4 D2) -- `claude -p --resume`, `codex exec ... resume`, `opencode run --session`; none of
    the three is verified yet on the pinned versions, so the caller falls back to a new session when one fails."""
    if mode not in MODES:
        raise ValueError(f"network mode must be one of {MODES}")
    main = account.models["main"]
    env = dict(proxy_env)
    if account.cli == "claude-code":
        command = ["claude", "-p", prompt, "--model", main, "--output-format", "stream-json", "--verbose",
                   "--dangerously-skip-permissions", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        if session:
            command += ["--resume", session]
        if mode == "offline":
            command += ["--disallowedTools", "WebSearch,WebFetch"]
        env.update(CLAUDE_CONFIG_DIR=f"{AGENT_HOME}/.claude-run", DISABLE_AUTOUPDATER="1",
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
        return Launch(command, env, [account.credential], [], {
            "config_dir": "fresh per run", "mcp": "none (strict)", "updates": "off",
            "nonessential_traffic": "off", "web_tools": "off" if mode == "offline" else "on"},
            skills_dir=f"{AGENT_HOME}/.claude-run/skills")
    if account.cli == "codex":
        home = run_dir / "codex-home"
        home.mkdir(mode=0o700, exist_ok=True)
        (home / "config.toml").write_text(codex_config(mode), encoding="utf-8")
        resume = f"resume {shlex.quote(session)} " if session else ""
        command = ["sh", "-c", "exec codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "
                               f"-C {WORKDIR} -m {shlex.quote(main)} {resume}{shlex.quote(prompt)} < /dev/null"]
        env.update(CODEX_HOME=CODEX_HOME)
        return Launch(command, env, [], [Mount(home, CODEX_HOME, read_only=False)], {
            "codex_home": "fresh per run, auth.json copied", "updates": "off", "analytics": "off",
            "otel_metrics": "none", "remote_plugin": "off", "apps": "off", "skill_dependency_install": "off",
            "memories": "off", "web_search": "disabled" if mode == "offline" else "live"},
            skills_dir=f"{CODEX_HOME}/skills")
    prov = account.options["provider"]
    command = ["opencode", "run", "--format", "json", "--auto", "--dir", WORKDIR, "-m", f"{prov}/{main}"]
    command += (["--session", session] if session else []) + [prompt]
    env.update(OPENCODE_CONFIG_CONTENT=json.dumps(opencode_config(account, mode), separators=(",", ":")),
               OPENCODE_DISABLE_AUTOUPDATE="1", OPENCODE_DISABLE_MODELS_FETCH="1",
               OPENCODE_DISABLE_DEFAULT_PLUGINS="1", OPENCODE_DISABLE_LSP_DOWNLOAD="1",
               OPENCODE_DISABLE_CLAUDE_CODE="1",
               XDG_CONFIG_HOME=f"{AGENT_HOME}/.config", XDG_DATA_HOME=f"{AGENT_HOME}/.local/share",
               XDG_CACHE_HOME=f"{AGENT_HOME}/.cache", XDG_STATE_HOME=f"{AGENT_HOME}/.local/state")
    return Launch(command, env, [account.credential], [], {
        "config": "inline only", "updates": "off", "models_fetch": "off", "default_plugins": "off",
        "lsp_download": "off", "reads_.claude": "off", "share": "off",
        "webfetch": "deny" if mode == "offline" else "allow"},
        skills_dir=f"{AGENT_HOME}/.config/opencode/skills")


def _events(out: str) -> list:
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


TOKEN_FIELDS = ("input_uncached", "cache_write", "cache_read", "output", "reasoning", "total")


def _counts(input_uncached, cache_write, cache_read, output, reasoning) -> dict:
    parts = (input_uncached, cache_write, cache_read, output)
    total = None if all(p is None for p in parts) else sum(p or 0 for p in parts)
    return dict(zip(TOKEN_FIELDS, (input_uncached, cache_write, cache_read, output, reasoning, total)))


def _sum(rows) -> dict:
    """Field by field: the sum of what was reported, None where nothing was."""
    out = {}
    for f in TOKEN_FIELDS:
        got = [r[f] for r in rows if r[f] is not None]
        out[f] = sum(got) if got else None
    return out


def tokens(cli: str, usage: Optional[dict], model_usage: Optional[dict] = None) -> dict:
    """The usage a CLI reported, in one schema, so that runs of different CLIs add up alike.

    Each field is a count of tokens, or None where the CLI did not report it.
    `output` always includes reasoning, and `reasoning` is the part of it that
    was; total = input_uncached + cache_write + cache_read + output, so nothing
    is counted twice. The raw `usage` is kept beside this, untouched.

      field           Claude Code 2.1.311        Codex 0.162.1                    opencode 1.19.0
                      `result`, per model        `turn.completed`                 `step_finish` parts, summed
      input_uncached  inputTokens                input_tokens minus the next two  input
      cache_write     cacheCreationInputTokens   cache_write_input_tokens         cache.write
      cache_read      cacheReadInputTokens       cached_input_tokens              cache.read
      output          outputTokens               output_tokens                    output + reasoning
      reasoning       thinkingTokens             reasoning_output_tokens          reasoning
      its output      includes thinking          includes reasoning               leaves reasoning out
      per model       modelUsage                 not reported                     not reported

    Claude Code (the Agent SDK reference's ModelUsage and Usage; no recorded
    output is in the repository): the totals are the sum over `modelUsage`,
    which counts subagents; the result's `usage` counts the top-level loop only,
    and is read (input_tokens, cache_creation_input_tokens,
    cache_read_input_tokens, output_tokens, output_tokens_details.thinking_tokens)
    only when there is no `modelUsage`. A model whose entry has no
    thinkingTokens adds nothing to the total's reasoning, and a thinking count of
    0 can mean the model gave no breakdown. A run stopped before its `result`
    event (at its budget) reports no usage at all.

    Codex (codex-rs at rust-v0.162.1: exec_events.rs, codex-api sse/responses.rs):
    the fields are the Responses API's, whose input_tokens holds the cached reads
    and the cache writes too, and whose output_tokens holds the reasoning. exec
    prints one turn.completed, carrying the thread's running total; a run stopped
    before its turn completed reports no usage at all. cache_write_input_tokens
    is 0 where the API reports no cache writes (models before GPT-5.6 bill none).

    opencode (packages/opencode/src/session/session.ts, getUsage): each part is
    one model call; its `input` is already uncached and its `output` has the
    reasoning taken out. A provider that reports no reasoning or cache writes
    gives 0 there, not None: an OpenAI-style API such as GLM's never reports
    cache writes, and whether GLM breaks reasoning out of its output is not
    verified (if it does not, the reasoning stays inside `output`).
    """
    u = usage or {}
    by_model = None
    if cli == "claude-code":
        if model_usage:
            by_model = {m: _counts(v.get("inputTokens"), v.get("cacheCreationInputTokens"),
                                   v.get("cacheReadInputTokens"), v.get("outputTokens"), v.get("thinkingTokens"))
                        for m, v in model_usage.items()}
            return {**_sum(by_model.values()), "by_model": by_model}
        counts = _counts(u.get("input_tokens"), u.get("cache_creation_input_tokens"), u.get("cache_read_input_tokens"),
                         u.get("output_tokens"), (u.get("output_tokens_details") or {}).get("thinking_tokens"))
    elif cli == "codex":
        whole, read, write = u.get("input_tokens"), u.get("cached_input_tokens"), u.get("cache_write_input_tokens")
        counts = _counts(None if whole is None else whole - (read or 0) - (write or 0), write, read,
                         u.get("output_tokens"), u.get("reasoning_output_tokens"))
    else:
        visible, reasoning = u.get("output"), u.get("reasoning")
        output = None if visible is None and reasoning is None else (visible or 0) + (reasoning or 0)
        counts = _counts(u.get("input"), u.get("cache.write"), u.get("cache.read"), output, reasoning)
    return {**counts, "by_model": by_model}


# Throttling, by the provider rather than by the agent: what each CLI prints
# when a request was refused for rate, load or quota. Matched only in error
# context (error events, a failed result, stderr), never in the agent's own
# text, which may well talk about rate limits. Most specific first. The
# patterns are the providers' documented error texts (HTTP 429/529, Anthropic's
# rate_limit_error and overloaded_error, OpenAI's insufficient_quota, GLM's codes
# 1113/1302/1305); no throttled run of the pinned CLIs has been recorded yet.
_THROTTLE = (
    ("usage-limit", re.compile(r"(?i)usage limit|(?:5-hour|weekly|daily|session) limit reached"
                               r"|insufficient[_ ]quota|exceeded your current quota"
                               r"|\\?\"code\\?\"\s*:\s*\\?\"?1113\b|余额不足")),
    ("overloaded", re.compile(r"(?i)\b529\b|overloaded")),
    ("rate-limit", re.compile(r"(?i)\b429\b|rate[_ ]?limit|too many requests"
                              r"|\\?\"code\\?\"\s*:\s*\\?\"?130[25]\b|速率限制|限流|并发|访问量过大|请求过于频繁")),
)


def _error_context(account: Account, out: str, err: str) -> list:
    """The texts in which a refusal by the provider would show: never the agent's own messages."""
    texts = []
    for e in _events(out):
        kind = f"{e.get('type', '')} {e.get('subtype', '')}".lower()
        if account.cli == "claude-code" and e.get("type") == "result":
            if e.get("is_error") or e.get("subtype") != "success":
                texts.append(json.dumps(e.get("result")) + " " + str(e.get("subtype")))
        elif "error" in kind or "retry" in kind or "failed" in kind:
            texts.append(json.dumps(e, ensure_ascii=False))
    # What the CLI printed outside its JSON stream (an error it gave up with), never the agent's messages.
    plain = [line for line in out.splitlines() if line.strip() and not line.lstrip().startswith("{")]
    texts.append("\n".join(plain[-20:]))
    texts.append((err or "")[-4000:])
    return texts


def throttle(account: Account, out: str, err: str) -> Optional[dict]:
    """Whether the provider refused the run's requests for rate, load or quota; None if nothing says so.

    A throttled run is not an agent that gave up or crashed: the record keeps
    the two apart, and a controller that resumes runs waits instead of counting
    it as idle. `retries` counts the retry events a CLI printed on its own
    (retrying inside one invocation); `evidence` is a short excerpt."""
    texts = _error_context(account, out, err)
    for kind, pattern in _THROTTLE:
        for text in texts:
            m = pattern.search(text)
            if m:
                lo = max(0, m.start() - 80)
                retries = sum(1 for e in _events(out) if "retry" in f"{e.get('type', '')} {e.get('subtype', '')}".lower())
                return {"kind": kind, "evidence": text[lo:lo + 200], "retries": retries}
    return None


def session_id(account: Account, out: str) -> Optional[str]:
    """The CLI's session, from its own events: what a next turn resumes."""
    for e in _events(out):
        if account.cli == "claude-code" and e.get("session_id"):
            return str(e["session_id"])
        if account.cli == "codex" and e.get("type") == "thread.started" and e.get("thread_id"):
            return str(e["thread_id"])
        if account.cli == "opencode" and e.get("sessionID"):
            return str(e["sessionID"])
    return None


_CODEX_TOOL_ITEMS = ("command_execution", "file_change", "mcp_tool_call", "web_search")


def tool_calls(account: Account, out: str) -> int:
    """How many tools the agent called in one turn: an island's idle count is turns with none (D7)."""
    n, seen = 0, set()
    for e in _events(out):
        if account.cli == "claude-code" and e.get("type") == "assistant":
            n += sum(1 for c in ((e.get("message") or {}).get("content") or [])
                     if isinstance(c, dict) and c.get("type") == "tool_use")
        elif account.cli == "codex" and e.get("type") == "item.completed":
            n += (e.get("item") or {}).get("type") in _CODEX_TOOL_ITEMS
        elif account.cli == "opencode":
            part = e.get("part") or {}
            if part.get("type") == "tool":
                key = part.get("callID") or part.get("id")
                if key not in seen:
                    seen.add(key)
                    n += 1
    return n


def codex_rollouts(run_dir: Path) -> tuple:
    """What Codex's own session logs say: the models that served, and the usage so far -- each session's last
    `token_count` total (a running total, a turn cut off before it completed included), summed over sessions.

    Decided 2026-11-09 from the R41-F3 smoke run: turn.completed carries the thread's running total, so adding up
    a resumed session's turns counted the earlier ones again (island 1: 20.5 M recorded, 13.7 M used), and a turn
    cut off at the time limit has no turn.completed at all (island 0: none recorded, 5.2 M used)."""
    models, total, found = set(), {}, False
    for rollout in sorted((run_dir / "codex-home" / "sessions").rglob("rollout-*.jsonl")):
        last = None
        for line in rollout.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                p = json.loads(line).get("payload") or {}
            except json.JSONDecodeError:
                continue
            if not isinstance(p, dict):
                continue
            if p.get("model"):
                models.add(p["model"])
            if p.get("type") == "token_count" and isinstance(p.get("info"), dict):
                last = p["info"].get("total_token_usage") or last
        if isinstance(last, dict):
            found = True
            for k, v in last.items():
                if k != "total_tokens" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    total[k] = total.get(k, 0) + v
    return models, (total if found else None)


def token_delta(now: Optional[dict], before: Optional[dict]) -> Optional[dict]:
    """One turn's share of two running totals in tokens()'s schema: each count's difference (None stays None)."""
    if now is None:
        return None
    before = before or {}
    out = {}
    for k, v in now.items():
        b = before.get(k)
        out[k] = (v - b if isinstance(b, (int, float)) else v) if isinstance(v, (int, float)) else (None if k == "by_model" else v)
    return out


def parse(account: Account, out: str, run_dir: Path, export_session=None) -> dict:
    """What the CLI answered last, which models served it, the usage it reported, and that usage in one schema."""
    r = _read(account, out, run_dir, export_session)
    r["tokens"] = tokens(account.cli, r.get("usage"), r.get("model_usage"))
    return r


def _read(account: Account, out: str, run_dir: Path, export_session=None) -> dict:
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
        turn_usage = None
        for e in events:
            if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message":
                r["text"] = e["item"].get("text")
            if e.get("type") == "turn.completed":
                turn_usage = e.get("usage") or turn_usage
        models, usage = codex_rollouts(run_dir)
        # The run's usage so far (a running total): the rollouts' when they have it, which a turn cut off before
        # it completed is in too; else the last turn.completed, which carries the same running total.
        r["usage"] = usage or turn_usage
        r["turn_usage"] = turn_usage
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


__all__ = ["codex_rollouts", "token_delta", "CODEX_HOME", "Launch", "MODES", "SKILLS_DIR", "TOKEN_FIELDS", "WORKDIR", "codex_config", "launch",
           "opencode_config", "parse", "session_id", "throttle", "tokens", "tool_calls"]
