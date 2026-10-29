"""stats: what the runs of each agent framework used, read from their records.

Every run directory under the given paths (one that holds a run.json, and its
certification.json when there is one) is grouped by agent framework: the CLI,
the requested model and the account label, or the pseudo-agent. Each group
gives its number of runs; per token field (adapters.tokens) the total, the mean
over the runs that report it, and how many do; and per occupancy role
(occupancy.py) the number of entries and the total and mean of their seconds
and of their queue seconds.

Only counts and sums leave it: no point id and no per-point result, so the
output can go into a report without showing the hidden benchmark. Intervals
(bootstrap, Wilson) are not computed here (README section 13).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .adapters import TOKEN_FIELDS, tokens

RUN_SCHEMA = "tritonchip-agent-run/1"


def run_dirs(paths) -> list:
    """Each directory under the paths that holds a run.json; nothing below one is searched."""
    found = set()
    for top in map(Path, paths):
        if not top.exists():
            raise OSError(f"{top} does not exist")
        for root, dirs, files in os.walk(top.parent if top.is_file() else top):
            if "run.json" in files:
                found.add(Path(root))
                dirs[:] = []
    return sorted(found)


def framework(run: dict) -> tuple:
    result = run.get("agent_result") or {}
    if result.get("pseudo"):
        return None, None, result["pseudo"]
    return result.get("cli"), result.get("requested_model"), result.get("label")


def _total_mean(values) -> dict:
    return {"total": round(sum(values), 1) if values else None,
            "mean": round(sum(values) / len(values), 1) if values else None}


def _group(key: tuple, runs: int, run_tokens: list, entries: list) -> dict:
    cli, model, label = key
    out = {"cli": cli, "model": model, "label": label, "runs": runs, "tokens": {}, "occupancy": {}}
    for f in TOKEN_FIELDS:
        got = [t[f] for t in run_tokens if t.get(f) is not None]
        out["tokens"][f] = {**_total_mean(got), "runs": len(got)}
    for role in sorted({e["role"] for e in entries}):
        mine = [e for e in entries if e["role"] == role]
        out["occupancy"][role] = {
            "entries": len(mine),
            "seconds": _total_mean([e["seconds"] for e in mine if e.get("seconds") is not None]),
            "queued_seconds": _total_mean([e.get("queued_seconds") or 0.0 for e in mine])}
    return out


def summarise(paths) -> dict:
    groups: dict = {}
    skipped = 0
    for d in run_dirs(paths):
        try:
            run = json.loads((d / "run.json").read_text(encoding="utf-8"))
            cert = json.loads((d / "certification.json").read_text(encoding="utf-8")) \
                if (d / "certification.json").is_file() else {}
        except (OSError, ValueError):
            skipped += 1
            continue
        if run.get("schema") != RUN_SCHEMA:
            skipped += 1
            continue
        g = groups.setdefault(framework(run), {"runs": 0, "tokens": [], "entries": []})
        g["runs"] += 1
        result = run.get("agent_result") or {}
        if result.get("cli"):
            # A record from before `tokens` existed is normalised from its raw usage.
            g["tokens"].append(result.get("tokens") or tokens(result["cli"], result.get("usage"),
                                                              result.get("model_usage")))
        g["entries"] += (run.get("occupancy") or []) + (cert.get("occupancy") or [])
    order = sorted(groups, key=lambda k: tuple(v or "" for v in k))
    return {"runs": sum(g["runs"] for g in groups.values()), "skipped": skipped,
            "groups": [_group(k, groups[k]["runs"], groups[k]["tokens"], groups[k]["entries"]) for k in order]}


__all__ = ["RUN_SCHEMA", "framework", "run_dirs", "summarise"]
