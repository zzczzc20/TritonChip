#!/usr/bin/env python3
"""Command-line entry point for the TritonChip agent harness."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tritonchip_agent_harness import doctor, loop, smoke
    from tritonchip_agent_harness.task import TaskError, load_task
    from tritonchip_agent_harness.provider import ProviderError
    from tritonchip_agent_harness.siteconfig import SiteError, credential_status, load_site
else:
    from . import doctor, loop, smoke
    from .task import TaskError, load_task
    from .provider import ProviderError
    from .siteconfig import SiteError, credential_status, load_site


def _summary(checks: list) -> str:
    lines = []
    for c in checks:
        lines.append(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['check']}" + ("" if c["ok"] else f"  -- {c.get('detail')}"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tritonchip-agent-harness")
    parser.add_argument("--site", type=Path, help="site configuration (default ~/.config/tritonchip-agent-harness/site.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="measure whether this host can give a run its guarantees")
    d.add_argument("--json", action="store_true")
    c = sub.add_parser("credentials", help="each account's credential, from its metadata only")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("smoke", help="one trivial request per account, through the gateway")
    s.add_argument("--account", action="append", help="account label; default every account")
    s.add_argument("--mode", choices=("offline", "open"), default="offline")
    s.add_argument("--json", action="store_true")
    t = sub.add_parser("run-task", help="one task run: workspace, agent, gate, evaluator, record")
    t.add_argument("task", type=Path, help="a task package directory or its task.json")
    t.add_argument("--agent", required=True, help="an account label, or noop / reference")
    t.add_argument("--mode", choices=("offline", "open"), default="offline")
    t.add_argument("--model", help="override the account's main model for this run")
    t.add_argument("--budget-seconds", type=int, help="override the task's wall-clock budget (the prompt says it too)")
    t.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        if args.command == "credentials":
            value = {"ok": None, "accounts": [credential_status(a) for a in site.accounts.values()]}
            value["ok"] = all(a["ok"] for a in value["accounts"])
            if not args.json:
                for a in value["accounts"]:
                    print(f"  {'PASS' if a['ok'] else 'FAIL'}  {a['label']} ({a['cli']}, {a['auth']})"
                          + ("" if a["ok"] else f"  -- {'; '.join(a['problems'])}"))
        elif args.command == "doctor":
            value = doctor.run(site)
            if not args.json:
                print(f"host class: {value['host_class']}   report: {value['report_path']}")
                print(_summary(value["checks"]))
        elif args.command == "run-task":
            task = load_task(args.task)
            if args.budget_seconds:
                task = dataclasses.replace(task, wall_seconds=args.budget_seconds)
            value = loop.run_task(site, task, args.agent, args.mode, args.model)
            value["ok"] = value.get("validity") == "valid"
            if not args.json:
                ev = value.get("evaluation") or {}
                print(f"{value['run_id']}: validity={value['validity']} gate={value['gate']['verdict']} "
                      f"taken={value['gate']['taken']}")
                for name in ("validation", "final"):
                    s = ev.get(name)
                    if not s:
                        continue
                    print(f"  {name}: passed {s['passed']}/{s['n']}  speedup (passed)={s['speedup_geomean']}  "
                          f"fast_1={s['fast_1']}")
                    for row in s["points"]:
                        print(f"    {row['point']:44s} {row['failure'] or 'pass':14s} "
                              f"{row['base_cycles']!s:>8} -> {row['cycles']!s:>8}")
                print(f"  record: {value['run_dir']}/run.json")
        else:
            labels = args.account or list(site.accounts)
            value = {"runs": [smoke.run(site, label, args.mode) for label in labels]}
            value["ok"] = all(r["ok"] for r in value["runs"])
            if not args.json:
                for r in value["runs"]:
                    print(f"{r['label']}: {'PASS' if r['ok'] else 'FAIL'}  {r.get('run_dir', '')}")
                    print(_summary(r.get("checks", [{"check": p, "ok": False} for p in r.get("problems", [])])))
        if args.json:
            print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["ok"] else 1
    except (SiteError, ProviderError, TaskError, OSError) as exc:
        print(f"agent-harness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
