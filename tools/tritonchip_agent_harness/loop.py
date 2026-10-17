"""One task run, end to end: workspace, agent, gate, evaluator, record.

    agent = an account label   a CLI, through the gateway, for the task's wall-clock budget
    agent = "noop"             changes nothing: must score exactly the base (speedup 1.0)
    agent = "reference"        applies the task's reference.patch: must score the known gain
    agent = "patch:<file>"     applies any patch: re-scores a recorded candidate under the current rules

The two pseudo-agents validate the loop itself, with no model involved, before
any model's run is believed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import __version__, evaluator, runner
from .services import Broker
from .siteconfig import Site, SiteError
from .task import Task, render
from .workspace import export, gate

PSEUDO = ("noop", "reference")


def run_task(site: Site, task: Task, agent: str, mode: str = "offline", model: Optional[str] = None) -> dict:
    if site.repository is None:
        raise SiteError("site.json has no `repository`: the clone workspaces are exported from")
    repo = site.repository
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
    label = f"patch-{Path(agent.split(':', 1)[1]).stem}" if agent.startswith("patch:") else agent
    run_dir = site.state_dir / "runs" / f"{task.task_id}-{label}-{stamp}"
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    prompt = render(task, mode)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    record: dict = {
        "schema": "tritonchip-agent-run/1", "run_id": run_dir.name, "harness_version": __version__,
        "task": {"id": task.task_id, "sha256": task.sha256, "base_commit": task.base_commit},
        "agent": agent, "mode": mode, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "started": time.time(),
    }
    record["workspace"] = export(repo, task.base_commit, task.hide, workspace)

    if agent == "noop":
        record["agent_result"] = {"pseudo": "noop"}
    elif agent == "reference" or agent.startswith("patch:"):
        patch = task.reference if agent == "reference" else Path(agent.split(":", 1)[1]).resolve()
        proc = subprocess.run(["git", "-C", str(workspace), "apply", str(patch)], capture_output=True, text=True)
        record["agent_result"] = {"pseudo": agent.split(":", 1)[0], "patch": str(patch),
                                  "applied": proc.returncode == 0, "detail": proc.stderr[-300:]}
    else:
        account = site.account(agent)
        if model:
            account = dataclasses.replace(account, models={**account.models, "main": model})
        broker = Broker(site, repo, task, workspace, run_dir) if task.services else None
        record["agent_result"] = runner.run_agent(site, account, prompt, mode, run_dir, workspace,
                                                  timeout=float(task.wall_seconds), services=broker)

    g = gate(repo, task.base_commit, task.hide, task.design, workspace, out=run_dir / "candidate",
             evaluator_owned=task.evaluator_owned)
    record["gate"] = {"verdict": g["verdict"], "taken": [c["path"] for c in g["taken"]],
                      "dropped": [c["path"] for c in g["dropped"]][:200],
                      "evaluator_owned_changes": [c["path"] for c in g["dropped"] if c.get("reason") == "evaluator-owned"],
                      "rejected": g["rejected"]}
    # Candidate RTL is simulated beside the observer's log: no file or system tasks.
    rtl = evaluator.rtl_check(run_dir / "candidate", repo) if g["verdict"] == "PASS" and g["taken"] else []
    record["gate"]["rtl_check"] = rtl

    pseudo = agent in PSEUDO or agent.startswith("patch:")
    if pseudo and agent != "noop" and not record["agent_result"]["applied"]:
        # The loop's own check failed; nothing it measured would mean anything.
        record["validity"] = "infra-invalid"
    elif g["verdict"] != "PASS" or rtl:
        record["validity"] = "integrity-rejected"
    else:
        candidate = run_dir / "candidate" if g["taken"] else None
        record["evaluation"] = evaluator.evaluate(site, repo, task, candidate, run_dir)
        # A changed machine's score stands once certify.py has run it on the card.
        record["hardware"] = evaluator.hardware_changed(repo, task, candidate)
        record["certification"] = "pending" if record["hardware"]["changed"] else "not-needed"
        if not record["evaluation"]["valid"]:
            record["validity"] = "infra-invalid"
        elif not pseudo and record["agent_result"].get("stop_reason") == "timeout":
            record["validity"] = "budget-exhausted"
        else:
            record["validity"] = "valid"
    record["finished"] = time.time()
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["run_dir"] = str(run_dir)
    return record


__all__ = ["PSEUDO", "run_task"]
