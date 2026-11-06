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

from . import __version__, certify, evaluator, occupancy, runner
from .services import Broker
from .siteconfig import Site, SiteError
from . import skills as skills_
from .task import Task, render
from .workspace import export, gate

PSEUDO = ("noop", "reference")


def run_task(site: Site, task: Task, agent: str, mode: str = "offline", model: Optional[str] = None,
             skills: Optional[list] = None) -> dict:
    """`skills`: the names of the skills this run provides; None takes the task's own set (`task.json`
    `skills`), a list overrides it -- [] runs without any, a pilot's control arm. The record says which."""
    if site.repository is None:
        raise SiteError("site.json has no `repository`: the clone workspaces are exported from")
    repo = site.repository
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)
    label = f"patch-{Path(agent.split(':', 1)[1]).stem}" if agent.startswith("patch:") else agent
    run_dir = site.state_dir / "runs" / f"{task.task_id}-{label}-{stamp}"
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    island_run = bool(task.islands) and agent not in PSEUDO and not agent.startswith("patch:")
    # An island run's workspaces are its islands' own (islands.py); the run has none of its own.
    exported = None if island_run else export(repo, task.base_commit, task.hide, workspace)
    # The skills this run gets: the task's default set, or the run's override; the prompt names them.
    source = "task" if skills is None else "run"
    names = list(task.skills) if skills is None else list(skills)
    selected = skills_.select(names)
    prompt = render(task, mode, skills=names)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    record: dict = {
        "schema": "tritonchip-agent-run/1", "run_id": run_dir.name, "harness_version": __version__,
        "task": {"id": task.task_id, "sha256": task.sha256, "base_commit": task.base_commit},
        "agent": agent, "mode": mode, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "skills": {"selected": names, "source": source, "hashes": skills_.manifest(selected),
                   "catalog": skills_.version()},
        "started": time.time(),
    }
    record["workspace"] = exported

    broker = None
    if island_run:
        record["prompt_sha256"] = None   # each island's prompt is its own, hashed in its row
        account = site.account(agent)
        if model:
            account = dataclasses.replace(account, models={**account.models, "main": model})
        return _islands(site, task, account, mode, names, record, run_dir, repo)
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
                                                  timeout=float(task.wall_seconds), services=broker,
                                                  skills=names)
        if broker is not None and broker.wait_outlived():
            # A slow request the agent left running ran to its end (report section 8.9, B10): its record,
            # unseen by the agent, joins the run's, and its image may serve the certification below.
            record["agent_result"]["services"] = broker.record()

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
    evaluated = None
    if pseudo and agent != "noop" and not record["agent_result"]["applied"]:
        # The loop's own check failed; nothing it measured would mean anything.
        record["validity"] = "infra-invalid"
    elif g["verdict"] != "PASS" or rtl:
        record["validity"] = "integrity-rejected"
    else:
        candidate = run_dir / "candidate" if g["taken"] else None
        started = time.time()
        record["evaluation"] = evaluator.evaluate(site, repo, task, candidate, run_dir)
        evaluated = (started, time.time())
        # A score stands once certify.py has run it on the card: a changed machine is built
        # first, one that kept the base's runs on the base's image (README section 10.4).
        record["hardware"] = evaluator.hardware_changed(repo, task, candidate)
        record["certification"] = "pending"
        if not record["evaluation"]["valid"]:
            record["validity"] = "infra-invalid"
        elif not pseudo and record["agent_result"].get("stop_reason") == "timeout":
            record["validity"] = "budget-exhausted"
        else:
            record["validity"] = "valid"
        slow = (task.services or {}).get("fpga_slow") or {}
        if (slow.get("certify_after_run") and slow.get("mode") == "board" and record["validity"] != "infra-invalid"
                and not pseudo):
            # The harness certifies the run's candidate on the card itself (report section 8.9, B1).
            try:
                record["certification"] = certify.after_run(site, repo, task, record, run_dir, broker)
            except Exception as exc:  # noqa: BLE001 -- the run's record is kept; certify by hand
                record["certification"] = {"verdict": "no-verdict", "why": f"{type(exc).__name__}: {str(exc)[:300]}"}
    record["occupancy"] = occupancy.of_run(site, task, record["agent_result"], evaluated)
    record["occupancy"] += occupancy.certification(site, record.get("certification"))
    record["finished"] = time.time()
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["run_dir"] = str(run_dir)
    return record


def _islands(site: Site, task: Task, account, mode: str, names: list, record: dict, run_dir: Path,
             repo: Path) -> dict:
    """A run of islands (islands.py): the base measured once before any island, every island at once, the
    slow requests that outlived their islands waited for, then the archive's top 3 by validation to the final
    evaluation and, when the task asks, to the card."""
    import types
    from . import islands as islands_, services
    evaluator.baseline(site, repo, task)
    shared = services.Shared(site, task)
    got = islands_.run(site, task, account, mode, run_dir, repo, names, Broker, shared)
    archive, brokers = got["archive"], [r["broker"] for r in got["islands"] if r]
    for b in brokers:
        b.wait_outlived()
        # Every commit has finished (or was cancelled) before the archive is ranked.
        for j in list(b.jobs.values()):
            if j["service"] != "fpga-slow" and j.get("thread"):
                j["thread"].join()
    rows = []
    for r in got["islands"]:
        if not r:
            continue
        agent_result = dict(r["agent_result"])
        agent_result["services"] = r["broker"].record()
        rows.append({**{k: v for k, v in r.items() if k not in ("broker", "agent_result")},
                     "agent_result": agent_result})
    record["islands"] = {"config": task.islands, "islands": rows}
    record["archive"] = archive.record()
    view = types.SimpleNamespace(images=shared.images,
                                 jobs={f"{b.island}:{k}": j for b in brokers for k, j in b.jobs.items()})
    record["champions"] = _champions(site, repo, task, archive, view, run_dir)
    record["occupancy"] = []
    for row in rows:
        for e in occupancy.of_run(site, task, row["agent_result"], None):
            record["occupancy"].append({**e, "island": row["island"]})
    for c in record["champions"]:
        if c.get("evaluated"):
            points = [*task.evaluation["validation"], *task.evaluation["final"]]
            record["occupancy"].append(occupancy.entry("final-evaluation", None, *c["evaluated"],
                                                       reserved=evaluator.reserved(site, task, points)))
        record["occupancy"] += occupancy.certification(site, c.get("certification"))
    valid = [c for c in record["champions"] if (c.get("evaluation") or {}).get("valid")]
    record["validity"] = "valid" if valid else ("no-commit" if not archive.ranked() else "infra-invalid")
    record["finished"] = time.time()
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                                      encoding="utf-8")
    record["run_dir"] = str(run_dir)
    return record


def _champions(site: Site, repo: Path, task: Task, archive, view, run_dir: Path) -> list:
    """The archive's top 3 by the validation aggregate, in order. Each gets the final evaluation (the final
    set runs only now); with the card, they go on until one is certified (D8, section 8.9 B1). Which commit
    is chosen never looks at the final set."""
    slow = (task.services or {}).get("fpga_slow") or {}
    on_card = bool(slow.get("certify_after_run")) and slow.get("mode") == "board"
    out, seen, top = [], set(), []
    for e in archive.ranked():
        # The same snapshot committed twice (a re-commit, a migrated patch applied as it was) is one candidate.
        if e.get("digest") in seen:
            continue
        seen.add(e.get("digest"))
        top.append(e)
        if len(top) == 3:
            break
    for rank, e in enumerate(top, 1):
        cand = Path(e["candidate"]) if e["candidate"] else None
        # The run's id in the directory name: the evaluator names its containers after it.
        here = run_dir / "champions" / f"{run_dir.name}-c{rank}"
        here.mkdir(parents=True, exist_ok=True)
        started = time.time()
        evaluation = evaluator.evaluate(site, repo, task, cand, here)
        c = {"rank": rank, "id": e["id"], "island": e["island"], "labels": e["labels"], "cell": e["cell"],
             "validation_in_run": e["validation"], "evaluation": evaluation, "evaluated": (started, time.time())}
        out.append(c)
        if not on_card:
            break   # the first is the run's result; nothing decides between it and the next without the card
        if not evaluation.get("valid"):
            continue
        taken = json.loads((cand / "candidate.json").read_text(encoding="utf-8"))["taken"] if cand else []
        rec = {"gate": {"taken": [t["path"] for t in taken]}, "evaluation": evaluation,
               "hardware": evaluator.hardware_changed(repo, task, cand)}
        try:
            c["certification"] = certify.after_run(site, repo, task, rec, cand.parent if cand else here, view)
        except Exception as exc:  # noqa: BLE001 -- recorded; the next champion is tried
            c["certification"] = {"verdict": "no-verdict", "why": f"{type(exc).__name__}: {str(exc)[:300]}"}
        if c["certification"].get("verdict") == "certified":
            break
    return out


__all__ = ["PSEUDO", "run_task"]
