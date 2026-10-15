"""A task package: where an agent starts, what it may change, what it is told, how it is measured.

    tasks/<id>/task.json        the definition (read strictly)
    tasks/<id>/instruction.md   the task, in the words every agent receives
    tasks/<id>/reference.patch  optional: the known answer, for validating the loop

The package lives inside the harness directory, which no workspace contains,
so an agent never sees its definition, its points or its reference. Every
point is hidden: `evaluation.validation` is the set an in-run scoring service
may report on (in aggregate only), `evaluation.final` is measured only after
the agent has stopped, and nothing about it ever reaches the agent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA = "tritonchip-agent-task/1"


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    base_commit: str
    design: tuple
    evaluator_owned: tuple
    hide: tuple
    instruction: str
    reference: Optional[Path]
    wall_seconds: int
    evaluation: dict
    path: Path
    sha256: str


def load_task(path: Path) -> Task:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "task.json"
    raw = path.read_bytes()
    v = json.loads(raw)
    required = {"schema", "task_id", "kind", "base_commit", "design", "hide", "instruction", "budget", "evaluation"}
    unknown = set(v) - required - {"note", "reference", "evaluator_owned"}
    missing = required - set(v)
    if missing or unknown:
        raise TaskError(f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if v["schema"] != SCHEMA:
        raise TaskError(f"{path}: schema must be {SCHEMA}")
    if not re.match(r"^[0-9a-f]{40}$", v["base_commit"]):
        raise TaskError(f"{path}: base_commit must be a full commit hash")
    ev = v["evaluation"]
    for k in ("profile", "validation", "final", "jobs", "timeout_seconds"):
        if k not in ev:
            raise TaskError(f"{path}: evaluation.{k} is missing")
    ref = path.parent / v["reference"] if v.get("reference") else None
    return Task(task_id=v["task_id"], kind=v["kind"], base_commit=v["base_commit"], design=tuple(v["design"]),
                evaluator_owned=tuple(v.get("evaluator_owned", ())), hide=tuple(v["hide"]), instruction=(path.parent / v["instruction"]).read_text(encoding="utf-8"),
                reference=ref, wall_seconds=int(v["budget"]["wall_seconds"]), evaluation=dict(ev),
                path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest())


def render(task: Task, mode: str) -> str:
    """The exact text every agent receives. Its sha256 goes in the run record."""
    net = ("Only the model service you are talking to is reachable: there is no web access, no package index "
           "and no download." if mode == "offline" else
           "The network is open. Every connection is logged.")
    design = ", ".join(f"`{d}`" for d in task.design)
    return task.instruction.rstrip() + f"""

## The rules of this run

- Your working directory is a fresh copy of the repository at one commit. Work there.
- What is evaluated is what you change under {design}. The evaluator rebuilds from the base commit plus
  those changes, with its own copies of everything else (tests, benchmarks, tools, and the code that builds
  and drives the simulated device); a change anywhere else is ignored.
- The evaluation runs a benchmark you are not shown. Each point is judged on its own: a point whose numbers
  are wrong, or that does not run on the device, is a failed point; the other points still count.
- {net}
- You have {task.wall_seconds // 60} minutes. When you are done, stop: your working directory as you leave
  it is your submission.
- If you conclude the task cannot be done, say so and stop. That is a valid result.
"""


__all__ = ["Task", "TaskError", "load_task", "render"]
