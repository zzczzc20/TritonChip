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
    services: dict
    path: Path
    sha256: str


def load_task(path: Path) -> Task:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "task.json"
    raw = path.read_bytes()
    v = json.loads(raw)
    required = {"schema", "task_id", "kind", "base_commit", "design", "hide", "instruction", "budget", "evaluation"}
    unknown = set(v) - required - {"note", "reference", "evaluator_owned", "services"}
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
    services = dict(v.get("services") or {})
    unknown_services = set(services) - {"score", "fpga_quick", "fpga_slow"}
    if unknown_services:
        raise TaskError(f"{path}: unknown services {sorted(unknown_services)}")
    slow = services.get("fpga_slow")
    if slow is not None:
        if not isinstance(slow, dict) or set(slow) != {"minutes", "job_minutes", "mode"}:
            raise TaskError(f"{path}: services.fpga_slow needs minutes, job_minutes and mode")
        for key in ("minutes", "job_minutes"):
            if (not isinstance(slow[key], (int, float)) or isinstance(slow[key], bool)
                    or slow[key] <= 0):
                raise TaskError(f"{path}: services.fpga_slow.{key} must be positive")
        if slow["job_minutes"] > slow["minutes"]:
            raise TaskError(f"{path}: fpga_slow.job_minutes exceeds its total budget")
        if slow["mode"] not in ("build", "board"):
            raise TaskError(f"{path}: services.fpga_slow.mode must be build or board")
        if int(v["budget"]["wall_seconds"]) < slow["job_minutes"] * 60:
            raise TaskError(f"{path}: agent wall budget is shorter than one FPGA slow request")
    ref = path.parent / v["reference"] if v.get("reference") else None
    return Task(task_id=v["task_id"], kind=v["kind"], base_commit=v["base_commit"], design=tuple(v["design"]),
                evaluator_owned=tuple(v.get("evaluator_owned", ())), hide=tuple(v["hide"]), instruction=(path.parent / v["instruction"]).read_text(encoding="utf-8"),
                reference=ref, wall_seconds=int(v["budget"]["wall_seconds"]), evaluation=dict(ev),
                services=services,
                path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest())


def render(task: Task, mode: str) -> str:
    """The exact text every agent receives. Its sha256 goes in the run record."""
    net = ("Only the model service you are talking to is reachable: there is no web access, no package index "
           "and no download." if mode == "offline" else
           "The network is open. Every connection is logged.")
    design = ", ".join(f"`{d}`" for d in task.design)
    services = _services_text(task.services)
    return task.instruction.rstrip() + services + f"""

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


def _services_text(cfg: dict) -> str:
    """How the agent may ask the evaluator for a measurement, and how much it may ask."""
    if not cfg:
        return ""
    score, fpga, slow = cfg.get("score") or {}, cfg.get("fpga_quick") or {}, cfg.get("fpga_slow") or {}
    lines = ["", "", "## Measurements you can ask for", "",
             "From your shell, `tnca-service` asks the evaluator to measure a snapshot of your working directory "
             "the way the final evaluation will (only changes under the paths it takes count). Whether and when "
             "to use it is up to you. Requests run in the background and your time keeps running while they do. "
             "A request still running when you stop is cancelled, and its result is lost.", ""]
    if score:
        lines.append(f"- `tnca-service submit score`: the hidden validation set, in simulation. You get how many "
                     f"points passed, the geometric-mean speedup over the points that passed, and the share of all "
                     f"points that passed and got faster. No point names, shapes or errors. You have "
                     f"{int(score.get('requests', 0))} such requests.")
    if fpga:
        lines.append(f"- `tnca-service submit fpga-quick`: out-of-context synthesis and timing of the whole device "
                     f"on the FPGA part at a {float(fpga.get('period_ns', 10.0)):g} ns clock, for the profile you are "
                     f"scored on: worst slack, the frequency it implies, the worst paths and the utilisation. A "
                     f"request may take up to {float(fpga.get('job_minutes', 30)):g} minutes. You have "
                     f"{float(fpga.get('minutes', 0)):g} minutes of FPGA time in total, charged as used; one request "
                     f"at a time, and one that could run past what is left is refused. Post-synthesis timing is "
                     f"optimistic: routing adds delay.")
    if slow:
        action = ("full implementation and validation-set board measurement" if slow["mode"] == "board"
                  else "full implementation and routed timing, without the board stage")
        lines.append(f"- `tnca-service submit fpga-slow`: {action}, using the FPGA harness. "
                     f"One request may take up to {float(slow['job_minutes']):g} minutes; the total slow-tier "
                     f"budget is {float(slow['minutes']):g} minutes, charged as used. The request runs in the "
                     f"background; it is cancelled if your agent run ends before it finishes. "
                     f"If a board stage would execute candidate host code, the site must explicitly permit it. "
                     f"You see build/timing information" +
                     (" and only an aggregate for the hidden validation set." if slow["mode"] == "board" else "."))
    lines += ["- `tnca-service budget`, `tnca-service status [ID]`, `tnca-service result ID`, "
              "`tnca-service wait ID`: what is left, and the results."]
    return "\n".join(lines)


__all__ = ["Task", "TaskError", "load_task", "render"]
