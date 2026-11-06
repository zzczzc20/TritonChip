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
from typing import Iterable, Optional

from . import skills as catalog_

SCHEMA = "tritonchip-agent-task/1"
# A block of instruction.md that only a run WITHOUT the skill receives: the syllabus's own
# measuring example, which the skill replaces. The control arm of a pilot gets today's text.
WITHOUT_SKILL = re.compile(r"<!-- without-skill -->\n(.*?)<!-- /without-skill -->\n", re.S)

# The skills a run gets are chosen in three layers (skills.py, decided 2026-11-03): the
# catalog, the task's default set (`task.json` `skills`), and a run's override. `render`
# names the selected ones from their front matter.


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
    skills: tuple = ()


def load_task(path: Path) -> Task:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "task.json"
    raw = path.read_bytes()
    v = json.loads(raw)
    required = {"schema", "task_id", "kind", "base_commit", "design", "hide", "instruction", "budget", "evaluation"}
    unknown = set(v) - required - {"note", "reference", "evaluator_owned", "services", "skills"}
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
    device = ev.get("device")
    if device is not None:
        # A multicore task: the machine is spec/devices/<device>.json, run in the
        # board-top bench behind a mailbox (evaluator._measure_multicore). That
        # bench and its observer are the platform's, so the task must pin one;
        # without it the points would run on the host's own simulator.
        if not isinstance(device, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", device):
            raise TaskError(f"{path}: evaluation.device must name a description in spec/devices")
        if not ev.get("platform"):
            raise TaskError(f"{path}: evaluation.device needs evaluation.platform (the board-top bench and its observer)")
        drain = ev.get("drain_cycles")
        if drain is not None and (isinstance(drain, bool) or not isinstance(drain, int) or drain < 0):
            raise TaskError(f"{path}: evaluation.drain_cycles must be a non-negative whole number of cycles")
    skills = v.get("skills", [])
    if not isinstance(skills, list) or not all(isinstance(s, str) and catalog_.NAME.match(s) for s in skills):
        raise TaskError(f"{path}: skills must be a list of skill names")
    try:
        catalog_.select(skills)
    except catalog_.SkillError as exc:
        raise TaskError(f"{path}: {exc}") from None
    services = dict(v.get("services") or {})
    unknown_services = set(services) - {"score", "fpga_quick", "fpga_slow"}
    if unknown_services:
        raise TaskError(f"{path}: unknown services {sorted(unknown_services)}")
    slow = services.get("fpga_slow")
    if slow is not None:
        _check_slow(path, slow, ev)
    ref = path.parent / v["reference"] if v.get("reference") else None
    return Task(task_id=v["task_id"], kind=v["kind"], base_commit=v["base_commit"], design=tuple(v["design"]),
                evaluator_owned=tuple(v.get("evaluator_owned", ())), hide=tuple(v["hide"]), instruction=(path.parent / v["instruction"]).read_text(encoding="utf-8"),
                reference=ref, wall_seconds=int(v["budget"]["wall_seconds"]), evaluation=dict(ev),
                services=services,
                path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest(), skills=tuple(skills))


_CHECK_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


def _check_slow(path: Path, slow, ev: dict) -> None:
    """services.fpga_slow (report section 8.9): counts, not minutes, for the run; caps for one request."""
    required = {"mode", "flows", "board_runs", "flow_minutes", "board_minutes"}
    optional = {"base_image_sha256", "board_checks", "certify_after_run"}
    if not isinstance(slow, dict) or not required <= set(slow) or set(slow) - required - optional:
        raise TaskError(f"{path}: services.fpga_slow needs {', '.join(sorted(required))} "
                        f"(and may have {', '.join(sorted(optional))})")
    if slow["mode"] not in ("build", "board"):
        raise TaskError(f"{path}: services.fpga_slow.mode must be build or board")
    for key in ("flows", "board_runs"):
        if not isinstance(slow[key], int) or isinstance(slow[key], bool) or slow[key] < 0:
            raise TaskError(f"{path}: services.fpga_slow.{key} must be a count")
    for key in ("flow_minutes", "board_minutes"):
        if not isinstance(slow[key], (int, float)) or isinstance(slow[key], bool) or slow[key] <= 0:
            raise TaskError(f"{path}: services.fpga_slow.{key} must be positive")
    sha = slow.get("base_image_sha256")
    if sha is not None and not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise TaskError(f"{path}: services.fpga_slow.base_image_sha256 must be a sha256")
    checks = slow.get("board_checks", [])
    if not isinstance(checks, list):
        raise TaskError(f"{path}: services.fpga_slow.board_checks must be a list")
    for c in checks:
        if (not isinstance(c, dict) or not _CHECK_NAME.match(str(c.get("name", "")))
                or not isinstance(c.get("command"), list) or not c["command"]
                or not all(isinstance(a, str) for a in c["command"])
                or not isinstance(c.get("env", {}), dict) or set(c) - {"name", "command", "env", "timeout_seconds"}):
            raise TaskError(f"{path}: each services.fpga_slow.board_checks entry needs a name and a command list")
    if not isinstance(slow.get("certify_after_run", False), bool):
        raise TaskError(f"{path}: services.fpga_slow.certify_after_run must be boolean")
    if slow["mode"] == "board" and not isinstance(ev.get("platform"), dict):
        raise TaskError(f"{path}: board mode needs evaluation.platform: the card host runs its code only")


def render(task: Task, mode: str, skills: Iterable[str] = ()) -> str:
    """The exact text every agent receives. Its sha256 goes in the run record.

    `skills`: the names of the skills the run provides (skills.py). With any, the
    instruction's `without-skill` block -- what a skill replaces -- is left out.
    """
    selected = catalog_.select(skills)
    instruction = WITHOUT_SKILL.sub("" if selected else r"\1", task.instruction)
    net = ("Only the model service you are talking to is reachable: there is no web access, no package index "
           "and no download." if mode == "offline" else
           "The network is open. Every connection is logged.")
    design = ", ".join(f"`{d}`" for d in task.design)
    skill_text = catalog_.prompt_lines(selected)
    services = _services_text(task.services)
    return instruction.rstrip() + services + f"""

## The rules of this run

- Your working directory is a fresh copy of the repository at one commit. Work there.
- What is evaluated is what you change under {design}. The evaluator rebuilds from the base commit plus
  those changes, with its own copies of everything else (tests, benchmarks, tools, and the code that builds
  and drives the simulated device); a change anywhere else is ignored.
- The evaluation runs a benchmark you are not shown. Each point is judged on its own: a point whose numbers
  are wrong, or that does not run on the device, is a failed point; the other points still count.
{skill_text}- {net}
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
             "A request still running when you stop is cancelled, and its result is lost (a board request "
             "excepted, below).", ""]
    if score:
        lines.append(f"- `tnca-service submit score`: the hidden validation set, in simulation. You get how many "
                     f"points passed, the geometric-mean speedup over the points that passed, and the share of all "
                     f"points that passed and got faster. When points fail, how many failed of each kind (such as "
                     f"numeric-error, timeout or measurement-rejected), and when your device did not build or "
                     f"staging stopped before any point ran, the last lines of that log. Never which point, its "
                     f"shape or its own error text. You have "
                     f"{int(score.get('requests', 0))} such requests.")
    if fpga:
        lines.append(f"- `tnca-service submit fpga-quick`: out-of-context synthesis and timing of the whole device "
                     f"on the FPGA part at a {float(fpga.get('period_ns', 10.0)):g} ns clock, for the profile you are "
                     f"scored on: worst slack, the frequency it implies, the worst paths and the utilisation. A "
                     f"request may take up to {float(fpga.get('job_minutes', 30)):g} minutes. You have "
                     f"{float(fpga.get('minutes', 0)):g} minutes of FPGA time in total, charged as used; one request "
                     f"at a time, and one that could run past what is left is refused. Post-synthesis timing is "
                     f"optimistic: routing adds delay.")
    if slow and slow["mode"] == "build":
        lines.append(f"- `tnca-service submit fpga-slow`: the full implementation of your machine on the FPGA "
                     f"(place and route at the target clock, one Vivado flow of up to "
                     f"{float(slow['flow_minutes']):g} minutes), and its routed timing. You have "
                     f"{int(slow['flows'])} such flows in this run.")
    elif slow:
        lines.append(
            f"- `tnca-service submit fpga-slow`: the real card. Board time is precious: a request that changed "
            f"the machine (RTL, the profile's parameters, the device description) is first fully implemented "
            f"(one Vivado flow, which takes hours; up to {float(slow['flow_minutes']):g} minutes), and you have "
            f"{int(slow['flows'])} flows in this run. Accumulate enough hardware change, and be confident in it, "
            f"before you ask. A request that kept the machine (compiler and software only), or that asks again "
            f"for hardware this run already built, needs no flow and runs on the card directly; you have "
            f"{int(slow['board_runs'])} such board runs. Before anything is charged, the snapshot's validation "
            f"set must pass in simulation, and a hardware request needs an `fpga-quick` result of this run on "
            f"exactly that hardware that met timing; otherwise it is refused and not charged. A flow that fails "
            f"on your design (timing or place-and-route does not converge, it does not fit, DRC) counts; one "
            f"that fails on the infrastructure does not, and is retried for you. One slow request at a time; "
            f"the quick tier stays open while it runs. The card is shared: a request waits until the card has "
            f"been idle for a while, and that wait is part of its time. On the card you get the hidden "
            f"validation set's aggregate against the base on the card (as the score service gives it), how "
            f"many points counted the cycles simulation counted, and any board check (such as FP8 bit "
            f"alignment) that did not pass; for a flow that failed, why: the failing stage, slack, the worst "
            f"paths and whether they cross an SLR, utilisation per SLR, the DRC rules that failed. A board "
            f"request still running when you stop runs to its end, but you will not see its result.")
    lines += ["- `tnca-service budget`, `tnca-service status [ID]`, `tnca-service result ID`, "
              "`tnca-service wait ID`: what is left, and the results."]
    return "\n".join(lines)


__all__ = ["Task", "TaskError", "WITHOUT_SKILL", "load_task", "render"]
