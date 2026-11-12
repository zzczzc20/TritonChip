"""A task package: where an agent starts, what it may change, what it is told, how it is measured.

    tasks/<id>/task.json        the definition (read strictly)
    tasks/<id>/instruction.md   the task, in the words every agent receives
    tasks/<id>/reference.patch  optional: the known answer, for validating the loop

The package lives inside the harness directory, which no workspace contains,
so an agent never sees its definition, its points or its reference. Every
point is hidden: `evaluation.validation` is the set an in-run scoring service
may report on (in aggregate only), `evaluation.final` is measured only after
the agent has stopped, and nothing about it ever reaches the agent.

A task with the card may split the card's points from simulation's (decided
2026-11-08): `evaluation.board_validation` is what an in-run board request
measures (points too slow for simulation; aggregate only, as `validation`),
`evaluation.board_final` the card's held-out set, measured on the result's
image only after the run. Without `board_validation` a board request measures
`validation`, as before.
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
    islands: Optional[dict] = None
    resume: Optional[dict] = None
    # The performance models (tools/performance_harness/systemc) are the agent's tools in this task: the run
    # mounts the site's SystemC and the model hook (runner.MODELS_DIR). Decided 2026-11-08 for R41's second
    # formal run; a task that does not ask gets neither.
    models: bool = False


def load_task(path: Path) -> Task:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "task.json"
    raw = path.read_bytes()
    v = json.loads(raw)
    required = {"schema", "task_id", "kind", "base_commit", "design", "hide", "instruction", "budget", "evaluation"}
    unknown = set(v) - required - {"note", "reference", "evaluator_owned", "services", "skills", "islands", "resume",
                                      "models"}
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
    _check_point_sets(path, ev, v.get("services") or {})
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
    _check_board_score(path, services, ev)
    islands = _check_islands(path, v.get("islands"), services)
    resume = _check_resume(path, v.get("resume"))
    models = v.get("models", False)
    if not isinstance(models, bool):
        raise TaskError(f"{path}: models must be true or false")
    ref = path.parent / v["reference"] if v.get("reference") else None
    return Task(task_id=v["task_id"], kind=v["kind"], base_commit=v["base_commit"], design=tuple(v["design"]),
                evaluator_owned=tuple(v.get("evaluator_owned", ())), hide=tuple(v["hide"]), instruction=(path.parent / v["instruction"]).read_text(encoding="utf-8"),
                reference=ref, wall_seconds=int(v["budget"]["wall_seconds"]), evaluation=dict(ev),
                services=services,
                path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest(), skills=tuple(skills), islands=islands, resume=resume,
                models=models)


POINT_SETS = ("validation", "final", "board_validation", "board_final")


def _check_point_sets(path: Path, ev: dict, services: dict) -> None:
    """Every point set is a list of distinct point ids, and no point is in two sets: a point is scored in one
    place or held out, never both. `board_validation` needs the card (services.fpga_slow in board mode)."""
    seen: dict = {}
    for name in POINT_SETS:
        if name not in ev:
            continue
        ids = ev[name]
        if not isinstance(ids, list) or not all(isinstance(p, str) and p for p in ids) or len(set(ids)) != len(ids):
            raise TaskError(f"{path}: evaluation.{name} must be a list of distinct point ids")
        if name.startswith("board_") and not ids:
            raise TaskError(f"{path}: evaluation.{name} is empty; leave it out instead")
        for p in ids:
            if p in seen:
                raise TaskError(f"{path}: {p} is in both evaluation.{seen[p]} and evaluation.{name}")
            seen[p] = name
    if "board_validation" in ev and ((services.get("fpga_slow") or {}).get("mode") != "board"):
        raise TaskError(f"{path}: evaluation.board_validation needs services.fpga_slow in board mode")


def _check_board_score(path: Path, services: dict, ev: dict) -> None:
    """The board as the score (2026-11-10): `services.score.on` "board" scores a commit on the card, by
    `evaluation.families` (B2); `evaluation.gate_points` are the short RTL gate before a flow and
    `evaluation.health_point` the card's known-answer check -- neither may be a held-out point."""
    on = (services.get("score") or {}).get("on", "simulation")
    if on not in ("simulation", "board"):
        raise TaskError(f"{path}: services.score.on must be simulation or board")
    fams = ev.get("families")
    if fams is not None:
        def good(spec) -> bool:
            w = spec.get("weight", 1) if isinstance(spec, dict) else None
            return (isinstance(spec, dict) and isinstance(spec.get("prefixes"), list) and bool(spec["prefixes"])
                    and all(isinstance(x, str) and x for x in spec["prefixes"])
                    and isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0)
        if not isinstance(fams, dict) or not all(isinstance(k, str) and k and good(v) for k, v in fams.items()):
            raise TaskError(f"{path}: evaluation.families maps a name to {{prefixes: [...], weight: > 0}}")
    held = set(ev.get("final") or ()) | set(ev.get("board_final") or ())
    gate = ev.get("gate_points")
    if gate is not None:
        if not isinstance(gate, list) or not gate or not all(isinstance(x, str) and x for x in gate) \
                or len(set(gate)) != len(gate):
            raise TaskError(f"{path}: evaluation.gate_points must be a list of distinct point ids")
        if held & set(gate):
            raise TaskError(f"{path}: evaluation.gate_points has held-out points: {sorted(held & set(gate))}")
    health = ev.get("health_point")
    if health is not None and (not isinstance(health, str) or not health or health in held):
        raise TaskError(f"{path}: evaluation.health_point must be one point id outside the held-out sets")
    if on == "board":
        if (services.get("fpga_slow") or {}).get("mode") != "board":
            raise TaskError(f"{path}: services.score.on board needs services.fpga_slow in board mode (the card)")
        if not ev.get("board_validation"):
            raise TaskError(f"{path}: services.score.on board needs evaluation.board_validation")
        if not gate:
            raise TaskError(f"{path}: services.score.on board needs evaluation.gate_points (the gate before a flow)")


def board_points(task: Task) -> list:
    """The points an in-run board request measures: the task's board validation set, else its validation set.
    With the board as the score (2026-11-10, B1) both: the simulated validation set's small points and the
    card's own medium and large ones, in that order."""
    ev = task.evaluation
    if ((task.services or {}).get("score") or {}).get("on") == "board":
        return list(dict.fromkeys([*ev["validation"], *(ev.get("board_validation") or ())]))
    return list(ev.get("board_validation") or ev["validation"])


# The descriptors an island grid may use (islands.py): the change's layer, and (2026-11-11, for R41-F5) the
# group of programs whose speedup was largest -- a result of the commit, not where it changed.
FEATURES = ("layer", "op_group")
# The groups `op_group` sorts points into, by point-id prefix; every other point is "elementwise" (element-wise
# operations, reductions and normalisations, data movement).
OP_GROUPS = ("elementwise", "matrix", "attention", "model")
DEFAULT_OP_GROUPS = {"matrix": ["op.matmul/", "op.bmm/", "op.linear/", "op.gemv/", "op.scaled_mm/",
                                "op.w8a8_block_linear/"],
                     "attention": ["op.attention"],
                     "model": ["model."]}
# The layers an island role may write (islands.LAYERS), and how the task text names them.
ROLE_LAYERS = {"software": "software (`compiler/`, `python/tnca/`)",
               "hardware": "hardware (`rtl/`, the scored profile in `spec/r1_profiles.json`, the device "
                           "description in `spec/devices/`)",
               "isa": "the ISA (`isa/`)"}


def _check_roles(path: Path, roles, n: int, services: dict) -> Optional[list]:
    """`islands.roles` (2026-11-09): one per island -- the layers it writes and its own budgets for the slow tier
    (flows, board runs) and the quick tier (minutes). The slow tier's counts add up to the task's."""
    if roles is None:
        return None
    if not isinstance(roles, list) or len(roles) != n:
        raise TaskError(f"{path}: islands.roles needs one entry per island ({n})")
    out = []
    for i, r in enumerate(roles):
        if (not isinstance(r, dict) or set(r) - {"name", "authors", "flows", "board_runs", "fpga_quick_minutes"}
                or not isinstance(r.get("authors"), list) or not r["authors"]
                or any(a not in ROLE_LAYERS for a in r["authors"])):
            raise TaskError(f"{path}: islands.roles[{i}] needs authors (a list of {', '.join(ROLE_LAYERS)}) and may "
                            f"have name, flows, board_runs, fpga_quick_minutes")
        for key in ("flows", "board_runs"):
            x = r.get(key, 0)
            if not isinstance(x, int) or isinstance(x, bool) or x < 0:
                raise TaskError(f"{path}: islands.roles[{i}].{key} must be a count")
        q = r.get("fpga_quick_minutes")
        if q is not None and (not isinstance(q, (int, float)) or isinstance(q, bool) or q < 0):
            raise TaskError(f"{path}: islands.roles[{i}].fpga_quick_minutes must be >= 0")
        out.append({"name": str(r.get("name") or f"island-{i}"), "authors": sorted(set(r["authors"])),
                    "flows": int(r.get("flows", 0)), "board_runs": int(r.get("board_runs", 0)),
                    **({"fpga_quick_minutes": float(q)} if q is not None else {})})
    slow = (services or {}).get("fpga_slow")
    if slow:
        for key in ("flows", "board_runs"):
            if sum(r[key] for r in out) != int(slow.get(key, 0)):
                raise TaskError(f"{path}: islands.roles' {key} must add up to services.fpga_slow.{key}")
    return out


def _check_islands(path: Path, value, services: dict) -> Optional[dict]:
    """`islands` (report section 8.5): how many islands, how they see each other, the grid's axes. The task's
    budgets are each island's: wall clock, scoring requests (its commits), quick-tier minutes; the slow
    tier's counts stay the run's (section 8.9, B7)."""
    if value is None:
        return None
    known = {"count", "topology", "migration_interval", "migration_rate", "features", "max_idle_turns",
             "max_restarts", "target", "roles", "op_groups", "min_gain", "software_exchange_seconds"}
    if not isinstance(value, dict) or set(value) - known or "count" not in value:
        raise TaskError(f"{path}: islands needs count (and may have {', '.join(sorted(known - {'count'}))})")
    n = value["count"]
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 16:
        raise TaskError(f"{path}: islands.count must be 1..16")
    topo = value.get("topology", "ring")
    if isinstance(topo, dict):
        ok = all(str(k).isdigit() and int(k) < n and isinstance(v, list)
                 and all(isinstance(x, int) and 0 <= x < n and x != int(k) for x in v) for k, v in topo.items())
        if not ok:
            raise TaskError(f"{path}: islands.topology as a map: island -> list of the islands it sees")
    elif topo not in ("ring", "all", "none"):
        raise TaskError(f"{path}: islands.topology must be ring, all, none or an adjacency map")
    for key, lo in (("migration_interval", 1), ("max_idle_turns", 1), ("max_restarts", 0)):
        x = value.get(key, lo)
        if not isinstance(x, int) or isinstance(x, bool) or x < lo:
            raise TaskError(f"{path}: islands.{key} must be an integer >= {lo}")
    rate = value.get("migration_rate", 0.3)
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 < rate <= 1:
        raise TaskError(f"{path}: islands.migration_rate must be in (0, 1]")
    features = value.get("features", ["layer"])
    if not isinstance(features, list) or not features or any(f not in FEATURES for f in features):
        raise TaskError(f"{path}: islands.features: known axes are {', '.join(FEATURES)}")
    groups = value.get("op_groups", DEFAULT_OP_GROUPS)
    if not isinstance(groups, dict) or set(groups) - set(OP_GROUPS[1:]) or not all(
            isinstance(v, list) and all(isinstance(x, str) and x for x in v) for v in groups.values()):
        raise TaskError(f"{path}: islands.op_groups maps {', '.join(OP_GROUPS[1:])} to point-id prefixes "
                        f"(every other point is elementwise)")
    gain = value.get("min_gain", 1.02)
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not 1.0 <= gain < 2.0:
        raise TaskError(f"{path}: islands.min_gain (a group's geometric-mean speedup a commit needs for a cell) "
                        f"must be in [1, 2)")
    exchange = value.get("software_exchange_seconds")
    if exchange is not None and (not isinstance(exchange, int) or isinstance(exchange, bool) or exchange < 60):
        raise TaskError(f"{path}: islands.software_exchange_seconds must be an integer >= 60")
    target = value.get("target")
    if target is not None and (not isinstance(target, dict) or set(target) - {"speedup_geomean", "fast_1"}
                               or not all(isinstance(x, (int, float)) for x in target.values())):
        raise TaskError(f"{path}: islands.target may name speedup_geomean and/or fast_1")
    if not ((services or {}).get("score") or {}).get("requests"):
        raise TaskError(f"{path}: islands need services.score.requests: an island's commits")
    roles = _check_roles(path, value.get("roles"), n, services)
    return {"count": n, "topology": topo, "migration_interval": int(value.get("migration_interval", 3)),
            "migration_rate": float(rate), "features": list(features),
            "op_groups": {k: list(v) for k, v in groups.items()}, "min_gain": float(gain),
            "software_exchange_seconds": exchange,
            "max_idle_turns": int(value.get("max_idle_turns", 3)), "max_restarts": int(value.get("max_restarts", 2)),
            "target": target, **({"roles": roles} if roles else {})}


def _check_resume(path: Path, value) -> Optional[dict]:
    """`resume` (a plain run's early-exit resume, 2026-11-07): when the agent's CLI stops before the time is up,
    the harness continues it, as an island's controller does (islands.Turns); an islands task has its own."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"max_idle_turns", "max_restarts"}:
        raise TaskError(f"{path}: resume may have max_idle_turns and max_restarts")
    for key, lo, default in (("max_idle_turns", 1, 3), ("max_restarts", 0, 2)):
        x = value.get(key, default)
        if not isinstance(x, int) or isinstance(x, bool) or x < lo:
            raise TaskError(f"{path}: resume.{key} must be an integer >= {lo}")
    return {"max_idle_turns": int(value.get("max_idle_turns", 3)), "max_restarts": int(value.get("max_restarts", 2))}


_CHECK_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


def _check_slow(path: Path, slow, ev: dict) -> None:
    """services.fpga_slow (report section 8.9): counts, not minutes, for the run; caps for one request."""
    required = {"mode", "flows", "board_runs", "flow_minutes", "board_minutes"}
    optional = {"base_image_sha256", "board_checks", "certify_after_run", "flows_at_once", "board_faster_than",
                "relay_cache_gib", "flows_at_once_per_island"}
    if not isinstance(slow, dict) or not required <= set(slow) or set(slow) - required - optional:
        raise TaskError(f"{path}: services.fpga_slow needs {', '.join(sorted(required))} "
                        f"(and may have {', '.join(sorted(optional))})")
    cache = slow.get("relay_cache_gib")
    if cache is not None and (isinstance(cache, bool) or not isinstance(cache, (int, float)) or cache <= 0):
        raise TaskError(f"{path}: services.fpga_slow.relay_cache_gib must be a positive number (GiB of the card "
                        f"host's chunk store, board_relay.py)")
    if slow["mode"] not in ("build", "board"):
        raise TaskError(f"{path}: services.fpga_slow.mode must be build or board")
    for key in ("flows", "board_runs"):
        if not isinstance(slow[key], int) or isinstance(slow[key], bool) or slow[key] < 0:
            raise TaskError(f"{path}: services.fpga_slow.{key} must be a count")
    for key in ("flow_minutes", "board_minutes"):
        if not isinstance(slow[key], (int, float)) or isinstance(slow[key], bool) or slow[key] <= 0:
            raise TaskError(f"{path}: services.fpga_slow.{key} must be positive")
    # How many of the run's flows may run at once, each on its own build machine (default one).
    at_once = slow.get("flows_at_once", 1)
    if not isinstance(at_once, int) or isinstance(at_once, bool) or at_once < 1:
        raise TaskError(f"{path}: services.fpga_slow.flows_at_once must be an integer >= 1")
    per = slow.get("flows_at_once_per_island")
    if per is not None and (not isinstance(per, int) or isinstance(per, bool) or not 1 <= per <= at_once):
        raise TaskError(f"{path}: services.fpga_slow.flows_at_once_per_island must be an integer from 1 to "
                        f"flows_at_once")
    faster = slow.get("board_faster_than", 1.0)
    if not isinstance(faster, (int, float)) or isinstance(faster, bool) or not 1.0 <= faster < 2.0:
        raise TaskError(f"{path}: services.fpga_slow.board_faster_than must be in [1, 2)")
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


def render(task: Task, mode: str, skills: Iterable[str] = (), island: Optional[int] = None) -> str:
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
    roles = (task.islands or {}).get("roles") if island is not None else None
    services = _services_text(task.services, islands=island is not None and bool(task.islands),
                              resumed=bool(task.resume), role=roles[island] if roles else None,
                              board_cycles=((task.evaluation or {}).get("certify") or {}).get("board_cycles", "equal"),
                              board_set=bool((task.evaluation or {}).get("board_validation")),
                              roofline=bool((task.evaluation or {}).get("device")),
                              board=_board_facts(task))
    return instruction.rstrip() + services + f"""

## The rules of this run

- Your working directory is a fresh copy of the repository at one commit. Work there.
- What is evaluated is what you change under {design}. The evaluator rebuilds from the base commit plus
  those changes, with its own copies of everything else (tests, benchmarks, tools, and the code that builds
  and drives the simulated device); a change anywhere else is ignored.
- The evaluation runs a benchmark you are not shown. Each point is judged on its own: a point whose numbers
  are wrong, or that does not run on the device, is a failed point; the other points still count.
{skill_text}- {net}
{_ending(task, island)}"""


def _board_facts(task: Task) -> Optional[dict]:
    """What the services text says of the board score, for a task that scores on the card; else None."""
    if ((task.services or {}).get("score") or {}).get("on") != "board":
        return None
    ev = task.evaluation
    return {"n": len(board_points(task)), "families": ev.get("families") or {},
            "gate": len(ev.get("gate_points") or ()), "health": bool(ev.get("health_point"))}


def _ending(task: Task, island: Optional[int]) -> str:
    if (island is None or not task.islands) and task.resume:
        return (f"- You have {task.wall_seconds // 60} minutes. Keep improving until the time runs out: your working "
                f"directory as it is then is your submission. If you stop earlier, you are asked to continue; a "
                f"turn that calls no tool counts as idle, and {task.resume['max_idle_turns']} idle turns in a row "
                f"end the run with your working directory as it is.\n")
    if island is None or not task.islands:
        return (f"- You have {task.wall_seconds // 60} minutes. When you are done, stop: your working directory as "
                f"you leave\n  it is your submission.\n"
                f"- If you conclude the task cannot be done, say so and stop. That is a valid result.\n")
    cfg = task.islands
    n = cfg["count"]
    others = ("" if n == 1 else
              f" You are island {island} of {n}; other islands work on the same task at the same time, and you "
              f"see the commits of the islands your island is connected to. Every island starts from the same "
              f"base at the same time, and the run's result is the best commit of all islands by the validation "
              f"set.")
    role = (cfg.get("roles") or [None] * n)[island]
    if role:
        writes = [ROLE_LAYERS[a] for a in role["authors"]]
        others += (f" Your island writes {', '.join(writes[:-1]) + ' and ' + writes[-1] if len(writes) > 1 else writes[0]}"
                   + (": a commit may change only that. " if len(writes) == 1 else ": a commit may change only those. ")
                   if len(writes) < len(ROLE_LAYERS) else " Your island may write every layer.")
        if len(writes) < len(ROLE_LAYERS):
            others += ("Another layer may differ from the base only when it is exactly that layer of a commit you can "
                       "see, taken whole (`tnca-service read-commit ID`, then `git apply`); anything else is refused, "
                       "uncharged.")
    target = cfg.get("target")
    by = "the board score" if _board_facts(task) else "the validation set"
    others = others.replace("by the validation set.", f"by {by}.")
    goal = (f" The run ends early if a commit reaches {', '.join(f'{k} {v:g}' for k, v in target.items())} on "
            f"{by}." if target else "")
    return (f"- You have {task.wall_seconds // 60} minutes and "
            f"{int(((task.services or {}).get('score') or {}).get('requests', 0))} commits.{others} Keep improving "
            f"until your commits or your time run out: a commit is what counts, not your working directory as you "
            f"leave it, and the best commit by {by} is the result.{goal}\n"
            f"- `tnca-service submit score --message \"one line on what this commit changes\"` commits a snapshot: "
            f"it is scored as the score service scores it and kept in the archive with your message (shown to other "
            f"islands as yours, unverified). Its reply, once done, also gives your best commit, your island's grid "
            f"({_cells_text(cfg)}), and "
            + ("with every commit the elites of connected islands you have not been given yet"
               if cfg["migration_interval"] == 1 else
               f"every {cfg['migration_interval']} commits some elites of connected islands")
            + " to consider.\n"
            f"- `tnca-service archive` lists the commits you can see; `tnca-service read-commit ID` prints one as a "
            f"patch against the base commit (`git apply` it on a clean tree to continue from it).\n"
            + _exchange_text(cfg)
            + f"- If you stop, you are asked to continue; a turn that calls no tool counts as idle, and "
            f"{cfg['max_idle_turns']} idle turns in a row end your island.\n")


def _cells_text(cfg: dict) -> str:
    """How the task text says what a cell of the island grid is."""
    if "op_group" not in cfg.get("features", ()):
        return "one elite per cell; cells are what a commit changed: compiler only, hardware only, or both"
    return (f"one elite per cell, every cell listed, empty ones included; a cell is the group of the scored "
            f"programs that sped up most -- element-wise, reduction and data-movement programs, matrix products, "
            f"attention, or whole models, by the geometric mean of their speedups -- and what the commit changed: "
            f"compiler only, hardware only, or both. A commit whose best group gained less than "
            f"{(cfg.get('min_gain', 1.02) - 1) * 100:.0f} % takes no cell; it stays in the archive")


def _exchange_text(cfg: dict) -> str:
    """When another island's commit becomes visible (2026-11-11, R41-F5)."""
    every = cfg.get("software_exchange_seconds")
    if not every:
        return ""
    hours = every / 3600
    when = f"{hours:g} hours" if hours >= 1 else f"{every // 60} minutes"
    return (f"- A commit of another island that changed hardware is visible to you (in the archive, to read, and "
            f"as an elite you are given) as soon as it is scored; one that changed only software becomes visible "
            f"at the next exchange, every {when} of the run, for every island at once. Your own commits are always "
            f"visible to you.\n")


def _services_text(cfg: dict, islands: bool = False, resumed: bool = False, board_cycles: str = "equal",
                   board_set: bool = False, role: Optional[dict] = None, roofline: bool = False,
                   board: Optional[dict] = None) -> str:
    """How the agent may ask the evaluator for a measurement, and how much it may ask. `role`: the island's
    (islands.roles), whose budgets are its own."""
    if not cfg:
        return ""
    score, fpga, slow = cfg.get("score") or {}, cfg.get("fpga_quick") or {}, cfg.get("fpga_slow") or {}
    quick_minutes = float((role or {}).get("fpga_quick_minutes", fpga.get("minutes", 0)) if fpga else 0)
    flows = int(role["flows"]) if role else int(slow.get("flows", 0)) if slow else 0
    board_runs = int(role["board_runs"]) if role else int(slow.get("board_runs", 0)) if slow else 0
    shared = (", your island's own" if role else ", shared by every island") if islands else ""
    at_once = int(slow.get("flows_at_once", 1)) if slow else 1
    # One at a time keeps the text it always had; more says how many, and that a flow may wait for a machine.
    concurrency = ("One slow request at a time; the quick tier stays open while it runs." if at_once == 1 else
                   f"Up to {at_once} slow requests at a time, each flow on a build machine of its own (a flow "
                   f"may wait for one to be free, and `tnca-service status` says when it does); the quick tier "
                   f"stays open while they run.")
    lines = ["", "", "## Measurements you can ask for", "",
             "From your shell, `tnca-service` asks the evaluator to measure a snapshot of your working directory "
             "the way the final evaluation will (only changes under the paths it takes count). Whether and when "
             "to use it is up to you. Requests run in the background and your time keeps running while they do. " +
             ("When you stop you are asked to continue, and your requests keep running; the budgets below are "
              "this island's unless they say they are shared." if islands else
              "When you stop you are asked to continue, and your requests keep running until the time runs out "
              "(a board request still running then runs to its end, unseen)." if resumed else
              "A request still running when you stop is cancelled, and its result is lost (a board request "
              "excepted, below)."), ""]
    if score and board is not None:
        lines += _board_lines(score, slow, board, flows, shared, islands, at_once)
    elif score:
        lines.append(f"- `tnca-service submit score`: the hidden validation set, in simulation. You get how many "
                     f"points passed, the geometric-mean speedup over the points that passed, and the share of all "
                     f"points that passed and got faster. When points fail, how many failed of each kind (such as "
                     f"numeric-error, timeout or measurement-rejected), and when your device did not build or "
                     f"staging stopped before any point ran, the last lines of that log. Never a point's name, its "
                     f"shape or its own error text. "
                     + ("The reply also lists every validation point in a fixed order, numbered and never named: the "
                        "resource that binds it and its η, its distance from the whole chip's roofline (η = L / T: L "
                        "the least cycles the base machine's cores and memory outlets could take for its work, T the "
                        "cycles measured; none for a point that did not pass), beside the base's η. "
                        if roofline else "")
                     + f"You have {int(score.get('requests', 0))} such requests, one running at a time. The "
                     + ("run (every island) " if islands else "run ")
                     + "simulates only a few validation sets at once, scoring requests and the simulations of "
                     "`fpga-slow` requests together, so a request may wait for its turn, and its time includes the "
                     "wait. A snapshot exactly like one this run already scored is not measured or charged again"
                     + (" and is not a new commit" if islands else "") + ": you get that request's result.")
    if fpga:
        lines.append(f"- `tnca-service submit fpga-quick`: out-of-context synthesis and timing of the whole device "
                     f"on the FPGA part at a {float(fpga.get('period_ns', 10.0)):g} ns clock, for the profile you are "
                     f"scored on: worst slack, the frequency it implies, the worst paths and the utilisation. A "
                     f"request may take up to {float(fpga.get('job_minutes', 30)):g} minutes. "
                     + (f"You have {quick_minutes:g} minutes of FPGA time in total, charged as used; one "
                        f"request at a time, and one that could run past what is left is refused. "
                        if quick_minutes > 0 else "Your island has no FPGA time for it. ")
                     + "Post-synthesis timing is optimistic: routing adds delay. It synthesises the core's top "
                     "level only, so a change it cannot apply there (the uncore's, or a parameter that top does not "
                     "take) is listed in its reply as `not_applied`.")
    if board is not None:
        pass   # the card is the score, and `flow` builds a machine (above): no separate slow service
    elif slow and slow["mode"] == "build":
        lines.append(f"- `tnca-service submit fpga-slow`: the full implementation of your machine on the FPGA "
                     f"(place and route at the target clock, one Vivado flow of up to "
                     f"{float(slow['flow_minutes']):g} minutes), and its routed timing. You have "
                     f"{int(slow['flows'])} such flows in this run."
                     + (f" Up to {at_once} at a time, each on a build machine of its own." if at_once > 1 else ""))
    elif slow:
        lines.append(
            f"- `tnca-service submit fpga-slow`: the real card. Board time is precious: a request that changed "
            f"the machine (RTL, the profile's parameters, the device description) is first fully implemented "
            f"(one Vivado flow, which takes hours; up to {float(slow['flow_minutes']):g} minutes), and you have "
            f"{flows} flows in this run{shared}. Accumulate enough hardware change, and be confident in it, "
            f"before you ask. A request that kept the machine (compiler and software only), or that asks again "
            f"for hardware this run already built, needs no flow and runs on the card directly; you have "
            f"{board_runs} such board runs{shared}. Before anything is charged, the snapshot's validation "
            f"set must pass in simulation, and a hardware request needs an `fpga-quick` result of this run on "
            f"exactly that hardware that met timing; otherwise it is refused and not charged. For a request "
            f"that needs a flow, the simulation runs beside the flow, and a simulation that does not pass stops "
            f"the flow, which is then not charged. A flow that fails "
            f"on your design (timing or place-and-route does not converge, it does not fit, DRC) counts; one "
            f"that fails on the infrastructure does not, and is retried for you. {concurrency} "
            f"The card is shared: a request waits until the card has "
            f"been idle for a while, and that wait is part of its time. On the card you get "
            + ("the hidden board validation set's aggregate against the base on the card (as the score service "
               "gives it; the card's own points, larger than the simulated ones and too slow to simulate)"
               if board_set else
               "the hidden validation set's aggregate against the base on the card (as the score service gives "
               "it), "
               + ("how many points counted the cycles simulation counted"
                  if board_cycles == "equal" else
                  "the card's cycle counts against simulation's in aggregate (the geometric mean and range of "
                  "board over simulation)"))
            + (", the same list of η for the card's points" if roofline else "")
            + f", and any board check (such as FP8 bit "
            f"alignment) that did not pass; for a flow that failed, why: the failing stage, slack, the worst "
            f"paths and whether they cross an SLR, utilisation per SLR, the DRC rules that failed. A board "
            f"request still running when you stop runs to its end, but you will not see its result."
            + (f" On the card a point counts as faster only when it is more than "
               f"{(float(slow['board_faster_than']) - 1) * 100:g}% faster than the base: the card's count of one "
               f"program differs by up to about half a percent from run to run."
               if float(slow.get("board_faster_than", 1.0)) > 1.0 else "")
            + " `tnca-service status` gives a running slow request's `ends_within_minutes`, the time within which "
            "it ends at the latest; asking again does not make it end sooner, and the time in between is yours "
            "to work in.")
    lines += ["- `tnca-service budget`, `tnca-service status [ID]`, `tnca-service result ID`, "
              "`tnca-service wait ID`: what is left, and the results."]
    return "\n".join(lines)


def _board_lines(score: dict, slow: dict, board: dict, flows: int, shared: str, islands: bool,
                 at_once: int) -> list:
    """The services text when the board is the score (2026-11-10): `score` on the card, `flow` for a changed
    machine's image."""
    named = [f"{', '.join('`' + x.rstrip('/') + '`' for x in spec['prefixes'])} points together form the family "
             f"`{name}`, of weight {float(spec.get('weight', 1)):g}" for name, spec in board["families"].items()]
    faster = float(slow.get("board_faster_than", 1.0))
    every = "every island's" if islands else "every"
    return [
        f"- `tnca-service submit score`: the hidden board validation set on the real card -- your machine on the "
        f"FPGA at its target clock, {board['n']} points of small, medium and large size (the large ones too slow "
        f"to simulate) -- against the base on the card. The score: each family of points (the operator a point "
        f"exercises"
        + ("; " + "; ".join(named) if named else "")
        + ", every other family of weight 1) gets the geometric mean of its points' speedups over the base, a point "
        "that does not pass counting as a speedup of 0.5; the score is the weighted geometric mean over the "
        "families, so a family counts once however many points it has. You get the score, each family's value, "
        "how many points passed, failures counted by kind, and every point of the set in a fixed order, numbered "
        "and never named: its family, the resource that binds it and its η, its distance from the whole chip's "
        "roofline (η = L / T: L the least cycles the base machine's cores and memory outlets could take for its "
        "work, T the cycles measured; none for a point that did not pass), beside the base's η. Never a point's "
        "shape or its own error text."
        + (f" A point counts as faster only when it is more than {(faster - 1) * 100:g}% faster than the base: "
           "the card's count of one program differs by up to about half a percent from run to run."
           if faster > 1.0 else "")
        + " Each point may run for 20 times the base's cycles on it; past that it fails as a timeout.",
        f"- A snapshot that kept the base's machine (compiler and software only) goes straight to the card. One "
        f"that changed the machine (RTL, the profile's parameters, the device description) runs on the card once "
        f"this run has built its image (`flow`, below); before that, `score` refuses it, uncharged. The card is "
        f"one, shared by {every} requests: one session at a time, first come first served, so a request may "
        f"wait for the card, and the wait is part of your time; a session takes about half an hour. You have "
        f"{int(score.get('requests', 0))} scoring requests, one running at a time. A snapshot exactly like one "
        f"this run already scored is not measured or charged again" + (" and is not a new commit" if islands else "")
        + ": you get that request's result. A session that fails on the infrastructure, or after which the card "
        "fails its own known-answer check, is not charged.",
        f"- `tnca-service submit flow`: the full implementation of a changed machine on the FPGA (place and route "
        f"at the target clock, one Vivado flow, which takes hours; up to {float(slow.get('flow_minutes', 480)):g} "
        f"minutes); the image it builds is then this run's, for any scoring request of exactly that hardware. "
        f"You have {flows} flows{shared}"
        + (f"; up to {at_once} run at a time across the run, each on a build machine of its own (a flow may wait "
           f"for one to be free)" if at_once > 1 else "; one runs at a time across the run")
        + ". Before it is charged, a flow needs an `fpga-quick` result of this run on exactly that hardware that "
        f"met timing, and a short functional gate ({board['gate']} tiny points in RTL simulation) runs beside it: "
        "a gate that does not pass stops the flow, which is then not charged. A flow that fails on your design "
        "(timing or place-and-route does not converge, it does not fit, DRC) counts; one that fails on the "
        "infrastructure does not, and is retried for you. You get its routed timing; for a flow that failed, why: "
        "the failing stage, slack, the worst paths and whether they cross an SLR, utilisation per SLR, the DRC "
        "rules that failed. A flow still running when you stop is stopped.",
        "- There is no simulated scoring service in this task: between card sessions, what you measure is your own "
        "-- the RTL simulation and the cycle models in your working directory.",
    ]


__all__ = ["POINT_SETS", "Task", "TaskError", "WITHOUT_SKILL", "board_points", "load_task", "render"]
