"""The evaluator, at CHIA's level: rebuild from source with no AI involved, measure with its own copies.

It builds a tree from the FULL base commit (nothing hidden) with the
candidate's files on top -- only files under the task's design paths, which
the gate already decided -- so the testbench, ArchBench, the tools and the
flow are the base's, never the agent's. It then runs ArchBench on the task's
validation and final points, in a container with no network and no
credential, and compares every point with the base's own cycles.

Each point is judged on its own. A point that is wrong, falls back to the
host, or produced no record is a failed point, and the other points still
count: a set reports how many points passed, the geometric-mean speedup over
the points that passed, and fast_1, the share of ALL points that passed and
beat the base. fast_1 is there because the passed-only mean alone would reward
breaking a point the candidate had made slower.

Known gap, to be closed by the boundary observer (README section 8, P2): the
candidate's Python runs in the same process that reports the cycles, so a
candidate that tampers with that process is not caught here. The diff audit
and the observer are what catch that.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Optional

from . import provider
from .provider import AgentContainer, Mount
from .siteconfig import Site
from .task import Task
from .workspace import apply_candidate, export, git

SCRIPT = """set -e
cd /work
make -s python/tnca/data/.staged > /out/stage.log 2>&1
/opt/torchenv/bin/tnca-sim prepare --profile {profile} > /out/prepare.log 2>&1
/opt/torchenv/bin/python -m archbench.run --platform tnca --profile {profile} --jobs {jobs} \\
    --point {points} --out /out/archbench.jsonl > /out/archbench.log 2>&1 || true
"""


def _measure(site: Site, tree: Path, task: Task, out: Path, name: str) -> dict:
    ev = task.evaluation
    points = list(ev["validation"]) + list(ev["final"])
    script = SCRIPT.format(profile=ev["profile"], jobs=int(ev["jobs"]), points=" ".join(points))
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluate.sh").write_text(script, encoding="utf-8")
    c = AgentContainer(name=name, image=site.images["toolchain"], cpus=site.cpus, memory=site.memory,
                       pids=site.pids, mounts=[Mount(tree, "/work", read_only=False), Mount(out, "/out", read_only=False)],
                       labels={"tritonchip-agent-harness": "evaluator"}, workdir="/work")
    proc = provider.docker(*c.run_args(["sh", "/out/evaluate.sh"], detach=False, remove=True),
                           check=False, timeout=float(ev["timeout_seconds"]))
    records = {}
    path = out / "archbench.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[r["point_id"]] = {"status": r.get("status"), "cycles": (r.get("time") or {}).get("device"),
                                      "fallbacks": r.get("fallbacks"),
                                      "max_rel_err": (r.get("correctness") or {}).get("max_rel_err")}
    return {"exit": proc.returncode, "points": records,
            "missing": [p for p in points if p not in records]}


def baseline(site: Site, repo: Path, task: Task) -> dict:
    """The base's own cycles, measured once per (base commit, profile, points) and kept."""
    ev = task.evaluation
    points = hashlib.sha256(json.dumps([ev["validation"], ev["final"]]).encode()).hexdigest()[:8]
    key = f"{task.base_commit[:12]}-{ev['profile']}-{points}"
    cache = site.state_dir / "baselines" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    work = site.state_dir / "baselines" / key
    shutil.rmtree(work, ignore_errors=True)
    export(repo, task.base_commit, (), work / "tree")
    result = _measure(site, work / "tree", task, work / "out", f"tnca-eval-base-{key}")
    shutil.rmtree(work / "tree", ignore_errors=True)
    if not result["missing"] and all(p["status"] == "pass" for p in result["points"].values()):
        cache.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _rtl_parameters(profiles_json: str, name: str) -> dict:
    p = json.loads(profiles_json)["profiles"].get(name) or {}
    params = dict(p.get("elaboration") or {})
    params.update((p.get("implementation") or {}).get("rtl_parameters") or {})
    return params


def machine_change(repo: Path, task: Task, tree: Path) -> dict:
    """Whether the candidate changed the machine it is measured on, and how.

    Allowed -- the tasks are cross-layer -- and recorded, because a faster
    machine and a faster program on the same machine are different results;
    the hardware's cost is not charged yet.
    """
    name = task.evaluation["profile"]
    before = _rtl_parameters(git("-C", str(repo), "show", f"{task.base_commit}:spec/r1_profiles.json").decode(), name)
    after = _rtl_parameters((tree / "spec/r1_profiles.json").read_text(encoding="utf-8"), name)
    changed = {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
    return {"profile": name, "changed": changed}


def _geomean(xs: list) -> Optional[float]:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs and all(x > 0 for x in xs) else None


def score(base: dict, got: dict, names) -> dict:
    """One set of points, each judged on its own against the base's cycles."""
    rows = []
    for p in names:
        b, c = base.get(p, {}), got.get(p)
        if c is None:
            failure = "missing"
        elif c.get("status") != "pass":
            failure = c.get("status") or "no-status"
        elif c.get("fallbacks"):
            failure = "fallback"          # part of it ran on the host: the chain was skipped
        elif not c.get("cycles") or not b.get("cycles"):
            failure = "no-cycles"
        else:
            failure = None
        speedup = b["cycles"] / c["cycles"] if failure is None else None
        rows.append({"point": p, "status": (c or {}).get("status"), "failure": failure,
                     "base_cycles": b.get("cycles"), "cycles": (c or {}).get("cycles"), "speedup": speedup})
    passed = [r["speedup"] for r in rows if r["failure"] is None]
    n = len(rows)
    return {"points": rows, "n": n, "passed": len(passed),
            "speedup_geomean": _geomean(passed),
            "fast_1": (sum(1 for x in passed if x > 1.0) / n) if n else None}


def evaluate(site: Site, repo: Path, task: Task, candidate: Optional[Path], run_dir: Path) -> dict:
    base = baseline(site, repo, task)
    tree = run_dir / "eval-tree"
    shutil.rmtree(tree, ignore_errors=True)
    if candidate is None:
        export(repo, task.base_commit, (), tree)
    else:
        apply_candidate(repo, task.base_commit, candidate, tree)
    machine = machine_change(repo, task, tree)
    got = _measure(site, tree, task, run_dir / "eval", f"tnca-eval-{run_dir.name}")
    shutil.rmtree(tree, ignore_errors=True)
    validation = score(base["points"], got["points"], task.evaluation["validation"])
    final = score(base["points"], got["points"], task.evaluation["final"])
    return {
        # The run means something only if the base itself measured cleanly.
        "valid": bool(base["points"]) and not base["missing"]
        and all(p.get("status") == "pass" for p in base["points"].values()),
        "validation": validation, "final": final,
        "all_passed": validation["passed"] == validation["n"] and final["passed"] == final["n"],
        "evaluator_exit": got["exit"],
        "machine": machine,
    }


__all__ = ["baseline", "evaluate", "score"]
