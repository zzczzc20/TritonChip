"""Islands: AgenticEvolve's persistent agents with an archive, MAP-Elites per island and migration (report
section 8, decided 2026-11-06/07; this is the first, simple version).

    task.json  "islands": {"count": 3, "topology": "ring", "migration_interval": 3, "migration_rate": 0.3,
                           "features": ["layer"], "max_idle_turns": 3, "max_restarts": 2,
                           "target": {"speedup_geomean": 1.5}}

Each island is one agent in a sandbox of its own with a workspace of its own, for the task's budgets read as
the island's (wall clock, scoring requests = its commits, quick-tier minutes); the slow tier's counts, its
images and its one-at-a-time stay the run's (section 8.9, B7/B8), and so does a cap on evaluations at once
(the site's `resources.evaluations`). One island and N islands take the same path (D1).

- A commit is a scoring request with a one-line message: scored as the score service scores it (the
  validation set: it doubles as the selection set, D3), then kept in the archive with the layer its change
  touched (computed here, never from the agent's words) and its cell in the island's grid. Its reply adds the
  island's best, the island's elites, and every `migration_interval` commits the top `migration_rate` of the
  connected islands' elites the island has not been given yet.
- What an island sees -- migration, the archive view, `read-commit` -- is one function of the topology (ring:
  the previous island, as AgenticEvolve; all; none; or an explicit map), and `read-commit` refuses any other id.
- The controller keeps the agent going (D2/D7): when the CLI stops, the same session is resumed with
  "continue"; a turn that called no tool is idle and `max_idle_turns` idle turns in a row end the island; a
  crash (the CLI failing without being throttled) is restarted up to `max_restarts` times, a new session when
  the old one cannot be resumed; a throttled turn waits and is not counted. An island also ends when its time
  or its commits run out, or when any commit reaches the target. The agent saying it is done is not an end.
- After the run, the archive's top 3 by the validation aggregate (fast_1, then the geometric-mean speedup)
  get the final evaluation and, when the task asks for it, the card, in that order until one is certified
  (D8, section 8.9 B1). The final set is never run while an agent does.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Optional

from . import evaluator, runner
from .task import Task, render
from .workspace import export, git

CELLS = ("compiler-only", "hardware-only", "cross-layer")
SOFTWARE = ("compiler/", "python/tnca/")
_RTL_SPACE = re.compile(r"\s+")


# -- what a commit changed: the layer descriptor (section 8.6/8.7) -------------------------------------------

def _rtl_code(text: str) -> str:
    return _RTL_SPACE.sub("", evaluator._RTL_COMMENT.sub(" ", text))


def layer_labels(repo: Path, task: Task, candidate: Optional[Path]) -> list:
    """The layers a snapshot really changed, from the gate's output and hardware_changed() -- never from the
    agent's own account. RTL counts only when it differs with comments and whitespace removed."""
    if candidate is None:
        return []
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    platform = set(((task.evaluation.get("platform") or {}).get("paths")) or [])
    labels = set()
    for c in report["taken"]:
        path = c["path"]
        if path.startswith(SOFTWARE) and path not in platform:
            labels.add("compiler")
        elif path.startswith("isa/"):
            labels.add("isa")
        elif path.startswith("rtl/"):
            try:
                before = git("-C", str(repo), "show", f"{task.base_commit}:{path}").decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 -- a file the base does not have
                before = ""
            after = "" if c["status"] == "deleted" else (candidate / "files" / path).read_text(
                encoding="utf-8", errors="replace")
            if _rtl_code(before) != _rtl_code(after):
                labels.add("rtl")
    hw = evaluator.hardware_changed(repo, task, candidate)
    if hw.get("profile"):
        labels.add("profile")
    if hw.get("device"):
        labels.add("device")
    return sorted(labels)


def layer_cell(labels: list) -> Optional[str]:
    """The grid's one axis in the first version: compiler only, hardware only, or both (section 8.7)."""
    if not labels:
        return None
    if set(labels) <= {"compiler"}:
        return "compiler-only"
    if set(labels) <= {"rtl", "profile", "device"}:
        return "hardware-only"
    return "cross-layer"


def fitness(agg: Optional[dict]) -> tuple:
    """The order commits are compared in (O12): fast_1 first, then the geometric-mean speedup."""
    agg = agg or {}
    return (agg.get("fast_1") or 0.0, agg.get("speedup_geomean") or 0.0)


# -- who sees whom (section 8.6) --------------------------------------------------------------------------

def visible(cfg: dict, island: int) -> list:
    """The other islands `island` sees: migration comes from them, the archive view lists them, read-commit
    accepts their ids. Ring is AgenticEvolve's: the previous island."""
    n, topo = cfg["count"], cfg.get("topology", "ring")
    if isinstance(topo, dict):
        return sorted(set(topo.get(str(island), [])))
    if topo == "all":
        return [j for j in range(n) if j != island]
    if topo == "ring" and n > 1:
        return [(island - 1) % n]
    return []


# -- the archive ------------------------------------------------------------------------------------------

class Archive:
    """Every commit of every island; each island's grid of elites; the best-so-far trajectory. One lock: an
    entry, its cell and the views built from them change together."""

    def __init__(self, repo: Path, task: Task):
        self.repo, self.task, self.cfg = repo, task, task.islands
        n = self.cfg["count"]
        self.entries: list = []
        self.by_id: dict = {}
        self.elites: dict = {i: {} for i in range(n)}
        self.given: dict = {i: set() for i in range(n)}
        self.commits: dict = {i: 0 for i in range(n)}
        self.trajectory: list = []
        self.lock = threading.Lock()
        self.started = time.time()
        self.target_hit = threading.Event()

    def _summary(self, e: dict) -> dict:
        return {"id": e["id"], "island": e["island"], "labels": e["labels"], "cell": e["cell"],
                "validation": e["validation"], "message": e["message"], "message_unverified": True}

    def add(self, island: int, job: dict, candidate: Optional[Path], result: dict) -> dict:
        """A finished commit, entered: its labels and cell, the grid, the trajectory, the target."""
        labels = layer_labels(self.repo, self.task, candidate)
        agg = result.get("aggregate") or {}
        with self.lock:
            self.commits[island] += 1
            seq = self.commits[island]
            e = {"id": f"{island}-{seq}", "island": island, "seq": seq, "job": job["id"], "labels": labels,
                 "cell": layer_cell(labels), "validation": agg, "message": (job.get("message") or "")[:200],
                 "listed": bool(agg.get("passed")) and not result.get("device_build_errors")
                 and not result.get("stage_errors"),
                 "candidate": str(candidate) if candidate else None, "digest": job.get("digest"),
                 "hardware": job.get("hardware"), "at": time.time()}
            self.entries.append(e)
            self.by_id[e["id"]] = e
            if e["listed"] and e["cell"]:
                cur = self.by_id.get(self.elites[island].get(e["cell"]))
                if cur is None or fitness(e["validation"]) > fitness(cur["validation"]):
                    self.elites[island][e["cell"]] = e["id"]
            best_island = self._best([x for x in self.entries if x["island"] == island])
            best_all = self._best(self.entries)
            self.trajectory.append({"at": round(e["at"] - self.started, 1), "island": island, "seq": seq,
                                    "id": e["id"], "fitness": list(fitness(agg)),
                                    "island_best": list(fitness(best_island["validation"])) if best_island else None,
                                    "run_best": list(fitness(best_all["validation"])) if best_all else None})
            target = self.cfg.get("target")
            if target and e["listed"] and all((agg.get(k) or 0) >= v for k, v in target.items()):
                self.target_hit.set()
        return e

    def _best(self, entries: list) -> Optional[dict]:
        listed = [x for x in entries if x["listed"]]
        return max(listed, key=lambda x: fitness(x["validation"])) if listed else None

    def reply(self, island: int, e: dict) -> dict:
        """What a commit's result adds: the commit in the archive, the island's best and grid, migration."""
        with self.lock:
            best = self._best([x for x in self.entries if x["island"] == island])
            grid = {cell: self._summary(self.by_id[cid]) for cell, cid in self.elites[island].items()}
            out = {"commit": {"id": e["id"], "labels": e["labels"], "cell": e["cell"], "listed": e["listed"]},
                   "island_best": self._summary(best) if best else None, "grid": grid}
            if e["seq"] % self.cfg["migration_interval"] == 0:
                out["migration"] = self._migration(island)
        return out

    def _migration(self, island: int) -> list:
        """The top `migration_rate` of the connected islands' current elites, each given to an island once."""
        pool = [self.by_id[cid] for j in visible(self.cfg, island) for cid in self.elites[j].values()]
        pool.sort(key=lambda x: fitness(x["validation"]), reverse=True)
        take = pool[:max(1, math.ceil(len(pool) * self.cfg["migration_rate"]))] if pool else []
        fresh = [x for x in take if x["id"] not in self.given[island]]
        self.given[island].update(x["id"] for x in fresh)
        return [self._summary(x) for x in fresh]

    def view(self, island: int) -> dict:
        """The listed commits of this island and the islands it sees; a failed commit only to its own island."""
        seen = {island, *visible(self.cfg, island)}
        with self.lock:
            rows = [self._summary(e) for e in self.entries if e["island"] in seen and (e["listed"] or e["island"] == island)]
            elites = {j: {cell: cid for cell, cid in self.elites[j].items()} for j in sorted(seen)}
        return {"commits": rows, "elites": elites, "you": island, "visible_islands": sorted(seen - {island})}

    def read(self, island: int, cid: str) -> dict:
        """One commit as a patch against the base commit -- its design overlay only, never an island's notes,
        scripts or cases -- if `island` may see it."""
        with self.lock:
            e = self.by_id.get(str(cid))
        if e is None or e["island"] not in {island, *visible(self.cfg, island)}:
            return {"error": f"no commit {cid!r} you can see"}
        if not e["listed"] and e["island"] != island:
            return {"error": f"no commit {cid!r} you can see"}
        if not e["candidate"]:
            return {"id": e["id"], "patch": "", "note": "the commit changed nothing"}
        return {"id": e["id"], "labels": e["labels"], "cell": e["cell"],
                "patch": overlay_patch(self.repo, self.task.base_commit, Path(e["candidate"]))}

    def ranked(self) -> list:
        with self.lock:
            return sorted([e for e in self.entries if e["listed"]], key=lambda x: fitness(x["validation"]),
                          reverse=True)

    def record(self) -> dict:
        with self.lock:
            return {"entries": [{k: v for k, v in e.items() if k != "candidate"} for e in self.entries],
                    "elites": {str(i): dict(g) for i, g in self.elites.items()},
                    "trajectory": list(self.trajectory), "target_hit": self.target_hit.is_set()}


def overlay_patch(repo: Path, base: str, candidate: Path) -> str:
    """A unified diff of the snapshot's taken files against the base commit, for `git apply`."""
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    out = []
    for c in sorted(report["taken"], key=lambda c: c["path"]):
        path = c["path"]
        try:
            before = "" if c["status"] == "added" else git("-C", str(repo), "show", f"{base}:{path}").decode("utf-8")
            after = "" if c["status"] == "deleted" else (candidate / "files" / path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, Exception):  # noqa: BLE001 -- a binary file
            out.append(f"# {path}: a binary or unreadable file changed; not shown\n")
            continue
        a = "/dev/null" if c["status"] == "added" else f"a/{path}"
        b = "/dev/null" if c["status"] == "deleted" else f"b/{path}"
        out.append(f"diff --git a/{path} b/{path}\n")
        out.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), a, b))
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n\\ No newline at end of file\n"
    return "".join(out)


# -- the controller ---------------------------------------------------------------------------------------

class Turns:
    """One island's turn policy for runner.run_agent: continue, restart, wait, or stop -- and why it stopped."""

    def __init__(self, island: int, task: Task, broker, archive: Archive, prompt: str):
        self.island, self.task, self.broker, self.archive, self.prompt = island, task, broker, archive, prompt
        self.cfg = task.islands
        self.idle = self.restarts = self.throttle_waits = 0
        self.stop_reason: Optional[str] = None
        self.last = "first"     # how the turn that just ended was started: first, resume, new-session

    def _commits_left(self) -> int:
        with self.broker.lock:
            running = any(j["service"] == "score" and j["state"] in ("queued", "running")
                          for j in self.broker.jobs.values())
            return self.broker.score_left + (1 if running else 0)

    def __call__(self, record: dict, left: float):
        if self.archive.target_hit.is_set():
            self.stop_reason = "target"
            return None
        if record.get("stop_reason") == "timeout" or left < 60:
            self.stop_reason = "wall-clock"
            return None
        if record.get("throttled"):
            # Refused by the provider, not stopped by the agent: wait, and do not count it (P9).
            self.throttle_waits += 1
            time.sleep(min(300.0, max(10.0, left / 10)))
            return self._next(fresh=False)
        if self._commits_left() <= 0:
            self.stop_reason = "commits-exhausted"
            return None
        crashed = record.get("exit") not in (0, None) and not record.get("tool_calls")
        if crashed:
            self.restarts += 1
            if self.restarts > self.cfg["max_restarts"]:
                self.stop_reason = "crashed"
                return None
            # A resume that failed is not tried again: a new session, with the task again.
            return self._next(fresh=self.last == "resume" or not record.get("session"))
        if record.get("tool_calls"):
            self.idle = 0
        else:
            self.idle += 1
            if self.idle >= self.cfg["max_idle_turns"]:
                self.stop_reason = "idle"
                return None
        return self._next(fresh=not record.get("session"))

    def _next(self, fresh: bool):
        self.last = "new-session" if fresh else "resume"
        b = self.broker.budget()
        status = (f"{b.get('score_requests_left')} commits and the rest of your time are left. "
                  f"`tnca-service archive` lists the commits you can see.")
        if fresh:
            return (self.prompt + "\n\nYou are resuming earlier work on this island: your working directory holds "
                    "it. " + status, True)
        return "Continue improving. " + status


def run(site, task: Task, account, mode: str, run_dir: Path, repo: Path, names: list, broker_cls, shared) -> dict:
    """Every island, at once; their records, the archive and the trajectory."""
    cfg = task.islands
    archive = Archive(repo, task)
    out = {"config": cfg, "islands": []}
    threads, results = [], [None] * cfg["count"]

    def one(i: int) -> None:
        idir = run_dir / f"island-{i}"
        idir.mkdir(parents=True, exist_ok=True)
        workspace = idir / "workspace"
        exported = export(repo, task.base_commit, task.hide, workspace)
        broker = broker_cls(site, repo, task, workspace, idir, island=i, archive=archive, shared=shared)
        prompt = render(task, mode, skills=names, island=i)
        (idir / "prompt.md").write_text(prompt, encoding="utf-8")
        turns = Turns(i, task, broker, archive, prompt)
        try:
            agent = runner.run_agent(site, account, prompt, mode, idir, workspace, timeout=float(task.wall_seconds),
                                     services=broker, skills=names, turns=turns, name=f"{run_dir.name}-i{i}")
        except Exception as exc:  # noqa: BLE001 -- one island failing ends that island, not the run (D7)
            agent = {"error": f"{type(exc).__name__}: {exc}"}
            turns.stop_reason = turns.stop_reason or "crashed"
        results[i] = {"island": i, "workspace": exported, "agent_result": agent, "broker": broker,
                      "stop_reason": turns.stop_reason or agent.get("stop_reason"), "idle": turns.idle,
                      "restarts": turns.restarts, "throttle_waits": turns.throttle_waits,
                      "commits": archive.commits[i]}

    for i in range(cfg["count"]):
        t = threading.Thread(target=one, args=(i,), name=f"island-{i}")
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    out["islands"] = results
    out["archive"] = archive
    return out


__all__ = ["Archive", "CELLS", "Turns", "fitness", "layer_cell", "layer_labels", "overlay_patch", "run", "visible"]
