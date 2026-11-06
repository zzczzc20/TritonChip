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
        self.halt = threading.Event()      # the target, or the operator: every island stops

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
                self.halt.set()
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
    """The snapshot's taken files against the base commit, as git itself writes the diff -- new and deleted
    files, modes, a missing final newline, binary files -- so `git apply` takes it back. Built in a scratch
    index of the harness's own repository (the base's tree, the candidate's files on top); the workspace and
    the repository's own index are not touched."""
    import os
    import subprocess
    import tempfile
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    paths = sorted(c["path"] for c in report["taken"])
    if not paths:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}

        def g(*args, data=None):
            return subprocess.run(["git", "-C", str(repo), *args], input=data, env=env, capture_output=True,
                                  check=True).stdout

        g("read-tree", base)
        for c in report["taken"]:
            if c["status"] == "deleted":
                g("update-index", "--force-remove", "--", c["path"])
                continue
            f = candidate / "files" / c["path"]
            oid = g("hash-object", "-w", "--stdin", data=f.read_bytes()).decode().strip()
            mode = "100755" if os.access(f, os.X_OK) else "100644"
            g("update-index", "--add", "--cacheinfo", f"{mode},{oid},{c['path']}")
        return g("diff-index", "--cached", "-p", "--binary", "--full-index", base, "--", *paths).decode(
            "utf-8", errors="replace")


# -- the controller ---------------------------------------------------------------------------------------

class Turns:
    """A turn policy for runner.run_agent -- an island's, or a plain run's with the task's `resume` block:
    continue, restart, wait, or stop, and why it stopped.

    D2: the same session is resumed; a CLI whose resume fails before it ever worked falls back to new sessions
    for the rest of the run (`mode`), which is recorded and is not a crash. A crash is a session that had
    started failing without being throttled; `max_restarts` of them end the run (or the island). A turn with
    no tool call is idle; `max_idle_turns` of them in a row end it. The agent saying it is done is not an end.
    `archive` is None for a plain run: no commits to count, no target."""

    def __init__(self, island: Optional[int], task: Task, broker, archive: Optional[Archive], prompt: str,
                 cfg: Optional[dict] = None):
        self.island, self.task, self.broker, self.archive, self.prompt = island, task, broker, archive, prompt
        self.cfg = cfg or task.islands
        self.idle = self.restarts = self.throttle_waits = 0
        self.throttle_seconds = 0.0
        self.stop_reason: Optional[str] = None
        self.last = "first"         # how the turn that just ended was started: first, resume, new-session
        self.mode = "resume"        # resume, or new-session once a resume has failed before ever working
        self.resume_worked = False

    def _commits_left(self) -> int:
        with self.broker.lock:
            running = any(j["service"] == "score" and j["state"] in ("queued", "running")
                          for j in self.broker.jobs.values())
            return self.broker.score_left + (1 if running else 0)

    def __call__(self, record: dict, left: float):
        if self.archive is not None and self.archive.halt.is_set():
            self.stop_reason = "target" if self.archive.target_hit.is_set() else "interrupted"
            return None
        if record.get("stop_reason") == "timeout" or left < 60:
            self.stop_reason = "wall-clock"
            return None
        if record.get("tool_calls"):
            self.idle = 0
        if record.get("throttled"):
            # Refused by the provider, not stopped by the agent: wait, and do not count it (P9).
            self.throttle_waits += 1
            wait = min(300.0, max(10.0, left / 10))
            self.throttle_seconds += wait
            if self.archive is not None:
                self.archive.halt.wait(wait)
            else:
                time.sleep(wait)
            return self._next(not record.get("session") or self.mode == "new-session", left - wait)
        if self.archive is not None and self._commits_left() <= 0:
            self.stop_reason = "commits-exhausted"
            return None
        crashed = record.get("exit") not in (0, None) and not record.get("tool_calls")
        if crashed and self.last == "resume" and not self.resume_worked:
            self.mode = "new-session"           # this CLI's resume does not work here: never again, no charge
            return self._next(True, left)
        if crashed:
            self.restarts += 1
            if self.restarts > self.cfg["max_restarts"]:
                self.stop_reason = "crashed"
                return None
            # D7: the same session when it can be resumed, else a new one with the task again.
            return self._next(self.mode == "new-session" or not record.get("session"), left)
        if self.last == "resume":
            self.resume_worked = True
        if not record.get("tool_calls"):
            self.idle += 1
            if self.idle >= self.cfg["max_idle_turns"]:
                self.stop_reason = "idle"
                return None
        return self._next(not record.get("session") or self.mode == "new-session", left)

    def _left(self, left: float) -> str:
        """D2's neutral status line: the minutes left and what is left of each service the task has."""
        parts = [f"about {max(0, int(left // 60))} minutes of your time"]
        if self.broker is not None:
            b, svc = self.broker.budget(), self.task.services or {}
            if "score" in svc:
                parts.append(f"{b.get('score_requests_left')} "
                             + ("commits" if self.archive is not None else "score requests"))
            if "fpga_quick" in svc:
                parts.append(f"{b.get('fpga_minutes_left')} minutes of fpga-quick")
            if "fpga_slow" in svc:
                parts.append(f"{b.get('fpga_slow_flows_left')} fpga-slow flows and "
                             f"{b.get('fpga_slow_board_runs_left')} board runs")
        return "Left: " + ", ".join(parts) + "."

    def _next(self, fresh: bool, left: float):
        self.last = "new-session" if fresh else "resume"
        if self.archive is not None:
            status = self._left(left) + " `tnca-service archive` lists the commits you can see."
            where = "your working directory holds it, and `tnca-service archive` lists your commits"
        else:
            status = (self._left(left) + " Your working directory as it is when the time runs out "
                      "is your submission.")
            where = "your working directory holds it"
        if fresh:
            return (self.prompt + f"\n\nYou are resuming earlier work: {where}. Keep a short NOTES.md at the top of "
                    "your working directory on what you tried and what you learned, for whoever continues. " + status,
                    True)
        return "Continue improving. " + status


def _stop_agents(run_dir: Path, count: int, cli: str) -> None:
    """Every island's CLI asked to end its turn (target reached, or the operator stopped the run)."""
    from . import provider
    for i in range(count):
        provider.docker("exec", f"tnca-agent-{run_dir.name}-i{i}", "pkill", "-TERM", "-f", runner.BINARY[cli],
                        check=False, timeout=30)


def run(site, task: Task, account, mode: str, run_dir: Path, repo: Path, names: list, broker_cls, shared) -> dict:
    """Every island, at once; their records, the archive and the trajectory."""
    import hashlib
    cfg = task.islands
    archive = Archive(repo, task)
    out = {"config": cfg, "islands": []}
    threads, results = [], [None] * cfg["count"]

    def one(i: int) -> None:
        idir = run_dir / f"island-{i}"
        broker, turns, agent, exported, prompt = None, None, {}, None, ""
        try:
            idir.mkdir(parents=True, exist_ok=True)
            workspace = idir / "workspace"
            exported = export(repo, task.base_commit, task.hide, workspace)
            # The run's id in every container name: two runs on one host never share one (the review's H2).
            broker = broker_cls(site, repo, task, workspace, idir, island=i, archive=archive, shared=shared,
                                run_id=f"{run_dir.name}-i{i}")
            prompt = render(task, mode, skills=names, island=i)
            (idir / "prompt.md").write_text(prompt, encoding="utf-8")
            turns = Turns(i, task, broker, archive, prompt)
            agent = runner.run_agent(site, account, prompt, mode, idir, workspace, timeout=float(task.wall_seconds),
                                     services=broker, skills=names, turns=turns, name=f"{run_dir.name}-i{i}")
        except Exception as exc:  # noqa: BLE001 -- one island failing ends that island, not the run (D7)
            agent = {**agent, "error": f"{type(exc).__name__}: {exc}"}
        results[i] = {"island": i, "workspace": exported, "agent_result": agent, "broker": broker,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
                      "stop_reason": (turns.stop_reason if turns else None) or agent.get("stop_reason")
                      or ("crashed" if agent.get("error") else None),
                      "idle": turns.idle if turns else None, "restarts": turns.restarts if turns else None,
                      "resume_mode": turns.mode if turns else None,
                      "throttle_waits": turns.throttle_waits if turns else None,
                      "throttle_seconds": round(turns.throttle_seconds, 1) if turns else None,
                      "commits": archive.commits[i]}

    for i in range(cfg["count"]):
        t = threading.Thread(target=one, args=(i,), name=f"island-{i}")
        threads.append(t)
        t.start()

    def watch() -> None:
        # The target ends the whole run (D7): every island's current turn is stopped, not waited for.
        while any(t.is_alive() for t in threads):
            if archive.halt.wait(15):
                _stop_agents(run_dir, cfg["count"], account.cli)
                return

    threading.Thread(target=watch, name="islands-watch", daemon=True).start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(5)
    except KeyboardInterrupt:
        # The operator: every island ends its turn and cleans up its sandbox in its own `finally`.
        archive.halt.set()
        _stop_agents(run_dir, cfg["count"], account.cli)
        for t in threads:
            t.join(120)
        raise
    out["islands"] = results
    out["archive"] = archive
    return out


__all__ = ["Archive", "CELLS", "Turns", "fitness", "layer_cell", "layer_labels", "overlay_patch", "run", "visible"]
