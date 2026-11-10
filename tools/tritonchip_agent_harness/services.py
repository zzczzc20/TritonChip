"""In-run services: the agent asks for a measurement, within its budget, and the harness makes it.

The broker runs in the harness process, beside the agent's run. It listens on
a Unix socket in a directory that the agent's container mounts read-only;
`tnca-service` (service_cli.py, mounted as one read-only file) is the agent's
side. One JSON line in, one JSON line out.

    submit score         the hidden validation set, in simulation, measured the
                         way the final evaluator measures (P2.1 included); the
                         agent sees an aggregate only (README section 6.4)
    submit fpga-quick    out-of-context synthesis and timing of the device
    submit fpga-slow     the full link and the card (fpga_slow.py, report section 8.9):
                         a hardware request is one Vivado flow, a software one a card
                         run only; each kind has its own count for the run
    status, result, budget

A request is a snapshot: the same gate as the final evaluation, so only
changes under the design paths count, and a rejected snapshot or RTL that
fails the check is refused before anything is charged. Budgets: a number of
scoring requests, charged when one is accepted; quick-tier minutes, charged as
the worker's actual time, one quick request at a time and none accepted that
could run past what is left; slow-tier flows and card runs, reserved when a
request is accepted and given back when it was refused by its gate or failed
on the infrastructure. One slow request at a time (one Vivado flow, one card);
the quick tier stays open while a slow request runs, since iterating on
hardware while a flow runs is what the slow tier's asynchrony is for. A slow
request still running when the agent stops runs to its end, unseen by the
agent, for the certification after the run. Everything asked and answered,
and the full results the agent does not see, go into the run record.

A run simulates each snapshot's validation set once (decided 2026-11-09, D1-D6): one memo for the run,
shared by its islands, keyed by the whole snapshot, the base's measurement and the evaluator's code. A
scoring request and an fpga-slow gate of the same snapshot take one simulation: a finished result is
reused (passed or not), a running one is joined, and one that was cancelled or failed on the
infrastructure is not -- the asker then simulates itself. At most `resources.evaluations` simulations
run at once, scores and gates together; a join takes no slot. A scoring request of a snapshot an
earlier one of the run already scored is not charged and is no new commit: its reply is that one's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import shutil
import socketserver
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import board, evaluator, fpga_quick, fpga_slow, provider, roofline
from .siteconfig import Site
from .task import Task, board_points
from .workspace import apply_candidate, export, gate

SERVICES = ("score", "fpga-quick", "fpga-slow", "flow")
# What of an FPGA result the agent may see: all of it is its own design's.
FPGA_VISIBLE = ("status", "profile", "period_ns", "part", "wns_ns", "tns_ns", "failing_endpoints",
                "total_endpoints", "fmax_est_mhz", "utilization", "worst_paths", "errors", "seconds",
                "not_applied")
FPGA_NOTE = ("post-synthesis estimate: delays leave out most of the wire, so the frequency is optimistic; "
             "the routed design is the slow tier's to judge")


# The evaluator's own code is part of what a simulation's result depends on (D1).
EVALUATOR_VERSION = hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest()[:16]


def sim_key(task: Task, candidate: Optional[Path]) -> str:
    """What a simulation of the validation set depends on (D1): the whole snapshot (every file the gate took,
    software and hardware), the base's measurement (evaluator.baseline_key: base commit, profile, the point
    lists, the platform, the device) and the evaluator's code."""
    return hashlib.sha256(json.dumps([fpga_slow.snapshot_digest(candidate), evaluator.baseline_key(task),
                                      EVALUATOR_VERSION]).encode()).hexdigest()


# Failures that say more about the host than about the snapshot (the review of 2026-11-09): a point that never
# reported, timed out or was not measured, and a stage that stopped before any point ran. A result with any of
# them is not kept for later askers; a point that ran and was wrong (numeric-error, a device fault, a fallback)
# is the snapshot's own, and is.
INFRA_LIKE = ("missing", "no-status", "no-observation", "not-measured", "timeout", "other")


def reusable(result: Optional[dict]) -> bool:
    """Whether a simulation's result is the snapshot's to hand to the next asker (D2)."""
    if not result or result.get("stage_errors"):
        return False
    return not any(k in INFRA_LIKE for k in (result.get("failures") or {}))


class Simulations:
    """A run's simulations of the validation set, one per snapshot (D1/D2/D6). `measure` runs `fn` once per key:
    a finished result is handed to every later asker (passed or not), a running one is waited for; a source
    that was cancelled or raised leaves nothing behind, and the next asker runs its own. Only a simulation that
    runs takes one of the run's slots."""

    def __init__(self, slots: Optional[threading.BoundedSemaphore]) -> None:
        self.slots = slots
        self.lock = threading.Lock()
        self.entries: dict = {}

    def measure(self, key: str, who: str, fn: Callable[[], dict], cancelled: Callable[[], bool] = lambda: False,
                keep: Callable[[dict], bool] = lambda r: True) -> tuple:
        """(result, how). `how`: {"cached_from": who} for a finished result, {"joined": who} for one waited for,
        {"ran": True, "eval_queued_seconds": s} for one this asker ran. The result is None when `cancelled`
        said so before a simulation of its own began, or while it waited on another."""
        while True:
            with self.lock:
                e = self.entries.get(key)
                if e is not None and e["state"] == "done":
                    return e["result"], {"cached_from": e["source"]}
                mine = e is None
                if mine:
                    e = {"state": "running", "source": who, "done": threading.Event(), "result": None}
                    self.entries[key] = e
            if mine:
                break
            while not e["done"].wait(10):
                if cancelled():
                    return None, {"cancelled_waiting": True}
            with self.lock:
                if e["state"] == "done":
                    return e["result"], {"joined": e["source"]}
            # The source was cancelled or failed on the infrastructure: never reused (D2); run it again.
        how: dict = {"ran": True}
        queued, kept, result, held = time.time(), False, None, False
        try:
            # Waiting for a slot, a cancel is still heard (the review of 2026-11-09: a stopped request's gate sat
            # in acquire() past the run's record).
            while self.slots is not None and not held:
                held = self.slots.acquire(timeout=10)
                if not held and cancelled():
                    break
            how["eval_queued_seconds"] = round(time.time() - queued, 1)
            if cancelled():
                return None, {**how, "cancelled_waiting": True}
            result = fn()
            kept = not cancelled() and keep(result)
            if not kept:
                how["not_kept"] = True
            return result, how
        finally:
            if held:
                self.slots.release()
            with self.lock:
                if kept:
                    e["state"], e["result"] = "done", result
                else:
                    e["state"] = "dropped"
                    if self.entries.get(key) is e:
                        del self.entries[key]
            e["done"].set()

    def kept(self, key: str) -> bool:
        """Whether the run keeps a result for `key` that later askers get."""
        with self.lock:
            e = self.entries.get(key)
            return e is not None and e["state"] == "done"


class Shared:
    """What a run's islands share (report sections 8.9 B7/B8 and 8.8 P10): the slow tier's counts and the
    images it built, one slow request at a time, the cap on evaluations at once, the run's simulations -- and
    one lock for all of their brokers' state. A run without islands has one of its own.

    The slow tier's counts are the run's, or with island roles (`islands.roles`, 2026-11-09) each island's own."""

    def __init__(self, site: Site, task: Task) -> None:
        fs = (task.services or {}).get("fpga_slow") or {}
        roles = (task.islands or {}).get("roles")
        if roles:
            self.counts = {i: {"flows": int(r.get("flows", 0)), "board_runs": int(r.get("board_runs", 0))}
                           for i, r in enumerate(roles)}
        else:
            self.counts = {None: {"flows": int(fs.get("flows", 0)), "board_runs": int(fs.get("board_runs", 0))}}
        self.images: dict = {}
        # The run's Vivado flows running now (the hardware requests holding them): at most flows_at_once,
        # each on a build machine of its own (fpga_slow.place); one by default (B8). Card runs are serial on
        # the card host anyway (its lease and idle wait).
        self.flows: set = set()
        self.flows_at_once = int(fs.get("flows_at_once", 1))
        # And at most this many of them one island's (2026-11-11: in R41-F3 one island held both flows of the run
        # 14 minutes in and the other hardware-writing island was refused nine times). None: no island limit.
        per = fs.get("flows_at_once_per_island")
        self.flows_per_island = int(per) if per else None
        # The quick tier's worker is one shared machine: one job at a time across islands (D8a), queued.
        self.quick = threading.BoundedSemaphore(1)
        self.lock = threading.Lock()
        n = getattr(site, "evaluations", None)
        self.evaluations = threading.BoundedSemaphore(int(n)) if n else None
        self.sims = Simulations(self.evaluations)
        # The board as the score (2026-11-10): one card run per snapshot for the run, the card one session at a
        # time for every island, and its health as the last check found it (None: healthy).
        self.boards = Simulations(None)
        self.card = threading.Lock()
        self.card_unhealthy: Optional[dict] = None
        self.health_tree: Optional[Path] = None
        self.brokers: list = []          # every island's broker: the run-wide checks look at all their requests
        # The whole chip's lower bound of each scored point (roofline.table), when the run computed it: each
        # reply then lists every point's eta (decided 2026-11-10).
        self.roofline: Optional[dict] = None
        self.commits: dict = {}          # sim_key -> (broker, job): the run's scoring requests, for D4

    def pool(self, island: Optional[int]) -> dict:
        """The slow tier's counts an island draws on: its own with roles, else the run's."""
        return self.counts[island] if island in self.counts else self.counts[None]

    # A run without roles: one pool (kept for the callers that read the run's counts).
    flows_left = property(lambda self: self.pool(None)["flows"],
                          lambda self, v: self.pool(None).__setitem__("flows", v))
    board_runs_left = property(lambda self: self.pool(None)["board_runs"],
                               lambda self, v: self.pool(None).__setitem__("board_runs", v))


class Broker:
    def __init__(self, site: Site, repo: Path, task: Task, workspace: Path, run_dir: Path,
                 runners: Optional[dict] = None, island: Optional[int] = None, archive=None,
                 shared: Optional[Shared] = None, run_id: Optional[str] = None) -> None:
        self.site, self.repo, self.task, self.workspace = site, repo, task, workspace
        self.run_id = run_id or run_dir.name
        self.dir = run_dir / "services"
        cfg = task.services or {}
        self.score_left = int((cfg.get("score") or {}).get("requests", 0))
        fq = cfg.get("fpga_quick") or {}
        roles = (task.islands or {}).get("roles") or []
        # The island's role (islands.roles, 2026-11-09): which layers it writes, and its own budgets.
        self.role = roles[island] if island is not None and island < len(roles) else None
        minutes = (self.role or {}).get("fpga_quick_minutes", fq.get("minutes", 0))
        self.fpga_left = float(minutes) * 60.0
        self.fpga_job = float(fq.get("job_minutes", 30)) * 60.0
        self.period = float(fq.get("period_ns", 10.0))
        fs = cfg.get("fpga_slow") or {}
        self.slow_mode = fs.get("mode", "board")
        self.flow_seconds = int(float(fs.get("flow_minutes", 480)) * 60)
        self.board_seconds = int(float(fs.get("board_minutes", 120)) * 60)
        # The slow tier's counts and images, and the lock, are the run's (islands share them).
        self.shared = shared or Shared(site, task)
        self.shared.brokers.append(self)
        self.island, self.archive = island, archive
        # The board as the score (services.score.on "board", decided 2026-11-10): a scoring request is a card run
        # and a commit; `flow` builds hardware, and no simulation scores.
        self.board_scoring = (cfg.get("score") or {}).get("on") == "board"
        self.runners: dict = runners or {"score": self._run_board_score if self.board_scoring else self._run_score,
                                         "fpga-quick": self._run_fpga, "fpga-slow": self._run_slow,
                                         "flow": self._run_flow}
        self.jobs: dict = {}
        self.busy: dict = {}
        # Every submission refused before it became a request (2026-11-09): when, which service, why. The agent
        # had the answer at once; the record keeps it, since a role's refusals are part of what a run shows.
        self.refusals: list = []
        self.lock = self.shared.lock
        self.socket_dir: Optional[Path] = None
        self._server = None

    # The slow tier's counts and images live in Shared: the run's pool, or this island's with roles.
    flows_left = property(lambda self: self.shared.pool(self.island)["flows"],
                          lambda self, v: self.shared.pool(self.island).__setitem__("flows", v))
    board_runs_left = property(lambda self: self.shared.pool(self.island)["board_runs"],
                               lambda self, v: self.shared.pool(self.island).__setitem__("board_runs", v))
    images = property(lambda self: self.shared.images)

    def _who(self, job: dict) -> str:
        return f"{self.island}:{job.get('id')}" if self.island is not None else str(job.get("id"))

    def _all_jobs(self) -> list:
        """Under the lock: every request of the run, every island's."""
        return [j for b in self.shared.brokers for j in b.jobs.values()]

    # -- lifetime ------------------------------------------------------------------
    def start(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        # A Unix socket's path is limited to about 100 bytes, and a run directory is
        # longer: the socket lives in a short directory of its own, mounted into the sandbox.
        self.socket_dir = Path(tempfile.mkdtemp(prefix="tnca-svc-"))
        self.socket_dir.chmod(0o755)
        broker = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request = json.loads(self.rfile.readline().decode() or "{}")
                    reply = broker.handle(request if isinstance(request, dict) else {})
                except Exception as exc:  # noqa: BLE001 -- the agent gets an answer, never a hang
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(reply, sort_keys=True) + "\n").encode())

        self._server = socketserver.ThreadingUnixStreamServer(str(self.socket_dir / "service.sock"), Handler)
        (self.socket_dir / "service.sock").chmod(0o777)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.socket_dir

    def stop(self) -> None:
        """At the end of the run: no new requests; a slow request runs to its end, unseen by the agent, for
        the certification after the run (report section 8.9, B10); anything else still running is stopped."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        stopped = []
        with self.lock:
            # Decided under the lock that _run takes to start a job, so a queued request is either
            # cancelled here (and its reservation given back once) or started there -- never both.
            for job in self.jobs.values():
                if job["state"] not in ("queued", "running"):
                    continue
                if (job["service"] in ("fpga-slow", "flow") and job["state"] == "running"
                        and job.get("waiting_for_build_host")):
                    # It never got a build machine: cancelled as a queued one is; its charge goes back when its
                    # thread ends (the flow did not run).
                    job["_cancel_requested"] = True
                    job["state"] = "cancelled"
                    job["note"] = "waiting for a free build machine when the agent stopped; not built, not charged"
                    self._release_flow(job)
                    stopped.append(job)
                    continue
                if job["service"] == "flow" and job["state"] == "running":
                    # An image built after the agent's end would never be scored: the flow is stopped, uncharged.
                    job["_cancel_requested"] = True
                    job["state"] = "cancelled"
                    job["note"] = "still building when the agent stopped; stopped, not charged"
                    self._release_flow(job)
                    stopped.append(job)
                    continue
                if (job["service"] == "fpga-slow" or (job["service"] == "score" and self.board_scoring
                                                       and not job.get("dedupe_of"))) and job["state"] == "running":
                    job["outlived_run"] = True
                    job["note"] = "still running when the agent stopped; it ran to its end, unseen by the agent"
                    continue
                if job["service"] == "fpga-slow" and job["state"] == "queued" and job.get("charge"):
                    if job["charge"] == "flow":
                        self.flows_left += 1
                    else:
                        self.board_runs_left += 1
                    job["charged"] = None
                job["state"] = "cancelled"
                job["note"] = "still running when the agent stopped; its result is not kept"
                self._release_flow(job)
                stopped.append(job)
        for job in stopped:
            if job["service"] == "fpga-quick" and self.site.fpga_worker:
                fpga_quick.cancel(self.site.fpga_worker, job["remote_id"])
            elif job["service"] in ("fpga-slow", "flow") and getattr(self.site, "fpga_pipeline", None):
                job["cleanup"] = fpga_slow.cancel(job, self.site.fpga_pipeline)
            else:
                for name in provider.docker("ps", "-a", "--format", "{{.Names}}", check=False).stdout.split():
                    if name.startswith(job["containers"]):
                        provider.remove(name)
        for job in stopped:
            if job.get("thread"):
                job["thread"].join(timeout=180)
        if self.socket_dir is not None:
            shutil.rmtree(self.socket_dir, ignore_errors=True)

    def wait_outlived(self) -> list:
        """The slow requests that outlived the agent, waited for to their end; their ids."""
        with self.lock:
            outlived = [j for j in self.jobs.values() if j.get("outlived_run")]
        for job in outlived:
            if job.get("thread"):
                job["thread"].join()
        return [j["id"] for j in outlived]

    # -- the protocol ----------------------------------------------------------------
    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "budget":
            return self.budget()
        if op == "submit":
            return self.submit(request.get("service"), message=request.get("message"))
        if op == "archive":
            return self.archive.view(self.island) if self.archive is not None else {"error": "this run has no islands"}
        if op == "read-commit":
            if self.archive is None:
                return {"error": "this run has no islands"}
            return self.archive.read(self.island, str(request.get("id")))
        if op == "status":
            if request.get("id"):
                return self._view(request["id"], with_result=False)
            with self.lock:
                return {"requests": [self._summary(j) for j in self.jobs.values()], **self.budget()}
        if op == "result":
            return self._view(request.get("id"), with_result=True)
        return {"error": f"unknown operation {op!r}; see `tnca-service --help`"}

    def budget(self) -> dict:
        if self.board_scoring:
            return {"score_requests_left": self.score_left,
                    "fpga_minutes_left": round(self.fpga_left / 60.0, 1),
                    "fpga_minutes_per_request": round(self.fpga_job / 60.0, 1),
                    "flows_left": self.flows_left, "busy": sorted(self.busy)}
        return {"score_requests_left": self.score_left,
                "fpga_minutes_left": round(self.fpga_left / 60.0, 1),
                "fpga_minutes_per_request": round(self.fpga_job / 60.0, 1),
                "fpga_slow_flows_left": self.flows_left,
                "fpga_slow_board_runs_left": self.board_runs_left,
                "busy": sorted(self.busy)}

    def submit(self, service, message=None) -> dict:
        if service not in SERVICES:
            return {"error": f"no service {service!r}; there are: {', '.join(SERVICES)}"}
        with self.lock:
            refused = self._refusal(service)
            if refused:
                self._refused(service, refused)
                return {"refused": refused, **self.budget()}
            n = len(self.jobs) + 1
            job_id = f"{ {'score': 's', 'fpga-quick': 'f', 'flow': 'b'}.get(service, 'l')}{n}"
            job_dir = self.dir / job_id
            self.busy.setdefault(service, []).append(job_id)
        try:
            reply = self._submit(service, message, job_id, job_dir)
            if "refused" in reply:
                with self.lock:
                    self._refused(service, reply["refused"], reply.get("files") or reply.get("rejected"))
            return reply
        except Exception:
            # Whatever failed after the reservation (a gate on a broken tree, a description that does not
            # parse) gives it back: one bad snapshot never blocks a service for the rest of the run.
            with self.lock:
                self._unbusy(service, job_id)
            raise

    def _refused(self, service: str, why: str, files=None) -> None:
        """Under the lock: a refused submission, for the record."""
        entry = {"at": round(time.time(), 1), "service": service, "why": str(why)[:300]}
        if files:
            entry["files"] = [str(f)[:200] for f in list(files)[:20]]
        self.refusals.append(entry)

    def _unbusy(self, service: str, job_id: str) -> None:
        """Under the lock: the request no longer counts as one of the service's running ones."""
        running = self.busy.get(service) or []
        if job_id in running:
            running.remove(job_id)
        if not running:
            self.busy.pop(service, None)

    def _at_once(self, service: str) -> int:
        """How many of a service's requests may run at once: the run's flows at once for the slow tier (no more than
        an island's own limit), else one."""
        if service not in ("fpga-slow", "flow"):
            return 1
        n, per = max(1, self.shared.flows_at_once), self.shared.flows_per_island
        return min(n, per) if per else n

    def _memo(self) -> "Simulations":
        """Where a snapshot's one measurement lives: the card's runs on the board, else the simulations."""
        return self.shared.boards if self.board_scoring else self.shared.sims

    def _submit(self, service: str, message, job_id: str, job_dir: Path) -> dict:
        g = gate(self.repo, self.task.base_commit, self.task.hide, self.task.design, self.workspace,
                 out=job_dir / "candidate", evaluator_owned=self.task.evaluator_owned)
        candidate = job_dir / "candidate" if g["taken"] else None
        problems = [] if g["verdict"] != "PASS" else evaluator.rtl_check(candidate, self.repo)
        snapshot = {"taken": [c["path"] for c in g["taken"]],
                    "dropped": [c["path"] for c in g["dropped"]][:50]}
        if g["verdict"] != "PASS" or problems:
            with self.lock:
                self._unbusy(service, job_id)
            return {"refused": "the snapshot cannot be measured", "rejected": g["rejected"][:20] or problems,
                    "snapshot": snapshot, **self.budget()}
        authorship = None
        if self.role is not None and self.archive is not None:
            # The island writes its own layers only; another layer it may take whole from a commit it can see
            # (2026-11-09, "adopt, don't author"). Refused before anything is charged.
            authorship = self.archive.authorship(self.island, candidate)
            if authorship.get("refused"):
                with self.lock:
                    self._unbusy(service, job_id)
                return {"refused": authorship["refused"], "files": authorship.get("files", [])[:20],
                        "snapshot": snapshot, **self.budget()}
        job = {"id": job_id, "service": service, "state": "queued", "submitted": time.time(),
               "snapshot": snapshot, "containers": f"tnca-svc-{self.run_id}-{job_id}",
               "remote_id": f"tnca-{secrets.token_hex(8)}"}
        if authorship is not None:
            job["authorship"] = {k: v for k, v in authorship.items() if k != "refused"}
        if self.island is not None:
            if service == "score":
                job["message"] = str(message or "")[:200]
                job["digest"] = fpga_slow.snapshot_digest(candidate)
                job["hardware"] = fpga_slow.hardware_digest(candidate)
                job["candidate"] = str(candidate) if candidate else None
        if service in ("fpga-quick", "fpga-slow", "flow"):
            job["hardware"] = fpga_slow.hardware_digest(candidate)
            job["digest"] = fpga_slow.snapshot_digest(candidate)
        if service in ("score", "fpga-slow", "flow"):
            job["sim_key"] = sim_key(self.task, candidate)
        if service == "score" and self.board_scoring:
            job["hardware"] = fpga_slow.hardware_digest(candidate)
            changed = evaluator.hardware_changed(self.repo, self.task, candidate)["changed"]
            if changed and job["hardware"] not in self.images:
                with self.lock:
                    self._unbusy(service, job_id)
                return {"refused": "this snapshot's hardware has no image yet: `tnca-service submit flow` builds it; "
                        "a scoring request runs on the card only the base's machine or one this run built",
                        "snapshot": snapshot, **self.budget()}
        if service == "flow":
            refused = self._flow_refusal(job, candidate)
            if refused:
                with self.lock:
                    self._unbusy(service, job_id)
                return {"refused": refused, "snapshot": snapshot, **self.budget()}
        if service == "score":
            with self.lock:
                src = self.shared.commits.get(job["sim_key"])
                if src is not None and not (src[1]["state"] in ("queued", "running")
                                            or (src[1]["state"] == "done"
                                                and self._memo().kept(self._memo_key(job)))):
                    src = None   # failed, cancelled, or a result not kept for reuse: this one is measured afresh
                if src is None:
                    # Registered now, in the same lock as the check: a second island's identical snapshot submitted
                    # at the same moment is this one's repeat, never a second charge.
                    self.shared.commits[job["sim_key"]] = (self, job)
                if src is not None:
                    # D4: the same snapshot as an earlier scoring request of the run -- not charged, no new
                    # commit, no simulation; its reply is that request's, once it has one.
                    self._unbusy(service, job_id)
                    job.update(charged=None, dedupe_of=src[0]._who(src[1]))
                    self.jobs[job_id] = job
                    job["thread"] = threading.Thread(target=self._run, args=(job, candidate, job_dir), daemon=True)
                    job["thread"].start()
                    return {"id": job_id, "state": "queued", "snapshot": snapshot,
                            "same_as": job["dedupe_of"], "note": "the same snapshot as an earlier scoring request "
                            "of this run: not charged, not a new commit; its result is that request's",
                            **self.budget()}
        if service == "fpga-slow":
            job["mode"] = self.slow_mode
            refused = self._slow_kind(job, candidate)
            if refused:
                with self.lock:
                    self._unbusy(service, job_id)
                return {"refused": refused, "snapshot": snapshot, **self.budget()}
        with self.lock:
            if service in ("fpga-slow", "flow") and job["kind"] == "hardware":
                at_once = max(1, self.shared.flows_at_once)
                per = self.shared.flows_per_island
                mine = sorted(f.split(":", 1)[1] for f in self.shared.flows if f.startswith(f"{self.run_id}:"))
                refused = None
                if per and len(mine) >= per:
                    refused = (f"this island's Vivado flow{'s' if len(mine) > 1 else ''} ({', '.join(mine)}) "
                               f"{'are' if len(mine) > 1 else 'is'} running: at most {per} per island at once, "
                               f"{at_once} for the run")
                elif len(self.shared.flows) >= at_once:
                    refused = ("another Vivado flow of this run is running; one at a time per run" if at_once == 1
                               else f"{at_once} Vivado flows of this run are running; at most {at_once} at once")
                elif any(j.get("hardware") == job["hardware"] and j.get("kind") == "hardware"
                         and j["state"] in ("queued", "running") for j in self._all_jobs()):
                    # Two flows of the same machine would build one image twice.
                    refused = ("a flow of this run is already building exactly this hardware; ask again when it "
                               "is done, and it will need no flow")
                if refused:
                    self._unbusy(service, job_id)
                    return {"refused": refused, "snapshot": snapshot, **self.budget()}
                self.shared.flows.add(f"{self.run_id}:{job_id}")
            if service == "score":
                self.score_left -= 1
            elif service == "flow":
                self.flows_left -= 1
            elif service == "fpga-slow":
                if job["charge"] == "flow":
                    self.flows_left -= 1
                else:
                    self.board_runs_left -= 1
            self.jobs[job_id] = job
        job["thread"] = threading.Thread(target=self._run, args=(job, candidate, job_dir), daemon=True)
        job["thread"].start()
        return {"id": job_id, "state": "queued", "snapshot": snapshot, **self.budget()}

    def _memo_key(self, job: dict) -> str:
        return job["sim_key"]

    def _flow_refusal(self, job: dict, candidate: Optional[Path]) -> Optional[str]:
        """Why a `flow` request is refused, if it is: it must change the machine, not one built already, and the
        quick tier must have met timing on exactly that hardware."""
        if not evaluator.hardware_changed(self.repo, self.task, candidate)["changed"]:
            return "the snapshot's machine is the base's: there is nothing to build; `score` runs it on the card"
        if job["hardware"] in self.images:
            return "this run already built exactly this hardware: `score` runs it on the card"
        if not self._timing_met(job["hardware"]):
            return ("a flow needs an fpga-quick result of this run on exactly this hardware that met timing; ask the "
                    "quick tier first")
        job["kind"], job["charge"] = "hardware", "flow"
        return None

    def _release_flow(self, job: dict) -> None:
        """Under the lock: the request's Vivado flow, given back to the run."""
        self.shared.flows.discard(f"{self.run_id}:{job['id']}")

    def _slow_kind(self, job: dict, candidate: Optional[Path]) -> Optional[str]:
        """Which kind a slow request is, what it is charged to, and why it is refused if it is."""
        changed = evaluator.hardware_changed(self.repo, self.task, candidate)["changed"]
        built = self.images.get(job["hardware"])
        if self.slow_mode == "build" or (changed and built is None):
            job["kind"], job["charge"] = "hardware", "flow"
            if self.flows_left <= 0:
                return "no Vivado flows left for this run"
            if changed and not self._timing_met(job["hardware"]):
                return ("a hardware request needs an fpga-quick result of this run on exactly this hardware "
                        "that met timing; ask the quick tier first")
            return None
        job["kind"], job["charge"] = "software", "board_run"
        job["image"] = built
        if self.board_runs_left <= 0:
            return "no board runs left for this run"
        if built is None:
            try:
                job["base_image"] = board.base_image(self.site, self.task)
            except board.BoardError as exc:
                return f"the base's image is not available on the card host: {exc}"
        return None

    def _timing_met(self, digest: str) -> bool:
        """A quick-tier result of this run -- any island's -- on exactly this hardware that met timing."""
        with self.lock:
            quick = [j for j in self._all_jobs()
                     if j["service"] == "fpga-quick" and j.get("hardware") == digest and j["state"] == "done"]
        for j in quick:
            r = j.get("full") or {}
            wns, failing = r.get("wns_ns"), r.get("failing_endpoints")
            if isinstance(wns, (int, float)) and wns >= 0 and not failing:
                return True
        return False

    def _refusal(self, service: str) -> Optional[str]:
        running, at_once = self.busy.get(service) or [], self._at_once(service)
        if len(running) >= at_once:
            if at_once == 1:
                return f"a {service} request is already running ({running[0]}); one at a time"
            return f"{len(running)} {service} requests are already running ({', '.join(running)}); at most {at_once} at once"
        if service == "score" and self.archive is not None and self.archive.halt.is_set():
            return "the run has ended (its target was reached): no more commits"
        if service == "score" and self.score_left <= 0:
            return "no scoring requests left"
        if service == "fpga-quick":
            if not self.site.fpga_worker:
                return "this site has no FPGA build worker"
            if self.fpga_left < self.fpga_job:
                return (f"an FPGA request may take up to {self.fpga_job / 60:.0f} min and "
                        f"{self.fpga_left / 60:.1f} min of FPGA time is left")
        if self.board_scoring and service == "fpga-slow":
            return "in this task `score` is the card run and `flow` builds hardware: ask those"
        if service == "flow":
            if not self.board_scoring:
                return "this task has no flow service; ask fpga-slow"
            if not getattr(self.site, "fpga_pipeline", None):
                return "this site has no FPGA pipeline"
            if self.flows_left <= 0:
                return "no Vivado flows left" + (" for your island" if self.island is not None else "")
        if service == "score" and self.board_scoring:
            if not (getattr(self.site, "fpga_pipeline", None) or {}).get("card"):
                return "this site has no card for the board score"
            bad = self.shared.card_unhealthy
            if bad and time.time() - bad["at"] < self.UNHEALTHY_SECONDS:
                return ("the card failed its health check after a recent session; it takes no board scores for "
                        f"about {round((self.UNHEALTHY_SECONDS - (time.time() - bad['at'])) / 60)} min; nothing "
                        "was charged")
        if service == "fpga-slow":
            if not (self.task.services or {}).get("fpga_slow"):
                return "this task does not enable the FPGA slow service"
            if not getattr(self.site, "fpga_pipeline", None):
                return "this site has no FPGA pipeline"
            if self.slow_mode == "board" and not (self.site.fpga_pipeline.get("board_host")
                                                  and self.site.fpga_pipeline.get("card")):
                return "the FPGA pipeline has no card host for board sessions"
            if self.flows_left <= 0 and self.board_runs_left <= 0:
                return "no Vivado flows and no board runs left for this run"
        return None

    def _run(self, job: dict, candidate: Optional[Path], job_dir: Path) -> None:
        with self.lock:
            if job["state"] != "queued":  # cancelled before it started
                self._unbusy(job["service"], job["id"])
                self._release_flow(job)
                return
            job["state"], job["started"] = "running", time.time()
        try:
            runner = self._run_dedupe if job.get("dedupe_of") else self.runners[job["service"]]
            visible, full = runner(job, candidate, job_dir)
            job["full"] = full
            # A cancelled request's numbers were measured while it was being torn
            # down: they go into the record as `full`, never as its result.
            if job["state"] == "running":
                job["state"], job["result"] = "done", visible
        except Exception as exc:  # noqa: BLE001
            error = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            # A slow request's failure can carry host names, paths and whatever a container wrote: the
            # agent learns that it failed and was not charged; the text stays in the record.
            shown = ({"status": "failed", "why": "the request failed on the infrastructure; not charged"}
                     if job["service"] in ("fpga-slow", "flow") or (job["service"] == "score" and self.board_scoring)
                     else error)
            if job["state"] == "running":
                job["state"], job["result"] = "failed", shown
                job["full"] = error
            else:
                job["full"] = error
        finally:
            job["finished"] = time.time()
            with self.lock:
                self._unbusy(job["service"], job["id"])
                self._release_flow(job)

    def _run_score(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        # One simulation per snapshot for the run (D1/D2), within the run's slots (D6).
        r, how = self.shared.sims.measure(
            job.get("sim_key") or sim_key(self.task, candidate), self._who(job),
            lambda: evaluator.measure_validation(self.site, self.repo, self.task, candidate, job_dir, job["containers"]),
            cancelled=lambda: job.get("state", "running") != "running", keep=reusable)
        job["simulation"] = how
        job["eval_queued_seconds"] = how.get("eval_queued_seconds", 0.0)
        if r is None:
            # Cancelled while it waited (its island stopped): nothing is measured or archived.
            return {"status": "cancelled"}, {"cancelled_waiting": True}
        visible = {"validation": r["aggregate"]}
        points = roofline.view(self.shared.roofline, (r.get("full") or {}).get("points") or [],
                               self.task.evaluation["validation"])
        if points is not None:
            visible["points"] = points
        # On failure, what may be said of why (README section 6.4): failed points counted by
        # kind, and the logs of what failed before any point ran (a device that did not
        # build, staging that stopped the host). Never a point's name or its own log.
        for key in ("failures", "device_build_errors", "stage_errors"):
            if r.get(key):
                visible[key] = r[key]
        if self.archive is not None and job.get("state", "running") == "running":
            # An island's scoring request is its commit: kept in the archive, its reply the island's view.
            entry = self.archive.add(self.island, job, candidate, r)
            job["commit"] = entry["id"]
            visible.update(self.archive.reply(self.island, entry))
        return visible, r

    # -- the board as the score (2026-11-10) ------------------------------------------------
    UNHEALTHY_SECONDS = 1800   # a card that failed its health check takes no board scores for this long

    def _card_health(self) -> Optional[dict]:
        """The known-answer check after each card session ({point, tree}), when the task names one."""
        point = self.task.evaluation.get("health_point")
        if not point:
            return None
        if self.shared.health_tree is None:
            self.shared.health_tree = board.health_tree(self.site, self.repo, self.task)
        return {"point": point, "tree": str(self.shared.health_tree)}

    def _card_image(self, job: dict) -> dict:
        built = self.images.get(job["hardware"])
        if built is None:
            return board.base_image(self.site, self.task)
        if not built.get("card"):
            built["card"] = board.put_image(self.site, built["build_run"], built["sha256"], built["xclbin"],
                                            job["remote_id"], build_host=built.get("build_host"))
        return built["card"]

    def _on_card(self, job: dict, candidate: Optional[Path], job_dir: Path) -> dict:
        """One card session for the snapshot, the card one session at a time for every island; the time it
        waited for the card is the agent's (decided 2026-11-10). The base itself (nothing taken, the base's
        image) is answered from the run's base measurement, with no card session and no wait (2026-11-11)."""
        if candidate is None and self.images.get(job.get("hardware")) is None:
            try:
                got = fpga_slow.on_card(self.site, self.repo, self.task, None, board.base_image(self.site, self.task),
                                        None, job_dir, job, self.board_seconds, roofline=self.shared.roofline,
                                        from_base=True)
                job["from_base"] = True
                return got
            except Exception as exc:  # noqa: BLE001 -- no base measurement to answer from: the card measures it
                job["from_base_error"] = f"{type(exc).__name__}: {exc}"
        queued = time.time()
        held = False
        while not held:
            held = self.shared.card.acquire(timeout=10)
            if not held and job.get("state", "running") != "running" and not job.get("outlived_run"):
                return {"outcome": "cancelled"}
        job["card_queued_seconds"] = round(time.time() - queued, 1)
        try:
            image = self._card_image(job)
            return fpga_slow.on_card(self.site, self.repo, self.task, candidate, image, None, job_dir, job,
                                     self.board_seconds,
                                     base_xclbin=fpga_slow.base_xclbin(self.site, self.repo, self.task, candidate,
                                                                       image),
                                     roofline=self.shared.roofline, health=self._card_health())
        finally:
            self.shared.card.release()

    def _run_board_score(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        """A scoring request with the board as the score: the snapshot on the card (the base's machine, or one a
        flow of this run built), once per snapshot for the run; a commit when it was measured. A session that
        failed on the infrastructure, or whose card then failed its health check, is not charged."""
        try:
            card, how = self.shared.boards.measure(
                self._memo_key(job), self._who(job), lambda: self._on_card(job, candidate, job_dir),
                cancelled=lambda: job.get("state", "running") != "running" and not job.get("outlived_run"),
                keep=lambda r: r.get("outcome") in ("ok", "candidate"))
        except Exception:
            self._refund_score(job)
            raise
        job["board"] = how
        if card is None or card.get("outcome") == "cancelled":
            self._refund_score(job)
            return {"status": "cancelled"}, {"cancelled_waiting": True}
        if card.get("health") and not card["health"].get("ok"):
            with self.lock:
                self.shared.card_unhealthy = {"at": time.time(), "job": self._who(job), "health": card["health"]}
        if card["outcome"] == "infra":
            self._refund_score(job)
            return {**card["visible"], "charged": None}, {k: v for k, v in card.items() if k != "visible"}
        visible = dict(card["visible"])
        if job.get("card_queued_seconds"):
            visible["waited_for_the_card_minutes"] = round(job["card_queued_seconds"] / 60.0, 1)
        agg = dict(visible.get("validation") or {"n": len(board_points(self.task)), "passed": 0})
        result = {"aggregate": agg, "full": card.get("score"), "families": visible.get("families"),
                  "outcome": card["outcome"]}
        if self.archive is not None and (job.get("state", "running") == "running" or job.get("outlived_run")):
            entry = self.archive.add(self.island, job, candidate, result)
            job["commit"] = entry["id"]
            visible.update(self.archive.reply(self.island, entry))
        return visible, {k: v for k, v in card.items() if k != "visible"}

    def _refund_score(self, job: dict) -> None:
        with self.lock:
            if job.get("charged", "score") is not None and not job.get("dedupe_of"):
                self.score_left += 1
            job["charged"] = None

    def _flow_gate(self, job: dict, candidate: Optional[Path], job_dir: Path) -> Optional[dict]:
        """The short functional RTL gate before a flow (2026-11-10): the task's gate points in simulation, the
        run's one such simulation per snapshot, within the run's slots."""
        ev = self.task.evaluation
        gtask = dataclasses.replace(self.task, evaluation={**ev, "validation": list(ev.get("gate_points") or ()),
                                                           "final": []})
        r, how = self.shared.sims.measure(
            "gate:" + job["sim_key"], self._who(job),
            lambda: evaluator.measure_validation(self.site, self.repo, gtask, candidate, job_dir / "gate",
                                                 job["containers"]),
            cancelled=lambda: bool(job.get("_cancel_requested")), keep=reusable)
        job["gate"] = how
        return r

    def _run_flow(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        """A hardware snapshot's image: the RTL gate beside its Vivado flow (a gate that does not pass stops the
        flow, uncharged); a built image is the run's, for any island's `score` of exactly this hardware."""
        visible, full = {}, {}
        try:
            visible, full = fpga_slow.run(self.site, self.repo, self.task, candidate, job_dir, job, with_board=False,
                                          image=None, flow_seconds=self.flow_seconds, board_seconds=0,
                                          gate=lambda: self._flow_gate(job, candidate, job_dir))
        except Exception:
            job["outcome"] = "infra"
            raise
        finally:
            kept = job.get("outcome") in ("ok", "design")
            with self.lock:
                job["charged"] = "flow" if kept else None
                if not kept:
                    self.flows_left += 1
                if job.get("image") and job.get("outcome") == "ok":
                    self.images[job["hardware"]] = job["image"]
        if "validation" in visible:
            # The gate's tiny points, not the validation set: whether they passed.
            g = visible.pop("validation")
            visible["gate"] = {"n": g.get("n"), "passed": g.get("passed")}
            if job.get("outcome") == "refused":
                visible["why"] = "the RTL gate's points do not pass in simulation; the flow was stopped, not charged"
        if job.get("outcome") == "ok":
            visible["next"] = "built: `tnca-service submit score` with this snapshot runs it on the card"
        return visible, full

    def _run_dedupe(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        """D4: a scoring request of a snapshot the run already scored, answered with that request's result once
        it has one; if that one does not finish (cancelled, failed), neither does this, uncharged."""
        with self.lock:
            src = self.shared.commits.get(job["sim_key"])
        src_job = src[1] if src else None
        while src_job is not None and src_job["state"] in ("queued", "running"):
            if job.get("state") != "running":
                return {"status": "cancelled"}, {"cancelled_waiting": True}
            time.sleep(2)
        if src_job is None or src_job["state"] != "done" or not self._memo().kept(self._memo_key(job)):
            return ({"status": "failed", "why": f"the same snapshot's request {job['dedupe_of']} gave no result to "
                     "share (it did not finish, or failed on the infrastructure); ask again; not charged"},
                    {"same_as": job["dedupe_of"], "source_state": (src_job or {}).get("state")})
        r = src_job.get("full") or {}
        visible = {k: v for k, v in (src_job.get("result") or {}).items()
                   if k in ("validation", "families", "points", "failures", "device_build_errors", "stage_errors",
                            "checks", "counted_as_in_simulation", "cycles_on_card")}
        visible["cached_from"] = job["dedupe_of"]
        if self.archive is not None and src_job.get("commit"):
            job["commit"] = src_job["commit"]
            entry = self.archive.by_id.get(src_job["commit"])
            if entry is not None:
                visible.update(self.archive.reply(self.island, entry))
        return visible, {"same_as": job["dedupe_of"], "commit": src_job.get("commit"),
                         "aggregate": r.get("aggregate")}

    def _gate(self, job: dict, candidate: Optional[Path], job_dir: Path) -> Optional[dict]:
        """An fpga-slow request's gate: the validation set in simulation, the run's one simulation of this
        snapshot (D1/D2), within the run's slots (D6). None when the request was cancelled first."""
        r, how = self.shared.sims.measure(
            job.get("sim_key") or sim_key(self.task, candidate), self._who(job),
            lambda: evaluator.measure_validation(self.site, self.repo, self.task, candidate, job_dir / "sim",
                                                 job["containers"]),
            cancelled=lambda: bool(job.get("_cancel_requested")), keep=reusable)
        job["gate"] = how
        return r

    def _run_fpga(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        tree = job_dir / "device-tree"
        if candidate is None:
            export(self.repo, self.task.base_commit, (), tree)
        else:
            apply_candidate(self.repo, self.task.base_commit, candidate, tree, only=evaluator.HARDWARE)
        try:
            # What this tier cannot see of the change (the uncore, parameters its top does not take): said, not
            # silently measured as the base's (2026-11-09).
            missing = fpga_quick.not_applied(tree, evaluator.hardware_changed(self.repo, self.task, candidate))
        except (OSError, ValueError, KeyError):
            missing = []
        seconds = int(min(self.fpga_job, self.fpga_left))
        queued = time.time()
        self.shared.quick.acquire()   # one quick-tier job at a time on the shared worker; the wait is not charged
        job["quick_queued_seconds"] = round(time.time() - queued, 1)
        started, r = time.monotonic(), {}
        try:
            r = fpga_quick.run(self.site.fpga_worker, tree, self.task.evaluation["profile"], self.period, seconds,
                               job_dir, job["remote_id"])
        finally:
            self.shared.quick.release()
            shutil.rmtree(tree, ignore_errors=True)
            # Charged even when run() raises (its remote clean-up can time out after the job ran):
            # the worker's own time when it reported one, else the time this request held the worker.
            charged = r.get("seconds", min(seconds, time.monotonic() - started))
            with self.lock:
                job["charged_seconds"] = round(charged, 1)
                self.fpga_left = max(0.0, self.fpga_left - charged)
        if missing:
            r["not_applied"] = missing
        visible = {k: r[k] for k in FPGA_VISIBLE if k in r}
        visible["note"] = FPGA_NOTE
        return visible, r

    def _run_slow(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        """The request, and its charge settled: kept when it was the candidate's (a flow that ran, a card
        run that measured or refused the image), given back when the gate refused it or the
        infrastructure failed it."""
        try:
            image = job.get("base_image")
            if job.get("image"):   # hardware an earlier request of this run built: its image, on the card host
                image = job["image"].get("card") or board.put_image(
                    self.site, job["image"]["build_run"], job["image"]["sha256"], job["image"]["xclbin"],
                    job["remote_id"], build_host=job["image"].get("build_host"))
            visible, full = fpga_slow.run(self.site, self.repo, self.task, candidate, job_dir, job,
                                          with_board=self.slow_mode == "board", image=image,
                                          flow_seconds=self.flow_seconds, board_seconds=self.board_seconds,
                                          gate=lambda: self._gate(job, candidate, job_dir),
                                          roofline=self.shared.roofline)
        except Exception:
            job["outcome"] = "infra"
            raise
        finally:
            kept = job.get("outcome") in ("ok", "design")
            with self.lock:
                job["charged"] = job["charge"] if kept else None
                if not kept:
                    if job["charge"] == "flow":
                        self.flows_left += 1
                    else:
                        self.board_runs_left += 1
                if job.get("kind") == "hardware" and job.get("image") and job.get("outcome") == "ok":
                    self.images[job["hardware"]] = job["image"]
        return visible, full

    # -- views ---------------------------------------------------------------------------
    def _summary(self, job: dict) -> dict:
        out = {"id": job["id"], "service": job["service"], "state": job["state"],
               "seconds": round((job.get("finished") or time.time()) - job["submitted"], 1)}
        if job["service"] in ("fpga-slow", "flow") and job["state"] == "running":
            # Never which machine: whether the flow waits for one, and once it runs, within how many minutes
            # from now the request ends at the latest (the flow's cap, then the card's in board mode). A `flow`
            # too (2026-11-11: R41-F3's b5 failed its link on one machine, waited 55 minutes for the other to try
            # again, and said only "running").
            if job.get("waiting_for_build_host"):
                out["waiting_for_a_build_machine"] = True
            tried = job.get("flow_attempts") or []
            if tried:
                out["earlier_attempts"] = tried
                if any(a.get("outcome") == "infra" for a in tried):
                    out["note"] = ("an earlier attempt ended on the build machine's side, not the design's (not "
                                   "charged); the flow is tried again on another build machine")
            board_s = self.board_seconds if self.slow_mode == "board" else 0
            if job.get("flow_started") and not job.get("flow_finished"):
                end = job["flow_started"] + self.flow_seconds + board_s
            elif job.get("flow_finished") and board_s:
                end = job["flow_finished"] + board_s
            else:
                end = None
            if end is not None:
                out["ends_within_minutes"] = max(0, round((end - time.time()) / 60.0))
        return out

    def _view(self, job_id, with_result: bool) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return {"error": f"no request {job_id!r}"}
        out = {**self._summary(job), "snapshot": job["snapshot"]}
        if with_result and job["state"] in ("done", "failed", "cancelled"):
            out["result"] = job.get("result")
        return out

    def record(self) -> dict:
        """For the run record: every request, what the agent saw, and what it did not."""
        with self.lock:
            # dict() copies in one step: a request's thread may add a key at the same time (an operator stop
            # records brokers whose requests are still ending).
            jobs = [{k: v for k, v in dict(j).items() if k != "thread" and not k.startswith("_")}
                    for j in list(self.jobs.values())]
        return {"config": self.task.services, "left": self.budget(), "requests": jobs,
                "refusals": [dict(r) for r in list(self.refusals)]}


__all__ = ["Broker", "SERVICES"]
