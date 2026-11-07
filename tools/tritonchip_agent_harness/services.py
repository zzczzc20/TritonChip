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
"""

from __future__ import annotations

import json
import secrets
import shutil
import socketserver
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import board, evaluator, fpga_quick, fpga_slow, provider
from .siteconfig import Site
from .task import Task
from .workspace import apply_candidate, export, gate

SERVICES = ("score", "fpga-quick", "fpga-slow")
# What of an FPGA result the agent may see: all of it is its own design's.
FPGA_VISIBLE = ("status", "profile", "period_ns", "part", "wns_ns", "tns_ns", "failing_endpoints",
                "total_endpoints", "fmax_est_mhz", "utilization", "worst_paths", "errors", "seconds")
FPGA_NOTE = ("post-synthesis estimate: delays leave out most of the wire, so the frequency is optimistic; "
             "the routed design is the slow tier's to judge")


class Shared:
    """What a run's islands share (report sections 8.9 B7/B8 and 8.8 P10): the slow tier's counts and the
    images it built, one slow request at a time, the cap on evaluations at once -- and one lock for all of
    their brokers' state. A run without islands has one of its own."""

    def __init__(self, site: Site, task: Task) -> None:
        fs = (task.services or {}).get("fpga_slow") or {}
        self.flows_left = int(fs.get("flows", 0))
        self.board_runs_left = int(fs.get("board_runs", 0))
        self.images: dict = {}
        # The run's Vivado flows running now (the hardware requests holding them): at most flows_at_once,
        # each on a build machine of its own (fpga_slow.place); one by default (B8). Card runs are serial on
        # the card host anyway (its lease and idle wait).
        self.flows: set = set()
        self.flows_at_once = int(fs.get("flows_at_once", 1))
        # The quick tier's worker is one shared machine: one job at a time across islands (D8a), queued.
        self.quick = threading.BoundedSemaphore(1)
        self.lock = threading.Lock()
        n = getattr(site, "evaluations", None)
        self.evaluations = threading.BoundedSemaphore(int(n)) if n else None


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
        self.fpga_left = float(fq.get("minutes", 0)) * 60.0
        self.fpga_job = float(fq.get("job_minutes", 30)) * 60.0
        self.period = float(fq.get("period_ns", 10.0))
        fs = cfg.get("fpga_slow") or {}
        self.slow_mode = fs.get("mode", "board")
        self.flow_seconds = int(float(fs.get("flow_minutes", 480)) * 60)
        self.board_seconds = int(float(fs.get("board_minutes", 120)) * 60)
        # The slow tier's counts and images, and the lock, are the run's (islands share them).
        self.shared = shared or Shared(site, task)
        self.island, self.archive = island, archive
        self.runners: dict = runners or {"score": self._run_score, "fpga-quick": self._run_fpga,
                                         "fpga-slow": self._run_slow}
        self.jobs: dict = {}
        self.busy: dict = {}
        self.lock = self.shared.lock
        self.socket_dir: Optional[Path] = None
        self._server = None

    # The slow tier's counts and images live in Shared: one pool for a run's islands.
    flows_left = property(lambda self: self.shared.flows_left,
                          lambda self, v: setattr(self.shared, "flows_left", v))
    board_runs_left = property(lambda self: self.shared.board_runs_left,
                               lambda self, v: setattr(self.shared, "board_runs_left", v))
    images = property(lambda self: self.shared.images)

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
                if job["service"] == "fpga-slow" and job["state"] == "running" and job.get("waiting_for_build_host"):
                    # It never got a build machine: cancelled as a queued one is; its charge goes back when its
                    # thread ends (the flow did not run).
                    job["_cancel_requested"] = True
                    job["state"] = "cancelled"
                    job["note"] = "waiting for a free build machine when the agent stopped; not built, not charged"
                    self._release_flow(job)
                    stopped.append(job)
                    continue
                if job["service"] == "fpga-slow" and job["state"] == "running":
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
            elif job["service"] == "fpga-slow" and getattr(self.site, "fpga_pipeline", None):
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
                return {"refused": refused, **self.budget()}
            n = len(self.jobs) + 1
            job_id = f"{'s' if service == 'score' else 'f' if service == 'fpga-quick' else 'l'}{n}"
            job_dir = self.dir / job_id
            self.busy.setdefault(service, []).append(job_id)
        try:
            return self._submit(service, message, job_id, job_dir)
        except Exception:
            # Whatever failed after the reservation (a gate on a broken tree, a description that does not
            # parse) gives it back: one bad snapshot never blocks a service for the rest of the run.
            with self.lock:
                self._unbusy(service, job_id)
            raise

    def _unbusy(self, service: str, job_id: str) -> None:
        """Under the lock: the request no longer counts as one of the service's running ones."""
        running = self.busy.get(service) or []
        if job_id in running:
            running.remove(job_id)
        if not running:
            self.busy.pop(service, None)

    def _at_once(self, service: str) -> int:
        """How many of a service's requests may run at once: the run's flows at once for the slow tier, else one."""
        return max(1, self.shared.flows_at_once) if service == "fpga-slow" else 1

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
        job = {"id": job_id, "service": service, "state": "queued", "submitted": time.time(),
               "snapshot": snapshot, "containers": f"tnca-svc-{self.run_id}-{job_id}",
               "remote_id": f"tnca-{secrets.token_hex(8)}"}
        if self.island is not None:
            if service == "score":
                job["message"] = str(message or "")[:200]
                job["digest"] = fpga_slow.snapshot_digest(candidate)
                job["hardware"] = fpga_slow.hardware_digest(candidate)
                job["candidate"] = str(candidate) if candidate else None
        if service in ("fpga-quick", "fpga-slow"):
            job["hardware"] = fpga_slow.hardware_digest(candidate)
            job["digest"] = fpga_slow.snapshot_digest(candidate)
        if service == "fpga-slow":
            job["mode"] = self.slow_mode
            refused = self._slow_kind(job, candidate)
            if refused:
                with self.lock:
                    self._unbusy(service, job_id)
                return {"refused": refused, "snapshot": snapshot, **self.budget()}
        with self.lock:
            if service == "fpga-slow" and job["kind"] == "hardware":
                at_once = max(1, self.shared.flows_at_once)
                refused = None
                if len(self.shared.flows) >= at_once:
                    refused = ("another Vivado flow of this run is running; one at a time per run" if at_once == 1
                               else f"{at_once} Vivado flows of this run are running; at most {at_once} at once")
                elif any(j.get("hardware") == job["hardware"] and j.get("kind") == "hardware"
                         and j["state"] in ("queued", "running") for j in self.jobs.values()):
                    # Two flows of the same machine would build one image twice.
                    refused = ("a flow of this run is already building exactly this hardware; ask again when it "
                               "is done, and it will need no flow")
                if refused:
                    self._unbusy(service, job_id)
                    return {"refused": refused, "snapshot": snapshot, **self.budget()}
                self.shared.flows.add(f"{self.run_id}:{job_id}")
            if service == "score":
                self.score_left -= 1
            elif service == "fpga-slow":
                if job["charge"] == "flow":
                    self.flows_left -= 1
                else:
                    self.board_runs_left -= 1
            self.jobs[job_id] = job
        job["thread"] = threading.Thread(target=self._run, args=(job, candidate, job_dir), daemon=True)
        job["thread"].start()
        return {"id": job_id, "state": "queued", "snapshot": snapshot, **self.budget()}

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
        with self.lock:
            quick = [j for j in self.jobs.values()
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
            visible, full = self.runners[job["service"]](job, candidate, job_dir)
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
                     if job["service"] == "fpga-slow" else error)
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
        slots = self.shared.evaluations
        queued = time.time()
        if slots is not None:
            slots.acquire()
        job["eval_queued_seconds"] = round(time.time() - queued, 1)
        if job.get("state", "running") != "running":
            # Cancelled while it waited for a slot (its island stopped): nothing is measured or archived.
            if slots is not None:
                slots.release()
            return {"status": "cancelled"}, {"cancelled_waiting": True}
        try:
            r = evaluator.measure_validation(self.site, self.repo, self.task, candidate, job_dir, job["containers"])
        finally:
            if slots is not None:
                slots.release()
        visible = {"validation": r["aggregate"]}
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

    def _run_fpga(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        tree = job_dir / "device-tree"
        if candidate is None:
            export(self.repo, self.task.base_commit, (), tree)
        else:
            apply_candidate(self.repo, self.task.base_commit, candidate, tree, only=evaluator.HARDWARE)
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
                                          flow_seconds=self.flow_seconds, board_seconds=self.board_seconds)
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
        if job["service"] == "fpga-slow" and job["state"] == "running":
            # Never which machine: whether the flow waits for one, and once it runs, within how many minutes
            # from now the request ends at the latest (the flow's cap, then the card's in board mode).
            if job.get("waiting_for_build_host"):
                out["waiting_for_a_build_machine"] = True
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
            jobs = [{k: v for k, v in j.items() if k != "thread" and not k.startswith("_")}
                    for j in self.jobs.values()]
        return {"config": self.task.services, "left": self.budget(), "requests": jobs}


__all__ = ["Broker", "SERVICES"]
