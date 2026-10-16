"""In-run services: the agent asks for a measurement, within its budget, and the harness makes it.

The broker runs in the harness process, beside the agent's run. It listens on
a Unix socket in a directory that the agent's container mounts read-only;
`tnca-service` (service_cli.py, mounted as one read-only file) is the agent's
side. One JSON line in, one JSON line out.

    submit score         the hidden validation set, in simulation, measured the
                         way the final evaluator measures (P2.1 included); the
                         agent sees an aggregate only (README section 6.4)
    submit fpga-quick    out-of-context synthesis and timing of the device on
                         the build worker (fpga_quick.py); the agent sees its
                         own design's timing and utilisation
    status, result, budget

A request is a snapshot: the same gate as the final evaluation, so only
changes under the design paths count, and a rejected snapshot or RTL that
fails the check is refused before anything is charged. Budgets: a number of
scoring requests, charged when one is accepted; FPGA minutes, charged as the
worker's actual time, with one FPGA request at a time and none accepted that
could run past what is left. Everything asked and answered, and the full
results the agent does not see, go into the run record.
"""

from __future__ import annotations

import json
import shutil
import socketserver
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import evaluator, fpga_quick, provider
from .siteconfig import Site
from .task import Task
from .workspace import apply_candidate, export, gate

SERVICES = ("score", "fpga-quick")
# What of an FPGA result the agent may see: all of it is its own design's.
FPGA_VISIBLE = ("status", "profile", "period_ns", "part", "wns_ns", "tns_ns", "failing_endpoints",
                "total_endpoints", "fmax_est_mhz", "utilization", "worst_paths", "errors", "seconds")
FPGA_NOTE = ("post-synthesis estimate: delays leave out most of the wire, so the frequency is optimistic; "
             "the routed design is the slow tier's to judge")


class Broker:
    def __init__(self, site: Site, repo: Path, task: Task, workspace: Path, run_dir: Path,
                 runners: Optional[dict] = None) -> None:
        self.site, self.repo, self.task, self.workspace = site, repo, task, workspace
        self.run_id = run_dir.name
        self.dir = run_dir / "services"
        cfg = task.services or {}
        self.score_left = int((cfg.get("score") or {}).get("requests", 0))
        fq = cfg.get("fpga_quick") or {}
        self.fpga_left = float(fq.get("minutes", 0)) * 60.0
        self.fpga_job = float(fq.get("job_minutes", 30)) * 60.0
        self.period = float(fq.get("period_ns", 10.0))
        self.runners: dict = runners or {"score": self._run_score, "fpga-quick": self._run_fpga}
        self.jobs: dict = {}
        self.busy: dict = {}
        self.lock = threading.Lock()
        self.socket_dir: Optional[Path] = None
        self._server = None

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
        """At the end of the run: no new requests, and what is still running is stopped and charged."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self.lock:
            running = [j for j in self.jobs.values() if j["state"] in ("queued", "running")]
        for job in running:
            job["state"] = "cancelled"
            job["note"] = "still running when the agent stopped; its result is not kept"
            if job["service"] == "fpga-quick" and self.site.fpga_worker:
                fpga_quick.cancel(self.site.fpga_worker, job["remote_id"])
            else:
                for name in provider.docker("ps", "-a", "--format", "{{.Names}}", check=False).stdout.split():
                    if name.startswith(job["containers"]):
                        provider.remove(name)
        for job in running:
            if job.get("thread"):
                job["thread"].join(timeout=180)
        if self.socket_dir is not None:
            shutil.rmtree(self.socket_dir, ignore_errors=True)

    # -- the protocol ----------------------------------------------------------------
    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "budget":
            return self.budget()
        if op == "submit":
            return self.submit(request.get("service"))
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
                "busy": sorted(self.busy)}

    def submit(self, service) -> dict:
        if service not in SERVICES:
            return {"error": f"no service {service!r}; there are: {', '.join(SERVICES)}"}
        with self.lock:
            refused = self._refusal(service)
            if refused:
                return {"refused": refused, **self.budget()}
            n = len(self.jobs) + 1
            job_id = f"{'s' if service == 'score' else 'f'}{n}"
            job_dir = self.dir / job_id
            self.busy[service] = job_id
        g = gate(self.repo, self.task.base_commit, self.task.hide, self.task.design, self.workspace,
                 out=job_dir / "candidate", evaluator_owned=self.task.evaluator_owned)
        candidate = job_dir / "candidate" if g["taken"] else None
        problems = [] if g["verdict"] != "PASS" else evaluator.rtl_check(candidate)
        snapshot = {"taken": [c["path"] for c in g["taken"]],
                    "dropped": [c["path"] for c in g["dropped"]][:50]}
        if g["verdict"] != "PASS" or problems:
            with self.lock:
                self.busy.pop(service, None)
            return {"refused": "the snapshot cannot be measured", "rejected": g["rejected"][:20] or problems,
                    "snapshot": snapshot, **self.budget()}
        job = {"id": job_id, "service": service, "state": "queued", "submitted": time.time(),
               "snapshot": snapshot, "containers": f"tnca-svc-{self.run_id}-{job_id}",
               "remote_id": f"tnca-fpga-{self.run_id}-{job_id}"}
        with self.lock:
            if service == "score":
                self.score_left -= 1
            self.jobs[job_id] = job
        job["thread"] = threading.Thread(target=self._run, args=(job, candidate, job_dir), daemon=True)
        job["thread"].start()
        return {"id": job_id, "state": "queued", "snapshot": snapshot, **self.budget()}

    def _refusal(self, service: str) -> Optional[str]:
        if service in self.busy:
            return f"a {service} request is already running ({self.busy[service]}); one at a time"
        if service == "score" and self.score_left <= 0:
            return "no scoring requests left"
        if service == "fpga-quick":
            if not self.site.fpga_worker:
                return "this site has no FPGA build worker"
            if self.fpga_left < self.fpga_job:
                return (f"an FPGA request may take up to {self.fpga_job / 60:.0f} min and "
                        f"{self.fpga_left / 60:.1f} min of FPGA time is left")
        return None

    def _run(self, job: dict, candidate: Optional[Path], job_dir: Path) -> None:
        with self.lock:
            if job["state"] != "queued":  # cancelled before it started
                self.busy.pop(job["service"], None)
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
            if job["state"] == "running":
                job["state"], job["result"] = "failed", error
            else:
                job["full"] = error
        finally:
            job["finished"] = time.time()
            with self.lock:
                self.busy.pop(job["service"], None)

    def _run_score(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        r = evaluator.measure_validation(self.site, self.repo, self.task, candidate, job_dir, job["containers"])
        visible = {"validation": r["aggregate"]}
        if r.get("device_build_errors"):
            visible["device_build_errors"] = r["device_build_errors"]
        return visible, r

    def _run_fpga(self, job: dict, candidate: Optional[Path], job_dir: Path) -> tuple:
        tree = job_dir / "device-tree"
        if candidate is None:
            export(self.repo, self.task.base_commit, (), tree)
        else:
            apply_candidate(self.repo, self.task.base_commit, candidate, tree, only=evaluator.HARDWARE)
        seconds = int(min(self.fpga_job, self.fpga_left))
        try:
            r = fpga_quick.run(self.site.fpga_worker, tree, self.task.evaluation["profile"], self.period, seconds,
                               job_dir, job["remote_id"])
        finally:
            shutil.rmtree(tree, ignore_errors=True)
        with self.lock:
            job["charged_seconds"] = r.get("seconds", 0.0)
            self.fpga_left = max(0.0, self.fpga_left - job["charged_seconds"])
        visible = {k: r[k] for k in FPGA_VISIBLE if k in r}
        visible["note"] = FPGA_NOTE
        return visible, r

    # -- views ---------------------------------------------------------------------------
    def _summary(self, job: dict) -> dict:
        return {"id": job["id"], "service": job["service"], "state": job["state"],
                "seconds": round((job.get("finished") or time.time()) - job["submitted"], 1)}

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
            jobs = [{k: v for k, v in j.items() if k not in ("thread",)} for j in self.jobs.values()]
        return {"config": self.task.services, "left": self.budget(), "requests": jobs}


__all__ = ["Broker", "SERVICES"]
