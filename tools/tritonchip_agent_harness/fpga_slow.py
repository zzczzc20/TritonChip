"""In-run split-host FPGA link and optional validation-set board measurement.

The agent submits a snapshot. The trusted controller builds the same pinned
candidate/platform tree as final certification, gives the remote pipeline a
known run ID, and keeps the point-level board records outside the agent view.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import certify, evaluator, provider
from .siteconfig import Site
from .task import Task


class SlowError(RuntimeError):
    pass


def host_code_paths(candidate: Optional[Path]) -> list[str]:
    """Candidate paths that would run as host-side code during a board stage."""
    if candidate is None:
        return []
    taken = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))["taken"]
    return sorted(c["path"] for c in taken if not c["path"].startswith(evaluator.HARDWARE))


def manifest(task: Task, label: str, commit: str, *, board: bool) -> dict:
    value = certify.manifest(task, label, commit)
    if board:
        value["board"]["archbench"]["points"] = list(task.evaluation["validation"])
    else:
        value.pop("board")
    return value


def _pipeline_args(tree: Path, mpath: Path, output: Path, cfg: dict, remote_id: str,
                   *, board: bool, command_timeout: int) -> list[str]:
    cli = Path(__file__).resolve().parents[1] / "tritonchip_fpga_harness" / "cli.py"
    argv = [sys.executable, str(cli), "--repo", str(tree), "pipeline", str(mpath),
            "--hosts", str(cfg["hosts"]), "--build-host", cfg["build_host"],
            "--run-id", remote_id, "--timeout", str(command_timeout),
            "--poll", "30", "--output", str(output)]
    if board:
        argv += ["--board-host", cfg["board_host"]]
    else:
        argv += ["--skip-board"]
    return argv


def _terminate_local(job: dict) -> None:
    proc = job.get("_process")
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=8)


def cancel(job: dict, cfg: dict) -> dict:
    """Stop the local controller and reconcile any detached remote stages."""
    job["_cancel_requested"] = True
    # The simulation precheck may still be using a Docker container when the
    # agent stops or its slow-tier deadline expires.
    try:
        names = provider.docker("ps", "-a", "--format", "{{.Names}}", check=False, timeout=10).stdout.split()
        for name in names:
            if name == job["containers"] or name.startswith(job["containers"] + "-"):
                provider.docker("rm", "-f", name, check=False, timeout=15)
    except (provider.ProviderError, OSError, subprocess.SubprocessError):
        pass
    _terminate_local(job)
    commit = job.get("tree_commit")
    if not commit:
        return {"remote": "not-started"}
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    from tritonchip_fpga_harness.pipeline import remote

    result = {}
    try:
        hosts = load_hosts(cfg["hosts"])
    except (OSError, ValueError) as exc:
        return {"remote_error": f"{type(exc).__name__}: {exc}"}
    board_host = cfg.get("board_host") if job.get("mode") == "board" else None
    for role, name in (("build", cfg["build_host"]), ("board", board_host)):
        if not name:
            continue
        try:
            host = host_for(hosts, name, role)
            argv = [host.python, host.cli(commit), "--repo", host.checkout(commit),
                    "reconcile", job["remote_id"], "--state-dir", host.state_dir, "--terminate"]
            reply = remote(host, argv, check=False, timeout=90)
            result[role] = {"exit": reply.returncode,
                            "detail": reply.stdout.decode(errors="replace")[-500:]}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result[role] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def _simulated(task: Task, result: dict) -> dict:
    measured = (result.get("full") or {}).get("points") or {}
    return {p: {"passed": (measured.get(p) or {}).get("status") == "pass",
                "done_cycles": (measured.get(p) or {}).get("record_cycles")}
            for p in task.evaluation["validation"]}


def run(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path,
        job: dict, seconds: int, *, board: bool) -> tuple[dict, dict]:
    cfg = site.fpga_pipeline
    if not cfg:
        raise SlowError("this site has no FPGA pipeline")
    if board and not cfg.get("board_host"):
        raise SlowError("the FPGA pipeline has no board host")
    if board and host_code_paths(candidate) and not cfg.get("allow_candidate_host_code", False):
        raise SlowError("the card host is not configured to run unreviewed candidate host code")

    started = time.monotonic()
    job_dir.mkdir(parents=True, exist_ok=True)
    sim = None
    if board:
        expired = threading.Event()

        def stop_precheck() -> None:
            expired.set()
            cancel(job, cfg)

        timer = threading.Timer(seconds, stop_precheck)
        timer.daemon = True
        timer.start()
        try:
            sim = evaluator.measure_validation(site, repo, task, candidate, job_dir / "sim", job["containers"])
        finally:
            timer.cancel()
        if expired.is_set() or time.monotonic() - started >= seconds:
            return {"status": "TIMEOUT", "mode": "board"}, {"simulation": sim,
                   "error": "the slow-tier budget expired during simulation"}
        if job.get("_cancel_requested"):
            return {"status": "cancelled"}, {"simulation": sim}
    tree = job_dir / "tree"
    commit = certify.build_tree(repo, task, candidate, tree)
    job["tree_commit"] = commit
    m = manifest(task, job["id"], commit, board=board)
    mpath, output = job_dir / "manifest.json", job_dir / "pipeline.json"
    mpath.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if job.get("_cancel_requested"):
        return {"status": "cancelled"}, {"simulation": sim, "tree_commit": commit}

    remaining = int(seconds - (time.monotonic() - started))
    if remaining <= 0:
        return {"status": "TIMEOUT", "mode": "board" if board else "build"}, {
            "simulation": sim, "tree_commit": commit, "error": "the slow-tier budget expired before the pipeline started"}
    # The FPGA harness limits each remote command too; the enclosing budget
    # limits the complete pipeline, including sync, link, transfer and board.
    argv = _pipeline_args(tree, mpath, output, cfg, job["remote_id"], board=board,
                          command_timeout=remaining)
    with (job_dir / "pipeline.log").open("wb") as log:
        proc = subprocess.Popen(argv, cwd=tree, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        job["_process"] = proc
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            job["cleanup"] = cancel(job, cfg)
            status = "timeout"
        else:
            status = "finished"
    job.pop("_process", None)
    pipe = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "status": "PIPELINE_FAIL", "error": f"controller {status}; no pipeline record"}
    if status == "timeout":
        pipe["status"] = "TIMEOUT"
    if pipe.get("status") == "PIPELINE_FAIL":
        job["cleanup"] = cancel(job, cfg)

    build = pipe.get("build") or {}
    visible = {"status": pipe.get("status"), "mode": "board" if board else "build",
               "build": {k: build.get(k) for k in ("status", "observed_clock_mhz", "timing", "elapsed_s")}}
    full = {"pipeline": pipe, "tree_commit": commit, "manifest": m, "simulation": sim}
    if board:
        records = certify.fetch_board_records(cfg["hosts"], cfg["board_host"], pipe["run_id"]) \
            if pipe.get("board") and pipe.get("run_id") else []
        rows = certify.compare(_simulated(task, sim or {}), records)
        verdict, why = certify.verdict(pipe, rows, records)
        if pipe.get("status") != "PASS" and verdict == "certified":
            verdict, why = "board-fail", f"pipeline status {pipe.get('status')}"
        passed = sum(1 for row in rows if not row.get("note") and not row["problem"])
        visible["board"] = {"verdict": verdict, "validation_passed": passed,
                            "validation_n": len(task.evaluation["validation"])}
        full.update(board_records=records, comparison=rows, verdict=verdict, why=why)
    return visible, full


__all__ = ["SlowError", "cancel", "host_code_paths", "manifest", "run"]
