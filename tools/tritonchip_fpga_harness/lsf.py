"""A build host that is a batch cluster: the link runs as an LSF job.

A plain build host is one machine: the controller starts the link there over
SSH, detached, and asks that machine how it stands. A cluster is reached
through a login node that must not run Vivado itself; the work goes to a node
LSF picks. For a host with an ``lsf`` block (hosts.py) the controller therefore
does three things differently, and only these::

    submit   writes the manifest and a job script into the host's state
             directory and hands the script to ``bsub``; the job is named
             after the run, so a submit repeated after a lost answer finds
             the job the first one queued
    wait     asks ``bjobs`` until the job has ended (PEND and RUN are live)
    kill     ``bkill`` by the job's name

The job script runs the harness of the commit's checkout exactly as the
detached worker would (``submit --skip-board``), in the foreground, on the
node. That checkout is the task's pinned platform code and is not changed: it
judges whether a run is alive by the process and boot identity of the machine
it is asked on, so ``status`` on the login node would declare a run on another
node lost, and write that down. The controller never asks it while the job
lives. Once LSF says the job has ended the answer is the run's own record,
read on the login node from the file system the nodes share: the verdict of a
finished run, or ORPHANED for one whose job was killed under it.

Everything else a split run does on a build host -- sync, doctor, export, the
packet's transfer, the reports -- is plain file work and stays on the login
node.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any, Callable

from .hosts import Host
from .worker import check_run_id

# LSF's states for a job that is not over. Anything else -- DONE, EXIT, or a
# job LSF no longer lists -- ends the wait, and the run's record is read.
LIVE = ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT", "PROV", "UNKWN")
MAX_POLL_FAILURES = 10
# Sources LSF's profile, runs one command, and always exits 0 with the
# command's own exit status on the last line: an SSH failure (255) stays
# distinct from a bjobs that found nothing.
_RUN = '. "$1" >/dev/null 2>&1; shift; "$@"; echo "LSF_EXIT=$?"'
_JOB = re.compile(r"Job <(\d+)>")


class LsfError(RuntimeError):
    """The cluster did not take, or could not be asked about, a job."""


def _call(host: Host, argv: list[str], *, ssh_binary: str, timeout: int = 120) -> tuple[int, str]:
    """One LSF command on the login node: (its exit status, its output)."""
    from .pipeline import remote
    assert host.lsf is not None
    done = remote(host, ["sh", "-c", _RUN, "sh", host.lsf["profile"], *argv], ssh_binary=ssh_binary,
                  timeout=timeout, check=False)
    text = done.stdout.decode("utf-8", errors="replace")
    lines = text.strip().splitlines()
    if done.returncode != 0 or not lines or not lines[-1].startswith("LSF_EXIT="):
        detail = done.stderr.decode("utf-8", errors="replace").strip().splitlines()[-2:]
        raise LsfError(f"{host.name}: {argv[0]} could not be run ({done.returncode}): {' | '.join(detail)}")
    return int(lines[-1].split("=", 1)[1]), "\n".join(lines[:-1])


def paths(host: Host, run_id: str) -> dict[str, str]:
    """Where a run's job keeps its files on the cluster: beside the detached worker's own requests."""
    base = f"{host.state_dir}/requests/{check_run_id(run_id)}.lsf"
    return {"manifest": base + ".manifest.json", "script": base + ".sh", "log": base + ".log"}


def job(host: Host, run_id: str, *, ssh_binary: str = "ssh") -> dict[str, Any] | None:
    """The latest job named after ``run_id``: {job_id, stat, exec_host}, or None when LSF lists none."""
    _code, text = _call(host, ["bjobs", "-a", "-noheader", "-o", "jobid stat exec_host", "-J", check_run_id(run_id)],
                        ssh_binary=ssh_binary)
    found = None
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 2 and words[0].isdigit():
            node = words[2] if len(words) > 2 and words[2] != "-" else None
            # `8*cpu10`: the slots and the node.
            entry = {"job_id": int(words[0]), "stat": words[1], "exec_host": node.split("*")[-1] if node else None}
            if found is None or entry["job_id"] > found["job_id"]:
                found = entry
    return found


def script(host: Host, commit: str, run_id: str, adapter: str, timeout: int | None) -> str:
    """The job: the checkout's own ``submit --skip-board`` in the foreground, with the host's environment.

    Ended by LSF (bkill, a run limit), it first has the checkout stop the
    vendor tools it started -- on this node, where their process group is.
    """
    q = shlex.quote
    files = paths(host, run_id)
    checkout, state = host.checkout(commit), host.state_dir
    cli = [host.python, host.cli(commit), "--repo", checkout]
    submit = cli + ["submit", files["manifest"], "--state-dir", state, "--run-id", run_id, "--adapter", adapter,
                    "--skip-board"] + (["--timeout", str(timeout)] if timeout else [])
    stop = cli + ["reconcile", run_id, "--state-dir", state, "--terminate"]
    lines = ["#!/bin/sh"]
    lines += [f"export {key}={q(value)}" for key, value in sorted(host.remote_env().items())]
    lines += [
        f"cd {q(checkout)} || exit 97",
        " ".join(q(word) for word in submit) + " &",
        "child=$!",
        "stop() { " + " ".join(q(word) for word in stop) + ' >/dev/null 2>&1; kill -TERM "$child" 2>/dev/null; }',
        "trap stop TERM INT",
        'wait "$child"; code=$?',
        # A trapped signal ends the first wait early; the child's own end is waited for.
        'if [ "$code" -gt 128 ]; then wait "$child"; code=$?; fi',
        'exit "$code"',
    ]
    return "\n".join(lines) + "\n"


def submit(host: Host, commit: str, request: dict[str, Any], run_id: str, *, timeout: int | None = None,
           ssh_binary: str = "ssh") -> dict[str, Any]:
    """Queue the link of ``run_id``; a job already named after it is attached to, never doubled."""
    from .pipeline import remote
    assert host.lsf is not None
    check_run_id(run_id)
    present = job(host, run_id, ssh_binary=ssh_binary)
    if present:
        return {"run_id": run_id, "attached": True, **present}
    files = paths(host, run_id)
    remote(host, ["mkdir", "-p", f"{host.state_dir}/requests"], ssh_binary=ssh_binary)
    manifest = json.dumps(request["manifest"], indent=2, sort_keys=True) + "\n"
    remote(host, ["dd", f"of={files['manifest']}"], ssh_binary=ssh_binary, data=manifest.encode("utf-8"))
    remote(host, ["dd", f"of={files['script']}"], ssh_binary=ssh_binary,
           data=script(host, commit, run_id, request["adapter"], timeout).encode("utf-8"))
    code, text = _call(host, ["bsub", "-q", host.lsf["queue"], "-n", str(host.lsf["slots"]), "-R", "span[hosts=1]",
                              "-J", run_id, "-o", files["log"], "sh", files["script"]], ssh_binary=ssh_binary)
    found = _JOB.search(text)
    if code != 0 or not found:
        raise LsfError(f"{host.name}: bsub did not take the job ({code}): {text.strip()[-300:]}")
    return {"run_id": run_id, "attached": False, "job_id": int(found.group(1)), "stat": "PEND", "exec_host": None,
            "queue": host.lsf["queue"], "slots": host.lsf["slots"]}


def queue(host: Host, name: str, script_path: str, log_path: str, *, slots: int | None = None,
          ssh_binary: str = "ssh") -> dict[str, Any]:
    """Queue a job of the controller's own (a script already on the cluster) under ``name``, as ``submit`` does
    a link: a job already named so is attached to, never doubled. For work after a link that must not run on
    the login node either, such as reading the run's checkpoint in Vivado (agent harness, congestion.py)."""
    assert host.lsf is not None
    check_run_id(name)
    present = job(host, name, ssh_binary=ssh_binary)
    if present:
        return {"name": name, "attached": True, **present}
    n = int(slots or host.lsf["slots"])
    code, text = _call(host, ["bsub", "-q", host.lsf["queue"], "-n", str(n), "-R", "span[hosts=1]",
                              "-J", name, "-o", log_path, "sh", script_path], ssh_binary=ssh_binary)
    found = _JOB.search(text)
    if code != 0 or not found:
        raise LsfError(f"{host.name}: bsub did not take the job ({code}): {text.strip()[-300:]}")
    return {"name": name, "attached": False, "job_id": int(found.group(1)), "stat": "PEND", "slots": n}


def wait(host: Host, run_id: str, *, poll_s: float = 60.0, deadline: float | None = None, ssh_binary: str = "ssh",
         sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Ask LSF until the job of ``run_id`` has ended: {job_id, stat, exec_host, pend_s, run_s}.

    ``stat`` is LSF's last word, or GONE for a job it no longer lists (one
    that ended long ago, when a build is attached to again).
    """
    started = time.time()
    running_since = None
    failures = 0
    last: dict[str, Any] = {}
    seen = None
    while True:
        try:
            current = job(host, run_id, ssh_binary=ssh_binary)
            failures = 0
        except LsfError as exc:
            failures += 1
            if log:
                log(f"lsf poll on {host.name} failed ({failures}/{MAX_POLL_FAILURES}): {exc}")
            if failures >= MAX_POLL_FAILURES:
                raise LsfError(f"{host.name} did not answer {MAX_POLL_FAILURES} job polls in a row; "
                               f"the job of {run_id} may still be running there")
            current = last or {"stat": "UNKWN"}
        else:
            if current is None:
                current = {**last, "stat": "GONE"}
        if current.get("stat") == "RUN" and running_since is None:
            running_since = time.time()
        if log and (current.get("stat"), current.get("exec_host")) != seen:
            seen = (current.get("stat"), current.get("exec_host"))
            log(f"lsf job {current.get('job_id')} of {run_id} on {host.name}: {seen[0]}"
                + (f" on {seen[1]}" if seen[1] else ""))
        last = {**last, **{k: v for k, v in current.items() if v is not None}}
        if last.get("stat") not in LIVE:
            now = time.time()
            return {"job_id": last.get("job_id"), "stat": last.get("stat"), "exec_host": last.get("exec_host"),
                    "pend_s": round((running_since or now) - started, 1),
                    "run_s": round(now - running_since, 1) if running_since else 0.0}
        if deadline is not None and time.time() > deadline:
            raise LsfError(f"the job of {run_id} on {host.name} was still {last.get('stat')} at the controller's deadline")
        sleep(poll_s)


def kill(host: Host, run_id: str, *, ssh_binary: str = "ssh") -> dict[str, Any]:
    """End every job named after ``run_id`` (the job script stops the vendor tools on its node)."""
    code, text = _call(host, ["bkill", "-J", check_run_id(run_id), "0"], ssh_binary=ssh_binary)
    return {"exit": code, "detail": text.strip()[-300:]}


def log_tail(host: Host, run_id: str, *, lines: int = 30, ssh_binary: str = "ssh") -> list[str]:
    """The end of the job's own output: what a job that never wrote a run record said."""
    from .pipeline import remote
    done = remote(host, ["tail", "-n", str(lines), paths(host, run_id)["log"]], ssh_binary=ssh_binary, check=False)
    return done.stdout.decode("utf-8", errors="replace").splitlines() if done.returncode == 0 else []


__all__ = ["LIVE", "LsfError", "job", "kill", "log_tail", "paths", "queue", "script", "submit", "wait"]
