"""Server occupancy: which machine each part of a run held, from when to when.

Wall clock, one entry per segment of a run (decided 2026-10-28):

    sandbox            the agent's container, from `docker run` to its removal
    score              one scoring request, on the evaluation containers
    final-evaluation   the evaluation after the agent stops, both sets together
    fpga-quick         one quick-tier request, on the build worker
    fpga-slow-build    a slow-tier request's link: its pipeline's `build:submit` and `build` steps
    fpga-slow-board    its board stage on the card host: `board:measure` and `board`
    certify-link       the same two stages of certification's own pipeline, in
    certify-board      certification.json rather than run.json

The pipeline's other steps -- sync, doctor, export, transfer -- belong to
neither stage. A slow-tier request or a certification whose pipeline left no
step timing (it was stopped before it wrote its record, or never reached the
build) is one entry for the whole of it, `fpga-slow` or `certify`, and its
`covers` says what it spans.

`host` is the operator's label from the site configuration -- the worker's ssh
alias, the pipeline's host names -- or the hosts certify was given; never a name
looked up at run time, and null where none is given, as for the machine the
harness and its containers run on. `queued_seconds` runs from a request's
submission to its start. `reserved` is what the harness itself sets for the
role: CPUs from container limits or the job's threads, one card for a board
stage; null where it sets nothing. What a role actually used, its CPU time, is
not measured yet.
"""

from __future__ import annotations

from typing import Optional

from . import evaluator, fpga_quick

BUILD_STEPS = ("build:submit", "build")
BOARD_STEPS = ("board:measure", "board")
WHOLE_SLOW = "the whole request, from its start to its end: no pipeline record timed a build or board stage"
WHOLE_CERTIFY = "the whole pipeline, link and board stage together: its record timed no step"


def entry(role: str, host: Optional[str], started: Optional[float], finished: Optional[float], *,
          queued: float = 0.0, reserved: Optional[dict] = None, request_id: Optional[str] = None,
          covers: Optional[str] = None) -> dict:
    e = {"role": role, "host": host, "started": started, "finished": finished,
         "seconds": None if started is None or finished is None else round(finished - started, 1),
         "queued_seconds": round(max(0.0, queued), 1), "reserved": reserved}
    if request_id is not None:
        e["id"] = request_id
    if covers is not None:
        e["covers"] = covers
    return e


def stages(record: Optional[dict], build_host: Optional[str], board_host: Optional[str], roles: tuple,
           request_id: Optional[str] = None) -> list:
    """The link and the board stage, from the FPGA harness pipeline record's own step timing."""
    steps = {s.get("name"): s for s in (record or {}).get("steps") or [] if s.get("started_at") is not None}
    out = []
    for role, host, names, reserved in ((roles[0], build_host, BUILD_STEPS, None),
                                        (roles[1], board_host, BOARD_STEPS, {"cards": 1})):
        got = [steps[n] for n in names if n in steps]
        if got:
            out.append(entry(role, host, min(s["started_at"] for s in got),
                             max(s["started_at"] + (s.get("elapsed_s") or 0.0) for s in got),
                             reserved=reserved, request_id=request_id))
    return out


def services(site, task, requests) -> list:
    """One entry per service request that started; a slow-tier one in two, where its pipeline says how."""
    out = []
    for job in requests:
        if job.get("started") is None:
            continue  # cancelled before it started: it held nothing
        times = (job["started"], job.get("finished"))
        common = {"queued": job["started"] - job["submitted"], "request_id": job["id"]}
        if job["service"] == "score":
            out.append(entry("score", None, *times, **common,
                             reserved=evaluator.reserved(site, task, task.evaluation["validation"])))
        elif job["service"] == "fpga-quick":
            worker = site.fpga_worker or {}
            out.append(entry("fpga-quick", worker.get("ssh"), *times, **common,
                             reserved={"cpus": fpga_quick.threads(worker)}))
        else:
            cfg = site.fpga_pipeline or {}
            out += (stages((job.get("full") or {}).get("pipeline"), cfg.get("build_host"), cfg.get("board_host"),
                           ("fpga-slow-build", "fpga-slow-board"), job["id"])
                    or [entry("fpga-slow", None, *times, **common, covers=WHOLE_SLOW)])
    return out


def of_run(site, task, agent_result: dict, evaluated: Optional[tuple]) -> list:
    """run.json's `occupancy`: the sandbox, every service request, and the final evaluation."""
    out = []
    box = agent_result.get("container")
    if box:
        out.append(entry("sandbox", None, box["started"], box["finished"], reserved={"cpus": site.cpus}))
    out += services(site, task, (agent_result.get("services") or {}).get("requests") or [])
    if evaluated:
        points = [*task.evaluation["validation"], *task.evaluation["final"]]
        out.append(entry("final-evaluation", None, *evaluated, reserved=evaluator.reserved(site, task, points)))
    return out


__all__ = ["BOARD_STEPS", "BUILD_STEPS", "entry", "of_run", "services", "stages"]
