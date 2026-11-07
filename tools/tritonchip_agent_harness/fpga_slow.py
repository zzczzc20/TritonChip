"""The slow tier: an agent's request for the full link and the card (report section 8.9, decided 2026-11-06/07).

The agent asks; the request is asynchronous, and the agent goes on iterating while it runs. Two kinds:

    hardware   the snapshot changed the machine (RTL, the scored profile, the device description) and
               no earlier request of this run built that hardware: the FPGA harness links it on the
               build host (one Vivado flow), and in board mode its points then run on the card on the
               image it built. Charged to the run's flows (`services.fpga_slow.flows`).
    software   the machine is the base's, or this run already built exactly this hardware: no Vivado,
               only the card, on the base's image or that one. Charged to the board runs (`board_runs`).

Gate, before anything is charged for good: the snapshot's validation set must pass in simulation
(measured here, the score service's way), and a hardware request needs an fpga-quick result of
this run on the same hardware that met timing. A refused request is not charged.

Charging: a flow that failed on the candidate's own design (timing or place-and-route did not
converge, it does not fit, DRC, a lowered clock, the card refused the image) counts; one that
failed on the infrastructure (a host, the licence, a disk, the harness) does not, and is retried
by itself. The classification reads the build record's own checks, never the agent's account.

What the agent sees: on a failed flow, the quick tier's fields plus what only the routed design has
(the failing stage, hold slack, utilisation per SLR, whether the worst paths cross an SLR, the DRC
rules that failed), host paths and names scrubbed, capped; on the card, the validation set's
aggregate against the base on the card (the score service's rule), how many points counted on the
card what simulation counted, and each board check that did not pass. Point records stay in the run
record. No candidate code runs on the card host (board.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import board, certify, evaluator, provider
from .siteconfig import Site
from .task import Task

RETRIES = 2                # an infrastructure failure is tried again this many times
REPORT_BYTES = 4 << 20     # the most of one Vivado report fetched from the build host
PATHS = 8                  # worst paths returned
DRC_ROWS = 10
# Vivado's own messages for a design that does not place, route, fit, synthesise or pass DRC. Generic ids
# (`Common 17-69`, a VPL step's failure) also follow licence and disk failures, so they are not here.
_DESIGN_ERROR = re.compile(r"ERROR: \[(?:Place|Route|Synth|DRC|Opt|Phys|Timing|Constraints) \d+-\d+\]")
_STAGES = (("write_bitstream", "bitstream"), ("route_design", "route"), ("phys_opt", "phys_opt"),
           ("place_design", "place"), ("opt_design", "opt"), ("synth", "synthesis"))


class SlowError(RuntimeError):
    pass


def hardware_digest(candidate: Optional[Path]) -> str:
    """The snapshot's hardware, as one hash: the files under evaluator.HARDWARE it takes, and their bytes."""
    h = hashlib.sha256()
    if candidate is None:
        return "base"
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    for c in sorted(report["taken"], key=lambda c: c["path"]):
        if not c["path"].startswith(evaluator.HARDWARE):
            continue
        h.update(f"{c['path']}\0{c['status']}\0".encode())
        f = candidate / "files" / c["path"]
        if c["status"] != "deleted" and f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def snapshot_digest(candidate: Optional[Path]) -> str:
    """The whole snapshot (every file the gate took), as one hash: what a post-run certification matches an
    in-run board record by."""
    h = hashlib.sha256()
    if candidate is None:
        return "base"
    report = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    for c in sorted(report["taken"], key=lambda c: c["path"]):
        h.update(f"{c['path']}\0{c['status']}\0".encode())
        f = candidate / "files" / c["path"]
        if c["status"] != "deleted" and f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def manifest(task: Task, label: str, commit: str, *, board: bool) -> dict:
    value = certify.manifest(task, label, commit)
    if board:
        value["board"]["archbench"]["points"] = list(task.evaluation["validation"])
    else:
        value.pop("board")
    return value


def _pipeline_args(tree: Path, mpath: Path, output: Path, cfg: dict, remote_id: str, *, command_timeout: int) -> list:
    """The link alone: the card stage is board.py's, through the relay."""
    cli = Path(__file__).resolve().parents[1] / "tritonchip_fpga_harness" / "cli.py"
    return [sys.executable, str(cli), "--repo", str(tree), "pipeline", str(mpath),
            "--hosts", str(cfg["hosts"]), "--build-host", cfg["build_host"],
            "--run-id", remote_id, "--timeout", str(command_timeout),
            "--poll", "30", "--output", str(output), "--skip-board"]


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
    """Stop the local controller and reconcile the build host's detached stages."""
    job["_cancel_requested"] = True
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

    try:
        host = host_for(load_hosts(cfg["hosts"]), cfg["build_host"], "build")
        argv = [host.python, host.cli(commit), "--repo", host.checkout(commit),
                "reconcile", job["remote_id"], "--state-dir", host.state_dir, "--terminate"]
        reply = remote(host, argv, check=False, timeout=90)
        return {"build": {"exit": reply.returncode, "detail": reply.stdout.decode(errors="replace")[-500:]}}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"build": {"error": f"{type(exc).__name__}: {exc}"}}


# -- the build: one Vivado flow -------------------------------------------------------------------

def classify(pipe: dict, clock_mhz: Optional[float] = None) -> str:
    """'ok', 'design' (the candidate's: charged), 'infra' (not charged, tried again) or 'timeout' (not charged,
    not tried again), from the build record:
    the FPGA harness's own status, the routed timing it read, the clock it observed and its errors."""
    build = pipe.get("build") or {}
    status = build.get("status")
    if status == "PASS" and pipe.get("status") == "PASS" and build.get("xclbin_sha256"):
        return "ok"
    if pipe.get("status") == "TIMEOUT" or status == "TIMEOUT":
        # Ran past its cap: not a failure the spec names as the design's, nor one a retry would cure.
        return "timeout"
    if status == "TIMING_FAIL":
        return "design"
    if status != "BUILD_FAIL" or build.get("preflight_failed"):
        return "infra"
    metrics = ((build.get("timing") or {}).get("metrics")) or {}
    if any(isinstance(metrics.get(k), (int, float)) and metrics[k] < 0 for k in ("wns_ns", "tns_ns", "whs_ns", "ths_ns")):
        return "design"
    if clock_mhz is not None and build.get("observed_clock_mhz") not in (None, clock_mhz):
        return "design"   # the link lowered the clock
    text = "\n".join([str(build.get("error") or "")] + [str(x) for x in (build.get("log_tail") or [])])
    return "design" if _DESIGN_ERROR.search(text) else "infra"


def _host_words(site: Site) -> list:
    cfg = site.fpga_pipeline or {}
    words = [cfg.get("build_host"), cfg.get("board_host")]
    try:
        from tritonchip_fpga_harness.hosts import load_hosts
        for h in load_hosts(cfg["hosts"]).values():
            target = str(h.ssh)
            words += [h.name, target, target.split("@")[-1], target.split("@")[0] if "@" in target else None]
    except Exception:  # noqa: BLE001 -- the names above still go
        pass
    return sorted({w for w in words if w and len(w) > 2}, key=len, reverse=True)


def _scrub(text: str, site: Site, words: Optional[list] = None) -> str:
    text = re.sub(r"(?<![\w.])/(?:[\w.+-]+/)+([\w.+-]+)", r"\1", str(text))
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<address>", text)
    for name in words if words is not None else _host_words(site):
        text = text.replace(name, "<host>")
    return text[:300]


def _blocks(text: str) -> list:
    """Vivado's path reports, one block per path: from a `Slack` line to the next."""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.strip().startswith("Slack") and ("VIOLATED" in line or "MET" in line or "(" in line):
            if cur:
                blocks.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def worst_paths(text: str, n: int = PATHS) -> list:
    out = []
    for block in _blocks(text):
        row = {}
        for line in block:
            w = line.split()
            s = line.strip()
            if s.startswith("Slack") and len(w) >= 4:
                try:
                    row["slack_ns"] = float(w[3].rstrip("ns"))
                except ValueError:
                    pass
            elif s.startswith("Source:") and len(w) > 1:
                row.setdefault("start", w[1])
            elif s.startswith("Destination:") and len(w) > 1:
                row.setdefault("end", w[1])
            elif s.startswith("Data Path Delay:") and len(w) > 3:
                try:
                    row["delay_ns"] = float(w[3].rstrip("ns"))
                except ValueError:
                    pass
            elif s.startswith("Logic Levels:") and len(w) > 2:
                try:
                    row["levels"] = int(w[2])
                except ValueError:
                    pass
            elif s.startswith("Path Type:") and len(w) > 2:
                row["type"] = "hold" if "Min" in s else "setup"
        if "start" in row:
            row["crosses_slr"] = any("Inter-SLR Compensation" in line for line in block)
            out.append(row)
        if len(out) >= n:
            break
    return out


def slr_utilization(text: str) -> dict:
    """The utilisation report's per-SLR table: resource -> {SLR: percent}."""
    names = {"CLB LUTs": "lut", "CLB Registers": "ff", "Block RAM Tile": "bram", "URAM": "uram", "DSPs": "dsp",
             "CLB": "clb"}
    out, header = {}, None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")][1:-1]
        if not cells:
            if header and line.strip() == "":
                header = None
            continue
        if cells[0] == "Site Type" and any(re.fullmatch(r"SLR\d %", c) for c in cells):
            header = cells
            continue
        if header and cells[0].rstrip("*").strip() in names and len(cells) == len(header):
            key = names[cells[0].rstrip("*").strip()]
            row = {}
            for col, value in zip(header, cells):
                m = re.fullmatch(r"(SLR\d) %", col)
                if m:
                    try:
                        row[m.group(1)] = float(value)
                    except ValueError:
                        pass
            if row:
                out.setdefault(key, row)
    return out


def drc_rules(text: str) -> list:
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")][1:-1]
        if len(cells) >= 4 and cells[1] in ("Error", "Critical Warning"):
            try:
                count = int(cells[3])
            except ValueError:
                continue
            rows.append({"rule": cells[0], "severity": cells[1], "description": cells[2][:120], "violations": count})
    return rows[:DRC_ROWS]


def failing_stage(pipe: dict) -> Optional[str]:
    build = pipe.get("build") or {}
    text = "\n".join([str(build.get("error") or "")] + [str(x) for x in (build.get("log_tail") or [])])
    for needle, stage in _STAGES:
        if needle in text:
            return stage
    return None


def routed_report(site: Site, run_id: str) -> dict:
    """What only the routed design says, from the build host's reports for the run: worst paths (setup and
    hold, SLR crossings), utilisation per SLR, the DRC rules that failed."""
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    from tritonchip_fpga_harness.pipeline import remote
    cfg = site.fpga_pipeline
    host = host_for(load_hosts(cfg["hosts"]), cfg["build_host"], "build")
    found = remote(host, ["find", f"{host.state_dir}/runs/{run_id}", "-name", "*.rpt", "-size", "-64M"],
                   check=False, timeout=120)
    files = found.stdout.decode(errors="replace").split() if found.returncode == 0 else []

    def pick(*words):
        hits = [f for f in files if all(w in Path(f).name for w in words)]
        return sorted(hits, key=len)[0] if hits else None

    def read(path):
        if not path:
            return ""
        done = remote(host, ["head", "-c", str(REPORT_BYTES), path], check=False, timeout=180)
        return done.stdout.decode(errors="replace") if done.returncode == 0 else ""

    timing = read(pick("timing_summary", "routed") or pick("timing_summary"))
    util = read(pick("utilization", "placed") or pick("utilization", "routed") or pick("utilization"))
    drc = read(pick("drc", "routed") or pick("drc"))
    out = {}
    if timing:
        out["worst_paths"] = worst_paths(timing)
    if util:
        out["utilization_by_slr"] = slr_utilization(util)
    if drc:
        out["drc"] = drc_rules(drc)
    return out


def _build_visible(site: Site, pipe: dict, outcome: str, routed: dict) -> dict:
    build = pipe.get("build") or {}
    metrics = ((build.get("timing") or {}).get("metrics")) or {}
    visible = {"status": build.get("status") or pipe.get("status"),
               "observed_clock_mhz": build.get("observed_clock_mhz"), "elapsed_s": build.get("elapsed_s")}
    for key in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "failing_endpoints"):
        if key in metrics:
            visible[key] = metrics[key]
    if outcome == "design":
        words = _host_words(site)
        visible["stage"] = failing_stage(pipe)
        visible.update({k: v for k, v in routed.items() if v})
        if visible.get("drc"):
            visible["drc"] = [dict(r, description=_scrub(r["description"], site, words)) for r in visible["drc"]]
        for row in visible.get("worst_paths") or []:
            for key in ("start", "end"):
                if key in row:
                    row[key] = _scrub(row[key], site, words)[:160]
        if build.get("error"):
            visible["error"] = _scrub(build["error"], site, words)
        tail = [_scrub(x, site, words) for x in (build.get("log_tail") or []) if "ERROR" in str(x) or "CRITICAL" in str(x)]
        if tail:
            visible["errors"] = tail[-10:]
    return visible


def build(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path, job: dict,
          seconds: int) -> dict:
    """One Vivado flow, tried again on an infrastructure failure. {outcome, pipeline, visible, ...}."""
    cfg = site.fpga_pipeline
    tree = job_dir / "tree"
    commit = certify.build_tree(repo, task, candidate, tree)
    job["tree_commit"] = commit
    attempts = []
    for attempt in range(RETRIES + 1):
        remote_id = job["remote_id"] if attempt == 0 else f"{job['remote_id']}-r{attempt}"
        m = manifest(task, job["id"], commit, board=False)
        mpath, output = job_dir / f"manifest-{attempt}.json", job_dir / f"pipeline-{attempt}.json"
        mpath.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        argv = _pipeline_args(tree, mpath, output, cfg, remote_id, command_timeout=seconds)
        with (job_dir / f"pipeline-{attempt}.log").open("wb") as log:
            proc = subprocess.Popen(argv, cwd=tree, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            job["_process"] = proc
            try:
                proc.wait(timeout=seconds + 600)
                timed_out = False
            except subprocess.TimeoutExpired:
                job["cleanup"] = cancel(job, cfg)
                timed_out = True
        job.pop("_process", None)
        pipe = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
            "status": "PIPELINE_FAIL", "error": "the controller wrote no pipeline record"}
        if timed_out:
            pipe["status"] = "TIMEOUT"
        outcome = classify(pipe, (task.evaluation.get("certify") or {}).get("clock_mhz"))
        attempts.append({"remote_id": remote_id, "status": pipe.get("status"), "outcome": outcome})
        if outcome != "infra" or job.get("_cancel_requested"):
            break
    routed = routed_report(site, pipe.get("run_id") or remote_id) if outcome == "design" else {}
    return {"outcome": outcome, "pipeline": pipe, "attempts": attempts, "tree_commit": commit, "manifest": m,
            "routed": routed, "visible": _build_visible(site, pipe, outcome, routed),
            "run_id": pipe.get("run_id") or remote_id}


# -- the card ----------------------------------------------------------------------------------------

def _check_view(ran: dict) -> str:
    """A board check, as the agent sees it: whether it passed. Its log is the container's, so the
    candidate's to write: it stays in the run record."""
    if ran.get("exit") == 0:
        return "pass"
    return "did not finish" if ran.get("timed_out") or ran.get("exit") is None else "fail"


def _cycles_view(points: list, got: dict, sim_rows: dict) -> dict:
    """The card's cycle counts against simulation's, over the points that passed in both, for a task that
    records them (certify.board_cycles "record"): ratios in aggregate, never which point, as every view of the
    validation set is."""
    ratios = []
    for p in points:
        card, simulated = got.get(p) or {}, sim_rows.get(p) or {}
        o, s = (card.get("observer") or {}).get("done_cycles"), simulated.get("done_cycles")
        if card.get("status") == "pass" and simulated.get("failure") is None and o and s:
            ratios.append(o / s)
    if not ratios:
        return {"points": 0}
    ratios.sort()
    return {"points": len(ratios),
            "board_over_simulation_geomean": round(math.exp(sum(map(math.log, ratios)) / len(ratios)), 4),
            "board_over_simulation_range": [round(ratios[0], 4), round(ratios[-1], 4)]}


def on_card(site: Site, repo: Path, task: Task, candidate: Optional[Path], image: dict, sim: dict,
            job_dir: Path, job: dict, seconds: int, base_xclbin: Optional[str] = None) -> dict:
    """The validation points (and the task's board checks) on `image`, tried again on an infrastructure
    failure; the agent's view against the base on the card. outcome: ok, candidate (charged) or infra."""
    points = list(task.evaluation["validation"])
    rec, attempts = None, []
    for attempt in range(RETRIES + 1):
        try:
            rec = board.measure(site, repo, task, candidate, image, points, job_dir / f"card-{attempt}",
                                job["containers"], f"{job['remote_id']}-c{attempt}", seconds, base_xclbin=base_xclbin)
            outcome = board.status(rec)
        except Exception as exc:  # noqa: BLE001 -- a host, the forward, a parse: the infrastructure's
            rec, outcome = {"error": f"{type(exc).__name__}: {exc}"}, "infra"
        attempts.append({"outcome": outcome, "status": (rec.get("session") or {}).get("status"),
                         "error": rec.get("error")})
        if outcome != "infra" or job.get("_cancel_requested"):
            break
    detail = _scrub(((rec.get("session") or {}).get("detail") or (rec.get("session") or {}).get("reason") or ""), site)
    if outcome == "infra":
        return {"outcome": "infra", "record": rec, "attempts": attempts,
                "visible": {"status": "not measured", "why": "the card host could not give a session; not charged"}}
    if outcome == "candidate" and not (rec.get("measured") or {}).get("stopped"):
        return {"outcome": "candidate", "record": rec, "attempts": attempts,
                "visible": {"status": "the card did not take the image",
                            "session": (rec.get("session") or {}).get("status"), "why": detail}}
    got = rec["points"]
    visible: dict = {"status": "measured"}
    if (rec.get("measured") or {}).get("stopped"):
        visible["note"] = "the card session ran out of time before every point ran"
    try:
        base = board.baseline(site, repo, task)
        full = evaluator.score(base["points"], got, points)
        visible["validation"] = {k: full[k] for k in ("n", "passed", "speedup_geomean", "fast_1")}
        failures = evaluator.failure_counts(full)
        if failures:
            visible["failures"] = failures
    except Exception as exc:  # noqa: BLE001 -- no base on the card yet: passed points only
        full = None
        passed = sum(1 for p in points if (got.get(p) or {}).get("status") == "pass")
        visible["validation"] = {"n": len(points), "passed": passed, "note": "no base measured on the card to compare with"}
        job["baseline_error"] = f"{type(exc).__name__}: {exc}"
    sim_rows = {r["point"]: r for r in (((sim or {}).get("full") or {}).get("points") or [])}
    if certify.board_cycles(task) == "equal":
        agree = sum(1 for p in points if (got.get(p) or {}).get("status") == "pass" and sim_rows.get(p, {}).get(
            "failure") is None and ((got[p].get("observer") or {}).get("done_cycles") == sim_rows[p].get("done_cycles")))
        visible["counted_as_in_simulation"] = agree
    else:
        visible["cycles_on_card"] = _cycles_view(points, got, sim_rows)
    checks = ((rec.get("measured") or {}).get("checks")) or {}
    if checks:
        bad = {name: _check_view(ran) for name, ran in checks.items() if _check_view(ran) != "pass"}
        visible["checks"] = bad or "all passed"
    return {"outcome": outcome if outcome == "candidate" else "ok", "record": rec, "attempts": attempts,
            "score": full, "visible": visible}


def base_xclbin(site: Site, repo: Path, task: Task, candidate: Optional[Path], image: dict) -> Optional[str]:
    """The base's image on the card host, for the session to compare HBM maps with -- only for a built image
    whose device description kept the base's (the map follows the description)."""
    if image.get("kind") != "built":
        return None
    changed = evaluator.hardware_changed(repo, task, candidate).get("device") or {}
    # The HBM map follows the memory side of the description (ports, AXI, HBM); other uncore changes
    # (cache sets, ways, contexts) leave it as the base's, so the base's map is the one to match.
    if any(w in str(k).upper() for k in changed for w in ("MEM", "AXI", "HBM", "PORT")):
        return None
    try:
        return board.base_image(site, task)["path"]
    except board.BoardError:
        return None


def run(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path, job: dict,
        *, with_board: bool, image: Optional[dict], flow_seconds: int, board_seconds: int) -> tuple:
    """One request: the gate, the flow when it needs one, the card when the task asks for it.
    `image`: the image to run on when no flow is needed (the base's, or one this run built)."""
    cfg = site.fpga_pipeline
    if not cfg:
        raise SlowError("this site has no FPGA pipeline")
    job_dir.mkdir(parents=True, exist_ok=True)
    full: dict = {"kind": job["kind"]}
    # The gate: the validation set passes in simulation, measured as the score service measures it.
    sim = evaluator.measure_validation(site, repo, task, candidate, job_dir / "sim", job["containers"])
    full["simulation"] = sim
    agg = sim["aggregate"]
    if job.get("_cancel_requested"):
        job["outcome"] = "cancelled"
        return {"status": "cancelled"}, full
    if agg["passed"] < agg["n"]:
        job["outcome"] = "refused"
        visible = {"status": "refused", "charged": None,
                   "why": "the validation set does not pass in simulation; nothing was built",
                   "validation": agg}
        if sim.get("failures"):
            visible["failures"] = sim["failures"]
        return visible, full
    visible: dict = {"status": "done", "kind": job["kind"]}
    if job["kind"] == "hardware":
        built = build(site, repo, task, candidate, job_dir, job, flow_seconds)
        full["build"] = {k: v for k, v in built.items() if k != "visible"}
        job["tree_commit"] = built["tree_commit"]
        visible["build"] = built["visible"]
        job["outcome"] = built["outcome"]
        if built["outcome"] in ("infra", "timeout"):
            visible.update(status="not built", charged=None,
                           why=("the build infrastructure failed after retries" if built["outcome"] == "infra"
                                else "the flow ran past its time limit") + "; not charged")
            return visible, full
        if built["outcome"] == "design":
            visible.update(status="infeasible", charged="flow")
            return visible, full
        pipe = built["pipeline"]
        job["image"] = {"build_run": built["run_id"], "sha256": pipe["build"]["xclbin_sha256"],
                        "xclbin": certify.xclbin_name(repo, task)}
        visible["charged"] = "flow"
        if not with_board:
            return visible, full
        try:
            image = board.put_image(site, built["run_id"], job["image"]["sha256"], job["image"]["xclbin"],
                                    job["remote_id"])
        except Exception as exc:  # noqa: BLE001 -- the flow ran and is charged; its image waits on the build host
            full["card"] = {"error": f"{type(exc).__name__}: {exc}"}
            visible["board"] = {"status": "not measured", "why": "the image could not reach the card host; "
                                "ask again for this hardware: it needs no new flow"}
            return visible, full
        job["image"]["card"] = image
    elif not with_board:
        raise SlowError("a request that needs no flow has nothing to do without the board")
    card = on_card(site, repo, task, candidate, image, sim, job_dir, job, board_seconds,
                   base_xclbin=base_xclbin(site, repo, task, candidate, image))
    full["card"] = {k: v for k, v in card.items() if k != "visible"}
    visible["board"] = card["visible"]
    if job["kind"] == "software":
        job["outcome"] = "design" if card["outcome"] == "candidate" else card["outcome"]
        visible["charged"] = "board_run" if card["outcome"] in ("ok", "candidate") else None
        if card["outcome"] == "infra":
            visible["status"] = "not measured"
    return visible, full


__all__ = ["SlowError", "build", "cancel", "classify", "hardware_digest", "snapshot_digest", "manifest", "on_card", "run",
           "routed_report", "worst_paths", "slr_utilization", "drc_rules"]
