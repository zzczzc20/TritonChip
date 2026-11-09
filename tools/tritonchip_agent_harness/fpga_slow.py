"""The slow tier: an agent's request for the full link and the card (report section 8.9, decided 2026-11-06/07).

The agent asks; the request is asynchronous, and the agent goes on iterating while it runs. Two kinds:

    hardware   the snapshot changed the machine (RTL, the scored profile, the device description) and
               no earlier request of this run built that hardware: the FPGA harness links it on the
               build host (one Vivado flow), and in board mode its points then run on the card on the
               image it built. Charged to the run's flows (`services.fpga_slow.flows`).
    software   the machine is the base's, or this run already built exactly this hardware: no Vivado,
               only the card, on the base's image or that one. Charged to the board runs (`board_runs`).

Gate, before anything is charged for good: the snapshot's validation set must pass in simulation
(measured the score service's way, the run's one simulation of the snapshot: services.Simulations), and
a hardware request needs an fpga-quick result of this run on the same hardware that met timing. A
refused request is not charged. A hardware request's flow does not wait for the simulation (D3,
2026-11-09): both start at once, and a simulation that does not pass -- or that fails on the
infrastructure -- stops the flow, which is then not charged.

Charging: a flow that failed on the candidate's own design (timing or place-and-route did not
converge, it does not fit, DRC, a lowered clock, the card refused the image) counts; one that
failed on the infrastructure (a host, the licence, a disk, the harness) does not, and is retried
by itself on another build machine, never on one that failed it. The classification reads the
build record's own checks, never the agent's account. A flow that wrote no image (an
infrastructure failure after routing, or its cap reached) still tells the agent the routed
design's timing when it got that far.

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
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import board, certify, evaluator, provider, roofline as roofline_
from .siteconfig import Site
from .task import Task, board_points

RETRIES = 2               # an infrastructure failure is tried again this many times
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
        value["board"]["archbench"]["points"] = board_points(task)
    else:
        value.pop("board")
    return value


def _pipeline_args(tree: Path, mpath: Path, output: Path, cfg: dict, remote_id: str, build_host: str, *,
                   command_timeout: int) -> list:
    """The link alone: the card stage is board.py's, through the relay."""
    cli = Path(__file__).resolve().parents[1] / "tritonchip_fpga_harness" / "cli.py"
    return [sys.executable, str(cli), "--repo", str(tree), "pipeline", str(mpath),
            "--hosts", str(cfg["hosts"]), "--build-host", build_host,
            "--run-id", remote_id, "--timeout", str(command_timeout),
            "--poll", "30", "--output", str(output), "--skip-board"]


# -- the build machines: which one a flow runs on, one flow on each at a time ----------------------------
#
# The FPGA harness on a build machine takes its own full lease (`<state_dir>/leases/full.lock`) for a link
# and fails at once when it is held; it is the task's pinned platform code and stays as it is. So the
# machine is chosen here, before the link starts: the first, in the site's order, that no flow of this
# process holds and whose locks are free -- that lease, and the `shared_slots` another user's own Vivado
# jobs take there (flock). A flow on a machine with shared slots holds one of them while it runs, through
# an SSH connection of its own, so the other user's jobs see it. With every machine busy the flow waits:
# its outside wait, recorded apart.

POLL_SECONDS = 60        # how often a flow waiting for a free build machine looks again
_PLACING = threading.Lock()
_HELD: dict = {}         # build machine -> the request (remote id) of this process's flow on it

# The held locks, one path a line; a file that is not there is free.
_PROBE = 'for f in "$@"; do [ -e "$f" ] || continue; flock -n "$f" true 2>/dev/null || printf "%s\\n" "$f"; done'
# Vivado or v++ under the account at all: a flow that outlived an earlier run, the owner's job outside the slots.
_RUNNING = 'if pgrep -u "$(id -u)" -x "vivado|v\\+\\+" >/dev/null; then echo vivado-running; fi'
# Takes the first free slot and keeps it until the controller closes the connection (its stdin).
_HOLD = ('for f in "$@"; do exec 9>>"$f" || continue; if flock -n 9; then echo held; cat >/dev/null; exit 0; fi; '
         'exec 9>&-; done; echo none')


def build_hosts(cfg: dict) -> list:
    """The site's build machines, in the order a flow tries them."""
    return list(cfg.get("build_hosts") or [cfg["build_host"]])


def _locks(cfg: dict, host) -> list:
    return [f"{host.state_dir}/leases/full.lock", *((cfg.get("shared_slots") or {}).get(host.name) or [])]


def busy(cfg: dict, host, *, processes: bool = False) -> list:
    """Why `host` cannot take a flow now, [] when it can: the locks held there, and with `processes`
    Vivado running under the account."""
    from tritonchip_fpga_harness.pipeline import remote
    script = _PROBE + (f"; {_RUNNING}" if processes else "")
    done = remote(host, ["sh", "-c", script, "sh", *_locks(cfg, host)], check=False, timeout=90)
    if done.returncode != 0:
        return [f"could not be checked (exit {done.returncode})"]
    words = done.stdout.decode(errors="replace").split()
    return ["Vivado is running there" if w == "vivado-running" else f"{Path(w).name} is held" for w in words]


def _hold_slot(cfg: dict, host) -> tuple:
    """One of the shared slots on `host`, for a flow: (the connection holding it, True); (None, True) when the
    machine has none; (None, False) when every one was taken since it was looked at."""
    slots = (cfg.get("shared_slots") or {}).get(host.name)
    if not slots:
        return None, True
    from tritonchip_fpga_harness.pipeline import _remote_argv
    proc = subprocess.Popen(_remote_argv(host, ["sh", "-c", _HOLD, "sh", *slots], "ssh"), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    if proc.stdout.readline().strip() == b"held":
        return proc, True
    _drop_slot(proc)
    return None, False


def _drop_slot(proc) -> Optional[str]:
    """The shared slot given back: 'held' when the connection kept it to the end, 'lost' when it had ended."""
    if proc is None:
        return None
    lost = proc.poll() is not None
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    return "lost" if lost else "held"


def place(site: Site, job: dict, seconds: int, avoid=()) -> tuple:
    """The build machine for one attempt of a flow, taken for the request: (its name, the connection holding
    its shared slot or None). With every machine busy the request waits, looking again every POLL_SECONDS;
    the wait adds to job["outside_wait_seconds"], is never more than `seconds`, and a cancel ends it:
    then (None, None). The machines in `avoid` are never taken (each failed this flow already)."""
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    cfg = site.fpga_pipeline
    names = [n for n in build_hosts(cfg) if n not in avoid]
    if not names:
        return None, None
    loaded = load_hosts(cfg["hosts"])
    hosts = {n: host_for(loaded, n, "build") for n in names}
    started = time.monotonic()

    def waited() -> None:
        job["outside_wait_seconds"] = round(job.get("outside_wait_seconds", 0.0) + time.monotonic() - started, 1)
        job.pop("waiting_for_build_host", None)

    while True:
        seen = {}
        for name in names:
            with _PLACING:
                if name in _HELD:
                    seen[name] = ["another flow of this run"]
                    continue
            why = busy(cfg, hosts[name])
            if why:
                seen[name] = why
                continue
            with _PLACING:
                if name in _HELD:
                    seen[name] = ["another flow of this run"]
                    continue
                _HELD[name] = job["remote_id"]
            holder, ok = _hold_slot(cfg, hosts[name])
            if not ok:
                _release(job["remote_id"], name)
                seen[name] = ["its shared slots were taken"]
                continue
            job["build_host"], job["flow_started"] = name, time.time()
            waited()
            return name, holder
        job["waiting_for_build_host"], job["build_hosts_busy"] = True, seen
        for _ in range(POLL_SECONDS):
            if job.get("_cancel_requested") or time.monotonic() - started >= seconds:
                waited()
                return None, None
            time.sleep(1)
        if job.get("_cancel_requested") or time.monotonic() - started >= seconds:
            waited()
            return None, None


def _release(remote_id: str, name: Optional[str]) -> None:
    with _PLACING:
        if name and _HELD.get(name) == remote_id:
            del _HELD[name]


def budget_fit(task: Task) -> dict:
    """Whether the run's flows can come back within its time when each takes its cap: rounds of at most
    flows_at_once flows, then the card once."""
    slow = (task.services or {}).get("fpga_slow") or {}
    flows, at_once = int(slow.get("flows", 0)), int(slow.get("flows_at_once", 1))
    rounds = -(-flows // at_once) if flows else 0
    need = rounds * float(slow.get("flow_minutes", 0)) + (
        float(slow.get("board_minutes", 0)) if slow.get("mode") == "board" and rounds else 0.0)
    wall = task.wall_seconds / 60.0
    return {"rounds": rounds, "minutes_needed": need, "minutes_of_run": wall, "fits": need <= wall}


def preflight(site: Site, task: Task) -> dict:
    """Before a run that may ask for flows: the task's flows at once fit the site's build machines, and every
    one is idle -- no lock held, no Vivado of the account running (a flow that outlived an earlier run, the
    owner's own job). Raises SlowError when the run should not start. Also how the flows fit the run's time."""
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    slow = (task.services or {}).get("fpga_slow") or {}
    cfg = site.fpga_pipeline
    names = build_hosts(cfg)
    at_once = int(slow.get("flows_at_once", 1))
    if at_once > len(names):
        raise SlowError(f"the task runs up to {at_once} flows at once and the site has {len(names)} build "
                        f"machine(s): one flow on each at a time")
    loaded = load_hosts(cfg["hosts"])
    seen = {n: busy(cfg, host_for(loaded, n, "build"), processes=True) for n in names}
    out = {"build_hosts": names, "flows_at_once": at_once, "busy": {n: w for n, w in seen.items() if w},
           "budget": budget_fit(task)}
    if out["busy"]:
        raise SlowError("build machines not idle: " + "; ".join(f"{n}: {', '.join(w)}" for n, w in out["busy"].items()))
    return out


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
    """Stop the local controller and reconcile the build host's detached stages -- the attempt that is running
    (a retry runs under `<remote_id>-r<n>`), not the first."""
    job["_cancel_requested"] = True
    _terminate_local(job)
    try:
        names = provider.docker("ps", "-a", "--format", "{{.Names}}", check=False, timeout=10).stdout.split()
        for name in names:
            if name == job["containers"] or name.startswith(job["containers"] + "-"):
                provider.docker("rm", "-f", name, check=False, timeout=15)
    except (provider.ProviderError, OSError, subprocess.SubprocessError):
        pass
    commit, name = job.get("tree_commit"), job.get("build_host")
    if not commit or not name:
        return {"remote": "not-started"}   # no machine taken yet: nothing ran there
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    from tritonchip_fpga_harness.pipeline import remote

    try:
        # The machine the request's flow ran on, not the site's first.
        host = host_for(load_hosts(cfg["hosts"]), name, "build")
        argv = [host.python, host.cli(commit), "--repo", host.checkout(commit),
                "reconcile", job.get("current_remote_id") or job["remote_id"], "--state-dir", host.state_dir,
                "--terminate"]
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
    words = [cfg.get("build_host"), *(cfg.get("build_hosts") or []), cfg.get("board_host")]
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
    """The paths the routed design misses timing on, worst first (setup before hold); when none misses, the ones
    with the least slack. Not the report's first paths: its order is the clocks', the platform's first, and
    R41-F2's flow l8 (2026-11-09) showed the agent its frequency counters' +3.9 ns paths and none of the 4062
    failing ones in the core."""
    rows = []
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
        if "start" in row and isinstance(row.get("slack_ns"), float):
            row["crosses_slr"] = any("Inter-SLR Compensation" in line for line in block)
            rows.append(row)
    order = lambda r: (r.get("type") == "hold", r["slack_ns"])  # noqa: E731 -- setup first, then the least slack
    violated = sorted((r for r in rows if r["slack_ns"] < 0), key=order)
    return (violated or sorted(rows, key=order))[:n]


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


def routed_report(site: Site, run_id: str, build_host: Optional[str] = None) -> dict:
    """What only the routed design says, from the build host's reports for the run: worst paths (setup and
    hold, SLR crossings), utilisation per SLR, the DRC rules that failed."""
    from tritonchip_fpga_harness.hosts import host_for, load_hosts
    from tritonchip_fpga_harness.pipeline import remote
    cfg = site.fpga_pipeline
    host = host_for(load_hosts(cfg["hosts"]), build_host or cfg["build_host"], "build")
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


def design_timing(text: str) -> dict:
    """The routed report's Design Timing Summary (the FPGA harness's own reading of it) and whether it is clean
    by the harness's design-wide rule: WNS, TNS, WHS, THS and WPWS all >= 0 and no failing endpoint. {} when
    the report has no such table. The harness's own verdict may also exempt clocks a target declares; that
    verdict needs the finished build, which a flow without an image does not have."""
    from tritonchip_fpga_harness.adapters.u50 import U50Adapter
    metrics = U50Adapter._design_timing_summary(text) or {}
    if not all(k in metrics for k in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "failing_endpoints")):
        return {}
    keep = {k: metrics[k] for k in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "wpws_ns", "failing_endpoints")
            if k in metrics}
    keep["clean"] = (all(metrics[k] >= 0 for k in ("wns_ns", "tns_ns", "whs_ns", "ths_ns"))
                     and metrics.get("wpws_ns", 0) >= 0 and metrics["failing_endpoints"] == 0)
    return keep


def routed_timing(site: Site, run_id: str, build_host: Optional[str]) -> dict:
    """What the routed design's timing came to, for a flow that wrote no image (an infrastructure failure
    after routing, or the flow's cap reached in write_bitstream): design_timing of the run's routed timing
    summary on its build machine; {} when the flow never got that far or the machine cannot be read."""
    if not build_host:
        return {}
    try:
        from tritonchip_fpga_harness.hosts import host_for, load_hosts
        from tritonchip_fpga_harness.pipeline import remote
        cfg = site.fpga_pipeline
        host = host_for(load_hosts(cfg["hosts"]), build_host, "build")
        found = remote(host, ["find", f"{host.state_dir}/runs/{run_id}", "-name", "*timing_summary_routed*.rpt",
                              "-size", "-64M"], check=False, timeout=120)
        files = sorted(found.stdout.decode(errors="replace").split(), key=len) if found.returncode == 0 else []
        if not files:
            return {}
        done = remote(host, ["head", "-c", str(REPORT_BYTES), files[0]], check=False, timeout=180)
        return design_timing(done.stdout.decode(errors="replace")) if done.returncode == 0 else {}
    except Exception:  # noqa: BLE001 -- the flow's outcome stands without it
        return {}


def _build_visible(site: Site, pipe: dict, outcome: str, routed: dict) -> dict:
    build = pipe.get("build") or {}
    metrics = ((build.get("timing") or {}).get("metrics")) or {}
    visible = {"status": build.get("status") or pipe.get("status"),
               "observed_clock_mhz": build.get("observed_clock_mhz"), "elapsed_s": build.get("elapsed_s")}
    for key in ("wns_ns", "tns_ns", "whs_ns", "ths_ns", "failing_endpoints"):
        if key in metrics:
            visible[key] = metrics[key]
    if outcome in ("infra", "timeout") and routed.get("routed_timing"):
        # The design was routed and no image came: the agent still learns whether it met timing.
        visible["routed_timing"] = {**routed["routed_timing"],
                                    "note": "the routed design's Design Timing Summary; no image was written"}
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
    """One Vivado flow, tried again on an infrastructure failure, each attempt on a build machine taken for it
    (place). {outcome, pipeline, visible, build_host, ...}."""
    cfg = site.fpga_pipeline
    tree = job_dir / "tree"
    commit = certify.build_tree(repo, task, candidate, tree)
    job["tree_commit"] = commit
    attempts, failed_on = [], []
    m = manifest(task, job["id"], commit, board=False)
    for attempt in range(RETRIES + 1):
        remote_id = job["remote_id"] if attempt == 0 else f"{job['remote_id']}-r{attempt}"
        if job.get("_cancel_requested"):
            # Stopped before this attempt began (a gate that did not pass, D3, or the agent's end): none starts.
            pipe, outcome = {"status": "CANCELLED", "error": "stopped before the flow started"}, "infra"
            attempts.append({"remote_id": None, "status": "CANCELLED", "outcome": outcome})
            break
        if failed_on and not [n for n in build_hosts(cfg) if n not in failed_on]:
            # Every build machine has failed this flow on the infrastructure: a machine's own fault (a missing
            # part, a missing tool) would only repeat there, as RSCL's missing bitstream data did (2026-11-08).
            attempts.append({"remote_id": None, "status": "NOT_RETRIED", "outcome": "infra",
                             "why": "every build machine failed this flow already: " + ", ".join(failed_on)})
            break
        host, holder = place(site, job, seconds, avoid=failed_on)
        if host is None:
            # No machine became free within the flow's cap (or the request was cancelled while it waited).
            pipe, outcome = {"status": "NO_BUILD_HOST", "error": "no build machine became free"}, "infra"
            attempts.append({"remote_id": remote_id, "status": pipe["status"], "outcome": outcome,
                             "build_host": None, "busy": job.get("build_hosts_busy")})
            break
        if job.get("_cancel_requested"):
            # Stopped while the machine was being taken: it is given back before anything runs on it.
            _drop_slot(holder)
            _release(job["remote_id"], host)
            pipe, outcome = {"status": "CANCELLED", "error": "stopped before the flow started"}, "infra"
            attempts.append({"remote_id": None, "status": "CANCELLED", "outcome": outcome, "build_host": host})
            break
        try:
            mpath, output = job_dir / f"manifest-{attempt}.json", job_dir / f"pipeline-{attempt}.json"
            mpath.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            argv = _pipeline_args(tree, mpath, output, cfg, remote_id, host, command_timeout=seconds)
            job["current_remote_id"] = remote_id
            with (job_dir / f"pipeline-{attempt}.log").open("wb") as log:
                proc = subprocess.Popen(argv, cwd=tree, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                job["_process"] = proc
                if job.get("_cancel_requested"):
                    # Stopped between taking the machine and starting the flow (a gate that did not pass, D3,
                    # or the agent's end): what started is stopped here and on the build host.
                    job["cleanup"] = cancel(job, cfg)
                try:
                    proc.wait(timeout=seconds + 600)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    job["cleanup"] = cancel(job, cfg)
                    timed_out = True
            job.pop("_process", None)
        finally:
            slot = _drop_slot(holder)
            _release(job["remote_id"], host)
        pipe = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
            "status": "PIPELINE_FAIL", "error": "the controller wrote no pipeline record"}
        if timed_out:
            pipe["status"] = "TIMEOUT"
        outcome = classify(pipe, (task.evaluation.get("certify") or {}).get("clock_mhz"))
        attempts.append({"remote_id": remote_id, "status": pipe.get("status"), "outcome": outcome,
                         "build_host": host, **({"shared_slot": slot} if slot else {})})
        if outcome in ("infra", "timeout"):
            # No image, but the design may have been routed: what its timing came to is kept.
            summary = routed_timing(site, pipe.get("run_id") or remote_id, host)
            if summary:
                attempts[-1]["routed_timing"] = summary
        if outcome != "infra" or job.get("_cancel_requested"):
            break
        failed_on.append(host)
    job["flow_finished"] = time.time()
    routed = (routed_report(site, pipe.get("run_id") or remote_id, job.get("build_host"))
              if outcome == "design" else {})
    if outcome in ("infra", "timeout"):
        kept = [a["routed_timing"] for a in attempts if a.get("routed_timing")]
        if kept:
            routed = {"routed_timing": kept[-1]}
    return {"outcome": outcome, "pipeline": pipe, "attempts": attempts, "tree_commit": commit, "manifest": m,
            "routed": routed, "visible": _build_visible(site, pipe, outcome, routed),
            "run_id": pipe.get("run_id") or remote_id, "build_host": job.get("build_host"),
            "outside_wait_seconds": job.get("outside_wait_seconds", 0.0)}


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
    return certify.cycle_ratios(ratios)


def on_card(site: Site, repo: Path, task: Task, candidate: Optional[Path], image: dict, sim: dict,
            job_dir: Path, job: dict, seconds: int, base_xclbin: Optional[str] = None,
            roofline: Optional[dict] = None) -> dict:
    """The board's validation points (and the task's board checks) on `image`, tried again on an infrastructure
    failure; the agent's view against the base on the card. outcome: ok, candidate (charged) or infra."""
    points = board_points(task)
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
        full = evaluator.score(base["points"], got, points,
                               faster_than=float(((task.services or {}).get("fpga_slow") or {}).get("board_faster_than", 1.0)))
        visible["validation"] = {k: full[k] for k in ("n", "passed", "speedup_geomean", "fast_1")}
        listed = roofline_.view(roofline, full.get("points") or [], points)
        if listed is not None:
            visible["points"] = listed     # each card point's eta against the whole chip's roofline, by number
        failures = evaluator.failure_counts(full)
        if failures:
            visible["failures"] = failures
    except Exception as exc:  # noqa: BLE001 -- no base on the card yet: passed points only
        full = None
        passed = sum(1 for p in points if (got.get(p) or {}).get("status") == "pass")
        visible["validation"] = {"n": len(points), "passed": passed, "note": "no base measured on the card to compare with"}
        job["baseline_error"] = f"{type(exc).__name__}: {exc}"
    sim_rows = {r["point"]: r for r in (((sim or {}).get("full") or {}).get("points") or [])}
    if not any(p in sim_rows for p in points):
        pass   # the card's own set (board_validation): simulation never ran these points, nothing to compare
    elif certify.board_cycles(task) == "equal":
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


def _refused(sim: dict, why: str) -> dict:
    visible = {"status": "refused", "charged": None, "why": why, "validation": sim["aggregate"]}
    if sim.get("failures"):
        visible["failures"] = sim["failures"]
    return visible


def _passes(sim: Optional[dict]) -> bool:
    return bool(sim) and sim["aggregate"]["passed"] >= sim["aggregate"]["n"]


def _gate_beside_flow(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path, job: dict,
                      gate, flow_seconds: int, full: dict) -> tuple:
    """D3: the gate's simulation and the Vivado flow at once; a gate that does not pass (or fails on the
    infrastructure) stops the flow. (built or None, sim or None, the visible refusal or None)."""
    box: dict = {}

    def judge() -> None:
        try:
            box["sim"] = gate()
        except Exception as exc:  # noqa: BLE001 -- the infrastructure's: the flow is stopped, not charged
            box["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        if not _passes(box.get("sim")) and not job.get("_cancel_requested") and not box.get("flow_over"):
            job["gate_stopped_flow"] = True
            job["gate_stop_cleanup"] = cancel(job, site.fpga_pipeline)

    t = threading.Thread(target=judge, name=f"gate-{job.get('id')}", daemon=True)
    t.start()
    try:
        built = build(site, repo, task, candidate, job_dir, job, flow_seconds)
    except BaseException:
        box["flow_over"] = True      # the request ends here (the broker: infrastructure); the gate stops nothing
        raise
    t.join()
    sim = box.get("sim")
    full["simulation"] = sim
    full["build"] = {k: v for k, v in built.items() if k != "visible"}
    if box.get("error"):
        full["gate_error"] = box["error"]
    if job.get("gate_stopped_flow"):
        if box.get("error") or sim is None:
            job["outcome"] = "infra"
            return built, sim, {"status": "not built", "charged": None,
                                "why": "the simulation gate failed on the infrastructure and the flow was stopped; "
                                       "not charged"}
        job["outcome"] = "refused"
        return built, sim, _refused(sim, "the validation set does not pass in simulation; the flow was stopped "
                                         "and is not charged")
    return built, sim, None


def run(site: Site, repo: Path, task: Task, candidate: Optional[Path], job_dir: Path, job: dict,
        *, with_board: bool, image: Optional[dict], flow_seconds: int, board_seconds: int, gate=None,
        roofline: Optional[dict] = None) -> tuple:
    """One request: the gate, the flow when it needs one, the card when the task asks for it.
    `image`: the image to run on when no flow is needed (the base's, or one this run built). `gate`: the
    validation set's simulation (the broker's, through the run's one simulation per snapshot); by default
    measured here."""
    cfg = site.fpga_pipeline
    if not cfg:
        raise SlowError("this site has no FPGA pipeline")
    job_dir.mkdir(parents=True, exist_ok=True)
    full: dict = {"kind": job["kind"]}
    if gate is None:
        def gate():
            return evaluator.measure_validation(site, repo, task, candidate, job_dir / "sim", job["containers"])
    visible: dict = {"status": "done", "kind": job["kind"]}
    if job["kind"] == "hardware":
        built, sim, refusal = _gate_beside_flow(site, repo, task, candidate, job_dir, job, gate, flow_seconds, full)
        if refusal is not None:
            return refusal, full
        # A flow that ran past its cap is stopped by build() itself (cancel): that is its outcome, said below,
        # not a cancelled request. Only a flow that built while the gate gave no answer is one.
        if sim is None and built["outcome"] == "ok":
            job["outcome"] = "cancelled"
            return {"status": "cancelled"}, full
    else:
        # The gate: the validation set passes in simulation, measured as the score service measures it.
        sim = gate()
        full["simulation"] = sim
        if sim is None or job.get("_cancel_requested"):
            job["outcome"] = "cancelled"
            return {"status": "cancelled"}, full
        if not _passes(sim):
            job["outcome"] = "refused"
            return _refused(sim, "the validation set does not pass in simulation; nothing was built"), full
    if job["kind"] == "hardware":
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
        # Which machine built it: its run and artifacts stay there, for the card host to fetch.
        job["image"] = {"build_run": built["run_id"], "sha256": pipe["build"]["xclbin_sha256"],
                        "xclbin": certify.xclbin_name(repo, task), "build_host": built["build_host"]}
        visible["charged"] = "flow"
        if not with_board:
            return visible, full
        try:
            image = board.put_image(site, built["run_id"], job["image"]["sha256"], job["image"]["xclbin"],
                                    job["remote_id"], build_host=built["build_host"])
        except Exception as exc:  # noqa: BLE001 -- the flow ran and is charged; its image waits on the build host
            full["card"] = {"error": f"{type(exc).__name__}: {exc}"}
            visible["board"] = {"status": "not measured", "why": "the image could not reach the card host; "
                                "ask again for this hardware: it needs no new flow"}
            return visible, full
        job["image"]["card"] = image
    elif not with_board:
        raise SlowError("a request that needs no flow has nothing to do without the board")
    card = on_card(site, repo, task, candidate, image, sim, job_dir, job, board_seconds,
                   base_xclbin=base_xclbin(site, repo, task, candidate, image), roofline=roofline)
    full["card"] = {k: v for k, v in card.items() if k != "visible"}
    visible["board"] = card["visible"]
    if job["kind"] == "software":
        job["outcome"] = "design" if card["outcome"] == "candidate" else card["outcome"]
        visible["charged"] = "board_run" if card["outcome"] in ("ok", "candidate") else None
        if card["outcome"] == "infra":
            visible["status"] = "not measured"
    return visible, full


__all__ = ["SlowError", "budget_fit", "build", "build_hosts", "busy", "cancel", "classify", "design_timing", "hardware_digest",
           "snapshot_digest", "manifest", "on_card", "place", "preflight", "run", "routed_report", "routed_timing", "worst_paths",
           "slr_utilization", "drc_rules"]
