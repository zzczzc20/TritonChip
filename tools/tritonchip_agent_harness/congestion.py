"""What a link says about routing congestion, for the agent (2026-11-23).

A flow that failed in routing returned v++'s ERROR lines, and nothing else: the routed reports the other
feedback is read from (fpga_slow.routed_report) are never written when routing fails. Those ERROR lines name
"the first 10" conflicted nets, which are the first in name order. In R41-HPC-demo's b3 they were all in
`device/control`, while the router's overlaps sat in tile 1's home-cache pipe and the placer's worst congestion
(level 7) in tile 0's transfer engine, next to the LSU path b3 had widened.

The platform's flow is not changed for this (the task pins it). The build machine already holds what is needed:

1. the implementation run's own log (`impl_1/runme.log`): the timing estimate after placement, the placer's
   estimated congestion for the device and each SLR, the router's initial estimate, the router's verdict, and
   the routing resources most contended when it gave up, with the nets on each;
2. the run's last checkpoint, on which Vivado attributes each congested window to the cells that fill it
   (`report_design_analysis -congestion`, both the placer's final and the router's initial estimate). Opening
   a checkpoint of this design takes about 8 minutes on the cluster's nodes, so this runs after a flow that
   failed in placement or routing only, as a batch job of its own on a build machine that is an LSF cluster
   (a plain machine has given its Vivado slot back by then and gets the log only).

Levels are AMD's (UG949): a congested window of 2^n x 2^n interconnect tiles is level n.
"""

from __future__ import annotations

import re
import shlex
import time
from typing import Optional

REPORT_BYTES = 8 << 20          # the most of the run's log or the analysis report fetched
ANALYSIS_SECONDS = 45 * 60      # the analysis job's cap: queue, opening the checkpoint (~8 min), the report
ANALYSIS_SLOTS = 4
ANALYSED_STAGES = ("place", "phys_opt", "route")
WINDOWS = 6                     # congested windows kept from each table
CELLS = 3                       # cells named per window
NODES = 10                      # contended routing resources kept
MIN_LEVEL = 5

NOTE = ("Vivado's own estimate of routing congestion for this link. A congested window of 2^n x 2^n "
        "interconnect tiles is level n; AMD's methodology guide (UG949) says level 5 and above often costs "
        "timing and above 6 a design is unlikely to meet timing and may fail to route. `windows` is Vivado's "
        "attribution (report_design_analysis -congestion) of the most congested windows to the cells that fill "
        "them, after placement (`placer`) and when routing began (`router`); paths are below the kernel's top, "
        "`platform/...` is the shell's. `route.worst_nodes` are the routing resources the router found most "
        "contended when it stopped, with the nets on each; a net that only crosses a congested region is listed "
        "there too. Interconnect rows Y: 60 to a clock-region row; on the U280, SLR n holds clock-region rows "
        "4n to 4n+3.")

# The kernel's own instance in the shell (`tnca_u50_1`); every other cell under level0_i is the platform's.
_KERNEL = re.compile(r"^level0_i/ulp/tnca_[^/]*/inst/")
_SIZE = re.compile(r"^(\d+)x\1$")
_WNS = re.compile(r"\[Physopt 32-619\] Estimated Timing Summary \| WNS=(-?[\d.]+) \| TNS=(-?[\d.]+)")
_OVERLAPS = re.compile(r"Design is not legally routed\. There are (\d+) node overlaps")
_NODE = re.compile(r"^\s*\d+\.\s+\(.*?\)\s+(\S+)\s+Hist:\s*(\d+)\s+Tile Name:\s*(\S+)\s+Node:\s*(\S+)"
                   r"\s+Overlapping Nets:\s*(\d+)")
_TILE = re.compile(r"_X(\d+)Y(\d+)$")
_DIRECTIONS = ("north", "south", "east", "west")


def short(path: str) -> str:
    """A cell or net name as the agent writes it: below the kernel's top, or `platform/...` for the shell's."""
    path = path.strip()
    m = _KERNEL.match(path)
    if m:
        return path[m.end():]
    if path.startswith("level0_i/"):
        return "platform/" + "/".join(path.split("/")[1:5])
    return path


def module_of(net: str) -> str:
    """The instance a net belongs to: its name without the last part; the kernel's top for a top-level net."""
    name = short(net)
    head = name.rsplit("/", 1)[0] if "/" in name else ""
    return head or "(kernel top)"


def level(size: str) -> Optional[int]:
    m = _SIZE.match(size.strip())
    if not m:
        return None
    n = int(m.group(1))
    return n.bit_length() - 1 if n > 0 and n & (n - 1) == 0 else None


def _cells(line: str) -> list:
    return [c.strip() for c in line.split("|")[1:-1]]


def _worst(rows: list, columns: dict) -> dict:
    """{type: {level, direction}} from a congestion table's rows: the highest level of each type."""
    out: dict = {}
    for row in rows:
        direction = row[0].lower()
        for kind, idx in columns.items():
            lv = level(row[idx]) if idx < len(row) else None
            if lv is not None and lv > out.get(kind, {}).get("level", -1):
                out[kind] = {"level": lv, "direction": direction}
    return out


def _placer_tables(lines: list, start: int) -> dict:
    """The placer's post-placement tables after line `start`: the device's, then one per SLR."""
    scope, rows, out = "device", [], {}
    for line in lines[start + 1:]:
        s = line.strip()
        m = re.match(r"^(SLR\d+):$", s)
        if m:
            if rows:
                out[scope] = _worst(rows, {"global": 1, "long": 2, "short": 3})
            scope, rows = m.group(1), []
            continue
        if s.startswith(("Phase", "INFO", "Time (s)", "Command:", "WARNING", "CRITICAL", "Ending")):
            break
        cells = _cells(line) if s.startswith("|") else []
        if cells and cells[0].lower() in _DIRECTIONS:
            rows.append(cells)
    if rows:
        out[scope] = _worst(rows, {"global": 1, "long": 2, "short": 3})
    return out


def _router_table(lines: list, start: int) -> dict:
    rows = []
    for line in lines[start + 1:start + 20]:
        cells = _cells(line) if line.strip().startswith("|") else []
        if cells and cells[0].lower() in _DIRECTIONS:
            rows.append(cells)
    return _worst(rows, {"global": 1, "long": 3, "short": 5})


def _overlap_nodes(lines: list, start: int) -> list:
    nodes, cur, in_nets = [], None, False
    for line in lines[start + 1:]:
        s = line.strip()
        m = _NODE.match(line)
        if m:
            if cur:
                nodes.append(cur)
            if len(nodes) >= NODES:
                cur = None
                break
            tile = m.group(3)
            t = _TILE.search(tile)
            cur = {"tile": tile, "node": m.group(4), "kind": m.group(1), "history": int(m.group(2)),
                   "nets": []}
            if t and tile.startswith("INT_"):
                cur["clock_region_row"] = int(t.group(2)) // 60
            in_nets = False
            continue
        if cur is None:
            continue
        if s == "Nets:":
            in_nets = True
            continue
        if in_nets and s and not s.startswith(("CRITICAL", "INFO", "WARNING", "ERROR", "Phase", "Resolution")):
            cur["nets"].append(short(s))
            continue
        if s.startswith(("CRITICAL", "INFO", "WARNING", "ERROR", "Phase")):
            break
    if cur:
        nodes.append(cur)
    return nodes[:NODES]


def parse_log(text: str) -> dict:
    """What an implementation run's log says of congestion: {post_place_estimate, placer, router_initial,
    route}; a key is left out when the log never got that far."""
    lines = text.splitlines()
    route_at = next((i for i, line in enumerate(lines) if line.startswith("Command: route_design")), len(lines))
    out: dict = {}
    est = [m for m in (_WNS.search(line) for line in lines[:route_at]) if m]
    if est:
        out["post_place_estimate"] = {"wns_ns": float(est[-1].group(1)), "tns_ns": float(est[-1].group(2))}
    placer = [i for i, line in enumerate(lines[:route_at]) if "Post-Placement Estimated Congestion" in line]
    if placer:
        out["placer"] = _placer_tables(lines, placer[-1])
    router = next((i for i, line in enumerate(lines) if "[Route 35-449] Initial Estimated Congestion" in line), None)
    if router is not None:
        out["router_initial"] = _router_table(lines, router)
    route: dict = {}
    if any("Router Completed Successfully" in line for line in lines):
        route["routed"] = True
    m = next((_OVERLAPS.search(line) for line in lines if _OVERLAPS.search(line)), None)
    if m:
        route.update(routed=False, overlaps=int(m.group(1)))
    block = next((i for i, line in enumerate(lines) if "physical nodes with signal overlaps" in line), None)
    if block is not None:
        nodes = _overlap_nodes(lines, block)
        if nodes:
            route["worst_nodes"] = nodes
            count: dict = {}
            for node in nodes:
                for module in {module_of(n) for n in node["nets"]}:
                    count[module] = count.get(module, 0) + 1
            route["modules_on_worst_nodes"] = [[k, v] for k, v in
                                               sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))]
    if route:
        out["route"] = route
    return out


def _named_cells(text: str) -> list:
    found = []
    for m in re.finditer(r"\s*,?\s*([^,]+?)\((\d+)%\)", text):
        found.append([short(m.group(1)), int(m.group(2))])
    return found[:CELLS]


def parse_design_analysis(text: str, min_level: int = MIN_LEVEL) -> dict:
    """`report_design_analysis -congestion`: {placer: [...], router: [...]}, every window at level >= min_level
    once (the directions and congestion types it was reported for merged), with the cells that fill it, most
    congested first."""
    out: dict = {}
    section, header = None, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("1. Placer Final"):
            section, header = "placer", None
            continue
        if s.startswith("2. Router Initial"):
            section, header = "router", None
            continue
        if re.match(r"^\d+\. ", s) and section and not s.startswith(("1.", "2.")):
            section = None
            continue
        if section is None or not s.startswith("|"):
            continue
        cells = _cells(line)
        if cells and cells[0] == "Direction":
            header = {name: i for i, name in enumerate(cells)}
            continue
        if header is None or len(cells) < len(header) or not cells[0]:
            continue
        try:
            lv = int(cells[header["Level"]])
        except (KeyError, ValueError):
            continue
        if lv < min_level:
            continue
        window = cells[header["Window"]]
        kind = f"{cells[header['Direction']].lower()} {cells[header['Type']].lower()}"
        rows = out.setdefault(section, [])
        row = next((r for r in rows if r["window"] == window), None)
        if row is None:
            rows.append({"level": lv, "window": window, "kinds": [kind],
                         "cells": _named_cells(cells[header["Cell Names"]]) if "Cell Names" in header else []})
        else:
            row["level"] = max(row["level"], lv)
            if kind not in row["kinds"]:
                row["kinds"].append(kind)
    return {k: sorted(v, key=lambda r: -r["level"]) for k, v in out.items()}


def _bounds(window: str) -> Optional[tuple]:
    xy = [(int(x), int(y)) for x, y in re.findall(r"X(\d+)Y(\d+)", window)]
    if len(xy) != 2:
        return None
    (x0, y0), (x1, y1) = xy
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def rank(windows: dict, nodes: list) -> dict:
    """Each table's windows, the most congested first and, at one level, those holding a routing resource the
    router was still fighting over when it stopped (`at_worst_nodes`) first; WINDOWS of each. Interconnect and
    logic tiles share their row numbers, and on UltraScale+ their column numbers closely enough for this."""
    points = [(int(m.group(1)), int(m.group(2))) for m in (_TILE.search(n.get("tile", "")) for n in nodes) if m]
    out = {}
    for table, rows in windows.items():
        marked = []
        for row in rows:
            b = _bounds(row["window"])
            hit = bool(b and any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in points))
            marked.append({**row, **({"at_worst_nodes": True} if hit else {})})
        marked.sort(key=lambda r: (-r["level"], not r.get("at_worst_nodes")))
        out[table] = marked[:WINDOWS]
    return out


def visible(found: dict) -> dict:
    """What the agent sees of `collect`'s record; {} when it holds nothing."""
    log = found.get("log") or {}
    out: dict = {}
    for key in ("post_place_estimate", "placer", "router_initial", "route"):
        if log.get(key):
            out[key] = log[key]
    if found.get("windows"):
        out["windows"] = rank(found["windows"], (log.get("route") or {}).get("worst_nodes") or [])
    elif out and found.get("analysis", {}).get("why"):
        out["windows_not_attributed"] = found["analysis"]["why"]
    if out:
        out = {"note": NOTE, **out}
    return out


ANALYZE_TCL = """# Written by the agent harness (congestion.py): Vivado's congestion attribution of a link's checkpoint.
lassign $argv dcp out
open_checkpoint $dcp
report_design_analysis -congestion -file $out
"""


# Where the Vitis link of the U280 binding leaves its implementation run's log (flow/fpga/link.sh's --temp_dir):
# read in place while the link runs, without searching the run's tree every poll.
IMPL_LOG = "artifacts/tmp/link/vivado/vpl/prj/prj.runs/impl_1/runme.log"
# Placement has ended once the placer closed its task or the next step began.
_PLACED = ("Ending Placer Task", "Command: phys_opt_design", "Command: route_design")
PLACEMENT_NOTE = ("Read from the link's own log while it runs, once placement has ended: Vivado's timing estimate "
                  "after placement and its estimated congestion for the device and each SLR (a window of 2^n x 2^n "
                  "interconnect tiles is level n; AMD's UG949: level 5 and above often costs timing, above 6 a "
                  "design is unlikely to meet timing and may fail to route). The link goes on; its result brings "
                  "the full congestion record.")


def placement(site, run_id: str, build_host: Optional[str], *, ssh_binary: str = "ssh") -> dict:
    """What a running link's log says once placement has ended: {note, post_place_estimate, placer}; {} before
    that, or when the log cannot be read. One read of one file; never raises."""
    try:
        from tritonchip_fpga_harness.hosts import host_for, load_hosts
        from tritonchip_fpga_harness.pipeline import remote
        if not build_host:
            return {}
        host = host_for(load_hosts(site.fpga_pipeline["hosts"]), build_host, "build")
        done = remote(host, ["tail", "-c", str(REPORT_BYTES), f"{host.state_dir}/runs/{run_id}/{IMPL_LOG}"],
                      ssh_binary=ssh_binary, check=False, timeout=120)
        if done.returncode != 0:
            return {}
        text = done.stdout.decode(errors="replace")
        if "Post-Placement Estimated Congestion" not in text or not any(m in text for m in _PLACED):
            return {}
        log = parse_log(text)
        seen = {k: log[k] for k in ("post_place_estimate", "placer") if log.get(k)}
        return {"note": PLACEMENT_NOTE, **seen} if seen else {}
    except Exception:  # noqa: BLE001 -- the flow goes on without it
        return {}


def _job_script(host, folder: str, dcp: str) -> str:
    q = shlex.quote
    lines = ["#!/bin/sh"] + [f"export {k}={q(v)}" for k, v in sorted(host.remote_env().items())]
    lines += [f". {q(host.vitis_root + '/settings64.sh')} >/dev/null 2>&1",
              f"cd {q(folder)} || exit 97",
              "vivado -mode batch -nojournal -log vivado.log -source analyze.tcl -tclargs "
              f"{q(dcp)} {q(folder + '/design_analysis.rpt')}"]
    return "\n".join(lines) + "\n"


def collect(site, run_id: str, build_host: Optional[str], *, analyse: bool, wait_s: float = ANALYSIS_SECONDS,
            poll_s: float = 60.0, ssh_binary: str = "ssh") -> dict:
    """{log, windows, analysis} for the run `run_id` on its build machine; {} when nothing can be read. Never
    raises: the flow's outcome stands without it."""
    try:
        from tritonchip_fpga_harness import lsf
        from tritonchip_fpga_harness.hosts import host_for, load_hosts
        from tritonchip_fpga_harness.pipeline import remote
        if not build_host:
            return {}
        host = host_for(load_hosts(site.fpga_pipeline["hosts"]), build_host, "build")
        run = f"{host.state_dir}/runs/{run_id}"
        found = remote(host, ["find", run, "-path", "*impl_1/*", "(", "-name", "runme.log", "-o", "-name", "*.dcp",
                              ")"], ssh_binary=ssh_binary, check=False, timeout=120)
        files = found.stdout.decode(errors="replace").split() if found.returncode == 0 else []
        logs = sorted((f for f in files if f.endswith("impl_1/runme.log")), key=len)
        if not logs:
            return {}
        done = remote(host, ["tail", "-c", str(REPORT_BYTES), logs[0]], ssh_binary=ssh_binary, check=False,
                      timeout=180)
        out: dict = {"log": parse_log(done.stdout.decode(errors="replace")) if done.returncode == 0 else {}}
        if not analyse:
            return out
        dcp = next((f for suffix in ("_routed_error.dcp", "_routed.dcp", "_physopt.dcp", "_placed.dcp")
                    for f in files if f.endswith(suffix)), None)
        if dcp is None:
            out["analysis"] = {"why": "the run left no placed or routed checkpoint"}
            return out
        if host.lsf is None or not host.vitis_root:
            out["analysis"] = {"why": "only a build machine with a batch queue runs the attribution"}
            return out
        folder, name = f"{run}/congestion", f"{run_id}-congestion"
        report = f"{folder}/design_analysis.rpt"
        have = remote(host, ["test", "-s", report], ssh_binary=ssh_binary, check=False, timeout=60)
        started = time.time()
        if have.returncode != 0:
            remote(host, ["mkdir", "-p", folder], ssh_binary=ssh_binary)
            remote(host, ["dd", f"of={folder}/analyze.tcl"], ssh_binary=ssh_binary, data=ANALYZE_TCL.encode())
            remote(host, ["dd", f"of={folder}/job.sh"], ssh_binary=ssh_binary,
                   data=_job_script(host, folder, dcp).encode())
            queued = lsf.queue(host, name, f"{folder}/job.sh", f"{folder}/job.log",
                               slots=min(ANALYSIS_SLOTS, int(host.lsf["slots"])), ssh_binary=ssh_binary)
            out["analysis"] = {"job_id": queued.get("job_id")}
            try:
                ended = lsf.wait(host, name, poll_s=poll_s, deadline=started + wait_s, ssh_binary=ssh_binary)
                out["analysis"].update(stat=ended.get("stat"), run_s=ended.get("run_s"))
            except lsf.LsfError as exc:
                lsf.kill(host, name, ssh_binary=ssh_binary)
                out["analysis"].update(why=f"the attribution did not finish within {wait_s / 60:g} minutes",
                                       detail=str(exc)[:200])
                return out
        read = remote(host, ["head", "-c", str(REPORT_BYTES), report], ssh_binary=ssh_binary, check=False,
                      timeout=180)
        windows = parse_design_analysis(read.stdout.decode(errors="replace")) if read.returncode == 0 else {}
        if windows:
            out["windows"] = windows
        else:
            out.setdefault("analysis", {})["why"] = "Vivado wrote no congestion report for the checkpoint"
        out.setdefault("analysis", {})["seconds"] = round(time.time() - started, 1)
        return out
    except Exception as exc:  # noqa: BLE001 -- the flow's outcome stands without it
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


__all__ = ["ANALYSED_STAGES", "IMPL_LOG", "NOTE", "PLACEMENT_NOTE", "collect", "level", "module_of",
           "parse_design_analysis", "parse_log", "placement", "rank", "short", "visible"]
