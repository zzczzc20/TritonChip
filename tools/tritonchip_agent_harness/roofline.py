"""Each scored point's distance from the whole chip's roofline (decided 2026-11-10).

    L   = max over resources of demand / chip capacity      (cycles)
    eta = L / T                                              (T: the point's measured device cycles)

Demand is the evaluation's own: each ArchBench point's semantic work (archbench.registry, Point.demand),
read in the toolchain container, the one place torch is. Capacity is the base machine's WHOLE chip, not one
core's (archbench.bounds is one core's): each core's spec_peak (archbench.bounds.tnca_peak, by the point's
operand format) times the device description's core_count, and memory MEM_PORTS outlets of AXI_DATA_W bits
per cycle -- the roofs of the R41 roofline figure (three cores at 100 MHz: FP16 28.8 GFLOP/s, 3 x 64 B per
cycle = 19.2 GB/s). The base machine's for every commit, so a machine change shows in eta against one roof.

What an agent is given (services.py, fpga_slow.py): one entry per point of the set it was measured on, in
the set's fixed order and never named -- its number, the resource that binds, eta, and the base's eta.

And, in its environment (`ENV`, runner.run_agent), the roof itself: the same chip rates per operand format,
the machine and never a point, so the test-case skill's eta for a case of its own is against the card's roof
(decided 2026-11-11: one eta, the whole chip's; the skill's one-core bound had a 64-bit memory port that R41's
512-bit core ports no longer have).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "chip-roofline/2"
# How the bound's resources are named to an agent.
NAMES = {"fp_lane": "compute", "mem_bytes": "memory", "int_lane": "integer", "sfu_lane": "special-function",
         "mma_mac": "matrix", "dependency": "dependency"}
# The variable that carries the chip's roof into an agent's container (agent_roof), read by the test-case
# skill's scripts/bound.py chip_peak.
ENV = "TNCA_CHIP_ROOF"

# The chip's rates come before any point, and the registry (torch) only when there are points, so the roof
# alone can be computed -- and tested -- without torch.
SCRIPT = r'''
import json, sys
sys.path[:0] = ["/work", "/work/python"]
import archbench.bounds as b
args = json.loads(sys.argv[1])
dev = json.load(open("/work/" + args["device"]))
cores = int(dev["platform"]["core_count"])
mem = int(dev["uncore"]["MEM_PORTS"]) * int(dev["platform"]["AXI_DATA_W"]) // 8


def chip(dtype):
    """One core's spec_peak for the operand format times the cores; memory through the outlets."""
    peak = dict(b.tnca_peak(args["profile"], dtype))
    rates = {k: v * cores for k, v in peak["rates"].items() if k != "mem_bytes"}
    rates["mem_bytes"] = mem
    peak["rates"] = rates
    return peak


out = {"cores": cores, "mem_bytes_per_cycle": mem, "points": {},
       "chip": {f or "any": chip(f)["rates"] for f in [None, *sorted(b._r38_peaks.MMA_GROUP)]}}
if args["points"]:
    import archbench.registry as reg
for pid in args["points"]:
    try:
        p = reg.get(pid)
        peak = chip(p.dtype)
        d = p.demand
        w = b._r38_bounds.Workload(p.id, p.family, p.kind, fp_ops=d.macs + d.fp_ops, macs=d.macs,
                                   sfu_ops=d.sfu_ops, read_bytes=d.read_bytes, write_bytes=d.write_bytes)
        cycles, binding, terms = b._r38_bounds.lower_bound(w, peak)
        out["points"][pid] = {"cycles": cycles, "binding": binding, "terms": terms}
    except Exception as exc:
        out["points"][pid] = {"error": type(exc).__name__ + ": " + str(exc)[:200]}
json.dump(out, open("/out/roofline.json", "w"), indent=1, sort_keys=True)
'''


def scored_points(task) -> list:
    """The points a run's agents get an eta for: the validation set and the card's own set. Never the final set."""
    from .task import board_points
    ev = task.evaluation
    return list(dict.fromkeys(list(ev["validation"]) + (board_points(task) if ev.get("board_validation") else [])))


def key(task) -> str:
    ev = task.evaluation
    return hashlib.sha256(json.dumps([VERSION, task.base_commit, ev["profile"], ev.get("device"),
                                      scored_points(task)]).encode()).hexdigest()[:16]


def table(site, repo: Path, task, run=None) -> dict:
    """{point: {cycles, binding, terms}} for the task's scored points, computed once per base and kept.
    `run(container, command, timeout)`: the evaluator's container run (a test passes its own)."""
    from . import evaluator
    from .provider import AgentContainer, Mount
    from .workspace import export
    k = key(task)
    root = site.state_dir / "rooflines"
    cache = root / f"{k}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tnca-roofline-", dir=str(site.state_dir)))
    try:
        tree, out = work / "tree", work / "out"
        export(repo, task.base_commit, (), tree)
        out.mkdir()
        out.chmod(0o777)
        args = {"profile": task.evaluation["profile"], "device": evaluator.device_path(task),
                "points": scored_points(task)}
        c = AgentContainer(name=f"tnca-roofline-{k}", image=site.images["toolchain"], cpus=2, memory="4g", pids=256,
                           mounts=[Mount(tree, "/work", read_only=True), Mount(out, "/out", read_only=False)],
                           labels={"tritonchip-agent-harness": "roofline"}, workdir="/work")
        status = (run or evaluator._run)(c, ["/opt/torchenv/bin/python", "-c", SCRIPT, json.dumps(args)], 900)
        got = out / "roofline.json"
        if status != 0 or not got.exists():
            raise RuntimeError(f"the roofline container ended with {status!r} and wrote "
                               f"{'nothing' if not got.exists() else 'a table'}")
        value = json.loads(got.read_text(encoding="utf-8"))
        value.update(version=VERSION, base_commit=task.base_commit, profile=task.evaluation["profile"])
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(cache)
        return value
    finally:
        shutil.rmtree(work, ignore_errors=True)


def view(bounds: Optional[dict], rows: list, order: list, family=None) -> Optional[list]:
    """What an agent sees: one entry per point of `order` (the set's fixed order), by number -- the binding
    resource, eta and the base's eta. `rows`: evaluator.score's rows (point, failure, cycles, base_cycles). A
    point that did not pass has no eta."""
    if not bounds:
        return None
    pts, by = bounds.get("points") or {}, {r.get("point"): r for r in rows or ()}
    out = []
    for n, p in enumerate(order, 1):
        b, r = pts.get(p) or {}, by.get(p) or {}
        L = b.get("cycles")
        entry = {"n": n, **({"family": family(p)} if family else {}),
                 "binds": NAMES.get(b.get("binding"), b.get("binding"))}
        entry["eta"] = (round(L / r["cycles"], 4) if L and r.get("failure") is None and r.get("cycles") else None)
        entry["eta_base"] = round(L / r["base_cycles"], 4) if L and r.get("base_cycles") else None
        out.append(entry)
    return out


def agent_roof(value: Optional[dict]) -> Optional[dict]:
    """What an agent's environment gets as `ENV`: the base machine's whole-chip rates per operand format ("any":
    no format, the matrix engine's generous end), the roof of every eta the card replies with. The machine only:
    no point, no point's bound. None when the table has no chip rates (a table of chip-roofline/1)."""
    if not value or not value.get("chip"):
        return None
    return {"version": value.get("version"), "profile": value.get("profile"), "cores": value.get("cores"),
            "mem_bytes_per_cycle": value.get("mem_bytes_per_cycle"), "rates": value["chip"]}


__all__ = ["ENV", "NAMES", "VERSION", "agent_roof", "key", "scored_points", "table", "view"]
