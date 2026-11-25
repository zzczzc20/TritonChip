#!/usr/bin/env python3
"""What a test case reaches in the CA model: which mechanisms of the machine it exercises, and how often.

    python /opt/tnca-skills/cycle-model-alignment/scripts/mechanisms.py CASE.py [CASE2.py ...] [--config CFG]
        [--profile P] [--active-cores N] [--model-build DIR] [--dim NAME=VALUE ...] [--json OUT]

Run it from the repository checkout you work in (or set TNCA_ROOT). Each case runs once on the CA model built
from your tree (incrementally, as align_case.py builds it), in a process of its own, under the evaluator's
environment; the model's event counters at the end of the run are read into mechanisms. A mechanism is
reached when its counters are not zero. The mechanisms and their rules are those of the task's
CALIBRATION.json (`RULES` below), so a case you write can be set beside the calibration cases.

What a mechanism's count says: that the path was taken, and how often. It is not a time: the model's cycles
are its own prediction, with its error; whether a path is taken does not depend on that error.

Matrix instructions are counted from the compiled code (`static`: every matrix instruction of every kernel,
by mnemonic), not from the counters: this base's models leave their matrix-command counters
(`*_mma_commands`, `*_matrix_engine_commands`, `*_matrix_dp_commands`) at 0. `matrix engine waits` (from
the stall counters) is counted.

`--dim NAME=VALUE` builds the case at another size: a case of the development set (`_point_make(seed, ...)`)
or one whose `make()` takes that keyword. The other scripts here import this one: touched.py and
shrink_case.py measure with `measure()`.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import align_case as A  # noqa: E402  the model build and the evaluator's environment

CORE = re.compile(r"core_(\d+)_(.+)")
MATRIX = re.compile(r"^\s*(?:@\S+\s+)?(m\.mma(?:\.\w+)*)\b", re.M)
DEADLINE = "200000000000"

# name -> (what it is, in the CA model's counters; the rule over the per-core sums `p` and the chip's `g`).
# The first twenty are CALIBRATION.json's; the last three no development point reached on the base.
RULES = {
    "all three cores": ("every core runs part of the program (core_N_instructions on all three)",
                        lambda p, g, c: 1 if sum(1 for k, v in c.items()
                                                 if CORE.match(k) and k.endswith("_instructions") and v) >= 3 else 0),
    "context parks": ("contexts parked and resumed (core_N_context_parks)", lambda p, g, c: p("context_parks")),
    "dirty writebacks": ("dirty lines written back from the home caches (uncore_writebacks)",
                         lambda p, g, c: g("uncore_writebacks")),
    "fences": ("fence stalls and drains (core_N_stall_fence, fence_drain_cycles)",
               lambda p, g, c: p("stall_fence") + p("fence_drain_cycles")),
    "fill chains": ("the home's chained line fills (uncore_fill_chains)", lambda p, g, c: g("uncore_fill_chains")),
    "fpu / sfu stalls": ("waits on the FPU or the SFU (core_N_stall_fpu, sfu_queue_cycles)",
                         lambda p, g, c: p("stall_fpu") + p("sfu_queue_cycles")),
    "home blocked": ("a home request blocked (uncore_blocked)", lambda p, g, c: g("uncore_blocked")),
    "home cache hits": ("home cache hits (uncore_hits)", lambda p, g, c: g("uncore_hits")),
    "home cache misses": ("home cache misses (uncore_misses)", lambda p, g, c: g("uncore_misses")),
    "home replays": ("home requests replayed (uncore_replays)", lambda p, g, c: g("uncore_replays")),
    "lsu stalls": ("waits on the load/store unit (core_N_stall_lsu, lsu_wait_cycles)",
                   lambda p, g, c: p("stall_lsu") + p("lsu_wait_cycles")),
    "matrix engine waits": ("waits on the matrix engine (core_N_stall_mtx, matrix_engine_wait_cycles)",
                            lambda p, g, c: p("matrix_engine_wait_cycles") + p("stall_mtx")),
    "memory request stalls (mreq)": ("requests to memory stalled (uncore_mreq_stall)",
                                     lambda p, g, c: g("uncore_mreq_stall")),
    "mshr merges": ("misses merged into an outstanding MSHR (uncore_merges)", lambda p, g, c: g("uncore_merges")),
    "reservation stalls": ("issue waits on a reservation (core_N_stall_rsv)", lambda p, g, c: p("stall_rsv")),
    "response / writeback queue full": ("the home's response or writeback queue full (uncore_rsq_full, wbq_full)",
                                        lambda p, g, c: g("uncore_rsq_full") + g("uncore_wbq_full")),
    "several launches": ("a program of more than one launch (chip_grid_launches + chip_single_launches)",
                         lambda p, g, c: max(0, g("chip_grid_launches") + g("chip_single_launches") - 1)),
    "transfer engine stalls": ("waits on the transfer engine (core_N_stall_txf)", lambda p, g, c: p("stall_txf")),
    "vrf bank conflicts": ("register-file bank conflicts (core_N_stall_vrf, vrf_bank_conflict_cycles)",
                           lambda p, g, c: p("stall_vrf") + p("vrf_bank_conflict_cycles")),
    "writeback steals": ("writeback port steals (core_N_stall_wb_steal)", lambda p, g, c: p("stall_wb_steal")),
    "barriers": ("barrier waits (core_N_stall_arrive, barrier_sleep_cycles)",
                 lambda p, g, c: p("stall_arrive") + p("barrier_sleep_cycles")),
    "memory port queueing": ("requests queued at a core's memory port (core_N_memory_port_queue_cycles)",
                             lambda p, g, c: p("memory_port_queue_cycles")),
    "home engine queueing": ("queueing in a home's engines (home_*_queue_cycles)",
                             lambda p, g, c: sum(v for k, v in c.items() if k.startswith("home_")
                                                 and k.endswith("queue_cycles") and isinstance(v, (int, float)))),
}


def reached(counters: dict) -> dict:
    """mechanism -> its count, for every mechanism the counters show taken (non-zero)."""
    per: dict = {}
    for k, v in counters.items():
        m = CORE.match(k)
        if m and isinstance(v, (int, float)):
            per[m.group(2)] = per.get(m.group(2), 0) + v

    def p(name):
        return per.get(name, 0)

    def g(name):
        v = counters.get(name) or 0
        return v if isinstance(v, (int, float)) else 0
    out = {}
    for name, (_, rule) in RULES.items():
        v = rule(p, g, counters)
        if v:
            out[name] = v
    return out


def static_matrix(dump_dir: Path) -> dict:
    """Every matrix instruction in the compiled assembly of the kernels dumped there, by mnemonic."""
    census: dict = {}
    for s in sorted(Path(dump_dir).glob("**/*.s")):
        for m in MATRIX.finditer(s.read_text(errors="ignore")):
            census[m.group(1)] = census.get(m.group(1), 0) + 1
    return census


def case_module(case: Path):
    """The case file as a module, with what a case may import on the path (files beside it; the test-case
    skill's scripts, where bound and protocol are)."""
    for extra in (case.resolve().parent, HERE.parents[1] / "testcase-design" / "scripts"):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(f"_case_{case.stem}", case)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make"):
        raise SystemExit(f"{case}: defines no make()")
    return mod


def sized(mod, dims: dict):
    """The callable that builds the case at `dims`, and the size it has without them: a development case's
    `_point_make(seed, **dims)`, else `make(**dims)`. Without dims it is the case's own `make()` (a case
    shrink_case.py wrote redefines it)."""
    fn, args = (mod._point_make, (0,)) if hasattr(mod, "_point_make") else (mod.make, ())
    if not dims:
        params = inspect.signature(fn).parameters
        return mod.make, {n: q.default for n, q in params.items()
                          if n != "seed" and q.default is not inspect.Parameter.empty}
    params = inspect.signature(fn).parameters
    for name in dims:
        if name not in params:
            raise SystemExit(f"the case has no dimension {name!r}: {'_point_make' if args else 'make'}() takes "
                             f"{[n for n in params if n != 'seed'] or 'none'}")
    defaults = {n: q.default for n, q in params.items() if n != "seed" and q.default is not inspect.Parameter.empty}
    return (lambda: fn(*args, **dims)), defaults


def child(case: str, out: Path, dims: dict) -> None:
    """Inside the case's process: compile it with TNCA as the backend, run it once, read the model's counters."""
    import tlm_backend                                       # noqa: E402  the model's backend, before torch
    tlm_backend.register()
    import torch
    import triton                                            # noqa: F401
    import tnca.inductor
    from tnca.triton_backend.driver import launch_log
    from torch._inductor.utils import run_and_get_code
    build, _ = sized(case_module(Path(case)), dims)
    module, inputs = build()
    rec = dict(status="not-measured", problems=[], launches=None, device_cycles=None, counters={})
    tnca.inductor.configure()
    torch._dynamo.reset()
    try:
        with torch.no_grad(), launch_log() as log:
            _, code = run_and_get_code(torch.compile(module), *tuple(inputs))
    except Exception as exc:                                 # noqa: BLE001 -- the failure is the result
        rec["status"] = "failed"
        rec["problems"].append(f"{type(exc).__name__}: {str(exc)[:800]}")
    else:
        rec["launches"] = len(log)
        rec["device_cycles"] = sum(e["device_cycles"] or 0 for e in log)
        fallbacks = tnca.inductor.host_fallbacks(code)
        rec["status"] = "pass" if log and not fallbacks else "unsupported"
        if fallbacks:
            rec["problems"].append(f"host fallback: {fallbacks}")
    for inst in tlm_backend.backend_class().instances:
        try:
            if inst.alive():
                for k, v in inst.counters_snapshot().items():
                    if isinstance(v, (int, float)):
                        rec["counters"][k] = rec["counters"].get(k, 0) + v
            inst.stop()
        except Exception as exc:                             # noqa: BLE001
            rec["problems"].append(f"counters: {type(exc).__name__}: {exc}")
    (out / "result.json").write_text(json.dumps(rec, indent=1) + "\n")


def measure(r: Path, case: Path, *, binary: Path, config: Path, profile: str, active_cores: int = 3,
            dims: dict | None = None, timeout: float = 7200) -> dict:
    """One case on the CA model in a fresh process (tree `r`, the model `binary` with `config`): its status, its
    cycles, its counters, the mechanisms they show and the matrix instructions of its compiled code."""
    work = Path(tempfile.mkdtemp(prefix="tnca-mech-"))
    try:
        env = A.side_env(r, work, active_cores, profile, timeout)
        env.update(TNCA_BACKEND="systemc-tlm-grid", TNCA_TLM_BINARY=str(binary), TNCA_TLM_CONFIG=str(config),
                   TNCA_TLM_OUT=str(work / "tlm"), TNCA_KERNEL_DUMP="1", TNCA_DEADLINE=DEADLINE)
        cmd = [sys.executable, __file__, "--child", str(case), "--out", str(work)]
        for k, v in (dims or {}).items():
            cmd += ["--dim", f"{k}={v}"]
        t0 = time.time()
        with (work / "run.log").open("w") as log:
            p = subprocess.run(cmd, cwd=r, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout + 600)
        res = work / "result.json"
        rec = json.loads(res.read_text()) if res.exists() else dict(
            status="no-result", problems=[f"exit {p.returncode}: " + (work / "run.log").read_text()[-800:]],
            counters={})
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["mechanisms"] = reached(rec.get("counters") or {})
        rec["static"] = static_matrix(work / "dump")
        rec.update(case=str(case), dims=dims or {}, config=str(config))
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


def model_for(r: Path, build: str, config: str) -> tuple:
    """The CA model of tree `r` (built incrementally in `build`) and its configuration's path in that tree."""
    cfg = Path(config) if Path(config).is_absolute() else r / config
    if not cfg.is_file():
        raise SystemExit(f"no model configuration {cfg}")
    return A.build_model(r, Path(build).resolve()), cfg.resolve()


def profile_default() -> str:
    return os.environ.get("TNCA_PROFILE", "core-vrf-sync-v5-x8-g2c4-dp-board")


def parse_dims(items: list) -> dict:
    out = {}
    for item in items or ():
        name, _, value = item.partition("=")
        if not name or not value:
            raise SystemExit(f"--dim {item!r}: NAME=VALUE")
        out[name] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return out


def show(rec: dict) -> str:
    dims = "".join(f" {k}={v}" for k, v in rec.get("dims", {}).items())
    head = (f"{rec['status']:12} {Path(rec['case']).name}{dims}  cycles {rec.get('device_cycles')}  "
            f"launches {rec.get('launches')}  ({rec.get('wall_s')} s)")
    lines = [head]
    for name, v in sorted(rec["mechanisms"].items()):
        lines.append(f"    {name:34} {v:>12}")
    if rec["static"]:
        lines.append(f"    matrix instructions (compiled code): {json.dumps(rec['static'])}")
    lines += [f"    ! {x}" for x in rec.get("problems") or ()]
    return "\n".join(lines)


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=A.CONFIG, help="the model configuration, in your tree (default: the CA model's)")
    ap.add_argument("--profile", default=profile_default())
    ap.add_argument("--active-cores", type=int, default=3)
    ap.add_argument("--model-build", default="/tmp/tnca-model", help="the model's build directory (built incrementally)")
    ap.add_argument("--timeout", type=float, default=7200, help="seconds per case")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+", help="case files, each with make()")
    common_args(ap)
    ap.add_argument("--dim", action="append", default=[], help="NAME=VALUE: build the case at this size")
    ap.add_argument("--json", type=Path, help="write one JSON record per case here")
    ap.add_argument("--list", action="store_true", help="print the mechanisms and their counters, and stop")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        child(a.cases[0], Path(a.out), parse_dims(a.dim))
        return 0
    if a.list:
        for name, (what, _) in RULES.items():
            print(f"{name:34} {what}")
        return 0
    r = A.root()
    binary, config = model_for(r, a.model_build, a.config)
    recs = []
    for case in a.cases:
        rec = measure(r, Path(case).resolve(), binary=binary, config=config, profile=a.profile,
                      active_cores=a.active_cores, dims=parse_dims(a.dim), timeout=a.timeout)
        print(show(rec), flush=True)
        recs.append(rec)
    if a.json:
        a.json.write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
    return 0 if all(x["status"] == "pass" for x in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
