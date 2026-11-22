"""Export a task's development set as cases for the agent (decided 2026-11-23: the points split by role).

The development set is the set the run is scored on -- `evaluation.validation` and `evaluation.board_validation`
-- given to the agent, one case per point exactly as skill_cases.py writes the skill's benchmark cases (the
module and make() copied from the benchmark as code, seed 0, its tolerance and declared demand), plus the
stand-ins (dev_standins.py) of the points too long for the CA model. INDEX.json says of each case what the
agent may use to choose what to run: its family and op group (the island directions), its base machine's
cycles on the card and on the CA model, how long the CA model took on it, and for a stand-in which point it
stands in for. Held-out points are never among them (task._check_dev_cases).

Run where torch and archbench import (the toolchain image), against the base commit's tree:

    /opt/torchenv/bin/python -m tritonchip_agent_harness.dev_cases --archbench TREE --task TASK_DIR \
        --base-card BOARD_BASELINE.json --base-model RESULTS.log [--base-model ...] --out TASK_DIR/dev_cases

`--base-card`: the harness's cached base measurement on the card (board-baselines/<key>.json); `--base-model`:
run_graphs.py results (one JSON row per point: key, device_cycles, process_wall_s) of the base on the CA model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

SCHEMA = "tritonchip-dev-cases/1"
HERE = Path(__file__).resolve().parent


def _rows(paths: list) -> dict:
    out: dict = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                if r.get("device_cycles"):
                    out[r["key"]] = r
    return out


def export(archbench_root: Path, task_dir: Path, base_card: Path, base_model: list, out: Path,
           source_commit=None, calibration: dict | None = None) -> dict:
    sys.path.insert(0, str(archbench_root))
    sys.path.insert(0, str(HERE.parent))
    from archbench import registry  # noqa: F401 -- the registry the stand-ins register into
    from tritonchip_agent_harness import skill_cases
    from tritonchip_agent_harness.islands import op_group
    from tritonchip_agent_harness.task import DEFAULT_OP_GROUP, DEV_SETS
    mods = {}
    for name in ("dev_standins", "calib_points"):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    standins, calib = mods["dev_standins"], mods["calib_points"]
    raw = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ev, isl = raw["evaluation"], raw.get("islands") or {}
    dev = list(dict.fromkeys(p for s in DEV_SETS for p in ev.get(s, [])))
    held = {p for s in ("final", "board_final") for p in ev.get(s, [])}
    card = json.loads(base_card.read_text(encoding="utf-8"))["points"]
    model = _rows(base_model)
    ids = dev + [s for s, p in standins.STANDS_IN_FOR.items() if p in dev] + list(calib.CALIBRATION)
    cases, sources = [], {}
    for pid in ids:
        target = standins.STANDS_IN_FOR.get(pid, pid)
        if pid in held or target in held:
            raise skill_cases.ExportError(f"{pid}: a held-out point is never a development case")
        point = registry.get(pid)
        name = skill_cases.file_name(pid)
        if name in sources:
            raise skill_cases.ExportError(f"two points map to {name}")
        sources[name] = skill_cases.case_source(point, source_commit)
        m = model.get(pid) or {}
        c = (card.get(pid) or {}) if pid == target else {}
        entry = {"point": pid, "file": name, "kind": point.kind, "family": point.family, "tier": point.tier,
                 "dtype": point.dtype, "out_dtype": point.out_dtype, "tolerance": point.tolerance,
                 "group": op_group(target, isl.get("op_groups") or {}, isl.get("default_op_group") or DEFAULT_OP_GROUP),
                 "base_card_cycles": c.get("cycles"),
                 "base_model_cycles": m.get("device_cycles"),
                 "model_minutes": round(m["process_wall_s"] / 60.0, 1) if m.get("process_wall_s") else None}
        if pid in calib.CALIBRATION:
            entry["calibration"] = True       # not scored: a case for the trace gate (CALIBRATION.json)
        elif pid != target:
            entry["stands_in_for"] = target
        elif c.get("cycles") and m.get("device_cycles"):
            # How far the CA model was from the card on the base: the model's error on this program.
            entry["base_model_over_card"] = round(m["device_cycles"] / c["cycles"], 4)
        cases.append(entry)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, src in sources.items():
        (out / name).write_text(src, encoding="utf-8")
    index = {"schema": SCHEMA, "seed": skill_cases.SEED, "source_commit": source_commit, "cases": cases}
    (out / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if calibration is not None:
        (out / "CALIBRATION.json").write_text(json.dumps(calibration_index(calibration, cases), indent=1) + "\n",
                                              encoding="utf-8")
    return index


# Where a calibration case is, in the sandbox: the task's development cases or the test-case skill's own.
SKILL_CASES = "/opt/tnca-skills/testcase-design/benchmark_cases"
DEV_MOUNT = "/opt/tnca-task/dev_cases"
# What each mechanism is, in the CA model's counters (counters.json), for the agent's choice of cases.
MECHANISMS = {
    "all three cores": "every core runs part of the program (core_N_instructions on all three)",
    "context parks": "contexts parked and resumed (core_N_context_parks)",
    "dirty writebacks": "dirty lines written back from the home caches (uncore_writebacks)",
    "fences": "fence stalls and drains (core_N_stall_fence, fence_drain_cycles)",
    "fill chains": "the home's chained line fills (uncore_fill_chains)",
    "fpu / sfu stalls": "waits on the FPU or the SFU (core_N_stall_fpu, sfu_queue_cycles)",
    "home blocked": "a home request blocked (uncore_blocked)",
    "home cache hits": "home cache hits (uncore_hits)",
    "home cache misses": "home cache misses (uncore_misses)",
    "home replays": "home requests replayed (uncore_replays)",
    "lsu stalls": "waits on the load/store unit (core_N_stall_lsu, lsu_wait_cycles)",
    "matrix engine waits": "waits on the matrix engine (core_N_stall_mtx, matrix_engine_wait_cycles)",
    "memory request stalls (mreq)": "requests to memory stalled (uncore_mreq_stall)",
    "mshr merges": "misses merged into an outstanding MSHR (uncore_merges)",
    "reservation stalls": "issue waits on a reservation (core_N_stall_rsv)",
    "response / writeback queue full": "the home's response or writeback queue full (uncore_rsq_full, wbq_full)",
    "several launches": "a program of more than one launch (chip_grid_launches + chip_single_launches)",
    "transfer engine stalls": "waits on the transfer engine (core_N_stall_txf)",
    "vrf bank conflicts": "register-file bank conflicts (core_N_stall_vrf, vrf_bank_conflict_cycles)",
    "writeback steals": "writeback port steals (core_N_stall_wb_steal)",
}


def calibration_index(sel: dict, cases: list) -> dict:
    """CALIBRATION.json: the chosen cases, and for every mechanism the cases that reach it, smallest first."""
    here = {c["point"]: f"{DEV_MOUNT}/{c['file']}" for c in cases}
    skill = json.loads((HERE / "skills" / "testcase-design" / "benchmark_cases" / "INDEX.json").read_text())
    here.update({c["point"]: f"{SKILL_CASES}/{c['file']}" for c in skill["cases"] if c["point"] not in here})

    def at(row):
        return {"point": row["point"], "cycles": row["cycles"], "case": here.get(row["point"])}
    return {"about": ("The calibration set (2026-11-23): small cases that together reach every mechanism the "
                      "development points reach in the CA model's counters on the base; run the ones that reach "
                      "what you changed, on the RTL and on the model, with the trace gate. `chosen` covers all "
                      "of them that any case the RTL simulator can run reaches; `by_mechanism` lists every such "
                      "case, smallest first; `unreached` names what only larger programs reach."),
            "mechanisms": {m: MECHANISMS.get(m, m) for m in sel["mechanisms"]},
            "chosen": [dict(at(r), covers=r["covers"]) for r in sel["chosen"]],
            "by_mechanism": {m: [at(r) for r in rows if here.get(r["point"])] for m, rows in sel["by_mechanism"].items()},
            "unreached": sel.get("unreached", {}),
            "reached_by_larger_points": sel.get("support", {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archbench", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True, help="the task's directory (task.json)")
    ap.add_argument("--base-card", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit")
    ap.add_argument("--calibration", type=Path, help="the calibration selection (calib_cover.py --json)")
    a = ap.parse_args(argv)
    sel = json.loads(a.calibration.read_text()) if a.calibration else None
    index = export(a.archbench.resolve(), a.task.resolve(), a.base_card, a.base_model, a.out, a.source_commit, sel)
    missing = [c["point"] for c in index["cases"] if not c["base_model_cycles"]]
    print(f"{len(index['cases'])} cases -> {a.out}" + (f"; no model cycles for {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
