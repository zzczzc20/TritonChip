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
           source_commit=None) -> dict:
    sys.path.insert(0, str(archbench_root))
    sys.path.insert(0, str(HERE.parent))
    from archbench import registry  # noqa: F401 -- the registry the stand-ins register into
    from tritonchip_agent_harness import skill_cases
    from tritonchip_agent_harness.islands import op_group
    from tritonchip_agent_harness.task import DEFAULT_OP_GROUP, DEV_SETS
    spec = importlib.util.spec_from_file_location("dev_standins", HERE / "dev_standins.py")
    standins = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standins)
    raw = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ev, isl = raw["evaluation"], raw.get("islands") or {}
    dev = list(dict.fromkeys(p for s in DEV_SETS for p in ev.get(s, [])))
    held = {p for s in ("final", "board_final") for p in ev.get(s, [])}
    card = json.loads(base_card.read_text(encoding="utf-8"))["points"]
    model = _rows(base_model)
    ids = dev + [s for s, p in standins.STANDS_IN_FOR.items() if p in dev]
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
        if pid != target:
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
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archbench", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True, help="the task's directory (task.json)")
    ap.add_argument("--base-card", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-commit")
    a = ap.parse_args(argv)
    index = export(a.archbench.resolve(), a.task.resolve(), a.base_card, a.base_model, a.out, a.source_commit)
    missing = [c["point"] for c in index["cases"] if not c["base_model_cycles"]]
    print(f"{len(index['cases'])} cases -> {a.out}" + (f"; no model cycles for {missing}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
