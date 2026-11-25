#!/usr/bin/env python3
"""Run development points of the task (its cases in /opt/tnca-task/dev_cases) on a performance model, several
at once, and set each beside the base machine's cycles on the same model.

    python /opt/tnca-skills/testcase-design/scripts/dev_points.py [ID or FILE ...] [--group G ...] [--all]
        [--points] [--jobs N] [--json OUT] [--prediction OUT] [--profile P]

Which cases: the ids or case files you name, every case of an op group (`--group`, the index's `group`),
or `--all`. A point that has a stand-in (a smaller program of the same operator, for the models) runs as its
stand-in unless you pass `--points`; the index says how long each case took on the CA model for the base
(`model_minutes`), so you can tell what a run will cost before you start it.

Each case runs through tnca_case.py in a process of its own (a model keeps its caches from one case to the
next within a process), up to `--jobs` at once (default: half the CPUs). The model is the one your environment
selects, as for tnca_case.py: set TNCA_BACKEND and the model's binary and configuration as the task text says.
A large point needs more device cycles than the driver's default deadline allows; unless you set
TNCA_DEADLINE yourself it is set high here.

For each case: its status, its cycles, the base machine's cycles on the CA model (`base_model_cycles` in
the index, measured on the CA model with the base's configuration), their ratio -- the speedup this model
predicts for your machine and compiler against the base -- and how far the CA model was from the card on that
program for the base (`base_model_over_card`). The ratio is a prediction only if the model you ran is the
CA model with a configuration (and source) that matches your machine; on any other model it is that model's
estimate. `--prediction OUT` writes the passing cases' ratios as a prediction file for `tnca-service submit
score|flow --prediction OUT` (a stand-in's id counts for its point).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV_CASES = Path(os.environ.get("TNCA_DEV_CASES", "/opt/tnca-task/dev_cases"))
DEADLINE = "200000000000"


def cpus() -> int:
    """The CPUs this container may use: its cgroup quota when it has one (a sandbox's --cpus), else the host's."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 2


def load_index(root: Path) -> dict:
    index = root / "INDEX.json"
    if not index.is_file():
        sys.exit(f"no development cases at {root} (this task gives none, or set TNCA_DEV_CASES)")
    return json.loads(index.read_text(encoding="utf-8"))


def select(index: dict, names: list, groups: list, every: bool, points_only: bool) -> list:
    cases = index["cases"]
    by_point = {c["point"]: c for c in cases}
    by_file = {c["file"]: c for c in cases}
    standin_of = {c["stands_in_for"]: c for c in cases if c.get("stands_in_for")}
    chosen = []
    for n in names:
        c = by_point.get(n) or by_file.get(Path(n).name)
        if c is None:
            sys.exit(f"{n}: not a development case (see INDEX.json)")
        chosen.append(c)
    for c in cases:
        if c.get("stands_in_for") or c.get("calibration"):
            continue          # a stand-in comes with its point; a calibration case is not a development point
        if every or c.get("group") in groups:
            chosen.append(c)
    if not points_only:
        chosen = [standin_of.get(c["point"], c) if not c.get("stands_in_for") else c for c in chosen]
    return list({c["point"]: c for c in chosen}.values())


def run_one(case: dict, root: Path, profile: str, out: Path) -> dict:
    env = dict(os.environ)
    env.setdefault("TNCA_DEADLINE", DEADLINE)
    rec_file = out / (case["file"] + ".jsonl")
    cmd = [sys.executable, str(HERE / "tnca_case.py"), str(root / case["file"]), "--profile", profile,
           "--kernels", "0", "--json", str(rec_file)]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rec = None
    if rec_file.exists():
        lines = [x for x in rec_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        rec = json.loads(lines[-1]) if lines else None
    cycles = ((rec or {}).get("time") or {}).get("device_cycles")
    status = (rec or {}).get("status") or f"no-record (exit {proc.returncode})"
    base = case.get("base_model_cycles")
    row = {"point": case["point"], "stands_in_for": case.get("stands_in_for"), "status": status,
           "cycles": cycles, "base_model_cycles": base,
           "speedup_on_model": round(base / cycles, 4) if status == "pass" and cycles and base else None,
           "base_model_over_card": case.get("base_model_over_card"), "eta": (rec or {}).get("eta")}
    if status != "pass":
        row["tail"] = proc.stdout[-800:]
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="development point ids or case files")
    ap.add_argument("--group", action="append", default=[], help="every case of this op group")
    ap.add_argument("--all", action="store_true", help="every development point")
    ap.add_argument("--points", action="store_true", help="the points themselves, not their stand-ins")
    ap.add_argument("--jobs", type=int, default=max(1, cpus() - 1), help="cases at once (default: the CPUs less one)")
    ap.add_argument("--profile", default=os.environ.get("TNCA_PROFILE", "core-vrf-sync-v5-x8-g2c4-dp-board"))
    ap.add_argument("--json", type=Path, help="write the rows here, one JSON object per line")
    ap.add_argument("--prediction", type=Path, help="write the passing cases' model speedups as a prediction file")
    a = ap.parse_args(argv)
    root = DEV_CASES
    index = load_index(root)
    cases = select(index, a.names, a.group, a.all, a.points)
    if not cases:
        ap.error("name cases, --group G or --all")
    minutes = sum(c.get("model_minutes") or 0 for c in cases)
    longest = max((c.get("model_minutes") or 0 for c in cases), default=0)
    print(f"{len(cases)} cases, {a.jobs} at once; the CA model took {minutes:.0f} min in all on the base, the "
          f"longest {longest:.0f} min", flush=True)
    with tempfile.TemporaryDirectory(prefix="dev-points-") as tmp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            rows = list(pool.map(lambda c: run_one(c, root, a.profile, Path(tmp)), cases))
    for r in rows:
        ratio = f"{r['speedup_on_model']:.3f}x" if r["speedup_on_model"] else "-"
        card = f"model/card on base {r['base_model_over_card']:.3f}" if r.get("base_model_over_card") else ""
        name = r["point"] + (f" (for {r['stands_in_for']})" if r.get("stands_in_for") else "")
        print(f"{r['status']:14} {name:70} {r['cycles'] or '-':>12} vs base {r['base_model_cycles'] or '-':>12}  "
              f"{ratio:>8}  {card}")
    if a.json:
        a.json.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if a.prediction:
        pred = {r["point"]: r["speedup_on_model"] for r in rows if r["speedup_on_model"]}
        a.prediction.write_text(json.dumps(pred, indent=1) + "\n", encoding="utf-8")
        print(f"prediction for {len(pred)} points -> {a.prediction}")
    return 0 if all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
