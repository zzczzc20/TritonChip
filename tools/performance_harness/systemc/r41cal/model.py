#!/usr/bin/env python3
"""The SystemC model on the calibration items: the same programs truth.py measured, the backend swapped.

    python3 model.py --host-tree WORK/<items>-<variant>/host-tree --model-dir SYSTEMC_DIR \
        --binary SYSTEMC_DIR/build/tnca-chip --config CFG --items items.json --skill SKILL --out OUT \
        [--jobs 12] [--env KEY=VALUE ...]

The host tree is truth.py's for that variant (the base, or the base plus the
candidate): its compiler and profile compile the programs the RTL ran. The
model is this branch's (SYSTEMC_DIR, built for the agent image), registered
in every Python process by a sitecustomize, as run_graphs.py's children do.
Points run with `python -m archbench.run --point`, cases with the skill's
tnca_case.py, with the evaluator's runtime environment (three active cores,
the physical-grid backend). The model's count is the device's count as the
driver sees it, so it is compared with the RTL's record_cycles (the
observer's done cycles), not with the boundary cycles.

OUT/results.json: {item id: {status, cycles, launches, wall_s}}.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

SITE = '''import os, sys
if os.environ.get("TNCA_BACKEND", "").startswith("systemc-tlm"):
    sys.path.append("/model")
    try:
        import tlm_backend
        tlm_backend.register()
        try:
            from archbench.platforms import tnca as _platform
            for _name in tlm_backend.NAMES:
                _platform.EVIDENCE[_name] = tlm_backend.EVIDENCE
        except ImportError:
            pass
    except Exception as exc:
        sys.stderr.write(f"sitecustomize: tlm_backend.register failed: {type(exc).__name__}: {exc}\\n")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tree", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--image", default="tritonchip-agent:dev")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--profile", default="core-vrf-sync-v5-x8-g2c4-dp-board")
    a = ap.parse_args()

    out = a.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    (out / "site").mkdir(parents=True)
    (out / "site" / "sitecustomize.py").write_text(SITE)
    cases = out / "cases"
    cases.mkdir()
    shutil.copy2(a.config, out / "model.cfg")
    items = json.loads(a.items.read_text())
    env = (f'TNCA_BACKEND=systemc-tlm-grid TNCA_TLM_BINARY=/bin-model/{a.binary.name} TNCA_TLM_CONFIG=/out/model.cfg '
           f'TNCA_TLM_OUT=/out/tlm-$1 TNCA_COMPILE_CACHE_DIR=/out/caches/tnca-inductor TNCA_ACTIVE_CORES=3 '
           f'TNCA_HOST_TIMEOUT=10800 PYTHONPATH=/out/site ' + " ".join(a.env))
    lines = ["set -e", "cd /work", "mkdir -p /out/caches", "make -s python/tnca/data/.staged > /out/stage.log 2>&1",
             "run_point() {", "  t0=$(date +%s)",
             f'  {env} /opt/torchenv/bin/python -m archbench.run --platform tnca --profile {a.profile} '
             f'--point "$2" --out /out/archbench-$1.jsonl > /out/run-$1.log 2>&1 || true',
             '  echo "$1 $(( $(date +%s) - t0 ))" >> /out/times.txt', "}",
             "run_case() {", "  t0=$(date +%s)",
             f'  {env} TNCA_ROOT=/work /opt/torchenv/bin/python /opt/tnca-skills/testcase-design/scripts/tnca_case.py '
             f'"/out/cases/$2" --profile {a.profile} --json /out/case-$1.jsonl > /out/run-$1.log 2>&1 || true',
             '  echo "$1 $(( $(date +%s) - t0 ))" >> /out/times.txt', "}"]
    for n in range(0, len(items), a.jobs):
        for i in range(n, min(n + a.jobs, len(items))):
            it = items[i]
            if it["kind"] == "case":
                shutil.copy2(it["file"], cases / f"{i}-{Path(it['file']).name}")
                lines.append(f"run_case {i} '{i}-{Path(it['file']).name}' &")
            else:
                lines.append(f"run_point {i} '{it['id']}' &")
        lines.append("wait")
    (out / "run.sh").write_text("\n".join(lines) + "\n")
    name = f"r41cal-model-{int(time.time())}"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none",
           "-v", f"{a.host_tree.resolve()}:/work", "-v", f"{out}:/out",
           "-v", f"{a.model_dir.resolve()}:/model:ro", "-v", f"{a.binary.resolve().parent}:/bin-model:ro",
           "-v", f"{a.skill.resolve()}:/opt/tnca-skills/testcase-design:ro",
           "-w", "/work", a.image, "sh", "/out/run.sh"]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    times = {}
    if (out / "times.txt").exists():
        for line in (out / "times.txt").read_text().split("\n"):
            if line.strip():
                i, s = line.split()
                times[int(i)] = int(s)
    res = {}
    for i, it in enumerate(items):
        rec = None
        path = out / (f"archbench-{i}.jsonl" if it["kind"] == "point" else f"case-{i}.jsonl")
        for line in (path.read_text().splitlines() if path.exists() else []):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if it["kind"] == "case" or r.get("point_id") == it["id"]:
                rec = r
        if rec is None:
            res[it["id"]] = {"status": "no-result", "cycles": None}
            continue
        t = rec.get("time") or {}
        cycles = t.get("device") if it["kind"] == "point" else t.get("device_cycles")
        res[it["id"]] = {"status": rec.get("status"), "cycles": cycles,
                         "launches": t.get("launches") if it["kind"] == "case" else None, "wall_s": times.get(i)}
    (out / "results.json").write_text(json.dumps({"config": str(a.config), "binary": str(a.binary),
                                                   "host_tree": str(a.host_tree), "exit": rc,
                                                   "wall_s": round(time.time() - t0), "items": res}, indent=1) + "\n")
    print(f"exit {rc}; {sum(r['status'] == 'pass' for r in res.values())}/{len(res)} pass; {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()
