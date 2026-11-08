#!/usr/bin/env python3
"""The model on every variant: its own configuration (make_config.py from the variant's device and
profile) and its own host tree (the base plus the candidate, as the evaluator builds it).

    python3 run_models.py --harness H --site SITE --variants VARIANTS_DIR --model-tree MODEL_TREE \
        --skill SKILL --items items/visible17.json [--items items/holdout.json] --out OUT [--jobs 6] [--only base,lsu2]

OUT/<variant>/config.cfg, OUT/<variant>/<items stem>/results.json (model.py's output). The trees
are rebuilt only when missing.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVICE = "spec/devices/u280-3core-r41.json"
PROFILES = "spec/r1_profiles.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--model-tree", type=Path, required=True)
    ap.add_argument("--skill", type=Path, required=True)
    ap.add_argument("--items", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only")
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness))
    from tritonchip_agent_harness import evaluator as E
    from tritonchip_agent_harness.siteconfig import load_site
    from tritonchip_agent_harness.task import load_task
    site = load_site(a.site)
    task = load_task(a.harness / "tritonchip_agent_harness/tasks/r41-dp-formal/task.json")
    repo = Path(os.path.expanduser(str(site.repository)))
    systemc = a.model_tree / "tools/performance_harness/systemc"
    names = ["base"] + sorted(p.name for p in a.variants.iterdir() if (p / "candidate.json").exists())
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    for v in names:
        cand = None if v == "base" else (a.variants / v).resolve()
        d = (a.out / v).resolve()
        d.mkdir(parents=True, exist_ok=True)
        host = d / "trees" / "host-tree"
        if not host.exists():
            E._trees(repo, task, cand, d / "trees")
        dev = host / DEVICE
        subprocess.run([sys.executable, str(HERE / "make_config.py"), "--device", str(dev), "--profiles", str(host / PROFILES),
                        "--base", str(systemc / "configs/r41-hbm8-device-service.cfg"), "--out", str(d / "config.cfg"),
                        "--sources", str(d / "config.sources.json")], check=True)
        for items in a.items:
            out = d / items.stem
            print(f"== {v} {items.stem}", flush=True)
            subprocess.run([sys.executable, str(HERE / "model.py"), "--host-tree", str(host), "--model-dir", str(systemc),
                            "--binary", str(a.model_tree / ".model-build/tnca-chip"), "--config", str(d / "config.cfg"),
                            "--items", str(items), "--skill", str(a.skill), "--out", str(out), "--jobs", str(a.jobs)],
                           check=True)


if __name__ == "__main__":
    main()
