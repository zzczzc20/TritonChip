#!/usr/bin/env python3
"""The change-acceptance variants of the R41 calibration (decided 2026-11-08), as harness candidates.

    python3 make_variants.py REPO OUT [--compiler-candidate DIR]

Each variant is the base a076cd64 with one or more hardware parameters changed,
written as a candidate directory the agent harness evaluator takes
(candidate.json + files/):

  lsu2      profile LSU_INFLIGHT 1 -> 2
  sets64    device uncore.SETS 32 -> 64
  vrfreg0   profile MTX_VRF_REG 1 -> 0
  combo     all three (the hardware of run 6180's best snapshot 0-8)
  compiler  a copy of a compiler-only candidate (run 6180's s3: python/tnca/triton_backend/compiler.py)
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

BASE = "a076cd6448e00e4aa7a9ac9edb3e0048387b146c"
PROFILE = "core-vrf-sync-v5-x8-g2c4-dp-board"
PROFILES = "spec/r1_profiles.json"
DEVICE = "spec/devices/u280-3core-r41.json"
HARDWARE = ("rtl/", "spec/r1_profiles.json", "spec/devices/")


def show(repo, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{BASE}:{path}"], check=True,
                          capture_output=True).stdout.decode()


def write(out, files):
    shutil.rmtree(out, ignore_errors=True)
    for path, text in files.items():
        f = out / "files" / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    (out / "candidate.json").write_text(json.dumps(
        {"base_commit": BASE, "taken": [{"path": p, "status": "modified"} for p in sorted(files)],
         "dropped": [], "made_by": "r41cal/make_variants.py"}, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--compiler-candidate", type=Path)
    a = ap.parse_args()
    prof_text, dev_text = show(a.repo, PROFILES), show(a.repo, DEVICE)

    def profile(**params):
        d = json.loads(prof_text)
        rp = d["profiles"][PROFILE]["implementation"]["rtl_parameters"]
        for k, v in params.items():
            assert k in rp, k
            rp[k] = v
        return json.dumps(d, indent=2) + "\n"

    def device(**uncore):
        d = json.loads(dev_text)
        for k, v in uncore.items():
            assert k in d["uncore"], k
            d["uncore"][k] = v
        return json.dumps(d, indent=2) + "\n"

    base_prof = json.loads(prof_text)["profiles"][PROFILE]["implementation"]["rtl_parameters"]
    base_dev = json.loads(dev_text)["uncore"]
    assert (base_prof["LSU_INFLIGHT"], base_prof["MTX_VRF_REG"], base_dev["SETS"]) == (1, 1, 32)
    write(a.out / "lsu2", {PROFILES: profile(LSU_INFLIGHT=2)})
    write(a.out / "sets64", {DEVICE: device(SETS=64)})
    write(a.out / "vrfreg0", {PROFILES: profile(MTX_VRF_REG=0)})
    write(a.out / "combo", {PROFILES: profile(LSU_INFLIGHT=2, MTX_VRF_REG=0), DEVICE: device(SETS=64)})
    if a.compiler_candidate:
        src = a.compiler_candidate
        rep = json.loads((src / "candidate.json").read_text())
        assert rep["base_commit"] == BASE
        assert not any(c["path"].startswith(HARDWARE) for c in rep["taken"]), "not compiler-only"
        dst = a.out / "compiler"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    print("variants in", a.out)


if __name__ == "__main__":
    main()
