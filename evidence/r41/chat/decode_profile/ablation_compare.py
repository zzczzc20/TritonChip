#!/usr/bin/env python3
"""Per-launch device cycles of two decode_profile.py runs of the same program (default vs a model-config ablation).
usage: ablation_compare.py DEFAULT_launches.json[.gz] ABLATION_launches.json[.gz] STEP [STEP ...]
Launches are matched by position within the step; the kernel names must agree."""
import gzip, json, sys
def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))["launches"]
a, b = load(sys.argv[1]), load(sys.argv[2])
for step in map(int, sys.argv[3:]):
    ra = [r for r in a if r["step"] == step]; rb = [r for r in b if r["step"] == step]
    assert [r["kernel"] for r in ra] == [r["kernel"] for r in rb], "launch sequences differ"
    ta = sum(r["device_cycles"] for r in ra); tb = sum(r["device_cycles"] for r in rb)
    print(f"step {step} ({'prefill' if step == 0 else 'decode'}): {len(ra)} launches, total {ta} -> {tb} cycles, ratio {tb / ta:.4f}")
    fa = fb = 0
    for x, y in zip(ra, rb):
        if x["kernel"].startswith("w8a8"):
            fa += x["device_cycles"]; fb += y["device_cycles"]
            w = sum(t["view_bytes"] for t in x["args"] if t.get("role") == "weight")
            print(f"  #{x['index']:>4} w8a8 grid {x['grid']} weights {w}: {x['device_cycles']} -> {y['device_cycles']} "
                  f"ratio {y['device_cycles'] / x['device_cycles']:.4f}")
    print(f"  all FP8 launches: {fa} -> {fb} ratio {fb / fa:.4f}; non-FP8: {ta - fa} -> {tb - fb} ratio {(tb - fb) / (ta - fa):.4f}")
