#!/usr/bin/env python3
"""R42 issue: per-point cycle partition of single-core RTL runs (archbench.run --counters records), and the upper bounds
that removing the vector-operand stalls would give.  Domain: rtl-1c (single-core RTL simulation), never board.

  python3 cnt_table.py RECORDS.jsonl [RECORDS2.jsonl ...]

Columns are fractions of perf_cycles: issued, x_hazard (R waits for a producer), x_vrf (operands not yet collected: the
second R cycle of a non-prefetched read), x_lsu/x_fpu/x_mtx/x_txf/x_fence/x_arrive (other stalls), x_rsv/x_wb (writeback
reservation / steal), front (nothing to issue).  bound_v = cycles / (cycles - x_vrf), bound_vh = cycles / (cycles - x_vrf -
x_hazard): what removing those stall cycles would give if nothing else moved (a loose upper bound: the cycles saved may
be waited out elsewhere).
"""
import json
import math
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("status") != "pass" or not r.get("counters"):
            continue
        out[r["point_id"]] = r["counters"]
    return out


def main():
    recs = {}
    for p in sys.argv[1:]:
        recs.update(load(p))
    cols = ["perf_issued", "perf_x_hazard", "perf_x_vrf", "perf_x_lsu", "perf_x_fpu", "perf_x_mtx", "perf_x_txf",
            "perf_x_fence", "perf_x_arrive", "perf_x_rsv", "perf_x_wb", "perf_x_frontend"]
    hdr = ["point", "cycles"] + [c.replace("perf_", "").replace("x_", "") for c in cols] + ["bound_v", "bound_vh"]
    print("\t".join(hdr))
    gv, gvh, n = 0.0, 0.0, 0
    tot = {c: 0 for c in cols}
    totc = 0
    for pid in sorted(recs):
        c = recs[pid]
        cyc = c["perf_cycles"]
        fr = [c.get(k, 0) / cyc for k in cols]
        bv = cyc / (cyc - c.get("perf_x_vrf", 0))
        bvh = cyc / (cyc - c.get("perf_x_vrf", 0) - c.get("perf_x_hazard", 0))
        gv += math.log(bv)
        gvh += math.log(bvh)
        n += 1
        totc += cyc
        for k in cols:
            tot[k] += c.get(k, 0)
        print("\t".join([pid, str(cyc)] + [f"{x:.3f}" for x in fr] + [f"{bv:.3f}", f"{bvh:.3f}"]))
    if n:
        print("\t".join(["TOTAL(cycle-weighted)", str(totc)] + [f"{tot[k] / totc:.3f}" for k in cols] +
                        [f"{totc / (totc - tot['perf_x_vrf']):.3f}", f"{totc / (totc - tot['perf_x_vrf'] - tot['perf_x_hazard']):.3f}"]))
        print("\t".join(["GEOMEAN(per point)", str(n)] + [""] * len(cols) + [f"{math.exp(gv / n):.3f}", f"{math.exp(gvh / n):.3f}"]))


if __name__ == "__main__":
    main()
