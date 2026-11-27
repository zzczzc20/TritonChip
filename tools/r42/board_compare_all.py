#!/usr/bin/env python3
"""Combined R42 yardstick: the 113 original ArchBench points plus the 32 xl points (145 points), each measured on the board
once for the R41 baseline (b41-run3-base, xl-base) and once for a candidate round (ab-intN, xl-candN).

usage: board_compare_all.py --orig evidence/r42/board/ab-int2/comparison.json --xl evidence/r42/board/xl-cand2/comparison.json [--out F]
Domain: board (U280), device-cycles-serial-grid, as computed by board_compare.py / board_compare_xl.py. Nothing is typed in.
"""
import argparse, json, math, sys

def rows_of(path):
    d = json.load(open(path))
    out = []
    for r in d["rows"]:
        out.append(dict(point=r["point"], family=r.get("family", r["point"].split("/")[0]), tier=r.get("tier", "xl"),
                        old=r["old_cycles"], new=r["new_cycles"]))
    return d, out

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True); ap.add_argument("--xl", required=True); ap.add_argument("--out")
a = ap.parse_args()
d1, r1 = rows_of(a.orig); d2, r2 = rows_of(a.xl)
for d, n in ((d1, a.orig), (d2, a.xl)):
    if d.get("failed_points") or d.get("problems"):
        print("NOT CERTIFIED: failures/problems in", n, file=sys.stderr)
rows = r1 + r2
sp = [r["old"] / r["new"] for r in rows]
res = dict(schema="r42-board-combined-comparison/1", domain="board (U280), device-cycles-serial-grid",
           inputs=dict(orig=a.orig, xl=a.xl), points=len(rows),
           speedup_geomean=gm(sp), speedup_geomean_113=gm([r["old"] / r["new"] for r in r1]),
           speedup_geomean_xl=gm([r["old"] / r["new"] for r in r2]),
           speedup_min=min(sp), speedup_max=max(sp), slower_than_2pct=[r["point"] for r in rows if r["new"] > 1.02 * r["old"]])
fam = {}
for r in rows:
    fam.setdefault(r["family"], []).append(r["old"] / r["new"])
res["by_family"] = {k: dict(points=len(v), speedup_geomean=gm(v)) for k, v in sorted(fam.items())}
tier = {}
for r in rows:
    tier.setdefault(r["tier"], []).append(r["old"] / r["new"])
res["by_tier"] = {k: dict(points=len(v), speedup_geomean=gm(v)) for k, v in sorted(tier.items())}
print(f"points {len(rows)}  speed-up geomean {res['speedup_geomean']:.3f}x  (113 original {res['speedup_geomean_113']:.3f}x, xl {res['speedup_geomean_xl']:.3f}x)"
      f"  min {res['speedup_min']:.2f} max {res['speedup_max']:.2f}  slower>2%: {len(res['slower_than_2pct'])}")
for k, v in res["by_tier"].items():
    print(f"  tier {k:8s} {v['points']:3d} points {v['speedup_geomean']:.3f}x")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
