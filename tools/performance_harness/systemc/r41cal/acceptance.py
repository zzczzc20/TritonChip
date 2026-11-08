#!/usr/bin/env python3
"""Judge the R41 model against the RTL ground truth: the author's absolute rule and the change rules.

    python3 acceptance.py --store ~/tritonchip-rtl-truth --model base=DIR [--model lsu2=DIR ...] \
        --items items/visible17.json --items items/holdout.json [--out report.json]

DIR is a model.py output (results.json) for that variant; a variant may be
given several times (its visible and its holdout runs), and the runs merge. RTL cycles are the
records' record_cycles (the device's count as the driver sees it, what the
model emulates).

Absolute (doc/r40_tasks_and_acceptance.md sec 3.4, the author's rule): on the
holdout, the median |model/RTL - 1| <= 10 % and its p95 <= 20 %.

Change (decided 2026-11-08), per variant v against the base, on items both
measured and passed: S = cycles(base) / cycles(v), for the RTL and the model.
  1. direction: of the items whose RTL change |S_rtl - 1| >= 3 %, at least 90 %
     have the model's change in the same direction;
  2. size: over those items, the captured gain (G_model - 1) / (G_rtl - 1),
     G the geometric mean of S, lies in [0.75, 1.25];
  3. no false gains: every item whose RTL change is under 1 % has a model
     change of at most 3 %.
A variant with no item changing by 3 % passes rules 1-2 vacuously and is
reported as such. The verdict uses the holdout; the visible (calibration)
items are reported beside it. A failing variant is not a failing model: its
knob is marked "model blind: RTL only" in the error card.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def family(item_id):
    if item_id.startswith(("holdout:", "example:")):
        name = item_id.split(":", 1)[1]
        return "example" if item_id.startswith("example:") else name.split("_")[0] if not name.startswith(
            ("attention", "w8a8", "wfp8")) else "_".join(name.split("_")[:2])
    return item_id.split("/")[0].split(".", 1)[1]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def rtl_table(store):
    out = defaultdict(dict)
    for f in (store / "records").glob("*.json"):
        r = json.loads(f.read_text())
        res = r["result"]
        if res.get("status") == "pass" and res.get("record_cycles"):
            out[r["variant"]][r["item"]["id"]] = int(res["record_cycles"])
    return out


def model_table(spec):
    out = {}
    for s in spec:
        v, d = s.split("=", 1)
        res = json.loads((Path(d) / "results.json").read_text())["items"]
        out.setdefault(v, {}).update(
            {k: int(x["cycles"]) for k, x in res.items() if x.get("status") == "pass" and x.get("cycles")})
    return out


def change(rb, rv, mb, mv, ids):
    rows = []
    for i in ids:
        if i in rb and i in rv and i in mb and i in mv:
            rows.append((i, rb[i] / rv[i], mb[i] / mv[i]))
    big = [(i, sr, sm) for i, sr, sm in rows if abs(sr - 1) >= 0.03]
    small = [(i, sr, sm) for i, sr, sm in rows if abs(sr - 1) < 0.01]
    right = sum(1 for _, sr, sm in big if (sr - 1) * (sm - 1) > 0)
    g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    captured = None
    if big:
        gr, gm = g([sr for _, sr, _ in big]), g([sm for _, _, sm in big])
        captured = (gm - 1) / (gr - 1) if abs(gr - 1) > 1e-9 else None
    false = [(i, round(sm, 4)) for i, sr, sm in small if abs(sm - 1) > 0.03]
    r1 = (right / len(big) >= 0.9) if big else None
    r2 = (captured is not None and 0.75 <= captured <= 1.25) if big else None
    r3 = not false
    verdict = "no-signal" if not big and r3 else ("pass" if (r1 and r2 and r3) else "fail")
    return {"items": len(rows), "changed_3pct": len(big), "direction_right": right,
            "direction_rate": round(right / len(big), 3) if big else None,
            "captured_gain": round(captured, 3) if captured is not None else None,
            "geomean_rtl": round(g([sr for _, sr, _ in rows]), 4) if rows else None,
            "geomean_model": round(g([sm for _, _, sm in rows]), 4) if rows else None,
            "unchanged_1pct": len(small), "false_gains": false,
            "rules": {"direction": r1, "size": r2, "no_false_gain": r3}, "verdict": verdict,
            "per_item": [(i, round(sr, 4), round(sm, 4)) for i, sr, sm in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--items", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rtl, model = rtl_table(a.store.expanduser()), model_table(a.model)
    sets = defaultdict(list)
    for f in a.items:
        for it in json.loads(f.read_text()):
            sets[it.get("set", "visible")].append(it["id"])
    report = {"absolute": {}, "change": {}, "families": {}}
    rb, mb = rtl.get("base", {}), model.get("base", {})
    for s, ids in sets.items():
        errs = [(i, mb[i] / rb[i] - 1) for i in ids if i in rb and i in mb]
        ab = [abs(e) for _, e in errs]
        report["absolute"][s] = {"items": len(ids), "compared": len(errs),
                                 "median": round(pct(ab, 0.5), 4) if ab else None,
                                 "p95": round(pct(ab, 0.95), 4) if ab else None,
                                 "geomean_ratio": round(math.exp(sum(math.log(1 + e) for _, e in errs) / len(errs)), 4) if errs else None,
                                 "per_item": [(i, rb[i], mb[i], round(e, 4)) for i, e in errs]}
        for i, e in errs:
            report["families"].setdefault(family(i), []).append(round(1 + e, 3))
    h = report["absolute"].get("holdout")
    report["absolute_verdict"] = None if not h or h["median"] is None else \
        ("pass" if h["median"] <= 0.10 and h["p95"] <= 0.20 else "fail")
    for v in sorted(set(rtl) & set(model) - {"base"}):
        report["change"][v] = {s: change(rb, rtl[v], mb, model[v], ids) for s, ids in sets.items()}
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    for s, r in report["absolute"].items():
        print(f"absolute {s:9} {r['compared']}/{r['items']}  median {r['median']}  p95 {r['p95']}  geomean model/RTL {r['geomean_ratio']}")
    print("absolute verdict (holdout):", report["absolute_verdict"])
    for v, by in report["change"].items():
        for s, c in by.items():
            print(f"change {v:9} {s:8} items {c['items']:2}  changed>=3% {c['changed_3pct']:2}  dir {c['direction_rate']}  "
                  f"captured {c['captured_gain']}  G rtl {c['geomean_rtl']} model {c['geomean_model']}  "
                  f"false {len(c['false_gains'])}  -> {c['verdict']}")
    print("families (model/RTL):", "  ".join(f"{k}:{v}" for k, v in sorted(report["families"].items())))


if __name__ == "__main__":
    main()
