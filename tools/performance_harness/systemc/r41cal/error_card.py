#!/usr/bin/env python3
"""The model's error card from an acceptance.py report: per family, how far the model is from the RTL,
and per knob, whether the model can be trusted to rank that change.

    python3 error_card.py REPORT.json --out ERROR_CARD.md [--note TEXT]

Families: model/RTL on the base, calibration (visible) and holdout items apart.
Knobs: the change verdict on the holdout where the holdout was measured, else on
the visible items, said so. A failing knob reads "model blind: RTL only": an
agent must measure that change on the RTL, not on the model.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from acceptance import family

KNOBS = {
    "lsu2": "LSU_INFLIGHT 1 -> 2 (word units behind the hybrid LSU)",
    "sets64": "SETS 32 -> 64 (home cache sets)",
    "vrfreg0": "MTX_VRF_REG 1 -> 0 (matrix VRF read register)",
    "combo": "0-8: all three above together",
    "compiler": "compiler: TNCA_TILED_POINTWISE 0 -> 3 and TNCA_BANK_ALLOC 0 -> 1",
}
NOTES = {
    "compiler": "Fails on op.linear small only (RTL 1.038, model 1.000): TNCA_BANK_ALLOC renumbers vector "
                "registers and the RTL gains from fewer VRF bank conflicts; the model has no VRF bank ports beyond "
                "one instruction's sources, so it cannot see register placement. The tiled pointwise lowering is "
                "captured. Read: bank-aware register allocation is model blind, RTL only.",
}


def ratios(rows):
    return [(i, m / r) for i, r, m, _ in rows]


def fmt(xs):
    return ", ".join(f"{x:.3f}" for x in sorted(xs)) if xs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rep = json.loads(a.report.read_text())
    ab = rep["absolute"]
    fam = defaultdict(lambda: defaultdict(list))
    for s, r in ab.items():
        for i, x in ratios(r["per_item"]):
            fam[i if family(i) == "example" else family(i)][s].append(x)
    sets = sorted(ab)
    out = ["# R41 SystemC model: error card", "",
           "Model `configs/r41-sim.cfg` against RTL simulation cycles of the same programs (three active cores, "
           "the evaluator's measurement path). Ratio = model / RTL; below 1 the model is fast.", ""]
    if a.note:
        out += [a.note, ""]
    out += ["## Absolute (author's rule: holdout median <= 10 %, p95 <= 20 %)", "",
            "| items | compared | median abs error | p95 | geomean model/RTL |", "|---|---|---|---|---|"]
    for s in sets:
        r = ab[s]
        out.append(f"| {s} | {r['compared']}/{r['items']} | {r['median']:.1%} | {r['p95']:.1%} | {r['geomean_ratio']:.3f} |")
    out += ["", f"Verdict on the holdout: **{rep['absolute_verdict']}**.", "",
            "## Families", "", "| family | " + " | ".join(sets) + " | worst |", "|---|" + "---|" * (len(sets) + 1)]
    for f in sorted(fam):
        allx = [x for s in sets for x in fam[f][s]]
        worst = max(allx, key=lambda x: abs(x - 1))
        out.append(f"| {f} | " + " | ".join(fmt(fam[f][s]) for s in sets) + f" | {worst - 1:+.1%} |")
    out += ["", "## Knobs (change acceptance)", "",
            "Rules per knob, against the base: direction right on >= 90 % of items the RTL moves by >= 3 %; "
            "captured gain in [0.75, 1.25]; no model change > 3 % where the RTL moves < 1 %.", "",
            "| knob | judged on | items moved >= 3 % | direction | captured gain | false gains | verdict |",
            "|---|---|---|---|---|---|---|"]
    for v in sorted(rep["change"], key=lambda v: list(KNOBS).index(v) if v in KNOBS else len(KNOBS)):
        by = rep["change"][v]
        on = "holdout" if by.get("holdout", {}).get("items") else "visible"
        c = by[on]
        verdict = {"pass": "pass", "no-signal": "no RTL effect here (model also flat)",
                   "fail": "**model blind: RTL only**"}[c["verdict"]]
        if on == "visible" and "holdout" in by:
            verdict += "; holdout not measured"
        d = f"{c['direction_right']}/{c['changed_3pct']}" if c["changed_3pct"] else "-"
        g = f"{c['captured_gain']:.3f}" if c["captured_gain"] is not None else "-"
        out.append(f"| {KNOBS.get(v, v)} | {on} | {c['changed_3pct']}/{c['items']} | {d} | {g} | "
                   f"{len(c['false_gains'])} | {verdict} |")
    out += [""] + [f"- {KNOBS.get(v, v)}: {t}" for v, t in NOTES.items() if v in rep["change"]]
    a.out.write_text("\n".join(out) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
