#!/usr/bin/env python3
"""Compare a model instruction trace (TNCA_MODEL_ITRACE) with an RTL core trace
(tools/performance_harness/align, TNCA_CORE_TRACE 1) of the same point on one core.

    python3 itrace_cmp.py RTL.trace MODEL.itrace [--top 25] [--launch N]

Launches are paired in order and each Role Context's issue records in order;
cycles are taken from the launch's L. The drift (model - RTL) after each
instruction is attributed to that instruction's pc: the table names the pcs
where the model gains or loses cycles, with the model's instruction class.
Debugging aid only; nothing here feeds the model.
"""
import argparse
from collections import defaultdict


def launches(path):
    out, cur = [], None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "L":
            cur = {"L": int(t[1]), "I": defaultdict(list), "E": None}
            out.append(cur)
        elif t[0] == "I" and cur is not None:
            cur["I"][int(t[2])].append((int(t[1]) - cur["L"], int(t[3], 16), t[4] if len(t) > 4 else ""))
        elif t[0] == "E" and cur is not None:
            cur["E"] = int(t[1]) - cur["L"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtl")
    ap.add_argument("model")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--launch", type=int)
    a = ap.parse_args()
    R, M = launches(a.rtl), launches(a.model)
    print(f"launches rtl {len(R)} model {len(M)}")
    per_pc = defaultdict(lambda: [0, 0, ""])  # pc -> [sum of drift increase, count, class]
    for n, (r, m) in enumerate(zip(R, M)):
        if a.launch is not None and n != a.launch:
            continue
        print(f"launch {n}: rtl {r['E']} model {m['E']} ratio {m['E'] / r['E']:.3f}" if r["E"] and m["E"] else f"launch {n}")
        for role in sorted(set(r["I"]) | set(m["I"])):
            ri, mi = r["I"].get(role, []), m["I"].get(role, [])
            k = 0
            while k < min(len(ri), len(mi)) and ri[k][1] == mi[k][1]:
                k += 1
            last = 0
            for j in range(k):
                d = mi[j][0] - ri[j][0]
                e = per_pc[mi[j][1]]
                e[0] += d - last
                e[1] += 1
                e[2] = mi[j][2]
                last = d
            end = f"rtl {ri[-1][0] if ri else '-'} model {mi[-1][0] if mi else '-'}"
            div = "" if k == len(ri) == len(mi) else f"  DIVERGES at #{k} (rtl {len(ri)} model {len(mi)} records)"
            print(f"  role {role}: {k} paired, last issue {end}, drift {last}{div}")
    rows = sorted(per_pc.items(), key=lambda kv: -abs(kv[1][0]))[:a.top]
    print(f"{'pc':>8} {'class':>14} {'count':>6} {'drift':>8} {'per':>7}")
    for pc, (d, c, k) in rows:
        print(f"{pc:8x} {k:>14} {c:6} {d:8} {d / c:7.2f}")


if __name__ == "__main__":
    main()
